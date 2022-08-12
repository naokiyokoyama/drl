from functools import partial
from typing import Tuple, Union

import gym
import torch
import torch.nn as nn

from drl.utils.registry import drl_registry
from drl.utils.running_mean_std import RunningMeanStd


class NNBase(nn.Module):
    """All inheritors of this class must support these methods, and no other methods
    should be expected by the actor-critic that the NNBase is for."""

    def __init__(self, recurrent: bool, output_shape: Tuple):
        super().__init__()

        self.rnn_hx = None
        self.output_shape = output_shape
        self._recurrent = recurrent

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.output_shape if self._recurrent else (1,)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(".forward() must be defined!")

    def from_config(self, *args, **kwargs):
        raise NotImplementedError(".from_config() must be defined!")


ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": partial(nn.ELU, alpha=1.0)}


def construct_mlp_base(input_size, hidden_sizes, activation="elu"):
    activation_layer = ACTIVATIONS[activation]
    layers = []
    prev_size = input_size
    for out_size in hidden_sizes:
        layers.append(nn.Linear(int(prev_size), int(out_size)))
        layers.append(activation_layer())
        prev_size = out_size
    return nn.Sequential(*layers)


@drl_registry.register_nn_base
class MLPBase(NNBase):  # noqa
    def __init__(self, input_shape, hidden_sizes, activation="elu"):
        super().__init__(recurrent=False, output_shape=(hidden_sizes[-1],))
        assert len(input_shape) == 1, "MLPBase can only take 1D inputs!"
        self.mlp = construct_mlp_base(input_shape[0], hidden_sizes, activation)
        self.input_shape = input_shape

    def forward(self, net_input):
        return self.mlp(net_input)

    def convert_to_torchscript(self):
        self.mlp = torch.jit.script(self.mlp)

    def get_other(self, *args, **kwargs):
        return {}

    @classmethod
    def from_config(
        cls, config, nn_config, input_space: Union[Tuple, gym.Space], *args, **kwargs
    ):
        if not isinstance(input_space, tuple):
            input_shape = input_space.shape
        else:
            input_shape = input_space
        return cls(
            input_shape=input_shape,
            hidden_sizes=nn_config.hidden_sizes,
            activation=nn_config.activation,
            **kwargs,
        )


PRED2LABEL = {
    "value_preds": "returns",
    "value_terms_preds": "return_terms",
    "rewards_preds": "rewards",
    "reward_terms_preds": "reward_terms",
    "adv_preds": "advantages",
    "adv_terms_preds": "advantage_terms",
}


@drl_registry.register_nn_base
class MLPCritic(MLPBase):
    def __init__(
        self,
        input_shape,
        hidden_sizes,
        output_types=("value_preds",),
        num_reward_terms=None,
        activation="elu",
        normalize_value=False,
    ):
        self.output_types = output_types
        self.num_reward_terms = num_reward_terms
        self.section_sizes = []  # gets assigned by _calc_num_outputs()
        all_sizes = [*hidden_sizes, self._calc_num_outputs()]
        super().__init__(input_shape, all_sizes, activation)

        # Remove the final activation layer
        layers = list(self.mlp.children())
        self.mlp = nn.Sequential(*layers[:-1]) if len(layers) > 2 else layers[0]

        self.value_normalizer = None
        self.value_terms_normalizer = None
        self._setup_normalizers(normalize_value)
        self.pred2label = {k: v for k, v in PRED2LABEL.items() if k in output_types}

    def forward(self, x, unnorm: bool = True):
        x = super().forward(x)
        x = torch.split(x, self.section_sizes, dim=1)
        out = {}
        for key, value in zip(self.output_types, x):
            if key in self.normalizer_map and unnorm:
                out[key] = self.normalizer_map[key](value, unnorm=True)
            else:
                out[key] = value
        return out

    @classmethod
    def from_config(
        cls, config, nn_config, input_space: Union[Tuple, gym.Space], *args, **kwargs
    ):
        if nn_config.is_head:
            input_space = (config.ACTOR_CRITIC.net.hidden_sizes[-1],)
        return super().from_config(
            config=config,
            nn_config=nn_config,
            input_space=input_space,
            normalize_value=nn_config.normalize_value,
            output_types=nn_config.output_types,
            num_reward_terms=config.get("num_reward_terms", None),
            **kwargs,
        )

    def _calc_num_outputs(self):
        """We assume any output type with the word "terms" in it needs as many terms
        as there are reward terms. Otherwise, we assume it only has one term."""
        num_outputs = 0
        self.section_sizes = []
        for out_type in self.output_types:
            if "terms" in out_type:
                assert self.num_reward_terms is not None
                section_size = self.num_reward_terms
            else:
                section_size = 1
            num_outputs += section_size
            self.section_sizes.append(section_size)

        return num_outputs

    def _setup_normalizers(self, normalize_value):
        self.normalizer_map = {}
        if not normalize_value:
            return

        for out_type, section_size in zip(self.output_types, self.section_sizes):
            if out_type == "value_preds":
                self.value_normalizer = RunningMeanStd((section_size,))
                self.normalizer_map[out_type] = self.value_normalizer
            elif out_type == "value_terms_preds":
                self.value_terms_normalizer = RunningMeanStd((section_size,))
                self.normalizer_map[out_type] = self.value_terms_normalizer
