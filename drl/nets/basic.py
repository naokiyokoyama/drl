from functools import partial
from typing import Tuple

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
    def from_config(cls, nn_config, obs_space=None, input_shape=None, *args, **kwargs):
        return cls(
            input_shape=input_shape if input_shape is not None else obs_space.shape,
            hidden_sizes=nn_config.hidden_sizes,
            activation=nn_config.activation,
            **kwargs,
        )


@drl_registry.register_nn_base
class MLPCritic(MLPBase):  # noqa
    target_key = "returns"

    def __init__(
        self,
        input_shape,
        hidden_sizes,
        num_outputs=1,
        activation="elu",
        normalize_value=False,
    ):
        all_sizes = [*hidden_sizes, num_outputs]
        super().__init__(input_shape, all_sizes, activation)
        # Remove the final activation layer
        layers = list(self.mlp.children())
        self.mlp = nn.Sequential(*layers[:-1]) if len(layers) > 2 else layers[0]
        self.normalizer = RunningMeanStd(self.output_shape) if normalize_value else None

    def forward(self, x, unnorm: bool = True):
        x = super().forward(x)
        if self.normalizer is not None and unnorm:
            return self.normalizer(x, unnorm=True)
        return x

    @classmethod
    def from_config(cls, nn_config, obs_space, net, *args, **kwargs):  # noqa
        input_shape = net.output_shape if nn_config.is_head else None
        return super().from_config(
            nn_config,
            obs_space,
            input_shape,
            normalize_value=nn_config.normalize_value,
            **kwargs,
        )


@drl_registry.register_nn_base
class MLPCriticTermsHead(MLPCritic):  # noqa
    target_key = "return_terms"

    @classmethod
    def from_config(cls, nn_config, obs_space, net, *args, **kwargs):
        assert "num_reward_terms" in nn_config
        return super().from_config(
            nn_config,
            obs_space,
            net,
            num_outputs=nn_config.num_reward_terms,
        )

    def get_other(self, features):
        return {"value_terms_preds": self.forward(features)}
