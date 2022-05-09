from typing import Tuple

import numpy as np
import torch.nn as nn

from drl.utils.common import initialized_linear
from drl.utils.registry import drl_registry


def construct_mlp_base(input_size, hidden_sizes, activation="relu"):
    if activation == "relu":
        activation_layer = nn.ReLU
    elif activation == "tanh":
        activation_layer = nn.Tanh
    else:
        raise RuntimeError(f"Activation layer type {activation} not valid.")

    layers = []
    prev_size = input_size
    for out_size in hidden_sizes:
        layers.append(
            initialized_linear(int(prev_size), int(out_size), gain=np.sqrt(2))
        )
        # layers.append(activation_layer())
        layers.append(nn.ELU(alpha=1.0))
        prev_size = out_size
    mlp = nn.Sequential(*layers)

    return mlp


class NNBase(nn.Module):
    """All inheritors of this class must support these methods, and no other methods
    should be expected by the actor-critic that the NNBase is for."""

    def __init__(self, recurrent: bool, output_shape: Tuple):
        super().__init__()

        self.rnn_hx = None
        self._output_shape = output_shape
        self._recurrent = recurrent

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    def forward(self, *args, **kwargs):
        raise NotImplementedError(".forward() must be defined!")

    def from_config(self, *args, **kwargs):
        raise NotImplementedError(".from_config() must be defined!")


@drl_registry.register_nn_base
class MLPBase(NNBase):  # noqa
    def __init__(self, input_shape, hidden_sizes, activation="relu"):
        super().__init__(recurrent=False, output_shape=(hidden_sizes[-1],))
        assert len(input_shape) == 1, "MLPBase can only take 1D inputs!"
        self.mlp = construct_mlp_base(input_shape[0], hidden_sizes, activation)

    def forward(self, net_input):
        return self.mlp(net_input)

    @classmethod
    def from_config(cls, nn_config, obs_space):
        return cls(
            input_shape=obs_space.shape,
            hidden_sizes=nn_config.hidden_sizes,
            activation=nn_config.activation,
        )


@drl_registry.register_nn_base
class MLPCritic(MLPBase):  # noqa
    def __init__(self, input_shape, hidden_sizes, activation="relu"):
        all_sizes = [*hidden_sizes, 1]  # add one final layer with one output (value)
        super().__init__(input_shape, all_sizes, activation)
        # Remove the final activation layer
        self.mlp = nn.Sequential(*list(self.mlp.children())[:-1])
