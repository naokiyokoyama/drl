import numpy as np
import torch
from torch import nn

from drl.utils.registry import drl_registry


@drl_registry.register_init_layer
def zero_all_bias(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "bias"):
            torch.nn.init.zeros_(m.bias)


@drl_registry.register_init_layer
def orthogonal_all_weights(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "weights"):
            torch.nn.init.orthogonal_(m.weights, gain=np.sqrt(2))
