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
        if hasattr(m, "weight"):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))


@drl_registry.register_init_layer
def debug_random(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "bias"):
            torch.nn.init.zeros_(m.bias)
        if hasattr(m, "weight"):
            torch.manual_seed(1)
            torch.nn.init.uniform_(m.weight)
