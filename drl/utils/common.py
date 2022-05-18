import argparse
from typing import Union

import numpy as np
import os.path as osp

import torch
from torch import nn
from yacs.config import CfgNode as CN

UTILS_DIR = osp.dirname(osp.abspath(__file__))
DRL_DIR = osp.join(osp.dirname(UTILS_DIR))
CONFIGS_DIR = osp.join(DRL_DIR, "configs")
DEFAULT_CONFIG_FILE = osp.join(CONFIGS_DIR, "default.yaml")


def get_default_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opts", nargs="*", default=[])
    return parser


def construct_config(opts=None):
    if opts is None:
        opts = []
    config = CN()
    config.set_new_allowed(True)
    config.merge_from_file(DEFAULT_CONFIG_FILE)
    config.merge_from_list(opts)

    return config


def initialized_linear(in_features, out_features, gain, bias=0):
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)

    return layer


class MeanReturns:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.total_returns = None
        self.curr_returns = None

    def update(
        self,
        rewards: Union[np.ndarray, torch.Tensor],
        dones: Union[np.ndarray, torch.Tensor],
    ):
        rewards_tensor = torch.as_tensor(rewards).flatten()
        if self.curr_returns is None:
            self.curr_returns = torch.zeros_like(rewards_tensor)

        dones_mask = torch.as_tensor(dones).bool().flatten()

        # Update buffer of last window_size total returns
        if self.total_returns is None:
            self.total_returns = self.curr_returns[dones_mask]
        else:
            self.total_returns = torch.cat(
                [self.total_returns, self.curr_returns[dones_mask]]
            )
        self.total_returns = self.total_returns[-self.window_size :]

        self.curr_returns[dones_mask] = 0
        self.curr_returns += rewards_tensor

    def mean(self):
        if self.total_returns is None or self.total_returns.nelement() == 0:
            return 0
        return self.total_returns.mean().item()