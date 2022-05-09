import argparse
import os.path as osp

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


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
