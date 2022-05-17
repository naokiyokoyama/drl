import shutil
import os.path as osp

from torch.utils.tensorboard import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, path, erase_existing_tb, *args, **kwargs):
        if erase_existing_tb and osp.isdir(path):
            shutil.rmtree(path)
        super().__init__(path, *args, **kwargs)

    def add_multi_scalars(self, data, idx):
        for k, v in data.items():
            self.add_scalar(k, v, idx)

    @classmethod
    def from_config(cls, config):
        return cls(config.TENSORBOARD_DIR, config.ERASE_EXISTING_TB)
