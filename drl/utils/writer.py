import os
import os.path as osp
import shutil

from torch.utils.tensorboard import SummaryWriter


class Writer(SummaryWriter):
    def __init__(self, path, erase_existing_tb, *args, **kwargs):
        if erase_existing_tb and osp.isdir(path):
            shutil.rmtree(path)
        super().__init__(path, *args, **kwargs)

        # Generate path to a .csv file, removing it if it already exists
        self.csv_path = osp.join(path, osp.basename(path) + ".csv")
        if osp.isfile(self.csv_path):
            os.remove(self.csv_path)
        self.csv_header = []

    def add_multi_scalars(self, data, idx):
        new_data = [idx]
        record_header = len(self.csv_header) == 0
        for k, v in data.items():
            self.add_scalar(k, v, idx)
            new_data.append(v)
            if record_header:
                self.csv_header.append(k)
        new_data = self.list_to_csv_line(new_data)
        if record_header:
            new_data = self.list_to_csv_line(["idx"] + self.csv_header) + new_data
        with open(self.csv_path, "a") as f:
            f.write(new_data)

    @staticmethod
    def list_to_csv_line(in_list):
        return ",".join([str(i) for i in in_list]) + "\n"

    @classmethod
    def from_config(cls, config):
        return cls(config.TENSORBOARD_DIR, config.ERASE_EXISTING_TB)
