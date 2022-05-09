import torch

from drl.utils.registry import drl_registry


class BaseRunner:
    def __init__(self, config, envs=None):
        self.config = config

        torch.set_num_threads(1)
        self.device = torch.device("cuda:0" if config.CUDA else "cpu")

        if envs is None:
            self.envs = drl_registry.get_envs(config.ENVS_NAME)
            self.num_envs = config.NUM_ENVIRONMENTS
        else:
            self.envs = envs
            self.num_envs = None  # define later

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
