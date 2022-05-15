import torch

from drl.utils.registry import drl_registry
from drl.utils.common import MeanReturns
import tqdm


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
            self.num_envs = None  # define later in init_train

        self.mean_returns = MeanReturns()
        self.num_updates = None
        self.actor_critic = None

    def train(self):
        observations = self.init_train()
        for _ in tqdm.trange(self.num_updates):
            for step in range(self.config.RL.PPO.num_steps):
                observations = self.step(observations)
            self.update(observations)

    def init_train(self):
        self.num_updates = self.config.NUM_UPDATES
        observations = self.envs.reset()["obs"]
        if self.num_envs is None:
            self.num_envs = observations.shape[0]
        return observations

    def step(self, observations):
        raise NotImplementedError

    def update(self, observations):
        raise NotImplementedError


    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
