import time

import torch
import tqdm

from drl.utils.common import MeanReturns
from drl.utils.registry import drl_registry
from drl.utils.writer import Writer

class BaseRunner:
    """Base class for Trainers and Evaluators"""
    def __init__(self, config, envs=None):
        self.config = config

        torch.set_num_threads(1)
        self.device = torch.device("cuda:0" if config.CUDA else "cpu")

        if envs is None:
            self.envs = drl_registry.get_envs(config.ENVS_NAME)
            self.num_envs = config.NUM_ENVIRONMENTS
        else:
            self.envs = envs
            self.num_envs = None  # define later in init_train or init_eval

        if self.config.TENSORBOARD_DIR == "":
            self.writer = None
        else:
            self.writer = Writer.from_config(self.config)
        self.write_data = {}

        """ Create actor-critic """
        actor_critic_cls = drl_registry.get_actor_critic(config.ACTOR_CRITIC.name)
        self.actor_critic = actor_critic_cls.from_config(
            config, self.envs.observation_space, self.envs.action_space
        )
        self.actor_critic.to(self.device)
        if config.USE_TORCHSCRIPT:
            self.actor_critic.convert_to_torchscript()
        print("Actor-critic architecture:\n", self.actor_critic)

        self.preprocess_observations = drl_registry.get_obs_preprocessor(
            config.OBS_PREPROCESSOR
        )


class BaseTrainer(BaseRunner):
    def __init__(self, config, envs=None):
        super().__init__(config, envs)
        self.mean_returns = MeanReturns()
        self.update_idx = 0
        self.step_idx = 0

    def train(self):
        observations = self.init_train()
        frames_per_step = self.config.RL.PPO.num_steps * self.num_envs
        for _ in tqdm.trange(self.config.NUM_UPDATES):
            start_time = time.time()
            self.write_data = {}
            for step in range(self.config.RL.PPO.num_steps):
                observations = self.step(observations)
            self.update(observations)
            self.write()
            self.update_idx += 1
            self.step_idx += self.config.RL.PPO.num_steps * self.num_envs
            print("mean_returns:", self.mean_returns.mean())
            print(f"fps: {frames_per_step / (time.time() - start_time):.2f}")

    def init_train(self):
        observations = self.envs.reset()["obs"]
        if self.num_envs is None:
            self.num_envs = observations.shape[0]
        return observations

    def step_envs(self, actions):
        observations, rewards, dones, infos = self.envs.step(actions)
        observations = self.preprocess_observations(observations)
        self.mean_returns.update(rewards, dones)
        rewards *= self.config.RL.reward_scale
        return observations, rewards, dones, infos

    def write(self):
        if self.writer is not None:
            self.writer.add_multi_scalars(self.write_data, self.step_idx)

    def step(self, observations):
        raise NotImplementedError

    def update(self, observations):
        raise NotImplementedError

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
