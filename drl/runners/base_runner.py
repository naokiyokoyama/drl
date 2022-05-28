import random
import time

import numpy as np
import torch
import tqdm

from drl.utils.common import MeanReturns
from drl.utils.registry import drl_registry
from drl.utils.writer import Writer


class BaseRunner:
    """Base class for Trainers and Evaluators"""

    def __init__(self, config, envs=None):
        torch.set_num_threads(1)
        self.config = config
        self.set_seed(config.SEED)
        self.device = torch.device("cuda:0" if config.CUDA else "cpu")
        if self.config.TENSORBOARD_DIR == "":
            self.writer = None
        else:
            self.writer = Writer.from_config(self.config)
        self.write_data = {}

        self.preprocess_observations = drl_registry.get_obs_preprocessor(
            config.OBS_PREPROCESSOR
        )

        if envs is None:
            self.envs = drl_registry.get_envs(config.ENVS_NAME)
            self.num_envs = config.NUM_ENVIRONMENTS
        else:
            self.envs = envs
            self.num_envs = None  # define later in init_train or init_eval
        self.initial_observations = self.preprocess_observations(self.envs.reset())
        if self.num_envs is None:
            self.num_envs = self.initial_observations.shape[0]

        self.actor_critic = self.load_actor_critic()

    def load_actor_critic(self):
        """Create actor-critic"""
        actor_critic_cls = drl_registry.get_actor_critic(self.config.ACTOR_CRITIC.name)
        actor_critic = self.instantiate_actor_critic(actor_critic_cls)
        actor_critic.to(self.device)
        if self.config.USE_TORCHSCRIPT:
            actor_critic.convert_to_torchscript()
        print("Actor-critic architecture:\n", actor_critic)
        # TODO: Support loading pre-trained weights
        return actor_critic

    def instantiate_actor_critic(self, actor_critic_cls):
        return actor_critic_cls.from_config(
            self.config, self.envs.observation_space, self.envs.action_space
        )

    def write(self, idx):
        if self.writer is not None:
            self.writer.add_multi_scalars(self.write_data, idx)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)


class BaseTrainer(BaseRunner):
    def __init__(self, config, envs=None):
        super().__init__(config, envs)
        self.mean_returns = MeanReturns()
        self.update_idx = 0
        for i in self.config.INIT_LAYERS:
            drl_registry.get_init_layer(i)(self.actor_critic)

    def train(self):
        observations = self.initial_observations
        frames_per_update = self.config.RL.num_steps * self.num_envs
        step_idx = 0
        for _ in tqdm.trange(self.config.NUM_UPDATES):
            start_time = time.time()
            self.write_data = {}
            for step in range(self.config.RL.num_steps):
                observations = self.step(observations)
            self.update(observations)
            self.write_data["rewards/step"] = mean_return = self.mean_returns.mean()
            self.write(step_idx)
            self.update_idx += 1
            step_idx += frames_per_update
            print("mean_returns:", mean_return)
            print(f"fps: {frames_per_update / (time.time() - start_time):.2f}")

    def step_envs(self, actions):
        observations, rewards, dones, infos = self.envs.step(actions)
        observations = self.preprocess_observations(observations)
        self.mean_returns.update(rewards, dones)
        rewards *= self.config.RL.reward_scale
        return observations, rewards, dones, infos

    def step(self, observations):
        raise NotImplementedError

    def update(self, observations):
        raise NotImplementedError
