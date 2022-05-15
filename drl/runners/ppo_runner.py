import numpy as np
import torch
import tqdm

from drl.algo import PPO
from drl.runners.base_runner import BaseRunner
from drl.utils.common import update_linear_schedule
from drl.utils.registry import drl_registry
from drl.utils.rollout_storage import RolloutStorage


@drl_registry.register_runner
class PPORunner(BaseRunner):
    def __init__(self, config, envs=None):
        super().__init__(config, envs)

        """ Create actor-critic """
        obs_space = self.envs.observation_space
        action_space = self.envs.action_space

        actor_critic_cls = drl_registry.get_actor_critic(config.ACTOR_CRITIC.name)
        self.actor_critic = actor_critic_cls.from_config(
            config, obs_space, action_space
        )
        self.actor_critic.to(self.device)
        print("Actor-critic architecture:\n", self.actor_critic)

        """ Setup PPO """
        self.ppo = PPO(
            self.actor_critic,
            config.RL.PPO.clip_param,
            config.RL.PPO.ppo_epoch,
            config.RL.PPO.num_mini_batch,
            config.RL.PPO.value_loss_coef,
            config.RL.PPO.entropy_coef,
            lr=config.RL.PPO.lr,
            eps=config.RL.PPO.eps,
            max_grad_norm=config.RL.PPO.max_grad_norm,
        )
        self.rollouts = None  # initialize during training

    def train(self):
        # Prep for training loop
        observations = self.envs.reset()["obs"]  # TODO: Fix this hack
        if self.num_envs is None:
            self.num_envs = observations.shape[0]

        self.num_updates = 1000  # TODO: Fix this hack

        self.rollouts = RolloutStorage(
            self.config.RL.PPO.num_steps, self.num_envs, self.device
        )
        self.rollouts.to(self.device)
        self.rollouts.insert_initial_obs(observations)

        for update_idx in tqdm.trange(self.num_updates):
            self._update_lr(update_idx)
            for step in range(self.config.RL.PPO.num_steps):
                # Sample actions
                with torch.no_grad():
                    (
                        value,
                        actions,
                        action_log_probs,
                        other,
                    ) = self.actor_critic.act(observations)

                # Observe reward and next obs
                # TODO: support action transformation
                observations, rewards, dones, infos = self.envs.step(actions)
                observations = observations["obs"]

                self.mean_returns.update(rewards, dones)

                rewards *= self.config.RL.reward_scale

                # If done then clean the history of observations.
                self.rollouts.insert(
                    next_observations=observations,
                    actions=actions,
                    action_log_probs=action_log_probs,
                    value_preds=value,
                    rewards=rewards,
                    next_dones=dones,
                )
                self.rollouts.advance_rollout()

            with torch.no_grad():
                next_value = self.actor_critic.get_value(observations)

            self.rollouts.compute_returns(
                next_value,
                self.config.RL.use_gae,
                self.config.RL.gamma,
                self.config.RL.tau,
            )

            ppo_info = self.ppo.update(self.rollouts)
            print("mean_returns:", self.mean_returns.mean())

    def _update_lr(self, update_idx):
        if self.config.RL.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(
                self.ppo.optimizer,
                update_idx,
                self.num_updates,
                self.config.RL.PPO.lr,
            )
        if self.config.RL.PPO.use_linear_clip_decay:
            self.ppo.clip_param = self.config.RL.PPO.clip_param * (
                1 - self.percent_done()
            )
