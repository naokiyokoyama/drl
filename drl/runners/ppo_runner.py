import torch

from drl.algo import PPO
from drl.runners.base_runner import BaseRunner
from drl.utils.registry import drl_registry
from drl.utils.rollout_storage import RolloutStorage


@drl_registry.register_runner
class PPORunner(BaseRunner):
    def __init__(self, config, envs=None):
        super().__init__(config, envs)

        """ Create actor-critic """
        actor_critic_cls = drl_registry.get_actor_critic(config.ACTOR_CRITIC.name)
        self.actor_critic = actor_critic_cls.from_config(
            config, self.envs.observation_space, self.envs.action_space
        )
        self.actor_critic.to(self.device)
        print("Actor-critic architecture:\n", self.actor_critic)

        # Training-only attributes
        self.ppo = None
        self.rollouts = None  # initialize during training

    def init_train(self):
        observations = super().init_train()
        self.rollouts = RolloutStorage(
            self.config.RL.PPO.num_steps, self.num_envs, self.device, observations
        )
        scheduler_cls = drl_registry.get_scheduler(self.config.RL.scheduler.name)
        self.ppo = PPO(
            self.actor_critic,
            scheduler_cls.from_config(self.config),
            self.config.RL.PPO.clip_param,
            self.config.RL.PPO.ppo_epoch,
            self.config.RL.PPO.num_mini_batch,
            self.config.RL.PPO.value_loss_coef,
            self.config.RL.PPO.entropy_coef,
            lr=self.config.RL.PPO.lr,
            eps=self.config.RL.PPO.eps,
            max_grad_norm=self.config.RL.PPO.max_grad_norm,
        )
        return observations

    def step(self, observations):
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

        self.rollouts.insert(
            next_observations=observations,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value,
            rewards=rewards,
            next_dones=dones,
            other=other,
        )
        self.rollouts.advance_rollout()
        return observations

    def update(self, observations):
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
