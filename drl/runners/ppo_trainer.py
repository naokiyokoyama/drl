import torch

from drl.algo import PPO
from drl.runners.base_runner import BaseTrainer
from drl.utils.registry import drl_registry
from drl.utils.rollout_storage import RolloutStorage


@drl_registry.register_runner
class PPOTrainer(BaseTrainer):
    def __init__(self, config, envs=None):
        super().__init__(config, envs)
        self.ppo = PPO.from_config(self.config, self.actor_critic)
        self.rollouts = None  # initialize during training

    def init_train(self):
        observations = super().init_train()
        self.rollouts = RolloutStorage.from_config(
            self.config, self.num_envs, self.device, observations
        )
        return observations

    def step(self, observations):
        with torch.no_grad():
            (
                value,
                actions,
                action_log_probs,
                other,
            ) = self.actor_critic.act(observations)

        observations, rewards, dones, infos = self.step_envs(actions)
        other.update(infos)

        self.rollouts.insert(
            next_observations=observations,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value,
            rewards=rewards,
            next_dones=dones,
            other=other,
        )
        return observations

    def update(self, observations):
        with torch.no_grad():
            next_value = self.actor_critic.get_value(observations)
        self.rollouts.compute_returns(next_value)
        self.write_data.update(self.ppo.update(self.rollouts))
