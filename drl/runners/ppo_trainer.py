import torch

from drl.algo.ppo import PPO
from drl.runners.base_runner import BaseTrainer
from drl.utils.registry import drl_registry
from drl.utils.rollout_storage import RolloutStorage


@drl_registry.register_runner
class PPOTrainer(BaseTrainer):
    algo_cls = PPO

    def __init__(self, config, envs=None):
        super().__init__(config, envs)
        self.algo = self.algo_cls.from_config(self.config, self.actor_critic)
        self.rollouts = RolloutStorage.from_config(
            self.config, self.num_envs, self.device, self.initial_observations
        )
        self.curr_step = 0

    def step(self, observations):
        with torch.no_grad():
            (
                value,
                actions,
                action_log_probs,
                other,
            ) = self.actor_critic.act(observations)

        observations, rewards, dones, infos = self.step_envs(actions)
        self.curr_step += 1
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
        self.prepare_rollouts(observations)
        self.write_data.update(self.algo.update(self.rollouts))

    def prepare_rollouts(self, observations):
        with torch.no_grad():
            values_dict = self.actor_critic.get_value(observations, all_values=True)
        self.rollouts.compute_returns(values_dict["value_preds"])
        if "value_terms_preds" in values_dict:
            self.rollouts.compute_returns(
                values_dict["value_terms_preds"], term_by_term=True
            )
