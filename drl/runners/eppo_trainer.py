import torch

from drl.algo.eppo import EPPO
from drl.runners.ppo_trainer import PPOTrainer
from drl.utils.registry import drl_registry

from drl.utils.common import get_num_reward_terms


@drl_registry.register_runner
class EPPOTrainer(PPOTrainer):
    algo_cls = EPPO

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def instantiate_actor_critic(self, actor_critic_cls):
        # Count the amount of reward terms by doing a dummy step
        num_reward_terms = get_num_reward_terms(self.envs, self.num_envs)
        self.config.ACTOR_CRITIC.head["num_reward_terms"] = num_reward_terms

        # Update obs because counting the reward terms requires env resetting
        self.initial_observations = self.preprocess_observations(self.envs.reset())

        return actor_critic_cls.from_config(
            self.config,
            self.envs.observation_space,
            self.envs.action_space,
        )

    def prepare_rollouts(self, observations):
        # super().prepare_rollouts(observations)
        with torch.no_grad():
            next_value_terms = self.actor_critic.head(self.actor_critic.features)
        self.rollouts.compute_returns(next_value_terms, term_by_term_returns=True)

        with torch.no_grad():
            next_value = next_value_terms.sum(1, keepdims=True)
        self.rollouts.compute_returns(next_value)
