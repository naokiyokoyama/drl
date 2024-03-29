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
        self.config["num_reward_terms"] = num_reward_terms

        # Reset envs and update obs because dummy step was taken
        self.initial_observations = self.preprocess_observations(self.envs.reset())

        return actor_critic_cls.from_config(
            self.config,
            self.envs.observation_space,
            self.envs.action_space,
        )
