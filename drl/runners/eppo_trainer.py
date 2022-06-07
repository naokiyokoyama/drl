import torch

from drl.algo.eppo import EPPO
from drl.policies.actor_critic_q import ActorCriticQ
from drl.runners.ppo_trainer import PPOTrainer
from drl.utils.registry import drl_registry


@drl_registry.register_runner
class EPPOTrainer(PPOTrainer):
    algo_cls = EPPO

    def instantiate_actor_critic(self, actor_critic_cls):
        assert issubclass(
            actor_critic_cls, ActorCriticQ
        ), "EPPO requires actor-critic to be ActorCriticQ (or inherited from it)!"

        # Count the amount of reward terms by doing a dummy step
        actions = torch.as_tensor(self.envs.action_space.sample())
        if actions.dim() == 1:
            actions = actions.repeat(self.num_envs, 1)
        _, _, _, infos = self.envs.step(actions)
        assert "reward_terms" in infos, "Key 'reward_terms' must be in info for EPPO!"
        num_reward_terms = infos["reward_terms"].shape[1]

        self.initial_observations = self.preprocess_observations(self.envs.reset())

        return actor_critic_cls.from_config(
            self.config,
            self.envs.observation_space,
            self.envs.action_space,
            num_reward_terms=num_reward_terms,
        )
