from typing import Union

import torch.jit
from torch import nn as nn

from drl.nets.basic import NNBase, MLPCritic
from drl.policies.actor_critic import ActorCritic
from drl.utils.registry import drl_registry


@drl_registry.register_actor_critic
class ActorCriticQ(ActorCritic):
    """For use with EPPO, not PPO"""

    def __init__(
        self,
        net: NNBase,
        critic: Union[NNBase, nn.Module],
        q_critic: Union[NNBase, nn.Module],
        action_distribution: nn.Module,
        critic_is_head: bool = False,
    ):
        super().__init__(net, critic, action_distribution, critic_is_head)
        self.q_critic = q_critic

    def evaluate_actions(self, observations, action):
        value, action_log_probs, dist = super().evaluate_actions(observations, action)
        q_terms_pred = self.q_critic(torch.cat([observations, dist.rsample()], dim=1))
        return value, action_log_probs, dist, q_terms_pred

    def convert_to_torchscript(self):
        super().convert_to_torchscript()
        self.q_critic.convert_to_torchscript()

    @classmethod
    def from_config(
        cls,
        config,
        obs_space,
        action_space,
        critic_obs_space=None,
        num_reward_terms: int = 0,
        **kwargs,
    ):
        if config.RUNNER.name == "EPPOTrainer":
            num_outputs = num_reward_terms
        elif config.RUNNER.name == "RPGTrainer":
            num_outputs = 1
        else:
            raise RuntimeError(f"Unsure Q net output size for {config.RUNNER.name}!")
        q_critic = MLPCritic(
            (obs_space.shape[0] + action_space.shape[0],),
            config.ACTOR_CRITIC.q_critic.hidden_sizes,
            num_outputs,
            config.ACTOR_CRITIC.q_critic.activation,
        )
        return super().from_config(
            config=config,
            obs_space=obs_space,
            action_space=action_space,
            critic_obs_space=critic_obs_space,
            q_critic=q_critic,
            **kwargs,
        )
