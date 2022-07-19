from typing import Union, Optional

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
        action_distribution: nn.Module,
        critic: Union[NNBase, nn.Module],
        q_critic: Union[NNBase, nn.Module],
        critic_is_head: bool = False,
        head: Optional[NNBase] = None,
        normalize_obs: bool = True,
        normalize_value: bool = True,
    ):
        super().__init__(
            net=net,
            action_distribution=action_distribution,
            critic=critic,
            critic_is_head=critic_is_head,
            head=head,
            normalize_obs=normalize_obs,
            normalize_value=normalize_value,
        )
        self.q_critic = q_critic

    def evaluate_actions(self, observations, action):
        value, action_log_probs, dist = super().evaluate_actions(observations, action)
        q_terms_pred = self.q_critic(torch.cat([observations, dist.rsample()], dim=1))
        # return value, action_log_probs, dist, q_terms_pred
        return value, action_log_probs, dist

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
        critic_kwargs = {}
        if config.RUNNER.name == "EPPOTrainer":
            if config.RL.term_by_term_returns:
                num_outputs = num_reward_terms
                critic_kwargs["num_outputs"] = num_reward_terms
            elif config.RL.PPO.q_terms == 1:
                num_outputs = 1
            else:
                num_outputs = num_reward_terms + 1
        elif config.RUNNER.name == "ERPGTrainer":
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
            critic_kwargs=critic_kwargs,
            **kwargs,
        )
