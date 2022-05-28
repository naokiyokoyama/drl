import copy
from typing import Union

import torch.jit
from torch import nn as nn

from drl.nets.basic import NNBase
from drl.policies.actor_critic import ActorCritic
from drl.utils.registry import drl_registry


@drl_registry.register_actor_critic
class ActorCriticQ(ActorCritic):
    """For use with EPPO, not PPO"""

    def __init__(
        self,
        net: NNBase,
        critic: Union[NNBase, nn.Module],
        action_distribution: nn.Module,
        num_reward_terms: int,
        critic_is_head: bool = False,
        **kwargs,
    ):
        super().__init__(net, critic, action_distribution, critic_is_head)
        self.q_critic = self.generate_q_critic(num_reward_terms)

    def generate_q_critic(self, num_reward_terms):
        action_dim = self.action_distribution.num_actions
        state_dim = self.net.input_shape[0]
        first_hidden_size = self.net.mlp[0].out_features
        net_copy = copy.deepcopy(self.net)
        q_critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, first_hidden_size),
            *net_copy.mlp[1:],
            nn.Linear(self.net.output_shape[0], num_reward_terms + 1),
        )
        return q_critic

    def evaluate_actions(self, observations, action):
        (
            value,
            dist,
            _,  # other
            value_terms_pred,
            q_value_terms_pred,
            state_action,
        ) = self._process_observations(observations, get_terms=True)
        action_log_probs = dist.log_probs(action)
        distribution_entropy = dist.entropy().sum(dim=-1)

        return (
            value,
            action_log_probs,
            distribution_entropy,
            value_terms_pred,
            q_value_terms_pred,
            state_action,
        )

    def get_value(self, observations, features=None, get_terms=False):
        value_terms_pred = super().get_value(observations, features)
        value = value_terms_pred.sum(1, keepdim=True)
        if get_terms:
            return value, value_terms_pred
        return value

    def _process_observations(self, observations, get_terms=False):
        features = self.net(observations)
        value = self.get_value(observations, get_terms=get_terms)
        dist = self.action_distribution(features)
        other = self._get_other()
        if get_terms:
            new_action = dist.sample(rsample=True)
            state_action = torch.cat([observations, new_action], dim=1)
            q_value_terms_pred = self.q_critic(state_action)
            value, value_terms_pred = value
            return (
                value,
                dist,
                other,
                value_terms_pred,
                q_value_terms_pred,
                state_action,
            )
        return value, dist, other

    def convert_to_torchscript(self):
        super().convert_to_torchscript()
        self.q_critic = torch.jit.script(self.q_critic)

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
        assert num_reward_terms > 0
        return super().from_config(
            config=config,
            obs_space=obs_space,
            action_space=action_space,
            critic_obs_space=critic_obs_space,
            num_reward_terms=num_reward_terms,
            **kwargs,
        )
