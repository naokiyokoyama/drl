from typing import Union, Optional

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
        action_distribution: nn.Module,
        critic: NNBase,
        q_critic: Union[NNBase, nn.Module],
        critic_is_head: bool = True,
        head: Optional[NNBase] = None,
        normalize_obs: bool = True,
    ):
        super().__init__(
            net=net,
            action_distribution=action_distribution,
            critic=critic,
            critic_is_head=critic_is_head,
            head=head,
            normalize_obs=normalize_obs,
        )
        self.q_critic = q_critic

    def reparameterize_action(self, observations, actions):
        # Don't update the normalizer stats when this method is used
        prev_state = self.obs_normalizer.training
        self.obs_normalizer.training = False

        features = self.net(self._norm_obs(observations))
        dist = self.action_distribution(features)

        self.obs_normalizer.training = prev_state

        eps = (actions - dist.mu) / dist.sigma
        reparam_action = dist.mu + dist.sigma * eps.detach()
        return reparam_action

    def convert_to_torchscript(self):
        super().convert_to_torchscript()
        self.q_critic.convert_to_torchscript()

    @classmethod
    def from_config(
        cls, config, obs_space, action_space, critic_obs_space=None, **kwargs
    ):
        ac_cfg = config.ACTOR_CRITIC
        q_critic_cls = drl_registry.get_nn_base(ac_cfg.q_critic.name)
        q_critic = q_critic_cls.from_config(
            config, ac_cfg.q_critic, obs_space, action_space
        )
        return super().from_config(
            config=config,
            obs_space=obs_space,
            action_space=action_space,
            critic_obs_space=critic_obs_space,
            q_critic=q_critic,
            **kwargs,
        )
