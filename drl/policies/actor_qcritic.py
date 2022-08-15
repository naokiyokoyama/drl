from typing import Optional

import torch
from drl.utils.tensor_dict import TensorDict

from drl.policies.actor_critic import ActorCritic
from drl.utils.registry import drl_registry


@drl_registry.register_actor_critic
class ActorQCritic(ActorCritic):

    """
    At minimum, this class represents a policy and a critic. The critic can be a head
    that shares parameters with the policy, or it can be a separate network. Optionally,
    a head that does NOT act as a critic can be added to the policy. In such case, the
    actual critic must be a separate network.
    """

    def get_value(
        self,
        observations,
        features: Optional[torch.Tensor] = None,
        unnorm_value: bool = True,
        all_values: bool = False,
        norm_obs: bool = True,
        action: Optional[torch.Tensor] = None,
    ):
        if norm_obs:
            observations = self._norm_obs(observations)
        if features is None:
            features = self.net(observations)
        state = features if self.critic_is_head else observations
        if action is None:
            with torch.no_grad():
                action = self.action_distribution(features).deterministic_sample()
        value_dict = self.critic(torch.cat([state, action], dim=1), unnorm=unnorm_value)
        if all_values:
            return self._get_value_dict(value_dict, observations, features)
        return value_dict

    def evaluate_actions(self, batch: TensorDict):
        observations, action = batch["observations"], batch["actions"]
        observations = self._norm_obs(observations)
        self.features = features = self.net(observations)
        # mu, _ = torch.chunk(batch["mu_sigma"], 2, dim=1)
        mu = action
        value_dict = self.get_value(
            observations, features, unnorm_value=False, norm_obs=False, action=mu
        )
        dist = self.action_distribution(features)
        action_log_probs = dist.log_probs(action)
        return value_dict, action_log_probs, dist

    @classmethod
    def from_config(
        cls, config, obs_space, action_space, critic_obs_space=None, **kwargs
    ):
        """Observation and actions spaces needed to define the sizes of network inputs
        and outputs."""
        ac_cfg = config.ACTOR_CRITIC
        net_cls = drl_registry.get_nn_base(ac_cfg.net.name)
        act_dist_cls = drl_registry.get_act_dist(ac_cfg.action_distribution.name)
        head_cls = drl_registry.get_nn_base(ac_cfg.head.name)

        net = net_cls.from_config(config, ac_cfg.net, obs_space)
        if head_cls is None:
            head = None
        else:
            head = head_cls.from_config(config, ac_cfg.head, critic_obs_space)

        q_critic_cls = drl_registry.get_nn_base(ac_cfg.critic.name)
        q_critic = q_critic_cls.from_config(
            config, ac_cfg.critic, obs_space, action_space
        )

        return cls(
            net=net,
            critic=q_critic,
            action_distribution=act_dist_cls.from_config(
                config, net.output_shape[0], action_space.shape[0]
            ),
            critic_is_head=ac_cfg.critic.is_head,
            head=head,
            normalize_obs=ac_cfg.normalize_obs,
            **kwargs,
        )
