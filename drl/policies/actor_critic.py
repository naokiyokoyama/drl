from typing import Optional

import torch
from torch import nn as nn

from drl.nets.basic import NNBase
from drl.utils.registry import drl_registry
from drl.utils.running_mean_std import RunningMeanStd


@drl_registry.register_actor_critic
class ActorCritic(nn.Module):

    """
    At minimum, this class represents a policy and a critic. The critic can be a head
    that shares parameters with the policy, or it can be a separate network. Optionally,
    a head that does NOT act as a critic can be added to the policy. In such case, the
    critic must be its own network.
    """

    def __init__(
        self,
        net: NNBase,
        action_distribution: nn.Module,
        critic: NNBase,
        critic_is_head: bool = True,
        head: Optional[NNBase] = None,
        normalize_obs: bool = True,
    ):
        """
        :param net: "Encoder" net that passes features into all heads
        :param action_distribution: Net that outputs/parameterizes action distribution
        :param critic: Net estimating value of the state
        :param head: Net for some auxiliary task
        :param normalize_obs: Whether observations will be normalized
        :param normalize_value: Whether values will be normalized
        """
        super().__init__()
        self.net = net
        self.action_distribution = action_distribution
        self.critic = critic
        self.head = head
        self.features = None  # encoder features stored for use with head if needed
        self.critic_is_head = critic_is_head
        self.obs_normalizer = (
            RunningMeanStd(self.net.input_shape) if normalize_obs else None
        )

    def act(self, observations, deterministic=False):
        value, dist, other = self._process_observations(observations)
        actions = dist.deterministic_sample() if deterministic else dist.sample()
        action_log_probs = dist.log_probs(actions)

        return value, actions, action_log_probs, other

    def evaluate_actions(self, observations, action):
        value, dist, _ = self._process_observations(observations, unnorm_value=False)
        action_log_probs = dist.log_probs(action)
        return value, action_log_probs, dist

    def get_value(
        self,
        observations,
        features: Optional[torch.Tensor] = None,
        unnorm_value: bool = True,
        all_values: bool = False,
    ):
        if self.obs_normalizer is not None:
            observations = self.obs_normalizer(observations)
        if self.critic_is_head and features is None:
            features = self.net(observations)
        value = self.critic(
            features if self.critic_is_head else observations,
            unnorm=unnorm_value,
        )
        if all_values:
            return self.get_value_dict(value, observations, features)
        return value

    def get_value_dict(self, value, observations, features):
        values_dict = {}
        if value.shape[1] > 1:
            values_dict["value_terms_preds"] = value
        else:
            values_dict["value_preds"] = value
        if self.head is not None:
            if features is None:
                features = self.net(observations)
            other = self.head.get_other(features)
            values_dict.update({k: v for k, v in other.items() if k not in values_dict})
        if "value_preds" not in values_dict:
            values_dict["value_preds"] = values_dict["value_terms_preds"].sum(
                1, keepdims=True
            )
        return values_dict

    def _process_observations(self, observations, unnorm_value: bool = True):
        if self.obs_normalizer is not None:
            observations = self.obs_normalizer(observations)
        self.features = features = self.net(observations)
        value = self.get_value(observations, features, unnorm_value)
        dist = self.action_distribution(features)
        other = self.get_other()

        return value, dist, other

    def get_other(self):
        """Primarily for returning rnn_hx, but could be used for auxiliary
        tasks/losses/schedulers/etc."""
        other = {}
        if self.net.is_recurrent:
            other["net_rnn_hx"] = self.net.rnn_hx
        if getattr(self.critic, "is_recurrent", False):
            other["critic_rnn_hx"] = self.critic.rnn_hx
        if self.action_distribution.name == "GaussianActDist":
            other["mu_sigma"] = self.action_distribution.output_mu_sigma
        if self.head is not None:
            other.update(self.head.get_other(self.features))
        return other

    @property
    def is_recurrent(self):
        return self.net.is_recurrent

    def convert_to_torchscript(self):
        self.net.convert_to_torchscript()
        self.action_distribution.convert_to_torchscript()
        self.critic.convert_to_torchscript()

    @classmethod
    def from_config(
        cls, config, obs_space, action_space, critic_obs_space=None, **kwargs
    ):
        """Observation and actions spaces needed to define the sizes of network inputs
        and outputs."""
        ac_cfg = config.ACTOR_CRITIC
        net_cls = drl_registry.get_nn_base(ac_cfg.net.name)
        act_dist_cls = drl_registry.get_act_dist(ac_cfg.action_distribution.name)
        critic_cls = drl_registry.get_nn_base(ac_cfg.critic.name)
        head_cls = drl_registry.get_nn_base(ac_cfg.head.name)

        net = net_cls.from_config(ac_cfg.net, obs_space)
        critic_obs_space = obs_space if critic_obs_space is None else critic_obs_space
        critic = critic_cls.from_config(ac_cfg.critic, critic_obs_space, net)
        if head_cls is None:
            head = None
        else:
            head = head_cls.from_config(ac_cfg.head, critic_obs_space, net)

        return cls(
            net=net,
            critic=critic,
            action_distribution=act_dist_cls.from_config(
                config, net.output_shape[0], action_space.shape[0]
            ),
            critic_is_head=ac_cfg.critic.is_head,
            head=head,
            normalize_obs=ac_cfg.normalize_obs,
            **kwargs,
        )

    def forward(self, x):
        return self.act(x)
