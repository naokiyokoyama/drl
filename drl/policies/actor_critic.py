import warnings

import numpy as np
from torch import nn as nn

from drl.nets.basic import NNBase
from drl.utils.registry import drl_registry

from drl.utils.running_mean_std import RunningMeanStd


@drl_registry.register_actor_critic
class ActorCritic(nn.Module):
    def __init__(
        self,
        net: NNBase,
        critic: NNBase,
        action_distribution: nn.Module,
        critic_is_head: bool = False,
        normalize_obs: bool = True,
        normalize_value: bool = True,
    ):
        super().__init__()
        self.net = net
        self.action_distribution = action_distribution
        self.critic = critic
        self.critic_is_head = critic_is_head
        self.obs_normalizer = (
            RunningMeanStd(self.net.input_shape) if normalize_obs else None
        )
        self.value_normalizer = (
            RunningMeanStd(critic.output_shape) if normalize_value else None
        )

    def act(self, observations, deterministic=False, get_terms=False):
        value, dist, other = self._process_observations(observations, get_terms)
        actions = dist.deterministic_sample() if deterministic else dist.sample()
        action_log_probs = dist.log_probs(actions)

        return value, actions, action_log_probs, other

    def evaluate_actions(self, observations, action):
        value, dist, _ = self._process_observations(observations, get_terms=True)
        action_log_probs = dist.log_probs(action)
        return value, action_log_probs, dist

    def get_value(self, observations, features=None, get_terms=False):
        if self.obs_normalizer is not None:
            observations = self.obs_normalizer(observations)
        if self.critic_is_head and features is None:
            features = self.net(observations)
        value = self.critic(features if self.critic_is_head else observations)
        if value.shape[1] > 1 and not get_terms:
            value = value.sum(1, keepdim=True)
        return value

    def _process_observations(self, observations, get_terms=False):
        if self.obs_normalizer is not None:
            observations = self.obs_normalizer(observations)
        features = self.net(observations)
        value = self.get_value(observations, features, get_terms)
        dist = self.action_distribution(features)
        other = self._get_other()

        return value, dist, other

    def _get_other(self):
        """Primarily for returning rnn_hx, but could be used for auxiliary
        tasks/losses/schedulers/etc."""
        other = {}
        if self.net.is_recurrent:
            other["net_rnn_hx"] = self.net.rnn_hx
        if getattr(self.critic, "is_recurrent", False):
            other["critic_rnn_hx"] = self.critic.rnn_hx
        if self.action_distribution.name == "GaussianActDist":
            other["mu_sigma"] = self.action_distribution.output_mu_sigma
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
        critic_cls = drl_registry.get_nn_base(ac_cfg.critic.name)
        act_dist_cls = drl_registry.get_act_dist(ac_cfg.action_distribution.name)

        net = net_cls.from_config(ac_cfg.net, obs_space)

        if ac_cfg.critic.is_head:
            if critic_obs_space is not None:
                warnings.warn(
                    f"Overwriting provided critic_obs_space {critic_obs_space} with "
                    f"{net.outout_shape}"
                )
            critic_obs_space = np.zeros(net.output_shape)
        elif critic_obs_space is None:
            critic_obs_space = obs_space

        if "critic_kwargs" in kwargs:
            critic_kwargs = kwargs.pop("critic_kwargs")
        else:
            critic_kwargs = {}

        return cls(
            net=net,
            critic=critic_cls.from_config(
                ac_cfg.critic, critic_obs_space, **critic_kwargs
            ),
            action_distribution=act_dist_cls.from_config(
                config, net.output_shape[0], action_space.shape[0]
            ),
            critic_is_head=ac_cfg.critic.is_head,
            normalize_obs=ac_cfg.normalize_obs,
            normalize_value=ac_cfg.normalize_value,
            **kwargs,
        )

    def forward(self, x):
        return self.act(x)
