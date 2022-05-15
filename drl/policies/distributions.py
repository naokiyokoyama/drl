from abc import ABC

import torch
from torch import Size, Tensor
from torch import nn as nn

from drl.utils.common import initialized_linear
from drl.utils.registry import drl_registry


class CustomCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:  # noqa: B008
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def deterministic_sample(self):
        return self.probs.argmax(dim=-1, keepdim=True)


@drl_registry.register_act_dist
class CategoricalActDist(nn.Module):
    name = "CategoricalActDist"

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.linear = initialized_linear(num_inputs, num_outputs, gain=0.01)

    def forward(self, x: Tensor) -> CustomCategorical:
        x = self.linear(x)
        return CustomCategorical(logits=x)

    @classmethod
    def from_config(cls, config, hidden_size: int, num_outputs: int):
        return cls(
            num_inputs=hidden_size,
            num_outputs=num_outputs,
        )


class CustomGaussian(torch.distributions.normal.Normal, ABC):
    def sample(self, sample_shape: Size = torch.Size()) -> Tensor:
        return super().sample(sample_shape)

    def log_probs(self, actions: Tensor) -> Tensor:
        return super().log_prob(actions).sum(-1).unsqueeze(-1)

    def deterministic_sample(self):
        return self.mean


@drl_registry.register_act_dist
class GaussianActDist(nn.Module):
    name = "GaussianActDist"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        min_sigma: float = 1e-6,
        max_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.mu = torch.jit.script(
            initialized_linear(num_inputs, num_outputs, gain=0.01)
        )
        self.sigma = torch.jit.script(
            initialized_linear(num_inputs, num_outputs, gain=0.01)
        )
        self.output_mu_sigma = None

    def forward(self, x: Tensor) -> CustomGaussian:
        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = torch.exp(sigma)
        sigma = torch.clamp(sigma, min=self.min_sigma, max=self.max_sigma)
        # Store these for losses/schedulers
        self.output_mu_sigma = torch.cat([mu, sigma], dim=1)

        return CustomGaussian(mu, sigma)

    @classmethod
    def from_config(cls, config, hidden_size: int, num_outputs: int):
        return cls(
            num_inputs=hidden_size,
            num_outputs=num_outputs,
            min_sigma=config.ACTOR_CRITIC.action_distribution.min_sigma,
            max_sigma=config.ACTOR_CRITIC.action_distribution.max_sigma,
        )
