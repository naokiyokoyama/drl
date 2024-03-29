import math

import torch
from torch import Size, Tensor
from torch import nn as nn

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
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x: Tensor) -> CustomCategorical:
        x = self.linear(x)
        return CustomCategorical(logits=x)

    @classmethod
    def from_config(cls, config, hidden_size: int, num_outputs: int):
        return cls(
            num_inputs=hidden_size,
            num_outputs=num_outputs,
        )


@torch.jit.script
class CustomGaussian:
    def __init__(self, mu: Tensor, sigma: Tensor):
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> Tensor:
        with torch.no_grad():
            return torch.normal(self.mu, self.sigma)

    def rsample(self) -> Tensor:
        """Sampling using the re-parameterization trick to allow for backprop"""
        with torch.no_grad():
            unit_normal_sample = torch.normal(
                torch.zeros_like(self.mu), torch.ones_like(self.sigma)
            )
        return self.mu + unit_normal_sample * self.sigma

    def deterministic_sample(self):
        return self.mu

    def log_probs(self, actions: Tensor, sum_reduce: bool = True) -> Tensor:
        log_probs = (
            -((actions - self.mu) ** 2) / (2 * self.sigma**2)
            - math.log(math.sqrt(2 * math.pi))
            - self.sigma.log()
        )
        if sum_reduce:
            return log_probs.sum(1, keepdim=True)
        return log_probs

    def log_probs_to_reparam_action(self, actions: Tensor, log_probs: Tensor):
        detached_sigma = self.sigma.detach()
        detached_mu = self.mu.detach()
        error = torch.sqrt(
            (log_probs + detached_sigma.log() + math.log(math.sqrt(2 * math.pi)))
            * (-2 * detached_sigma**2)
        )
        # error will be completely positive, some will need to be flipped negative
        error = torch.where(
            torch.gt(actions - detached_mu, 0),
            error,
            error * -1,
        )
        reparam_action = error + detached_mu
        return reparam_action

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma)


@drl_registry.register_act_dist
class GaussianActDist(nn.Module):
    name = "GaussianActDist"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        min_sigma: float = 1e-6,
        max_sigma: float = 1.0,
        sigma_as_params: bool = True,
        clip_sigma: bool = True,
    ) -> None:
        super().__init__()
        self.num_actions = num_outputs
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.sigma_as_params = sigma_as_params
        self.clip_sigma = clip_sigma

        self.mu = nn.Linear(num_inputs, num_outputs)
        if sigma_as_params:
            self.sigma = nn.Parameter(
                torch.zeros(num_outputs, requires_grad=True, dtype=torch.float32)
            )
            nn.init.constant_(self.sigma, val=0)
        else:
            self.sigma = nn.Linear(num_inputs, num_outputs)

        self.output_mu_sigma = None

    def forward(self, x: Tensor) -> CustomGaussian:
        mu = self.mu(x)
        if self.sigma_as_params:
            num_envs = x.shape[0]
            sigma = self.sigma.reshape(1, -1).repeat(num_envs, 1)
        else:
            sigma = self.sigma(x)

        sigma = torch.exp(sigma)
        if self.clip_sigma:
            sigma = torch.clamp(sigma, min=self.min_sigma, max=self.max_sigma)

        # Store these for losses/schedulers
        self.output_mu_sigma = torch.cat([mu, sigma], dim=1)

        return CustomGaussian(mu, sigma)

    def convert_to_torchscript(self):
        self.mu = torch.jit.script(self.mu)
        if not self.sigma_as_params:
            self.sigma = torch.jit.script(self.sigma)

    @classmethod
    def from_config(cls, config, hidden_size: int, num_outputs: int):
        return cls(
            num_inputs=hidden_size,
            num_outputs=num_outputs,
            min_sigma=config.ACTOR_CRITIC.action_distribution.min_sigma,
            max_sigma=config.ACTOR_CRITIC.action_distribution.max_sigma,
            sigma_as_params=config.ACTOR_CRITIC.action_distribution.sigma_as_params,
            clip_sigma=config.ACTOR_CRITIC.action_distribution.clip_sigma,
        )
