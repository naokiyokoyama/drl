import torch

from drl.utils.registry import drl_registry


class RLScheduler:
    def update(self, *args, **kwargs):
        raise NotImplementedError


@drl_registry.register_scheduler
class IdentityScheduler(RLScheduler):
    name = "IdentityScheduler"

    def update(self, current_lr, *args, **kwargs):
        return current_lr

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls()

@torch.jit.script
def policy_kl(mu, sigma, prev_mu, prev_sigma, reduce: bool=True):
    mu, sigma, prev_mu, prev_sigma = [
        i.detach() for i in [mu, sigma, prev_mu, prev_sigma]
    ]
    c1 = torch.log(prev_sigma / sigma + 1e-5)
    c2 = (sigma ** 2 + (prev_mu - mu) ** 2) / (2.0 * (prev_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    if reduce:
        return kl.mean()
    else:
        return kl


@drl_registry.register_scheduler
class AdaptiveScheduler(RLScheduler):
    name = "AdaptiveScheduler"

    def __init__(self, kl_threshold=0.008, min_lr=1e-6, max_lr=1e-2):
        super().__init__()
        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr

    def update(self, current_lr, algo, batch, *args, **kwargs):
        mu_sigma = algo.actor_critic.action_distribution.output_mu_sigma
        prev_mu_sigma = batch["mu_sigma"]
        mu, sigma = torch.chunk(mu_sigma, 2, dim=1)
        prev_mu, prev_sigma = torch.chunk(prev_mu_sigma, 2, dim=1)
        kl_dist = policy_kl(mu, sigma, prev_mu, prev_sigma)
        if kl_dist > (2.0 * self.kl_threshold):
            current_lr = max(current_lr / 1.5, self.min_lr)
        elif kl_dist < (0.5 * self.kl_threshold):
            current_lr = min(current_lr * 1.5, self.max_lr)
        return current_lr

    @classmethod
    def from_config(cls, config):
        scheduler_config = config.RL.scheduler
        return cls(
            kl_threshold=scheduler_config.kl_threshold,
            min_lr=scheduler_config.min_lr,
            max_lr=scheduler_config.max_lr,
        )
