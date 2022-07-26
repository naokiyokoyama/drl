"""
Similar to PPO, but with the following differences:
- Additional bespoke optimizer is needed for the Q-critic
- Critic loss is now an MSE using the reward terms and the next expected value

"""
import torch

from drl.algo.ppo import PPO
from drl.utils.common import mse_loss

# import torch.nn as nn
# from torch.optim import Adam
from drl.utils.rollout_storage import RolloutStorage


class EPPO(PPO):
    def __init__(self, aux_coeff, *args, **kwargs):
        self.aux_coeff = aux_coeff
        super().__init__(*args, **kwargs)

    def update(self, rollouts: RolloutStorage):
        if self.actor_critic.head.normalizer is not None:
            num_terms = rollouts.buffers["return_terms"].shape[-1]
            rollouts.buffers["return_terms"] = self.actor_critic.head.normalizer(
                rollouts.buffers["return_terms"].reshape(-1, num_terms)
            ).reshape(rollouts.num_steps + 1, rollouts._num_envs, -1)
        return super().update(rollouts)

    def aux_loss(self, batch, values, action_log_probs, dist):
        value_terms_pred = self.actor_critic.head(
            self.actor_critic.features, unnorm=False
        )
        aux_loss = mse_loss(value_terms_pred, batch["return_terms"])
        self.losses_data["losses/aux_loss"] += aux_loss.item()
        return aux_loss * self.aux_coeff

    # def setup_optimizer(self, eps):
    #     actor_params, critic_params, q_params = [], [], []
    #     for name, p in self.actor_critic.named_parameters():
    #         if not p.requires_grad:
    #             continue
    #         elif "q_critic" in name:
    #             q_params.append(p)
    #         elif "critic" in name and not self.actor_critic.critic_is_head:
    #             critic_params.append(p)
    #         else:
    #             actor_params.append(p)
    #     optimizers = {
    #         "q_critic": Adam(q_params, lr=self.q_critic_lr, eps=eps),
    #         "actor": Adam(actor_params, lr=self.q_critic_lr, eps=eps),
    #     }
    #     if not self.actor_critic.critic_is_head:
    #         optimizers["critic"] = Adam(q_params, lr=self.q_critic_lr, eps=eps)
    #     return optimizers

    def cherry_pick(self, action_log_probs, batch):
        ratio = torch.exp(action_log_probs - batch["action_log_probs"])
        adv = batch["advantages"].repeat(1, action_log_probs.shape[1])
        assert adv.shape == ratio.shape, f"{adv.shape} != {ratio.shape}"
        bad = torch.logical_or(
            torch.logical_and(adv > 0, ratio > 1 + self.clip_param),
            torch.logical_and(adv < 0, ratio < 1 - self.clip_param),
        )
        mask = torch.logical_not(bad).detach()
        return mask

    @classmethod
    def from_config(cls, config, actor_critic, **kwargs):
        return super().from_config(
            config,
            actor_critic,
            aux_coeff=config.RL.PPO.aux_coeff,
        )
