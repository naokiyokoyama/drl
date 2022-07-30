"""
Similar to PPO, but with the following differences:
- Additional bespoke optimizer is needed for the Q-critic
- Critic loss is now an MSE using the reward terms and the next expected value

"""
import torch

from drl.algo.ppo import PPO
from drl.utils.common import mse_loss


class EPPO(PPO):
    def __init__(self, aux_coeff, *args, **kwargs):
        self.aux_coeff = aux_coeff
        super().__init__(*args, **kwargs)

    def aux_loss(self, batch, values, action_log_probs, dist):
        if self.actor_critic.head is None:
            return 0
        head_pred = self.actor_critic.head(
            self.actor_critic.features, unnorm=False
        )
        aux_loss = mse_loss(head_pred, batch[self.actor_critic.head.target_key])
        self.losses_data["losses/aux_loss"] += aux_loss.item()
        return aux_loss * self.aux_coeff

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
