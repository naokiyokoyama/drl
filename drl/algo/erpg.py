"""
Similar to EPPO, but with the following differences:
- Q Critic only outputs reward terms
- Using RPG loss function
"""
import torch
import torch.nn as nn

from drl.algo.eppo import EPPO
from drl.utils.common import mse_loss


class ERPG(EPPO):
    q_mse_key = "reward_terms"

    def single_update(self, batch):
        (
            values,
            action_log_probs,
            dist,
            q_value_terms_pred,
        ) = self.actor_critic.evaluate_actions(batch["observations"], batch["actions"])

        value_loss = mse_loss(values, batch["returns"])

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])
        too_high = torch.logical_and(
            batch["advantages"] > 0, ratio > 1 + self.clip_param
        )
        too_low = torch.logical_and(
            batch["advantages"] < 0, ratio < 1 - self.clip_param
        )
        ratio = torch.where(
            torch.logical_or(too_low, too_high), torch.zeros_like(ratio), ratio
        )
        obj = (
            q_value_terms_pred.sum(1, keepdims=True)
            + batch["advantages"] * action_log_probs
        )
        action_loss = -(ratio.detach() * obj).mean()

        # Entropy loss
        entropy_loss = dist.entropy().sum(dim=1).mean()

        for k, v in self.optimizers.items():
            if k != "q_critic":
                v.zero_grad()
        (
            0.5 * value_loss * self.value_loss_coef
            + action_loss
            - entropy_loss * self.entropy_coef
        ).backward()
        if self.truncate_grads:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        for k, v in self.optimizers.items():
            if k != "q_critic":
                v.step()

        state_action = torch.cat([batch["observations"], batch["actions"]], dim=1)
        q_value_terms_pred = self.actor_critic.q_critic(state_action)
        q_value_loss = mse_loss(q_value_terms_pred, batch[self.q_mse_key])

        self.optimizers["q_critic"].zero_grad()
        q_value_loss.backward()
        self.optimizers["q_critic"].step()

        self.advance_schedule(batch)
        self.losses_data["losses/c_loss"] += value_loss.item()
        self.losses_data["losses/eq_loss"] += q_value_loss.item()
        self.losses_data["losses/a_loss"] += action_loss.item()
        self.losses_data["losses/entropy"] += entropy_loss.item()


class RPG(ERPG):
    q_mse_key = "rewards"