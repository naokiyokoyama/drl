from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from drl.utils.rollout_storage import RolloutStorage

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):
        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_normalized_advantage = use_normalized_advantage
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        print("\nPPO Optimizer:")
        print(self.optimizer)

    def get_advantages(self, rollouts: RolloutStorage) -> Tensor:
        advantages = (
            rollouts.buffers["returns"][:-1] - rollouts.buffers["value_preds"][:-1]
        )
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts: RolloutStorage) -> Dict:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator
            else:
                data_generator = rollouts.feed_forward_generator

            for batch in data_generator(advantages, self.num_mini_batch):
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                ) = self.actor_critic.evaluate_actions(
                    batch["observations"], batch["actions"]
                )

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * batch["advantages"]
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(
                        2
                    )
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        rollouts.after_update()
        losses_data = {
            "value_loss": value_loss_epoch / num_updates,
            "action_loss": action_loss_epoch / num_updates,
            "dist_entropy": dist_entropy_epoch / num_updates,
        }

        return losses_data
