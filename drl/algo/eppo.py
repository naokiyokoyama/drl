"""
Similar to PPO, but with the following differences:
- Additional bespoke optimizer is needed for the Q-critic
- Critic loss is now an MSE using the reward terms and the next expected value

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from drl.algo import PPO


class EPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer, self.q_optimizer = self.optimizer

    def setup_optimizer(self, lr, eps):
        actor_critic_params, q_critic_params = [], []
        for name, p in self.actor_critic.named_parameters():
            if not p.requires_grad:
                continue
            if "q_critic" in name:
                q_critic_params.append(p)
            else:
                actor_critic_params.append(p)

        optimizer = optim.Adam(actor_critic_params, lr=lr, eps=eps)
        q_optimizer = optim.Adam(q_critic_params, lr=lr, eps=eps)
        print("\nPPO Optimizer:")
        print(optimizer)
        print("\nQ-Critic Optimizer:")
        print(q_optimizer)
        return optimizer, q_optimizer

    def single_update(self, epoch, idx, batch):
        (
            values,
            action_log_probs,
            dist_entropy,
            value_terms_pred,
            q_value_terms_pred,
            state_action,
        ) = self.actor_critic.evaluate_actions(batch["observations"], batch["actions"])
        next_value_gt = batch["returns"] - batch["reward_terms"].sum(1).unsqueeze(1)
        value_terms_gt = torch.cat([batch["reward_terms"], next_value_gt], 1)

        # Action loss
        # action_loss = -q_value_terms_pred.sum(1).mean()
        action_loss = self.get_action_loss(action_log_probs, batch, batch["advantages"])

        # Expressive value loss
        expressive_value_loss = (value_terms_gt - value_terms_pred).sum(1).pow(2).mean()

        # Normal value loss (for visualization only)
        value_loss = F.mse_loss(batch["returns"], values.detach())

        # Entropy loss
        entropy_loss = dist_entropy.mean()

        self.optimizer.zero_grad()
        (
            0.5 * expressive_value_loss * self.value_loss_coef
            + action_loss
            - entropy_loss * self.entropy_coef
        ).backward(retain_graph=True)
        if self.truncate_grads:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.advance_schedule(idx, epoch, batch)
        self.losses_data["losses/c_loss"] += value_loss.item()
        self.losses_data["losses/ec_loss"] += expressive_value_loss.item()
        self.losses_data["losses/a_loss"] += action_loss.item()
        self.losses_data["losses/entropy"] += entropy_loss.item()
