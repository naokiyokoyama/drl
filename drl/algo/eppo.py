"""
Similar to PPO, but with the following differences:
- Additional bespoke optimizer is needed for the Q-critic
- Critic loss is now an MSE using the reward terms and the next expected value

"""
import torch
import torch.nn as nn
from torch.optim import Adam

from drl.algo import PPO
from drl.utils.common import mse_loss


class EPPO(PPO):
    def __init__(self, q_critic_lr, *args, **kwargs):
        self.q_critic_lr = q_critic_lr
        super().__init__(*args, **kwargs)

    def setup_optimizer(self, eps):
        actor_params, critic_params, q_params = [], [], []
        for name, p in self.actor_critic.named_parameters():
            if not p.requires_grad:
                continue
            elif "q_critic" in name:
                q_params.append(p)
            elif "critic" in name:
                critic_params.append(p)
            else:
                actor_params.append(p)
        optimizers = {"q_critic": Adam(q_params, lr=self.q_critic_lr, eps=eps)}
        if self.actor_critic.critic_is_head:
            optimizers["actor"] = Adam(
                actor_params + critic_params, lr=self.actor_lr, eps=eps
            )
        else:
            optimizers["actor"] = Adam(actor_params, lr=self.actor_lr, eps=eps)
            optimizers["critic"] = Adam(critic_params, lr=self.critic_lr, eps=eps)
        return optimizers

    def single_update(self, epoch, idx, batch):
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
        ).backward(retain_graph=True)
        if self.truncate_grads:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        for k, v in self.optimizers.items():
            if k != "q_critic":
                v.step()

        state_action = torch.cat([batch["observations"], batch["actions"]], dim=1)
        q_value_terms_pred = self.actor_critic.q_critic(state_action)
        q_value_loss = mse_loss(q_value_terms_pred, batch["reward_terms"])
        assert torch.allclose(
            batch["reward_terms"].sum(1, keepdims=True),
            batch["rewards"],
            1e-6,
            1e-6,
        )

        self.optimizers["q_critic"].zero_grad()
        q_value_loss.backward()
        self.optimizers["q_critic"].step()

        self.advance_schedule(idx, epoch, batch)
        self.losses_data["losses/c_loss"] += value_loss.item()
        self.losses_data["losses/eq_loss"] += q_value_loss.item()
        self.losses_data["losses/a_loss"] += action_loss.item()
        self.losses_data["losses/entropy"] += entropy_loss.item()

    @classmethod
    def from_config(cls, config, actor_critic, **kwargs):
        q_critic_lr = config.RL.PPO.critic_lr
        return super().from_config(config, actor_critic, q_critic_lr=q_critic_lr)
