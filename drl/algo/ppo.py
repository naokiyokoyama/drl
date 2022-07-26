from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from drl.utils.common import mse_loss
from torch import Tensor

from drl.utils.registry import drl_registry
from drl.utils.rollout_storage import RolloutStorage

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        scheduler,
        clip_param,
        policy_epoch,
        critic_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        actor_lr,
        critic_lr,
        eps,
        truncate_grads=False,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):
        super().__init__()

        self.actor_critic = actor_critic
        self.scheduler = scheduler

        self.clip_param = clip_param
        self.policy_epoch = policy_epoch
        self.critic_epoch = 0 if self.actor_critic.critic_is_head else critic_epoch
        self.ppo_epoch = max(policy_epoch, critic_epoch)
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_normalized_advantage = use_normalized_advantage
        self.truncate_grads = truncate_grads
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizers = self.setup_optimizer(eps)
        self.losses_data = defaultdict(float)

    def setup_optimizer(self, eps: float):
        # Only one optimizer is needed if critic is part of policy
        if self.actor_critic.critic_is_head:
            params = [p for p in self.actor_critic.parameters() if p.requires_grad]
            return {"actor": Adam(params, lr=self.actor_lr, eps=eps)}

        # If actor and critic are separate, create two optimizers, one for each
        actor_params, critic_params = [], []
        for name, p in self.actor_critic.named_parameters():
            if not p.requires_grad:
                continue
            elif "critic" in name:
                critic_params.append(p)
            else:
                actor_params.append(p)
        actor_optim = Adam(actor_params, lr=self.actor_lr, eps=eps)
        critic_optim = Adam(critic_params, lr=self.critic_lr, eps=eps)
        return {"actor": actor_optim, "critic": critic_optim}

    def update(self, rollouts: RolloutStorage) -> Dict:
        self.losses_data = defaultdict(float)  # clear the loss data
        if self.actor_critic.is_recurrent:
            generator = rollouts.recurrent_generator
        else:
            generator = rollouts.feed_forward_generator
        if self.actor_critic.critic.normalizer is not None:
            # MUST be run AFTER get_advantages() due to mutation of "returns" buffer
            rollouts.normalize_values(self.actor_critic.critic.normalizer)
        for epoch in range(self.ppo_epoch):
            for batch in generator(self.num_mini_batch):
                values, action_log_probs, dist = self.actor_critic.evaluate_actions(
                    batch["observations"], batch["actions"]
                )
                if epoch < self.policy_epoch:
                    self.update_policy(batch, values, action_log_probs, dist)
                if epoch < self.critic_epoch and not self.actor_critic.critic_is_head:
                    self.update_critic(values, batch)
        rollouts.after_update()
        return self.get_losses_data()

    def update_policy(self, batch, values, action_log_probs, dist):
        action_loss = self.action_loss(action_log_probs, batch)
        if self.actor_critic.critic_is_head:
            value_loss = self.value_loss(batch, values)
        else:
            value_loss = 0.0
        entropy_loss = self.entropy_loss(dist)
        aux_loss = self.aux_loss(batch, values, action_log_probs, dist)

        loss = (
            0.5 * value_loss * self.value_loss_coef
            + action_loss
            - entropy_loss * self.entropy_coef
            + aux_loss
        )
        self.update_weights(loss, ["actor"])

        self.advance_schedule(batch)

    def update_critic(self, batch, values):
        self.update_weights(
            0.5 * self.value_loss(values, batch) * self.value_loss_coef, ["critic"]
        )

    def update_weights(self, loss, opt_keys):
        for name, opt in self.optimizers.items():
            if name in opt_keys:
                opt.zero_grad()
        loss.backward()
        if self.truncate_grads:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        for name, opt in self.optimizers.items():
            if name in opt_keys:
                opt.step()

    def action_loss(self, action_log_probs, batch):
        ratio = torch.exp(action_log_probs - batch["action_log_probs"])
        clipped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        surr1, surr2 = batch["advantages"] * ratio, batch["advantages"] * clipped_ratio
        action_loss = -torch.min(surr1.sum(1), surr2.sum(1)).mean()
        self.losses_data["losses/a_loss"] += action_loss.item()
        return action_loss

    def value_loss(self, batch, values):
        if self.use_clipped_value_loss:
            value_pred_clipped = batch["value_preds"] + (
                values - batch["value_preds"]
            ).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - batch["returns"]).pow(2)
            value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = mse_loss(values, batch["returns"])
        self.losses_data["losses/c_loss"] += value_loss.item()
        return value_loss

    def entropy_loss(self, dist):
        entropy_loss = dist.entropy().sum(dim=1).mean()
        self.losses_data["losses/entropy"] += entropy_loss.item()
        return entropy_loss

    def aux_loss(self, batch, values, action_log_probs, dist):
        return 0  # Should be overridden by descendant classes

    def get_losses_data(self):
        num_updates = self.ppo_epoch * self.num_mini_batch
        return {k: v / num_updates for k, v in self.losses_data.items()}

    def advance_schedule(self, batch):
        """This repo assumes only the learning rate for actor is manually modulated"""
        for param_group in self.optimizers["actor"].param_groups:
            param_group["lr"] = self.scheduler.update(
                param_group["lr"], algo=self, batch=batch
            )

    @classmethod
    def from_config(cls, config, actor_critic, **kwargs):
        """kwargs is added to support other classes that inherit this method but have
        different signatures for __init__"""
        scheduler_cls = drl_registry.get_scheduler(config.RL.scheduler.name)
        return cls(
            actor_critic=actor_critic,
            scheduler=scheduler_cls.from_config(config),
            clip_param=config.RL.PPO.clip_param,
            policy_epoch=config.RL.PPO.policy_epoch,
            critic_epoch=config.RL.PPO.critic_epoch,
            num_mini_batch=config.RL.PPO.num_mini_batch,
            value_loss_coef=config.RL.PPO.value_loss_coef,
            entropy_coef=config.RL.PPO.entropy_coef,
            actor_lr=config.RL.PPO.actor_lr,
            critic_lr=config.RL.PPO.critic_lr,
            eps=config.RL.PPO.eps,
            truncate_grads=config.RL.PPO.truncate_grads,
            max_grad_norm=config.RL.PPO.max_grad_norm,
            use_clipped_value_loss=config.RL.PPO.use_clipped_value_loss,
            use_normalized_advantage=config.RL.PPO.use_normalized_advantage,
            **kwargs,
        )
