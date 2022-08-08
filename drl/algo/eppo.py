"""
Similar to PPO, but with the following differences:
- Additional bespoke optimizer is needed for the Q-critic
- Critic loss is now an MSE using the reward terms and the next expected value

"""
import torch

from drl.algo.ppo import PPO
from drl.utils.common import mse_loss
from torch.optim import Adam


class EPPO(PPO):
    def __init__(self, aux_coeff, *args, **kwargs):
        self.aux_coeff = aux_coeff
        super().__init__(*args, **kwargs)

    def setup_optimizer(self, eps):
        actor_params, q_params, critic_params = [], [], []
        for name, p in self.actor_critic.named_parameters():
            if not p.requires_grad:
                continue
            if "q_critic" in name:
                q_params.append(p)
            elif "critic" in name and not self.actor_critic.critic_is_head:
                critic_params.append(p)
            else:
                actor_params.append(p)
        optimizers = {"actor": Adam(actor_params, lr=self.actor_lr, eps=eps)}
        if critic_params:
            optimizers["critic"] = Adam(critic_params, lr=self.critic_lr, eps=eps)
        if q_params:
            optimizers["q_critic"] = Adam(q_params, lr=self.critic_lr, eps=eps)
        return optimizers

    def aux_loss(self, batch, values, action_log_probs, dist):
        if "q_critic" in self.optimizers:
            action = self.actor_critic.reparameterize_action(
                batch["observations"], batch["actions"]
            )
            state_action = torch.cat([batch["observations"], action], dim=1)
            adv_pred = self.actor_critic.q_critic(state_action)
            if adv_pred.shape[1] > 1:
                adv_pred = adv_pred.sum(1, keepdims=True)
            obj = -adv_pred
            mask = self.cherry_pick(action_log_probs, batch)
            aux_loss = (mask * obj).mean()
            # aux_loss = obj.mean() * self.aux_coeff
            self.losses_data["losses/aux_loss"] += aux_loss.item()
            return aux_loss

        if self.actor_critic.head is not None:
            head_pred = self.actor_critic.head(self.actor_critic.features, unnorm=False)
            aux_loss = mse_loss(head_pred, batch[self.actor_critic.head.target_key])
            self.losses_data["losses/aux_loss"] += aux_loss.item()
            return aux_loss * self.aux_coeff

        return 0.0

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

    def update_other(self, batch):
        if "q_critic" not in self.optimizers:
            return
        state_action = torch.cat([batch["observations"], batch["actions"]], dim=1)
        adv_pred = self.actor_critic.q_critic(state_action)
        if adv_pred.shape[1] == 1:
            r_key, v_key = "returns", "value_preds"
        else:
            r_key, v_key = "return_terms", "value_terms_preds"
        q_loss = mse_loss(adv_pred, batch[r_key] - batch[v_key])
        self.update_weights(q_loss, "q_critic")

    @classmethod
    def from_config(cls, config, actor_critic, **kwargs):
        return super().from_config(
            config,
            actor_critic,
            aux_coeff=config.RL.PPO.aux_coeff,
        )
