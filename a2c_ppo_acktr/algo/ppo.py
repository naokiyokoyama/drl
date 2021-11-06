import torch
import torch.nn as nn
import torch.optim as optim


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
        loss_type="",
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
        print('\nPPO Optimizer:')
        print(self.optimizer)

        if loss_type == "":
            loss_type = "regular"
        print('\nCritic loss type:', loss_type)

        self.loss_type = loss_type
        self.mse_loss = torch.nn.MSELoss()

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if self.use_normalized_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    reward_terms_batch,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    reward_terms,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * adv_targ
                )

                action_loss = -torch.min(surr1, surr2).mean()

                if self.loss_type == "regular":
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + (
                            values - value_preds_batch
                        ).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (value_pred_clipped - return_batch).pow(
                            2
                        )
                        value_loss = (
                            0.5 * torch.max(value_losses, value_losses_clipped).mean()
                        )
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()
                else:
                    return_minus_reward = return_batch - torch.sum(
                        reward_terms_batch, 1
                    ).unsqueeze(1)
                    expressive_values = torch.cat(
                        [reward_terms_batch, return_minus_reward],
                        1,
                    )
                    value_loss = (
                        0.5 * (expressive_values - reward_terms).sum(1).pow(2).mean()
                    )

                # Special types of expressive critic losses
                if self.loss_type == "regular":
                    critic_loss = value_loss
                elif self.loss_type == "regularized":
                    critic_loss = value_loss / expressive_values.shape[1]
                elif self.loss_type == "mse":
                    critic_loss = self.mse_loss(expressive_values, reward_terms)
                elif self.loss_type == "no_reparam":
                    critic_loss = (
                        0.5
                        * (expressive_values.sum(1) - reward_terms.sum(1))
                        .pow(2)
                        .mean()
                    )
                else:
                    raise RuntimeError("Loss not understood: " + self.loss_type)

                self.optimizer.zero_grad()
                (
                    critic_loss * self.value_loss_coef
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

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
