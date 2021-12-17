import copy
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
        expressive_action_loss_coef=0.0,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
        loss_type="",
        use_q="",
    ):
        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.expressive_action_loss_coef = expressive_action_loss_coef

        self.use_normalized_advantage = use_normalized_advantage
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        print("\nPPO Optimizer:")
        print(self.optimizer)

        if loss_type == "":
            loss_type = "regular"
        elif loss_type == "q_mse":
            loss_type = "mse"
            use_q = "q"
        elif loss_type == "r_mse":
            loss_type = "mse"
        print("\nCritic loss type:", loss_type)

        self.loss_type = loss_type
        self.mse_loss = torch.nn.MSELoss()

        # Setup Q critic
        if use_q != "":
            print("\nUsing expressive action loss!")
            value_critic_copy = copy.deepcopy(actor_critic.base.critic)
            value_critic_linear_copy = copy.deepcopy(actor_critic.base.critic_linear)
            action_dim = actor_critic.dist.mu.out_features
            state_dim = actor_critic.base.actor[0].in_features
            first_hidden_size = actor_critic.base.actor[0].out_features
            self.q_critic = torch.nn.Sequential(
                torch.nn.Linear(state_dim + action_dim, first_hidden_size),
                *value_critic_copy[1:],
                value_critic_linear_copy,
            )
            self.q_critic.train()
            self.q_optimizer = optim.Adam(self.q_critic.parameters(), lr=lr, eps=eps)
            self.q_critic = self.q_critic.to(torch.device("cuda:0"))
        else:
            self.q_critic = None

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if self.use_normalized_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        expressive_action_loss_epoch = 0
        accuracy_coef_epoch = 0

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
                    new_action,
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
                expressive_action_loss = 0

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
                    critic_loss = value_loss
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
                    if self.loss_type == "regularized":
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

                    if self.q_critic is not None:
                        state_actions = torch.cat([obs_batch, new_action], dim=1)
                        q_reward_terms = self.q_critic(state_actions)

                        q_value = q_reward_terms.sum(1).mean()
                        q_value_gt = expressive_values.sum(1).mean()
                        accuracy_coef = 1 - torch.clip(
                            torch.abs((q_value.detach() - q_value_gt) / q_value_gt),
                            0,
                            1,
                        )
                        expressive_action_loss = q_value * accuracy_coef

                self.optimizer.zero_grad()
                (
                    critic_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                    - expressive_action_loss * self.expressive_action_loss_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                if self.q_critic is not None:
                    q_reward_terms = self.q_critic(state_actions.detach())
                    q_loss = self.mse_loss(expressive_values, q_reward_terms)
                    self.q_optimizer.zero_grad()
                    (q_loss * self.value_loss_coef).backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                    self.q_optimizer.step()

                    expressive_action_loss_epoch += expressive_action_loss.item()
                    accuracy_coef_epoch += accuracy_coef.item()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        losses_data = {
            "value_loss": value_loss_epoch / num_updates,
            "action_loss": action_loss_epoch / num_updates,
            "dist_entropy": dist_entropy_epoch / num_updates,
        }

        if self.q_critic is not None:
            losses_data.update(
                {
                    "expressive_action_loss": expressive_action_loss_epoch
                    / num_updates,
                    "accuracy_coef": accuracy_coef_epoch / num_updates,
                }
            )

        return losses_data
