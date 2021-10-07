#!/usr/bin/env python3

from collections import defaultdict

import torch
import math

class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""
    
    def __init__(self, 
        observation_space=None,
        num_actions=None,
        action_dtype=None,
        rollout_length=None
    ):

        observation_space=None
        num_actions=1
        action_dtype=torch.long
        rollout_length=20

        # For now, assume single observation space (Box)
        self.observations = torch.zeros(
            rollout_length+1, 
            4
            # *observation_space.shape
        )
        # For now, assume single action space
        self.actions = torch.zeros(
            rollout_length+1,
            num_actions,
            dtype = action_dtype
        )
        self.log_probs = torch.zeros(rollout_length+1, num_actions)

        self.value_preds = torch.zeros(rollout_length+1)
        self.rewards     = torch.zeros(rollout_length+1)
        self.dones       = torch.zeros(rollout_length+1)

        self.advantages = torch.zeros(rollout_length+1)

        self.rollout_length = rollout_length
        self.step = 0

    def batch_generator(self, batch_size, gamma, tau):
        self.compute_advantages(gamma, tau)
        randperm = torch.randperm(self.step-1)
        num_batches = math.ceil((self.step-1) / batch_size)

        for i in range(num_batches):
            indices = randperm[i*batch_size:(i+1)*batch_size]
            yield (
                self.observations[indices],
                self.actions[indices],
                self.log_probs[indices],
                self.value_preds[indices],
                self.rewards[indices],
                self.dones[indices],
                self.advantages[indices]
            )

    def insert(
        self, 
        observation, 
        action, 
        log_prob, 
        value, 
        reward, 
        done
    ):
        self.observations[self.step].copy_(observation)
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        self.value_preds[self.step].copy_(value)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)

        self.step += 1

    def after_update(self):
        self.step = 0

    # Uses GAE
    def compute_advantages(self, gamma, tau):
        gae = 0
        for step in reversed(range(0, self.step-1)):
            delta = (
                self.rewards[step]
                + gamma * self.value_preds[step+1] * self.dones[step+1]
                - self.value_preds[step]
            )
            gae = delta + gamma * tau * self.dones[step+1] * gae
            self.advantages[step] = gae 
