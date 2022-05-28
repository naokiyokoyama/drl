#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Optional, Union

import numpy as np
import torch

from drl.utils.tensor_dict import TensorDict

TensorLike = Union[torch.Tensor, np.ndarray]


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        device: Union[torch.device, str],
        use_gae: bool,
        gamma: float,
        tau: float,
        value_bootstrap: bool,
        observations=None,
    ):
        self.buffers = TensorDict()
        self.num_steps = num_steps
        self._num_envs = num_envs
        self.device = device
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        self.value_bootstrap = value_bootstrap

        # We can definitively define the shapes of these sub-buffers without a sample
        for i in ["rewards", "value_preds", "returns"]:
            self.buffers[i] = torch.zeros(num_steps + 1, num_envs, 1, device=device)
        self.buffers["not_dones"] = torch.zeros(
            num_steps + 1, num_envs, 1, device=device, dtype=torch.bool
        )

        self.current_rollout_step_idx = 0
        if observations is not None:
            self.insert_initial_data("observations", observations)

    @classmethod
    def from_config(cls, config, num_envs, device, observations=None):
        return cls(
            num_steps=config.RL.num_steps,
            num_envs=num_envs,
            device=device,
            use_gae=config.RL.use_gae,
            gamma=config.RL.gamma,
            tau=config.RL.tau,
            value_bootstrap=config.RL.value_bootstrap,
            observations=observations,
        )

    def insert_initial_data(self, key, data: Union[TensorDict, TensorLike]):
        assert isinstance(
            data, (TensorDict, np.ndarray, torch.Tensor)
        ), f"Got observations of invalid type {type(data)}"

        if isinstance(data, TensorDict):
            num_envs = list(data.values())[0].shape[0]
        else:
            num_envs = data.shape[0]

        assert num_envs == self._num_envs, (
            f"Rollout storage was created for {self._num_envs} envs, "
            f"but got an initial {key} for {num_envs} envs instead."
        )

        if isinstance(data, TensorDict):
            for k, v in data.items():
                self.buffers[key][k] = torch.zeros(
                    self.num_steps + 1,
                    self._num_envs,
                    *v.shape,
                    dtype=v.dtype,
                    device=self.device,
                )
        else:
            if isinstance(data, np.ndarray):
                data_type = torch.from_numpy(data).dtype
                if data.ndim == 1:
                    data = np.expand_dims(data, -1)
            else:
                data_type = data.dtype
                if data.dim() == 1:
                    data = data.unsqueeze(1)
            self.buffers[key] = torch.zeros(
                self.num_steps + 1,
                self._num_envs,
                *data.shape[1:],
                dtype=data_type,
                device=self.device,
            )

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        next_observations,
        next_dones,
        other,
        advance=True,
    ):
        # Automatically try to reshape rewards if they seem squeezed
        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards.unsqueeze(-1) if rewards.ndim == 1 else rewards,
            **other,
        )
        next_not_dones = torch.logical_not(next_dones.bool())
        next_step = dict(observations=next_observations, not_dones=next_not_dones)

        # Add new entries from dicts if they don't exist already within buffers
        for d in [current_step, next_step]:
            for k, v in d.items():
                if k not in self.buffers:
                    self.insert_initial_data(k, v)
                if v.dim() == 1:
                    d[k] = v.unsqueeze(1)

        for offset, data in [(0, current_step), (1, next_step)]:
            self.buffers.set(
                self.current_rollout_step_idx + offset,
                data,
                strict=False,
            )
        if advance:
            self.advance_rollout()

    def advance_rollout(self):
        self.current_rollout_step_idx += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]
        self.current_rollout_step_idx = 0

    def compute_returns(self, next_value):
        if self.use_gae:
            self.buffers["value_preds"][self.current_rollout_step_idx] = next_value
        self.buffers["returns"] = compute_returns(
            self.current_rollout_step_idx,
            next_value,
            self.buffers["rewards"],
            self.buffers["value_preds"],
            self.buffers["not_dones"],
            self.use_gae,
            self.gamma,
            self.tau,
            self.value_bootstrap,
            self.buffers.get("time_outs", None),
        )

    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            f"Trainer requires the number of environments ({num_environments}) "
            "to be greater than or equal to the number of "
            f"trainer mini batches ({num_mini_batch})."
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                f"Number of environments ({num_environments}) is not a multiple of the "
                f"number of mini batches ({num_mini_batch}). This results in mini "
                f"batches of different sizes, which can harm training performance."
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[: self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[: self.current_rollout_step_idx, inds]

            yield batch.map(lambda v: v.flatten(0, 1))

    def feed_forward_generator(self, advantages, num_mini_batch) -> TensorDict:
        batch_size = self._num_envs * self.num_steps
        assert batch_size >= num_mini_batch, (
            f"PPO requires the number of envs ({self._num_envs}) "
            f"* number of steps ({self.num_steps}) = {self._num_envs * self.num_steps} "
            "to be greater than or equal to the number"
            f"of PPO mini batches ({num_mini_batch})."
        )

        for inds in torch.randperm(self.current_rollout_step_idx).chunk(num_mini_batch):
            batch = self.buffers[inds]
            batch["advantages"] = advantages[inds]

            yield batch.map(lambda v: v.flatten(0, 1))


@torch.jit.script
def compute_returns(
    current_rollout_step_idx: int,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    value_preds: torch.Tensor,
    not_dones: torch.Tensor,
    use_gae: bool,
    gamma: float,
    tau: float,
    value_bootstrap: bool,
    time_outs: Optional[torch.Tensor] = None,
):
    with torch.no_grad():
        returns = torch.zeros_like(rewards)
        if use_gae:
            gae = torch.zeros_like(rewards[0])
        else:
            gae = torch.zeros(1)  # torchscript needs gae defined in false branch too
            returns[current_rollout_step_idx] = next_value
        for step in range(current_rollout_step_idx - 1, -1, -1):
            if value_bootstrap and time_outs is not None:
                curr_reward = rewards[step] + time_outs[step] * value_preds[step]
            else:
                curr_reward = rewards[step]
            if use_gae:
                delta = (
                    curr_reward
                    + gamma * value_preds[step + 1] * not_dones[step + 1]
                    - value_preds[step]
                )
                gae = delta + gamma * tau * gae * not_dones[step + 1]
                returns[step] = gae + value_preds[step]
            else:
                returns[step] = (
                    curr_reward + gamma * returns[step + 1] * not_dones[step + 1]
                )
    return returns
