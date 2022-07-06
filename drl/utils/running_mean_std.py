import torch
import torch.nn as nn

"""
updates statistic from a full data
"""


class RunningMeanStdAtomic(nn.Module):
    def __init__(self, in_shape, epsilon=1e-05, per_channel=False, norm_only=False):
        super().__init__()
        self.in_shape = in_shape
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.in_shape) == 3:
                self.axis = [0, 2, 3]
            if len(self.in_shape) == 2:
                self.axis = [0, 2]
            if len(self.in_shape) == 1:
                self.axis = [0]
            in_size = self.in_shape[0]
        else:
            self.axis = [0]
            in_size = in_shape

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    @staticmethod
    def _update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis)  # along channel axis
            var = input.var(self.axis)
            (
                self.running_mean,
                self.running_var,
                self.count,
            ) = self._update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size()[0],
            )

        # change shape
        if self.per_channel:
            if len(self.in_shape) == 3:
                current_mean = self.running_mean.view(
                    [1, self.in_shape[0], 1, 1]
                ).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.in_shape[0], 1, 1]
                ).expand_as(input)
            elif len(self.in_shape) == 2:
                current_mean = self.running_mean.view(
                    [1, self.in_shape[0], 1]
                ).expand_as(input)
                current_var = self.running_var.view([1, self.in_shape[0], 1]).expand_as(
                    input
                )
            elif len(self.in_shape) == 1:
                current_mean = self.running_mean.view([1, self.in_shape[0]]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.in_shape[0]]).expand_as(
                    input
                )
            else:
                raise RuntimeError(f"{self.in_shape} in_shape not valid length!")
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = (
                torch.sqrt(current_var.float() + self.epsilon) * y
                + current_mean.float()
            )
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(
                    current_var.float() + self.epsilon
                )
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y


class RunningMeanStd(RunningMeanStdAtomic):
    def __init__(self, obs_space, epsilon=1e-05, per_channel=False, norm_only=False):
        if isinstance(obs_space, dict):
            super(RunningMeanStdAtomic, self).__init__()
            self.running_mean_std = nn.ModuleDict(
                {
                    k: RunningMeanStdAtomic(v, epsilon, per_channel, norm_only)
                    for k, v in obs_space.items()
                }
            )
        else:
            in_shape = obs_space if isinstance(obs_space, tuple) else obs_space.shape
            super().__init__(
                in_shape, epsilon=1e-05, per_channel=False, norm_only=False
            )
            self.running_mean_std = None

    def forward(self, input, unnorm=False):
        if self.running_mean_std is None:
            return super().forward(input, unnorm)
        return {k: self.running_mean_std(v, unnorm) for k, v in input.items()}
