import torch

from drl.algo.ppo import PPO
from drl.runners.base_runner import BaseTrainer
from drl.utils.registry import drl_registry
from drl.utils.rollout_storage import RolloutStorage


@drl_registry.register_runner
class PPOTrainer(BaseTrainer):
    algo_cls = PPO
    term_by_term_returns = False

    def __init__(self, config, envs=None):
        super().__init__(config, envs)
        self.algo = self.algo_cls.from_config(self.config, self.actor_critic)
        self.rollouts = RolloutStorage.from_config(
            self.config, self.num_envs, self.device, self.initial_observations
        )
        self.curr_step = 0
        # torch.save(self.actor_critic.state_dict(), "test_weights.pth")
        pretrained_state = torch.load("test_weights.pth", map_location="cpu")
        orig = self.actor_critic.state_dict()
        orig.update({k: v for k, v in pretrained_state.items() if k in orig})
        self.actor_critic.load_state_dict(orig)

    def step(self, observations):
        self.set_seed(self.curr_step)
        observations = torch.rand_like(observations)
        with torch.no_grad():
            (
                value,
                actions,
                action_log_probs,
                other,
            ) = self.actor_critic.act(observations, deterministic=True)
        print("action:", actions)

        observations, rewards, dones, infos = self.step_envs(actions)
        self.set_seed(self.curr_step)
        rewards = torch.rand_like(rewards)
        dones = torch.zeros_like(dones)
        infos["reward_terms"] = torch.rand_like(infos["reward_terms"])
        self.curr_step += 1
        other.update(infos)

        self.rollouts.insert(
            next_observations=observations,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value,
            rewards=rewards,
            next_dones=dones,
            other=other,
        )
        return observations

    def update(self, observations):
        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                observations, get_terms=self.term_by_term_returns
            )
        self.rollouts.compute_returns(
            next_value, term_by_term_returns=self.term_by_term_returns
        )
        data = self.algo.update(self.rollouts)
        print(data)
        self.write_data.update(data)
