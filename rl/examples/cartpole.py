import gym
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, '/Users/naoki/repos/')

from pytorch_rl.algo.ppo import PPO
from pytorch_rl.rl.model import Policy
from pytorch_rl.rl.storage import RolloutStorage

NUM_ROLLOUTS = 50
ROLLOUT_LENGTH = 20
BATCH_SIZE = 5

def main():
    env = gym.make('CartPole-v0')

    # net = nn.Sequential(
    #     nn.Linear(4,256),
    #     nn.ReLU(),
    #     nn.Linear(256,256),
    #     nn.ReLU()
    # )
    # actor_critic = Policy(net, CategoricalNet, 2)
    actor_critic = Policy(
        obs_shape=env.observation_space,
        action_space=env.action_space,
        base=None, 
        base_kwargs=None
    )
    agent = PPO(
        actor_critic=actor_critic,
        clip_param=0.2,
        ppo_epoch=2,
        num_mini_batch=1,
        value_loss_coef=,0.5
        entropy_coef=0.01,
        lr=0.0003,
        eps=None,
        max_grad_norm=0.2,
        use_clipped_value_loss=True
    )


    rollout_storage = RolloutStorage(
        num_steps=ROLLOUT_LENGTH,
        num_processes=1,
        obs_shape=env.observation_space,
        action_space=env.action_space,
        recurrent_hidden_state_size=
    )

    score = 0
    scores = []
    observation = env.reset()
    # torch.autograd.set_detect_anomaly(True)
    for _ in range(NUM_ROLLOUTS):
        # env.render()
        for step in range(ROLLOUT_LENGTH+1):
            (
                value_pred,
                action,
                action_log_probs
            ) = actor_critic.act(
                inputs=torch.Tensor(observation.reshape(1,-1)), 
                rnn_hxs=,
                masks=,
                deterministic=False
            )
            new_observation, reward, done, _ = env.step(action.item())
            score += reward 
            rollout_storage.insert(
                torch.Tensor(observation),
                action.squeeze(),
                action_log_probs.squeeze(),
                value_pred.squeeze(),
                torch.Tensor([reward]).squeeze(),
                torch.Tensor([done]).squeeze(),
            )
            if not done:
                observation = new_observation
            else:
                scores.append(score)
                score = 0
                if len(scores) == 50:
                    print(f'Avg score: {np.mean(scores)}')
                    scores = []
                observation = env.reset()

        agent.update(rollout_storage)
        rollout_storage.after_update()

main()