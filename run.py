import os
import time
from collections import deque

import gym
import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.vector_env import VectorEnv
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env_name = 'MountainCarContinuous-v0'
    # env_name = 'CartPole-v0'
    vec_env_args = [tuple([env_name])]*args.num_processes
    envs = VectorEnv(
        make_env_fn=gym.make,
        env_fn_args=tuple(vec_env_args)
    )

    actor_critic = Policy(
        envs.observation_spaces[0].shape,
        envs.action_spaces[0],
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm
    )
    # 0.2 4 32 0.5 0.01 0.0007 1e-05 0.5
    # 0.2 2 2 0.5 0.0001 0.0003 1e-05 0.5

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_spaces[0].shape,
        envs.action_spaces[0],
        actor_critic.recurrent_hidden_state_size
    )

    obs = envs.reset()
    # obs = torch.from_numpy(obs).float()
    obs = torch.FloatTensor(obs)

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                args.lr
            )

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]
                )

            # Obser reward and next obs
            # obs, reward, done, infos = envs.step(action)
            outputs = envs.step(action)
            obs, reward, done, infos = [list(x) for x in zip(*outputs)]
            # envs.render(mode='rgb_array')
            for idx, d in enumerate(done):
                if d:
                    reward[idx] = -10.0
            # print(done, reward)

            episode_rewards.extend(reward)
            # print('mean reward:', np.mean(episode_rewards))
            obs = torch.FloatTensor(obs)
            reward = torch.FloatTensor(reward).unsqueeze(1)


            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

if __name__ == "__main__":
    main()
