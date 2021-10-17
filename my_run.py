from collections import deque
import numpy as np
import os
import time
import torch
from yacs.config import CfgNode as CN
import argparse
import gym
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.vector_env import VectorEnv
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from envs.knobs_env import KnobsEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Path to .yaml file containing parameters"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = CN(new_allowed=True)
    config.merge_from_file(args.config_file)
    if args.opts is not None:
        config.merge_from_list(args.opts)
    config.freeze()

    run(config, KnobsEnv)
    # run(config, 'MountainCarContinuous-v0')
    # run(config, 'Pendulum-v0')


def run(config, env_class):
    """Runs RL training base on config"""

    """ Set seeds """
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    """ CUDA vs. CPU """
    if config.CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.CUDA else "cpu")

    """ Create environments """
    if type(env_class) == str:
        make_env_fn = gym.make
        env_fn_args = tuple([tuple([env_class])] * config.NUM_ENVIRONMENTS)
    else:
        make_env_fn = env_class
        env_fn_args = tuple(
            [
                (config.ENVIRONMENT, config.SEED + seed)
                for seed in range(config.NUM_ENVIRONMENTS)
            ]
        )

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=env_fn_args,
    )

    """ Create policy """
    actor_critic = Policy(
        envs.observation_spaces[0].shape,
        envs.action_spaces[0],
        base_kwargs={
            "recurrent": config.RECURRENT_POLICY,
            "reward_terms": config.RL.PPO.reward_terms,
        },
    )
    actor_critic.to(device)

    """ Setup PPO """
    agent = algo.PPO(
        actor_critic,
        config.RL.PPO.clip_param,
        config.RL.PPO.ppo_epoch,
        config.RL.PPO.num_mini_batch,
        config.RL.PPO.value_loss_coef,
        config.RL.PPO.entropy_coef,
        lr=config.RL.PPO.lr,
        eps=config.RL.PPO.eps,
        max_grad_norm=config.RL.PPO.max_grad_norm,
        expressive_critic=config.RL.PPO.reward_terms > 0,
    )

    """ Set up rollout storage """
    rollouts = RolloutStorage(
        config.RL.PPO.num_steps,
        config.NUM_ENVIRONMENTS,
        envs.observation_spaces[0].shape,
        envs.action_spaces[0],
        actor_critic.recurrent_hidden_state_size,
        reward_terms=config.RL.PPO.reward_terms,
    )

    obs = envs.reset()
    obs = torch.FloatTensor(obs)

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=config.RL.PPO.reward_window_size)
    episode_successes = deque(maxlen=config.RL.PPO.reward_window_size)
    episode_cumul_rewards = deque(maxlen=config.RL.PPO.reward_window_size)

    """ Start training """
    # Create tensorboard if path was specified
    if config.TENSORBOARD_DIR != "":
        print(f"Creating tensorboard at '{config.TENSORBOARD_DIR}'...")
        if not os.path.isdir(config.TENSORBOARD_DIR):
            os.makedirs(config.TENSORBOARD_DIR)
        writer = SummaryWriter(config.TENSORBOARD_DIR)

    # Calculate number of updates
    if config.NUM_UPDATES < 0:
        num_updates = (
            int(config.TOTAL_NUM_STEPS)
            // config.RL.PPO.num_steps
            // config.NUM_ENVIRONMENTS
        )
    else:
        num_updates = config.NUM_UPDATES

    start = time.time()
    for j in range(num_updates):

        if config.RL.PPO.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, config.RL.PPO.lr
            )

        for step in range(config.RL.PPO.num_steps):
            # Sample actions
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Observe reward and next obs
            outputs = envs.step(action.cpu().numpy())
            obs, reward, done, infos = [list(x) for x in zip(*outputs)]

            if config.RL.PPO.reward_terms > 0:
                reward_terms = []
            else:
                reward_terms = None
            for info_ in infos:
                if info_.get("success", False):
                    episode_successes.append(1.0)
                    episode_cumul_rewards.append(info_["cumul_reward"])
                if info_.get("failed", False):
                    episode_successes.append(0.0)
                    episode_cumul_rewards.append(info_["cumul_reward"])
                if config.RL.PPO.reward_terms > 0:
                    reward_terms.append(info_["reward_terms"])

            # envs.render(mode='rgb_array')

            episode_rewards.extend(reward)
            obs = torch.FloatTensor(obs)
            reward = torch.FloatTensor(reward).unsqueeze(1)
            if config.RL.PPO.reward_terms > 0:
                reward_terms = torch.FloatTensor(reward_terms)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                reward_terms=reward_terms,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            config.RL.PPO.use_gae,
            config.RL.PPO.gamma,
            config.RL.PPO.tau,
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % config.LOG_INTERVAL == 0 and len(episode_rewards) > 1:
            total_num_steps = (
                (j + 1) * config.NUM_ENVIRONMENTS * config.RL.PPO.num_steps
            )
            end = time.time()
            print(
                "Update {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".format(
                    j,
                    total_num_steps,
                    int(
                        config.NUM_ENVIRONMENTS
                        * config.RL.PPO.num_steps
                        * config.LOG_INTERVAL
                        / (end - start)
                    ),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                )
            )

            if not episode_successes:
                mean_success = 0
                mean_cumul_reward = 0
            else:
                mean_success = np.mean(episode_successes)
                mean_cumul_reward = np.mean(episode_cumul_rewards)
            print(
                "Mean success: ",
                mean_success,
                "Mean cumul reward: ",
                mean_cumul_reward,
                "\n",
            )

            print(
                f"CSV:{j},{total_num_steps},{mean_cumul_reward},{mean_success},"
                f"{value_loss},{action_loss},{dist_entropy}"
            )

            # Update tensorboard
            if config.TENSORBOARD_DIR != "":
                data = {
                    "success": mean_success,
                    "cumulative_reward": mean_cumul_reward,
                    "value_loss": value_loss,
                    "action_loss": action_loss,
                    "dist_entropy": dist_entropy,
                }
                writer.add_scalars("steps", data, total_num_steps)
                writer.add_scalars("updates", data, j)

            start = time.time()


if __name__ == "__main__":
    main()
