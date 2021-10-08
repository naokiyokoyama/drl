import time
from collections import deque

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.vector_env import VectorEnv
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from yacs.config import CfgNode as CN
import argparse
import gym

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
    config.freeze()

    # run(config, env_class)
    run(config, 'MountainCarContinuous-v0')


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

    """ Create environment """
    # if type(env_class) == str:
    #     make_env_fn = gym.make
    # else:
    #     make_env_fn = env_class
    # envs = VectorEnv(
    #     make_env_fn=make_env_fn,
    #     env_fn_args=tuple([tuple([config])] * config.NUM_ENVIRONMENTS),
    # )
    env_name = 'MountainCarContinuous-v0'
    vec_env_args = [tuple([env_name])] * config.NUM_ENVIRONMENTS
    envs = VectorEnv(
        make_env_fn=gym.make,
        env_fn_args=tuple(vec_env_args)
    )

    """ Create policy """
    actor_critic = Policy(
        envs.observation_spaces[0].shape,
        envs.action_spaces[0],
        base_kwargs={"recurrent": config.RECURRENT_POLICY},
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
    )

    """ Set up rollout storage """
    rollouts = RolloutStorage(
        config.RL.PPO.num_steps,
        config.NUM_ENVIRONMENTS,
        envs.observation_spaces[0].shape,
        envs.action_spaces[0],
        actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    obs = torch.FloatTensor(obs)

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=config.RL.PPO.reward_window_size)

    """ Start training """

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

            # Obser reward and next obs
            outputs = envs.step(action)
            obs, reward, done, infos = [list(x) for x in zip(*outputs)]
            # envs.render(mode='rgb_array')

            episode_rewards.extend(reward)
            obs = torch.FloatTensor(obs)
            reward = torch.FloatTensor(reward).unsqueeze(1)

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
            total_num_steps = (j + 1) * config.NUM_ENVIRONMENTS * config.TOTAL_NUM_STEPS
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
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


if __name__ == "__main__":
    main()
