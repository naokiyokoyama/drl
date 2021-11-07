from collections import deque, defaultdict, OrderedDict
from functools import partial
import numpy as np
import os
import time
import torch
import argparse
from gym import spaces
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.vector_env import VectorEnv
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from envs.knobs_env import KnobsEnv

# Habitat-specific
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.config.default import get_config

HEADER = (
    "updates,steps,mean_cumul_reward,mean_reward,mean_success,"
    "value_loss,action_loss,dist_entropy\n"
)

# We need this to set the seeds for each VectorEnv independently
def make_env_fn(env_class, seed_offset):
    env = env_class(config)
    env.seed(config.TASK_CONFIG.SEED + seed_offset)
    return env


def run(config, env_class):
    """Runs RL training base on config"""

    """ Set seeds """
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.cuda.manual_seed_all(config.TASK_CONFIG.SEED)

    """ CUDA vs. CPU """
    if config.CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.CUDA else "cpu")

    """ Count reward terms for expressive critic """
    if config.RL.PPO.loss_type in ["", "regular"]:
        # Expressive critic not being used; don't predict reward terms
        num_reward_terms = 0
    else:
        # Create a temporary env just to count reward terms
        temp_env = env_class(config)
        temp_env.reset()
        _, _, _, info = temp_env.step(temp_env.action_space.sample())
        num_reward_terms = info["reward_terms"].shape[0]
        del temp_env

    """ Create vector environments """
    env_fn_args = tuple(
        [(env_class, seed_offset) for seed_offset in range(config.NUM_ENVIRONMENTS)]
    )
    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=env_fn_args,
    )

    """ Create actor-critic """
    actor_critic = Policy(
        envs.observation_spaces[0].shape,
        envs.action_spaces[0],
        base_kwargs={
            "recurrent": config.RECURRENT_POLICY,
            "reward_terms": num_reward_terms,
            "hidden_size": config.RL.PPO.hidden_size,
            "mlp_hidden_sizes": config.RL.PPO.mlp_hidden_sizes,
        },
    )
    print("\nActor-critic architecture:")
    print(actor_critic)
    actor_critic.to(device)

    """ Setup PPO """
    agent = algo.PPO(
        actor_critic,
        config.RL.PPO.clip_param,
        config.RL.PPO.ppo_epoch,
        config.RL.PPO.num_mini_batch,
        config.RL.PPO.value_loss_coef,
        config.RL.PPO.entropy_coef,
        config.RL.PPO.get('expressive_action_loss_coef', 0.0),
        lr=config.RL.PPO.lr,
        eps=config.RL.PPO.eps,
        max_grad_norm=config.RL.PPO.max_grad_norm,
        use_normalized_advantage=config.RL.PPO.use_normalized_advantage,
        loss_type=config.RL.PPO.loss_type,
    )
    if num_reward_terms > 0:
        print("# of reward terms (using expressive critic!):", num_reward_terms)

    """ Set up rollout storage """
    rollouts = RolloutStorage(
        config.RL.PPO.num_steps,
        config.NUM_ENVIRONMENTS,
        envs.observation_spaces[0].shape,
        envs.action_spaces[0],
        actor_critic.recurrent_hidden_state_size,
        reward_terms=num_reward_terms,
    )

    obs = envs.reset()
    obs = torch.FloatTensor(obs)

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    """ Setup metrics buffers """
    episode_cumul_rewards = defaultdict(
        partial(deque, maxlen=config.RL.PPO.reward_window_size)
    )
    episode_metrics = defaultdict(
        partial(deque, maxlen=config.RL.PPO.reward_window_size)
    )

    """ Start training """
    # Create tensorboard if path was specified
    if config.TENSORBOARD_DIR != "":
        print(f"Creating tensorboard at '{config.TENSORBOARD_DIR}'...")
        if not os.path.isdir(config.TENSORBOARD_DIR):
            os.makedirs(config.TENSORBOARD_DIR)
        writer = SummaryWriter(config.TENSORBOARD_DIR)
    else:
        writer = None

    # Create checkpoints folder if path was specified
    checkpoint_dir = config.CHECKPOINT_FOLDER
    if checkpoint_dir != "" and not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create log file
    if config.LOG_FILE != "":
        with open(config.LOG_FILE, "w") as f:
            f.write(HEADER)

    # Calculate number of updates
    if config.NUM_UPDATES < 0:
        num_updates = (
            int(config.TOTAL_NUM_STEPS)
            // config.RL.PPO.num_steps
            // config.NUM_ENVIRONMENTS
        )
    else:
        num_updates = config.NUM_UPDATES

    # Train!
    start = time.time()
    checkpoint_id = 0
    for update_idx in range(num_updates):

        if config.RL.PPO.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, update_idx, num_updates, config.RL.PPO.lr
            )

        # Collect a full rollout from every VectorEnv
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

            if num_reward_terms > 0:
                reward_terms = []
            else:
                reward_terms = None
            for idx, info_ in enumerate(infos):
                # Get rewards terms for expressive critic if needed
                if num_reward_terms > 0:
                    reward_terms.append(info_["reward_terms"])

                # Get terminal metrics for ended episodes (e.g., success, cumul_reward)
                if done[idx]:
                    for k, v in info_.items():
                        # Assumes keys in info with 'cumul' are about rewards
                        if "cumul" in k:
                            episode_cumul_rewards[k].append(v)
                        elif type(v) in [float, int, bool]:
                            episode_metrics[k].append(float(v))

            obs = torch.FloatTensor(obs)
            reward = torch.FloatTensor(reward).unsqueeze(1)
            if num_reward_terms > 0:
                reward_terms = torch.FloatTensor(reward_terms)

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

        losses_data = agent.update(rollouts)

        rollouts.after_update()

        if update_idx % config.LOG_INTERVAL == 0 and update_idx > 0:
            total_num_steps = (
                (update_idx + 1) * config.NUM_ENVIRONMENTS * config.RL.PPO.num_steps
            )
            end = time.time()
            fps = int(
                config.NUM_ENVIRONMENTS
                * config.RL.PPO.num_steps
                * config.LOG_INTERVAL
                / (end - start)
            )
            rewards_data = {k: np.mean(v) for k, v in episode_cumul_rewards.items()}
            metrics_data = {k: np.mean(v) for k, v in episode_metrics.items()}
            # Sort each dictionary alphabetically and collect into a list
            all_data = [
                OrderedDict(sorted(d.items(), key=lambda t: t[0]))
                for d in [rewards_data, metrics_data, losses_data]
            ]

            # Print fps, rewards, metrics, losses
            print("\nupdate:", update_idx, "steps:", total_num_steps, "fps:", fps)
            for print_data in all_data:
                if len(print_data) > 0:  # skip empty dicts
                    print(" ".join([f"{k}: {v:.3f}" for k, v in print_data.items()]))

            if config.LOG_FILE != "":
                # Create/overwrite file with header if this is the first log iteration
                if update_idx == config.LOG_INTERVAL:
                    csv_header = (
                        ",".join([",".join([k for k in d.keys()]) for d in all_data])
                        + ",steps"
                    )
                    with open(config.LOG_FILE, "w") as f:
                        f.write(csv_header + "\n")

                # Append to the end of the existing file
                csv_values = (
                    ",".join([",".join([str(v) for v in d.values()]) for d in all_data])
                    + f",{total_num_steps}"
                )
                with open(config.LOG_FILE, "a") as f:
                    f.write(csv_values + "\n")

            # Update tensorboard
            if writer is not None:
                writer.add_scalars("rewards", rewards_data, update_idx)
                writer.add_scalars("losses", losses_data, update_idx)
                writer.add_scalars("metrics", metrics_data, update_idx)
            start = time.time()

        # Save checkpoint
        if (
            checkpoint_dir != ""
            and update_idx % (num_updates // config.NUM_CHECKPOINTS) == 0
        ):
            checkpoint = {
                "state_dict": agent.state_dict(),
                "config": config,
            }

            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f"ckpt.{checkpoint_id}.pth"),
            )
            checkpoint_id += 1


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
if "JUNK" in args.opts:
    args.opts.pop(args.opts.index("JUNK"))
    args.opts.extend(["TENSORBOARD_DIR", "", "CHECKPOINT_FOLDER", "", "LOG_FILE", ""])

# Create config, overriding values with those provided through command line args
config = get_config(args.config_file)
config.merge_from_list(args.opts)
config.defrost()

# Assume expressive critic is not being used if not specified
if "loss_type" not in config.RL.PPO:
    config.RL.PPO.loss_type = ""
if "CUDA" not in config:
    config.CUDA = True
if "RECURRENT_POLICY" not in config:
    config.RECURRENT_POLICY = False

config.freeze()


class GymWrappedEnv(get_env_class(config.ENV_NAME)):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        We naively assume that every observation and action is 1D, and simply 
        concatenate them together.
        """
        self.observation_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(sum([v.shape[0] for v in self.observation_space.spaces.values()]),),
            dtype=np.float32,
        )

        self.action_space_description = {
            k: v.shape[0] for k, v in self.action_space.spaces.items()
        }
        self.action_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(sum(list(self.action_space_description.values())),),
            dtype=np.float32,
        )
        self._cumul_reward = 0

    def step(self, action):
        action_args = {}
        action_offset = 0
        for action_name, action_length in self.action_space_description.items():
            action_args[action_name] = action[action_offset:action_length]
            action_offset += action_length

        # Parent env may be updating self._cumul_reward, so we save prestep value
        cumul_reward_prestep = self._cumul_reward
        observations, reward, done, info = super().step("null", action_args)

        # Convert observation dictionary to array
        observations = np.concatenate(list(observations.values()))
        self._cumul_reward = cumul_reward_prestep + reward
        info["cumul_reward"] = self._cumul_reward
        return observations, reward, done, info

    def reset(self, *args, **kwargs):
        observations = super().reset(*args, **kwargs)
        observations = np.concatenate(list(observations.values()))
        self._cumul_reward = 0

        return observations


if __name__ == "__main__":
    run(config, GymWrappedEnv)
