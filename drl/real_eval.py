from locomotion.drl.a2c_ppo_acktr import model

from absl import app
from absl import logging
import numpy as np
import pybullet as p  # pytype: disable=import-error
from tqdm import tqdm

from locomotion.envs import locomotion_gym_config
from locomotion.robots import aliengo
from locomotion.robots import aliengo_robot
from locomotion.envs import locomotion_gym_env

import argparse
import torch

sim_params = locomotion_gym_config.SimulationParameters()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .pth file")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    """ Generate config """
    config = checkpoint["config"]
    config.merge_from_list(args.opts)
    config.defrost()

    """ Create environment """
    env = locomotion_gym_env.LocomotionGymEnv(
        gym_config=locomotion_gym_config.LocomotionGymConfig(
            simulation_parameters=sim_params
        ),
        robot_class=aliengo.Aliengo,
        is_render=True,
        on_rack=True,
    )

    """ Create actor-critic """
    num_reward_terms = (
        checkpoint["state_dict"]["actor_critic.base.critic_linear.bias"].shape[0] - 1
    )
    actor_critic = model.Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={
            "recurrent": config.RECURRENT_POLICY,
            "reward_terms": num_reward_terms,
            "hidden_size": config.RL.PPO.hidden_size,
            "mlp_hidden_sizes": config.RL.PPO.mlp_hidden_sizes,
        },
    )
    print("\nActor-critic architecture:")
    print(actor_critic)
    device = torch.device("cuda:0" if config.CUDA else "cpu")
    actor_critic.to(device)

    """ Load weights """
    actor_critic.load_state_dict(
        {
            k[len("actor_critic.") :]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("actor_critic")
        }
    )

    """ Execute episodes """
    num_episodes = 1
    for idx in range(num_episodes):
        observations = env.reset()
        recurrent_hidden_states = torch.zeros(
            1,
            actor_critic.base.recurrent_hidden_state_size,
            config.RL.PPO.hidden_size,
            device=device,
        )
        not_done = torch.ones(1, 1).to(device)
        step_count = 0
        while not_done[0]:
            step_count += 1
            (_, action, _, recurrent_hidden_states,) = actor_critic.act(
                torch.FloatTensor(observations).unsqueeze(0).to(device),
                recurrent_hidden_states,
                not_done,
                deterministic=True,
            )

            observations, _, done, _ = env.step(action[0].detach().cpu().numpy())
            not_done[0] = not done

        print(f"Episode #{idx + 1} finished in {step_count} steps")