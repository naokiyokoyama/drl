from locomotion.envs import walking_locomotion, locomotion_gym_config
from locomotion.robots import aliengo, aliengo_robot
from .policy_loader import make_flat_terrian_policy

import argparse
import torch

# sim_params = locomotion_gym_config.SimulationParameters()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to .pth file")
    # parser.add_argument("environment", type=str, help="Which task we want do: 'stand' or 'walk'")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint

    """ Create environment """
    env = walking_locomotion.LocomotionWalk(
        gym_config=locomotion_gym_config.LocomotionGymConfig(
            simulation_parameters=sim_params
        ),
        robot_class=aliengo_robot.AliengoRobot,
        is_render=False,
        on_rack=False,
    )

    """ Load weights """
    actor_critic = make_flat_terrian_policy(checkpoint)

    """ Execute episodes """
    num_episodes = 1
    for idx in range(num_episodes):
        observations = env.reset()

        recurrent_hidden_states = torch.zeros(
            1,
            128,
            48,
            device="cpu",
        )
        not_done = torch.ones(1, 1).to("cpu")
        step_count = 0
        while not_done[0]:
            step_count += 1
            (_, action, _, recurrent_hidden_states,) = actor_critic.act(
                torch.FloatTensor(observations).unsqueeze(0).to("cpu"),
                recurrent_hidden_states,
                not_done,
                deterministic=True,
            )

            observations, _, done, _ = env.step(action[0].detach().cpu().numpy())
            not_done[0] = not done
