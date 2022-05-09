import argparse

import torch
from run_old import GymWrappedEnv

from drl.nets.model import Policy

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

    checkpoint = torch.load(args.checkpoint)

    """ Generate config """
    config = checkpoint["config"]
    config.merge_from_list(args.opts)
    config.defrost()

    """ Create environment """
    env = GymWrappedEnv(config, render=True)

    """ Create actor-critic """
    num_reward_terms = (
        checkpoint["state_dict"]["actor_critic.base.critic_linear.bias"].shape[0] - 1
    )
    actor_critic = Policy(
        env.observation_space.shape,
        env.action_space,
        base_kwargs={
            "recurrent": config.RECURRENT_POLICY,
            "reward_terms": num_reward_terms,
            "hidden_size": config.RL.PPO.hidden_size,
            "mlp_hidden_sizes": config.RL.PPO.mlp_hidden_sizes,
            "init_weights": False,
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
