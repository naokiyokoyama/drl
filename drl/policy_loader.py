import torch
import torch.nn as nn

CKPT = "/Users/marcodelgado/vloco/locomotion_simulation/checkpoints/flat_anymal_c/Mar03_16-17-12_/model_300.pt"


def make_flat_terrian_policy(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    policy = nn.Module()
    policy.std = nn.Parameter(torch.zeros(12))
    policy.actor = nn.Sequential(
        nn.Linear(48, 128),
        nn.ELU(),
        nn.Linear(128, 64),
        nn.ELU(),
        nn.Linear(64, 32),
        nn.ELU(),
        nn.Linear(32, 12),
    )

    actor_state_dict = {
        k: v for k, v in ckpt["model_state_dict"].items() if "critic" not in k
    }
    policy.load_state_dict(actor_state_dict)

    return policy


if __name__ == "__main__":
    policy = make_flat_terrian_policy(CKPT)
