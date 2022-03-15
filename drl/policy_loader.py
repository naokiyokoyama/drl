import torch
import torch.nn as nn

CKPT = "/Users/marcodelgado/vloco/locomotion_simulation/checkpoints/flat_anymal_c/Mar03_16-17-12_/model_300.pt"


class FlatTerrainPolicy(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.net = make_flat_terrian_policy(ckpt_path)

    def act(self, observations, deterministic=True):
        mean = self.net.forward(observations)
        if deterministic:
            return mean
        distribution = torch.distributions.Normal(mean, self.net.std)
        return distribution.sample()


def make_flat_terrian_policy(ckpt_path):
    class my_policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.std = nn.Parameter(torch.zeros(12))
            self.actor = nn.Sequential(
                nn.Linear(48, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, 12),
            )

            ckpt = torch.load(ckpt_path, map_location="cpu")
            actor_state_dict = {
                k: v for k, v in ckpt["model_state_dict"].items() if "critic" not in k
            }
            self.load_state_dict(actor_state_dict)

        def forward(self, obs):
            return self.actor(obs)

    return my_policy()

if __name__ == "__main__":
    policy = make_flat_terrian_policy(CKPT)