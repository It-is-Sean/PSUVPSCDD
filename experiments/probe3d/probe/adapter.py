import torch
from torch import nn


class SmallAdapter(nn.Module):
    """Intentionally small probe adapter from frozen features to latent space."""

    def __init__(self, input_dim: int, latent_dim: int = 512, depth: int = 1):
        super().__init__()
        if depth not in (0, 1, 2):
            raise ValueError(f"depth must be 0, 1, or 2; got {depth}")

        if depth == 0:
            self.net = nn.Linear(input_dim, latent_dim)
        else:
            layers = []
            dim = input_dim
            for _ in range(depth):
                layers.extend([nn.Linear(dim, latent_dim), nn.GELU()])
                dim = latent_dim
            self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

