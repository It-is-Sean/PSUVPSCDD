import torch
from torch import nn


class PointDecoder(nn.Module):
    """Simple MLP decoder from a latent vector to a complete point cloud."""

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 1024, num_points: int = 8192):
        super().__init__()
        self.num_points = num_points
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_points * 3),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        points = self.net(latent)
        return points.view(latent.shape[0], self.num_points, 3)

