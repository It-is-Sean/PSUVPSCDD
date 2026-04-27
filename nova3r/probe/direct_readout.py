from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Direct3DReadout(nn.Module):
    """Shallow direct baseline for point/depth/pose readout."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.point_head = _MLP(input_dim, hidden_dim, 3)
        self.depth_head = _MLP(input_dim, hidden_dim, 1)
        self.pose_head = _MLP(input_dim, hidden_dim, 7)

    def forward(self, pooled_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "pointmap": self.point_head(pooled_tokens),
            "depth": self.depth_head(pooled_tokens),
            "relative_pose": self.pose_head(pooled_tokens),
        }
