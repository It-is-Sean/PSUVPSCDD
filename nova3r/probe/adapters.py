from __future__ import annotations

import torch
from torch import nn


class FeatureProjector(nn.Module):
    """Project arbitrary backbone tokens to the canonical token width."""

    def __init__(self, input_dim: int, token_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(tokens))


class SceneTokenAdapter(nn.Module):
    """
    Small adapter: frozen source tokens -> fixed number of scene tokens.

    This is intentionally lightweight so that the probe remains a measurement
    of the frozen representation rather than a large learned reconstruction model.
    """

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 768,
        scene_tokens: int = 32,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        refinement_layers: int = 0,
    ) -> None:
        super().__init__()
        self.projector = FeatureProjector(input_dim=input_dim, token_dim=token_dim)
        self.scene_queries = nn.Parameter(torch.randn(scene_tokens, token_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden_dim = int(token_dim * mlp_ratio)
        self.post = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.refinement = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(refinement_layers)
        ])

    def forward(self, source_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if source_tokens.ndim != 3:
            raise ValueError(f"Expected [B, L, C] tokens, got shape={tuple(source_tokens.shape)}")

        source_tokens = self.projector(source_tokens)
        batch_size = source_tokens.shape[0]
        scene_queries = self.scene_queries.unsqueeze(0).expand(batch_size, -1, -1)
        aggregated, _ = self.cross_attn(
            query=scene_queries,
            key=source_tokens,
            value=source_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        aggregated = aggregated + self.post(aggregated)
        for block in self.refinement:
            aggregated = block(aggregated)
        return aggregated
