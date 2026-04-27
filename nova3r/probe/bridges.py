from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroShotSceneTokenBridge(nn.Module):
    """Training-free bridge from arbitrary frozen tokens to canonical scene tokens.

    This is intentionally simple and lossy. It is not the final learned adapter from
    the proposal; it is a quick sanity bridge for wiring external representations
    (e.g. VGGT) directly into the frozen NOVA3R Stage-1 decoder.
    """

    def __init__(self, target_tokens: int, target_dim: int, normalize: bool = True):
        super().__init__()
        self.target_tokens = int(target_tokens)
        self.target_dim = int(target_dim)
        self.normalize = normalize

    def forward(self, source_tokens: torch.Tensor) -> torch.Tensor:
        if source_tokens.ndim == 4:
            b, s, p, c = source_tokens.shape
            source_tokens = source_tokens.reshape(b, s * p, c)
        elif source_tokens.ndim != 3:
            raise ValueError(
                f'Expected source tokens with shape [B, L, C] or [B, S, P, C], got {tuple(source_tokens.shape)}'
            )

        b, l, c = source_tokens.shape
        x = source_tokens.float()
        if self.normalize:
            x = F.layer_norm(x, (c,))

        if c != self.target_dim:
            x = x.reshape(b * l, 1, c)
            x = F.adaptive_avg_pool1d(x, self.target_dim)
            x = x.reshape(b, l, self.target_dim)

        if l != self.target_tokens:
            x = x.transpose(1, 2)  # [B, C, L]
            x = F.adaptive_avg_pool1d(x, self.target_tokens)
            x = x.transpose(1, 2)

        return x.contiguous()
