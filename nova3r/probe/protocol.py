from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class FrozenRepresentation:
    """Unified container for frozen backbone features."""

    tokens: torch.Tensor
    padding_mask: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeBatch:
    """Minimal batch contract for the probing pipeline."""

    representation: FrozenRepresentation
    complete_points: torch.Tensor
    visible_mask: Optional[torch.Tensor] = None
    unseen_mask: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalDecoderSpec:
    checkpoint_path: str
    latent_tokens: int
    token_dim: int
    decoder_query_budget: int
    inference_steps: int = 100
    solver: str = "euler"
