from __future__ import annotations

from typing import Dict, Optional

import torch


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Simple symmetric Chamfer distance for point sets `[B, N, 3]` / `[B, M, 3]`."""
    dist = torch.cdist(x, y)
    return dist.min(dim=-1).values.mean(dim=-1) + dist.min(dim=-2).values.mean(dim=-1)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    err = (pred - target).abs()
    if mask is None:
        return err.mean()
    mask = mask.to(dtype=err.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (err * mask).sum() / denom


def pose_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def visible_unseen_chamfer(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    visible_mask: Optional[torch.Tensor],
    unseen_mask: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    metrics = {"cd_all": chamfer_distance(pred_points, gt_points).mean()}
    if visible_mask is not None:
        metrics["cd_visible"] = chamfer_distance(pred_points, gt_points * visible_mask.unsqueeze(-1)).mean()
    if unseen_mask is not None:
        metrics["cd_unseen"] = chamfer_distance(pred_points, gt_points * unseen_mask.unsqueeze(-1)).mean()
    return metrics
