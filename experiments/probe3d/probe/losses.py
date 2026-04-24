import torch


def chamfer_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer L2 distance implemented with torch.cdist."""
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[-1] != 3 or gt.shape[-1] != 3:
        raise ValueError(f"Expected pred/gt shapes [B,N,3] and [B,M,3], got {pred.shape} and {gt.shape}")
    dists = torch.cdist(pred.float(), gt.float(), p=2).pow(2)
    return dists.min(dim=2).values.mean() + dists.min(dim=1).values.mean()

