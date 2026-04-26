import torch
from torch.utils.data import Dataset


class FeaturePointDataset(Dataset):
    """Dataset backed by a feature extraction .pt file."""

    def __init__(self, path: str, pool_features: bool = True, split: str | None = None):
        data = torch.load(path, map_location="cpu")
        required = {"scene_ids", "features", "target_points"}
        missing = required.difference(data.keys())
        if missing:
            raise KeyError(f"{path} is missing required keys: {sorted(missing)}")
        self.scene_ids = data["scene_ids"]
        self.features = data["features"]
        self.target_points = data["target_points"]
        self.pool_features = pool_features
        self.metadata = data.get("metadata")
        self.splits = data.get("splits")
        if len(self.scene_ids) != self.features.shape[0] or len(self.scene_ids) != self.target_points.shape[0]:
            raise ValueError("scene_ids, features, and target_points must have matching first dimensions")
        if self.splits is not None and len(self.splits) != len(self.scene_ids):
            raise ValueError("splits must have the same length as scene_ids")
        if self.metadata is not None and len(self.metadata) != len(self.scene_ids):
            raise ValueError("metadata must have the same length as scene_ids")

        self.indices = list(range(len(self.scene_ids)))
        if split is not None:
            if self.splits is None:
                raise ValueError(f"{path} does not contain split labels; cannot select split={split!r}")
            self.indices = [idx for idx, sample_split in enumerate(self.splits) if sample_split == split]
            if not self.indices:
                raise ValueError(f"{path} does not contain any samples for split={split!r}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        src_idx = self.indices[idx]
        features = self.features[src_idx].float()
        if self.pool_features and features.ndim == 2:
            features = features.mean(dim=0)
        item = {
            "scene_id": self.scene_ids[src_idx],
            "features": features,
            "target_points": self.target_points[src_idx].float(),
        }
        if self.splits is not None:
            item["split"] = self.splits[src_idx]
        if self.metadata is not None:
            item["metadata"] = self.metadata[src_idx]
        return item
