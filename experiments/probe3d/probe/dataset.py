import torch
from torch.utils.data import Dataset


class FeaturePointDataset(Dataset):
    """Dataset backed by a feature extraction .pt file."""

    def __init__(self, path: str, pool_features: bool = True):
        data = torch.load(path, map_location="cpu")
        required = {"scene_ids", "features", "target_points"}
        missing = required.difference(data.keys())
        if missing:
            raise KeyError(f"{path} is missing required keys: {sorted(missing)}")
        self.scene_ids = data["scene_ids"]
        self.features = data["features"]
        self.target_points = data["target_points"]
        self.pool_features = pool_features
        if len(self.scene_ids) != self.features.shape[0] or len(self.scene_ids) != self.target_points.shape[0]:
            raise ValueError("scene_ids, features, and target_points must have matching first dimensions")

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        features = self.features[idx].float()
        if self.pool_features and features.ndim == 2:
            features = features.mean(dim=0)
        return {
            "scene_id": self.scene_ids[idx],
            "features": features,
            "target_points": self.target_points[idx].float(),
        }

