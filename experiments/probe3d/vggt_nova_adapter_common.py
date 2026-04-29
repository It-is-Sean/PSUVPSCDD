from __future__ import annotations

import json
import importlib.machinery
import os
import random
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


PROBE3D_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROBE3D_ROOT.parents[1]
DEFAULT_REFERENCE_ROOT = REPO_ROOT
DEFAULT_VGGT_REPO = REPO_ROOT / "third_party" / "vggt"
DEFAULT_VGGT_WEIGHTS_CANDIDATES = [
    Path("/data1/jcd_data/cache/models/vggt/VGGT-1B/model.pt"),
    REPO_ROOT / "checkpoints" / "vggt" / "model.pt",
    REPO_ROOT / "artifacts" / "weights" / "vggt" / "VGGT-1B" / "model.pt",
]
DEFAULT_NOVA_CKPT_CANDIDATES = [
    REPO_ROOT / "checkpoints/scene_ae/checkpoint-last.pth",
    Path.home() / ".openclaw/workspace/projects/probe/checkpoints/scene_ae/checkpoint-last.pth",
]
DEFAULT_NOVA_CKPT = DEFAULT_NOVA_CKPT_CANDIDATES[0]
DEFAULT_DUST3R_SRC = Path.home() / "CUT3R/src"
DEFAULT_ADAPTER_DATA = PROBE3D_ROOT / "result/scrream_adapter_full_seed17.pt"
DEFAULT_SCANNET_ROOT = Path("/data1/jcd_data/scannet_processed_large")


def resolve_vggt_weights(weights_path: str | None = None) -> Path | None:
    if weights_path:
        return Path(weights_path)
    for candidate in DEFAULT_VGGT_WEIGHTS_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def resolve_nova_ckpt(ckpt_path: str | Path | None = None) -> Path:
    if ckpt_path is not None:
        path = Path(ckpt_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Requested NOVA checkpoint does not exist: {path}")
    for candidate in DEFAULT_NOVA_CKPT_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(p) for p in DEFAULT_NOVA_CKPT_CANDIDATES)
    raise FileNotFoundError(f"Could not find a NOVA Stage-1 checkpoint. Searched:\n{searched}")


def add_repo_paths() -> None:
    # Prefer the integrated repo copy first. Fall back to the existing CUT3R
    # checkout only if the local repo does not yet contain dust3r.datasets.
    search_paths = [DEFAULT_VGGT_REPO, REPO_ROOT]
    if not (REPO_ROOT / "dust3r" / "datasets").exists():
        search_paths.append(DEFAULT_DUST3R_SRC)
    for path in search_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
        elif path.exists():
            sys.path.remove(str(path))
            sys.path.insert(0, str(path))


def ensure_accelerate_stub() -> None:
    if "accelerate" in sys.modules:
        return
    module = types.ModuleType("accelerate")

    class Accelerator:
        num_processes = 1

    module.Accelerator = Accelerator
    module.__version__ = "0.0.0"
    module.__spec__ = importlib.machinery.ModuleSpec("accelerate", loader=None)
    sys.modules["accelerate"] = module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        print("Requested cuda but CUDA is not available; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(device)


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_vggt(device: torch.device, weights_path: str | None = None, model_id: str = "facebook/VGGT-1B"):
    add_repo_paths()
    from vggt.models.vggt import VGGT

    weights = resolve_vggt_weights(weights_path)
    if weights is not None and weights.exists():
        model = VGGT()
        payload = torch.load(weights, map_location="cpu")
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Failed to load VGGT weights cleanly: missing={missing}, unexpected={unexpected}")
        print(f"Loaded VGGT weights from {weights}")
    else:
        model = VGGT.from_pretrained(model_id)
        print(f"Loaded VGGT from pretrained model id {model_id}")
    model.to(device).eval().requires_grad_(False)
    return model


def extract_vggt_features(vggt, images: torch.Tensor, amp: bool = True) -> tuple[list[torch.Tensor], int]:
    if images.ndim == 4:
        images = images.unsqueeze(1)
    with torch.no_grad():
        with amp_context(images.device, amp):
            features, patch_start_idx = vggt.aggregator(images)
    return list(features), int(patch_start_idx)


def select_vggt_layer23(features: list[torch.Tensor]) -> tuple[torch.Tensor, int, str]:
    if len(features) < 1:
        raise ValueError("VGGT returned no intermediate features.")
    idx = 22
    if idx >= len(features):
        idx = len(features) - 1
        reason = (
            f"VGGT returned {len(features)} intermediate features; using the last available feature index {idx} "
            "instead of human block 23."
        )
    else:
        reason = (
            f"VGGT returned {len(features)} intermediate features; human 23rd block maps to zero-based index {idx}."
        )
    return features[idx], idx, reason


def load_experiment_config(ckpt_path: str | Path | None = None):
    ckpt_path = resolve_nova_ckpt(ckpt_path)
    config_path = ckpt_path.parent / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    return cfg.experiment if "experiment" in cfg else cfg


def build_decoder(device: torch.device, ckpt_path: str | Path | None = None):
    add_repo_paths()
    from nova3r.heads.pts3d_decoder import PointJointFMDecoderV2

    ckpt_path = resolve_nova_ckpt(ckpt_path)
    cfg = load_experiment_config(ckpt_path)
    params = dict(cfg.model.params.cfg.pts3d_head.params)
    decoder = PointJointFMDecoderV2(**params).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    decoder_state = {
        key[len("pts3d_head."):]: value
        for key, value in state.items()
        if key.startswith("pts3d_head.")
    }
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load NOVA3R decoder cleanly: missing={missing}, unexpected={unexpected}")
    decoder.eval().requires_grad_(False)
    meta = {
        "token_dim": int(params["dim_in"]),
        "num_scene_tokens": int(params["num_3d_tokens"]),
        "sample_size": int(params.get("sample_size", 8192)),
        "query_source": params.get("query_source", "src_complete"),
        "fm_step_size": float(cfg.get("fm_step_size", 0.04)),
    }
    return decoder, meta, cfg


def nova_flow_matching_loss(
    decoder,
    scene_tokens: torch.Tensor,
    target_points: torch.Tensor,
    seed: int = 0,
    num_views: int = 4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch, queries, _ = target_points.shape
    generator = torch.Generator(device=target_points.device)
    generator.manual_seed(int(seed))

    num_scene_tokens = int(scene_tokens.shape[1])
    token_dim = int(scene_tokens.shape[2])
    z0 = torch.randn(
        batch,
        num_scene_tokens,
        token_dim,
        device=target_points.device,
        dtype=target_points.dtype,
        generator=generator,
    )
    noise = torch.randn(
        batch,
        queries,
        3,
        device=target_points.device,
        dtype=target_points.dtype,
        generator=generator,
    )
    t_scalar = torch.rand(
        batch,
        1,
        device=target_points.device,
        dtype=target_points.dtype,
        generator=generator,
    )
    xt = t_scalar.unsqueeze(-1) * target_points + (1.0 - t_scalar.unsqueeze(-1)) * noise
    target_velocity = target_points - noise
    timestep = t_scalar.expand(batch, queries).contiguous()
    num_views_tensor = torch.full((batch,), int(num_views), device=target_points.device, dtype=target_points.dtype)
    pred_velocity = decoder([scene_tokens], query_points=xt, timestep=timestep, num_views=num_views_tensor)
    # Align all adapter variants to the same reduction: mean over (points, xyz)
    # per sample, then mean over the batch.
    pointwise_mse = F.mse_loss(pred_velocity.float(), target_velocity.float(), reduction="none")
    loss = pointwise_mse.mean(dim=(1, 2))
    return loss.mean(), {"pred_velocity": pred_velocity, "target_velocity": target_velocity, "xt": xt, "t": timestep}


def euler_sample(decoder, scene_tokens: torch.Tensor, num_queries: int, step_size: float, seed: int, num_views: int) -> torch.Tensor:
    decoder_dtype = next(decoder.parameters()).dtype
    scene_tokens = scene_tokens.to(dtype=decoder_dtype)

    batch = scene_tokens.shape[0]
    generator = torch.Generator(device=scene_tokens.device)
    generator.manual_seed(int(seed))
    num_scene_tokens = int(scene_tokens.shape[1])
    token_dim = int(scene_tokens.shape[2])
    z0 = torch.randn(
        batch,
        num_scene_tokens,
        token_dim,
        device=scene_tokens.device,
        dtype=decoder_dtype,
        generator=generator,
    )
    x = torch.randn(
        batch,
        num_queries,
        3,
        device=scene_tokens.device,
        dtype=decoder_dtype,
        generator=generator,
    )
    steps = max(1, int(round(1.0 / step_size)))
    times = torch.linspace(0.0, 1.0 - 1.0 / steps, steps, device=scene_tokens.device, dtype=decoder_dtype)
    num_views_tensor = torch.full((batch,), int(num_views), device=scene_tokens.device, dtype=decoder_dtype)
    for t_scalar in times:
        timestep = torch.full((batch, num_queries), t_scalar, device=scene_tokens.device, dtype=decoder_dtype)
        velocity = decoder([scene_tokens], query_points=x, timestep=timestep, num_views=num_views_tensor)
        x = x + (1.0 / steps) * velocity
    return x


def sample_decoder(decoder, scene_tokens, num_queries, step_size, seed, num_views):
    return euler_sample(decoder, scene_tokens, num_queries, step_size, seed, num_views)


def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred, target)
    return dist.min(dim=-1).values.mean() + dist.min(dim=-2).values.mean()


def write_point_cloud_ply(path: str | Path, points: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.detach().cpu().numpy()
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{float(x)} {float(y)} {float(z)}\n")


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def trainable_parameter_names(modules: dict[str, torch.nn.Module]) -> list[str]:
    names = []
    for prefix, module in modules.items():
        for name, param in module.named_parameters():
            if param.requires_grad:
                names.append(f"{prefix}.{name}")
    return names


def assert_only_adapter_trainable(adapter, vggt, decoder) -> None:
    trainable = list(trainable_parameter_names({"adapter": adapter, "vggt": vggt, "decoder": decoder}))
    bad = [name for name in trainable if not name.startswith("adapter.")]
    if bad:
        raise AssertionError(f"Unexpected non-adapter trainable parameters: {bad}")


class AdapterPrecomputedDataset(Dataset):
    def __init__(self, path: str | Path, split: str, image_root_map: tuple[str, str] | None = None):
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu")
        required = {"scene_ids", "target_points", "splits", "metadata"}
        missing = required.difference(payload.keys())
        if missing:
            raise KeyError(f"{self.path} is missing required keys: {sorted(missing)}")
        self.scene_ids = payload["scene_ids"]
        self.features = payload.get("features")
        self.target_points = payload["target_points"]
        self.splits = payload["splits"]
        self.metadata = payload["metadata"]
        self.image_root_map = image_root_map
        requested_split = split
        if split == "test" and not any(sample_split == "test" for sample_split in self.splits):
            print(f"Adapter dataset {self.path} has no test split; falling back to val split for evaluation.")
            split = "val"
        self.indices = [i for i, sample_split in enumerate(self.splits) if sample_split == split]
        print(f"Building adapter image/point loader from {self.path}, split={requested_split}->{split}, len={len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        sample_meta = self.metadata[i] or {}
        frame_paths = sample_meta.get("frame_paths") or []
        remapped_paths = []
        for path in frame_paths:
            if self.image_root_map is not None:
                old, new = self.image_root_map
                if path.startswith(old):
                    path = new + path[len(old):]
            remapped_paths.append(path)
        item = {
            "target_points": self.target_points[i].float(),
            "frame_paths": remapped_paths,
            "scene_id": self.scene_ids[i],
            "metadata": sample_meta,
        }
        if self.features is not None:
            item["features"] = self.features[i].float()
        return item


class ScanNetProcessedDataset(Dataset):
    def __init__(self, root: str | Path = DEFAULT_SCANNET_ROOT, split: str = "train", num_views: int = 4):
        root = Path(root)
        split_path = root / split
        if not split_path.exists():
            raise FileNotFoundError(f"ScanNet processed split not found: {split_path}")
        self.samples = sorted(path for path in split_path.iterdir() if path.is_dir())
        self.num_views = int(num_views)
        print(f"Building ScanNet processed loader from {split_path}, len={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample_dir = self.samples[idx]
        rgb_dir = sample_dir / "rgb"
        ply_path = sample_dir / "points.ply"
        if not rgb_dir.exists() or not ply_path.exists():
            raise FileNotFoundError(f"Missing rgb/points.ply under {sample_dir}")
        image_paths = sorted(rgb_dir.glob("*.png"))
        if len(image_paths) < self.num_views:
            raise ValueError(f"Need at least {self.num_views} views in {rgb_dir}, found {len(image_paths)}")
        if len(image_paths) == self.num_views:
            chosen = image_paths
        else:
            positions = np.linspace(0, len(image_paths) - 1, self.num_views)
            chosen = [image_paths[int(round(pos))] for pos in positions]
        images = []
        for path in chosen:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(arr).permute(2, 0, 1))
        images = torch.stack(images, dim=0)
        target_points = load_ply_xyz(ply_path)
        return {
            "images": images,
            "target_points": target_points,
            "frame_paths": [str(path) for path in chosen],
            "scene_id": sample_dir.name,
        }


def load_ply_xyz(path: str | Path) -> torch.Tensor:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        end_idx = lines.index("end_header")
    except ValueError as exc:
        raise RuntimeError(f"PLY header missing end_header: {path}") from exc
    points = []
    for line in lines[end_idx + 1 :]:
        if not line.strip():
            continue
        x, y, z, *_ = line.split()
        points.append((float(x), float(y), float(z)))
    return torch.tensor(points, dtype=torch.float32)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
        "target_points": torch.stack([item["target_points"] for item in batch], dim=0),
        "frame_paths": [item["frame_paths"] for item in batch],
        "scene_id": [item["scene_id"] for item in batch],
    }
    if "images" in batch[0]:
        collated["images"] = torch.stack([item["images"] for item in batch], dim=0)
    if "features" in batch[0]:
        collated["features"] = torch.stack([item["features"] for item in batch], dim=0)
    if "metadata" in batch[0]:
        collated["metadata"] = [item["metadata"] for item in batch]
    return collated


def build_loader(
    cfg,
    batch_size: int,
    num_workers: int,
    test: bool = False,
    image_root_map: tuple[str, str] | None = None,
    dataset_name: str = "scrream_adapter",
    data_root: str | None = None,
    seed: int = 17,
    num_views: int = 4,
):
    split = "test" if test else "train"
    if dataset_name == "scrream_adapter":
        dataset_path = Path(data_root) if data_root else DEFAULT_ADAPTER_DATA
        dataset = AdapterPrecomputedDataset(dataset_path, split=split, image_root_map=image_root_map)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not test,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )
        args = types.SimpleNamespace(data_root=str(dataset_path), test_dataset_name="adapter_precomputed_train")
        return loader, args
    if dataset_name == "scannet":
        root = Path(data_root) if data_root else DEFAULT_SCANNET_ROOT
        dataset = ScanNetProcessedDataset(root=root, split=split, num_views=num_views)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=not test,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )
        args = types.SimpleNamespace(data_root=str(root), test_dataset_name=f"scannet_{split}")
        return loader, args
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def images_from_batch(batch: dict[str, Any]) -> torch.Tensor:
    images = batch["images"]
    if images.ndim == 4:
        images = images.unsqueeze(0)
    return images


def get_targets(batch: dict[str, Any], query_source: str, max_points: int = 2048) -> torch.Tensor:
    del query_source
    points = batch["target_points"]
    if points.shape[1] > max_points:
        points = points[:, :max_points]
    elif points.shape[1] < max_points:
        pad = max_points - points.shape[1]
        points = F.pad(points, (0, 0, 0, pad))
    return points.contiguous()


def scene_ids_from_batch(batch: dict[str, Any], fallback_step: int) -> list[str]:
    scene_ids = batch.get("scene_id")
    if isinstance(scene_ids, list) and scene_ids:
        return scene_ids
    return [f"step{fallback_step:06d}"]
