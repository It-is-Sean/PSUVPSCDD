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
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


PROBE3D_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROBE3D_ROOT.parents[1]
DEFAULT_REFERENCE_ROOT = REPO_ROOT
DEFAULT_VGGT_REPO = REPO_ROOT / "third_party" / "vggt"
DEFAULT_VGGT_WEIGHTS_CANDIDATES = [
    Path("/data1/jcd_data/cache/models/vggt/VGGT-1B/model.pt"),
    REPO_ROOT / "checkpoints" / "vggt" / "model.pt",
    REPO_ROOT / "artifacts" / "weights" / "vggt" / "VGGT-1B" / "model.pt",
]
DEFAULT_NOVA_CKPT = REPO_ROOT / "checkpoints/scene_ae/checkpoint-last.pth"
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


def init_distributed_mode(device: str = "cuda") -> dict[str, Any]:
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env = os.environ.get("RANK")
    if local_rank_env is None or world_size_env is None or rank_env is None:
        resolved = resolve_device(device)
        return {
            "enabled": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": resolved,
            "is_main": True,
        }

    local_rank = int(local_rank_env)
    world_size = int(world_size_env)
    rank = int(rank_env)
    backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND")
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.set_device(local_rank)
        resolved = torch.device(f"cuda:{local_rank}")
    else:
        resolved = resolve_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return {
        "enabled": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": resolved,
        "is_main": rank == 0,
    }


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(value: float | torch.Tensor, enabled: bool) -> float:
    if torch.is_tensor(value):
        tensor = value.detach().float().clone()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    if enabled and dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def barrier_if_distributed(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()


def sampler_set_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


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
    if not features:
        raise RuntimeError("VGGT aggregator returned no intermediate features.")
    if len(features) >= 23:
        idx = 22
        reason = (
            f"VGGT returned {len(features)} intermediate features; human 23rd block maps "
            "to zero-based index 22."
        )
    else:
        idx = len(features) - 1
        reason = (
            f"VGGT returned only {len(features)} features; using final available index {idx} "
            "instead of human block 23."
        )
    return features[idx], idx, reason


def load_experiment_config(ckpt_path: str | Path = DEFAULT_NOVA_CKPT):
    ckpt_path = Path(ckpt_path)
    config_path = ckpt_path.parent / ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    return cfg.experiment if "experiment" in cfg else cfg


def build_decoder(device: torch.device, ckpt_path: str | Path = DEFAULT_NOVA_CKPT):
    add_repo_paths()
    from nova3r.heads.pts3d_decoder import PointJointFMDecoderV2

    ckpt_path = Path(ckpt_path)
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
        "norm_mode": str(params.get("norm_mode", "none")),
        "target_sampling": str(params.get("target_sampling", "none")),
        "fm_step_size": float(cfg.get("fm_step_size", 0.04)),
    }
    return decoder, meta, cfg


class AdapterImagePointDataset(Dataset):
    def __init__(
        self,
        path: str | Path = DEFAULT_ADAPTER_DATA,
        split: str | None = None,
        max_samples: int | None = None,
        image_root_map: tuple[str, str] | None = None,
    ) -> None:
        add_repo_paths()
        self.path = Path(path)
        payload = torch.load(self.path, map_location="cpu")
        self.scene_ids = payload["scene_ids"]
        self.targets = payload["target_points"].float()
        self.metadata = payload.get("metadata", [{} for _ in self.scene_ids])
        self.splits = payload.get("splits", ["train" for _ in self.scene_ids])
        self.image_root_map = image_root_map
        self.indices = list(range(len(self.scene_ids)))
        if split is not None:
            self.indices = [idx for idx in self.indices if self.splits[idx] == split]
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
        if not self.indices:
            raise ValueError(f"No samples available in {self.path} for split={split!r}")

    def __len__(self) -> int:
        return len(self.indices)

    def _map_path(self, path: str) -> Path:
        if self.image_root_map is None:
            return Path(path)
        old, new = self.image_root_map
        if path.startswith(old):
            return Path(new + path[len(old):])
        return Path(path)

    def _resolve_frame_paths(self, meta: dict[str, Any]) -> list[str]:
        paths = [Path(p) for p in meta.get("frame_paths", [])]
        mapped = [self._map_path(str(p)) for p in paths]
        if mapped and all(p.exists() for p in mapped):
            return [str(p) for p in mapped]
        missing = [str(p) for p in mapped if not p.exists()]
        sample_id = meta.get("sample_id", "<unknown>")
        raise FileNotFoundError(
            "Missing real input images for sample "
            f"{sample_id}. First missing paths: {missing[:4]}. "
            "Pass --image_root_map OLD=NEW if the SCRREAM dataset was moved."
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from vggt.utils.load_fn import load_and_preprocess_images

        src_idx = self.indices[idx]
        paths = self._resolve_frame_paths(self.metadata[src_idx])
        return {
            "images": load_and_preprocess_images(paths, mode="pad"),
            "target_points": self.targets[src_idx],
            "scene_id": self.scene_ids[src_idx],
            "split": self.splits[src_idx],
            "frame_paths": paths,
        }


def collate_adapter_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([sample["images"] for sample in samples], dim=0),
        "target_points": torch.stack([sample["target_points"] for sample in samples], dim=0),
        "scene_ids": [sample["scene_id"] for sample in samples],
        "splits": [sample["split"] for sample in samples],
        "frame_paths": [sample["frame_paths"] for sample in samples],
    }


def _limit_scannet_dataset_scenes(dataset, max_scenes: int | None):
    if max_scenes is None or max_scenes <= 0:
        return []
    seen_scene_ids = []
    seen = set()
    for start_id in dataset.start_img_ids:
        scene_id = int(dataset.sceneids[start_id])
        if scene_id not in seen:
            seen.add(scene_id)
            seen_scene_ids.append(scene_id)
    allowed = set(seen_scene_ids[:max_scenes])
    if not allowed:
        raise ValueError(f"Cannot limit ScanNet dataset to max_scenes={max_scenes}: no scenes found")
    dataset.start_img_ids = [
        start_id for start_id in dataset.start_img_ids if int(dataset.sceneids[start_id]) in allowed
    ]
    if not dataset.start_img_ids:
        raise ValueError(f"ScanNet scene limit max_scenes={max_scenes} removed all samples")
    return [dataset.scenes[scene_id] for scene_id in seen_scene_ids[:max_scenes]]


def build_scannet_loader(
    data_root: str | Path,
    batch_size: int,
    num_workers: int,
    test: bool = False,
    seed: int = 17,
    num_views: int = 4,
    split_override: str | None = None,
    max_scenes: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    scannet_target_mode: str = "complete_zpos",
    scannet_frustum_margin: float = 1.0,
    scannet_min_views: int = 2,
    scannet_complete_points: int = 10000,
    scannet_max_interval: int = 1,
):
    add_repo_paths()
    from dust3r.datasets.scannet import ScanNet_Multi

    root = Path(data_root)
    split = split_override or ("test" if test else "train")
    dataset = ScanNet_Multi(
        ROOT=str(root),
        split=split,
        num_views=num_views,
        resolution=518,
        aug_crop=0,
        seed=seed,
        allow_repeat=False,
        complete_gt_target_mode=scannet_target_mode,
        complete_gt_frustum_margin=scannet_frustum_margin,
        complete_gt_min_views=scannet_min_views,
        complete_gt_points=scannet_complete_points,
        max_interval=scannet_max_interval,
    )
    limited_scenes = _limit_scannet_dataset_scenes(dataset, max_scenes)
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=not test,
            seed=seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not test) if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    data_args = OmegaConf.create(
        {
            "data_root": str(root),
            "test_dataset_name": f"scannet_{split}_n{num_views}",
            "split": split,
            "scene_count": len(set(int(dataset.sceneids[start_id]) for start_id in dataset.start_img_ids)),
            "limited_scenes": limited_scenes,
            "scannet_target_mode": scannet_target_mode,
            "scannet_frustum_margin": float(scannet_frustum_margin),
            "scannet_min_views": int(scannet_min_views),
            "scannet_complete_points": int(scannet_complete_points),
            "scannet_max_interval": int(scannet_max_interval),
        }
    )
    scene_msg = f", scene_count={data_args.scene_count}, target_mode={scannet_target_mode}, frustum_margin={scannet_frustum_margin}, min_views={scannet_min_views}, complete_points={scannet_complete_points}, max_interval={scannet_max_interval}"
    if limited_scenes:
        scene_msg += f", limited_scenes={limited_scenes}"
    print(f"Building ScanNet loader from {root}, split={split}, len={len(dataset)}, num_views={num_views}{scene_msg}")
    return loader, data_args


def build_loader(
    cfg,
    batch_size: int,
    num_workers: int,
    test: bool = False,
    image_root_map: tuple[str, str] | None = None,
    dataset_name: str = "scrream_adapter",
    data_root: str | Path | None = None,
    seed: int = 17,
    num_views: int = 4,
    split_override: str | None = None,
    max_scenes: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    scannet_target_mode: str = "complete_zpos",
    scannet_frustum_margin: float = 1.0,
    scannet_min_views: int = 2,
    scannet_complete_points: int = 10000,
    scannet_max_interval: int = 1,
):
    del cfg
    if dataset_name == "scannet":
        return build_scannet_loader(
            data_root or DEFAULT_SCANNET_ROOT,
            batch_size=batch_size,
            num_workers=num_workers,
            test=test,
            seed=seed,
            num_views=num_views,
            split_override=split_override,
            max_scenes=max_scenes,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            scannet_target_mode=scannet_target_mode,
            scannet_frustum_margin=scannet_frustum_margin,
            scannet_min_views=scannet_min_views,
            scannet_complete_points=scannet_complete_points,
            scannet_max_interval=scannet_max_interval,
        )
    if dataset_name != "scrream_adapter":
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}")
    split = split_override or ("test" if test else "train")
    try:
        dataset = AdapterImagePointDataset(DEFAULT_ADAPTER_DATA, split=split, image_root_map=image_root_map)
    except ValueError:
        dataset = AdapterImagePointDataset(DEFAULT_ADAPTER_DATA, split=None, image_root_map=image_root_map)
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=not test,
            seed=seed,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(not test) if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_adapter_samples,
    )
    data_args = OmegaConf.create(
        {
            "data_root": str(DEFAULT_ADAPTER_DATA),
            "test_dataset_name": f"adapter_precomputed_{split}",
        }
    )
    print(f"Building adapter image/point loader from {DEFAULT_ADAPTER_DATA}, split={split}, len={len(dataset)}")
    return loader, data_args


def build_nova_eval_loader(cfg, batch_size: int, num_workers: int, test: bool = False):
    add_repo_paths()
    ensure_accelerate_stub()
    from eval.mv_recon.test_nova3r import _apply_eval_defaults, build_dataset

    args = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    _apply_eval_defaults(args)
    if "data_root" not in args:
        args.data_root = "/data1/jcd_data/datasets"
    if "test_dataset_name" not in args:
        args.test_dataset_name = "scrream_n2"
    data_root = args.data_root
    name = args.test_dataset_name
    if name == "scrream_n1":
        ds = (
            "SCRREAM(split='train', ROOT='"
            f"{data_root}/eval_scrream', train_list_path='data/scrream/scrream_n1_list.json', "
            "test_list_path='data/scrream/scrream_n1_list.json', resolution=518, aug_crop=0, "
            "input_n=1, n_ldi_layers=0, enforce_img_reso_for_eval=[518,392], max_pts=100000)"
        )
    else:
        ds = (
            "SCRREAM_MULTI(split='train', ROOT='"
            f"{data_root}/eval_scrream', train_list_path='data/scrream/scrream_n2_list.json', "
            "test_list_path='data/scrream/scrream_n2_list.json', resolution=518, aug_crop=0, "
            "input_n=2, n_ldi_layers=1, enforce_img_reso_for_eval=[518,392], max_pts=100000)"
        )
    return build_dataset(args, ds, batch_size, num_workers, test=test), args


def move_batch_to_device(batch: list[dict[str, Any]], device: torch.device):
    if isinstance(batch, dict):
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(device, non_blocking=True)
        return batch
    ignore = {"dataset", "label", "instance", "idx", "true_shape", "rng", "view_label"}
    for view in batch:
        for key, value in list(view.items()):
            if key not in ignore and torch.is_tensor(value):
                view[key] = value.to(device, non_blocking=True)
    return batch


def images_from_batch(batch: list[dict[str, Any]]) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["images"]
    imgs = []
    for view in batch:
        labels = view.get("view_label", [""])
        label = labels[0] if isinstance(labels, (list, tuple)) else str(labels)
        if "input" in label:
            imgs.append(view["img"] * 0.5 + 0.5)
    if not imgs:
        imgs = [view["img"] * 0.5 + 0.5 for view in batch]
    return torch.stack(imgs, dim=1)


def get_targets(
    batch: list[dict[str, Any]],
    query_source: str,
    max_points: int | None = None,
    norm_mode: str = "none",
):
    if isinstance(batch, dict):
        targets = batch["target_points"]
        if max_points is not None and targets.shape[1] > max_points:
            targets = targets[:, :max_points]
        return targets
    from nova3r.inference import get_all_pts3d, normalize_input

    effective_query_source = query_source
    first_view = batch[0]
    if (
        isinstance(first_view, dict)
        and query_source.startswith("src_complete")
        and "pts3d_complete" not in first_view
        and "pts3d" in first_view
        and "valid_mask" in first_view
    ):
        effective_query_source = "src_view"
        print(
            f"query_source fallback: requested {query_source} but batch lacks pts3d_complete; "
            f"using {effective_query_source} instead"
        )

    targets, valid = get_all_pts3d(batch, mode=effective_query_source)
    if norm_mode and norm_mode != "none":
        targets, _ = normalize_input(targets, valid, targets, valid, mode=norm_mode)
    compact = []
    for pts, mask in zip(targets, valid):
        pts = pts[mask]
        if max_points is not None and pts.shape[0] > max_points:
            pts = pts[:max_points]
        compact.append(pts)
    min_len = min(p.shape[0] for p in compact)
    return torch.stack([p[:min_len] for p in compact], dim=0)


def _value_for_sample(value: Any, sample_idx: int):
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        item = value[sample_idx]
        return item.item() if torch.is_tensor(item) and item.ndim == 0 else item
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        item = value[sample_idx]
        return item.item() if np.ndim(item) == 0 else item
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return value[sample_idx]
    return value


def sample_keys_from_batch(batch: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    if isinstance(batch, dict):
        if "frame_paths" in batch:
            return [tuple(str(p) for p in paths) for paths in batch["frame_paths"]]
        if "scene_ids" in batch:
            return [(str(x),) for x in batch["scene_ids"]]
    first = batch[0]
    bsz = int(first["img"].shape[0])
    keys: list[tuple[str, ...]] = []
    for sample_idx in range(bsz):
        tokens = []
        for view_idx, view in enumerate(batch):
            value = view.get("label", view.get("instance", view.get("idx", f"view{view_idx}")))
            tokens.append(str(_value_for_sample(value, sample_idx)))
        keys.append(tuple(tokens))
    return keys


def infer_batch_device(batch: list[dict[str, Any]]) -> torch.device:
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value):
                return value.device
        return torch.device("cpu")
    for view in batch:
        for value in view.values():
            if torch.is_tensor(value):
                return value.device
    return torch.device("cpu")


def get_targets_cached(
    batch: list[dict[str, Any]],
    query_source: str,
    max_points: int | None = None,
    cache: dict | None = None,
    norm_mode: str = "none",
):
    if cache is None:
        return get_targets(batch, query_source, max_points=max_points, norm_mode=norm_mode)
    cache_key = (tuple(sample_keys_from_batch(batch)), query_source, max_points, norm_mode)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.to(infer_batch_device(batch), non_blocking=True)
    targets = get_targets(batch, query_source, max_points=max_points, norm_mode=norm_mode)
    cache[cache_key] = targets.detach().cpu()
    return targets


def scene_ids_from_batch(batch: list[dict[str, Any]], start_idx: int) -> list[str]:
    if isinstance(batch, dict):
        return [str(x) for x in batch["scene_ids"]]
    first = batch[0]
    value = first.get("instance", first.get("label", first.get("idx")))
    if torch.is_tensor(value):
        return [str(x.item()) for x in value.view(-1)]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if value is not None:
        return [str(value)]
    bsz = first["img"].shape[0]
    return [f"sample_{start_idx + i:06d}" for i in range(bsz)]


def sample_decoder(
    decoder,
    tokens: torch.Tensor,
    num_queries: int,
    step_size: float,
    seed: int,
    num_views: int | None = None,
) -> torch.Tensor:
    bsz = tokens.shape[0]
    steps = max(2, int(1 // step_size))
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(seed)
    x = torch.rand(bsz, num_queries, 3, device=tokens.device, generator=generator) * 2 - 1
    view_tensor = None
    if num_views is not None:
        view_tensor = torch.full((bsz,), float(num_views), device=tokens.device)
    time_grid = torch.linspace(0.0, 1.0, steps, device=tokens.device)
    for idx in range(len(time_grid) - 1):
        t = torch.full((bsz, x.shape[1]), float(time_grid[idx]), device=tokens.device, dtype=x.dtype)
        velocity = decoder([tokens.float()], query_points=x.float(), timestep=t.float(), num_views=view_tensor)
        x = x + (time_grid[idx + 1] - time_grid[idx]) * velocity.to(dtype=x.dtype)
    return x


def chamfer_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[-1] != 3 or gt.shape[-1] != 3:
        raise ValueError(f"Expected [B,N,3] and [B,M,3], got {tuple(pred.shape)} and {tuple(gt.shape)}")
    dists = torch.cdist(pred.float(), gt.float(), p=2).pow(2)
    return dists.min(dim=2).values.mean() + dists.min(dim=1).values.mean()


def nova_flow_matching_loss(decoder, tokens: torch.Tensor, target_points: torch.Tensor, seed: int, num_views: int | None = None):
    from nova3r.flow_matching.path import AffineProbPath
    from nova3r.flow_matching.path.scheduler import CosineScheduler

    bsz, num_points, _ = target_points.shape
    generator = torch.Generator(device=target_points.device)
    generator.manual_seed(seed)
    x0 = torch.rand(target_points.shape, device=target_points.device, generator=generator, dtype=target_points.dtype) * 2 - 1
    t = torch.rand((bsz,), device=target_points.device, generator=generator, dtype=target_points.dtype)
    path = AffineProbPath(scheduler=CosineScheduler())
    path_sample = path.sample(x_0=x0.float(), x_1=target_points.float(), t=t.float())
    timestep = t[:, None].expand(bsz, num_points).float()
    view_tensor = None
    if num_views is not None:
        view_tensor = torch.full((bsz,), float(num_views), device=target_points.device)
    pred_velocity = decoder([tokens.float()], query_points=path_sample.x_t.float(), timestep=timestep, num_views=view_tensor)
    # Align all adapter variants to the same reduction: mean over (points, xyz)
    # per sample, then mean over the batch.
    pointwise_mse = F.mse_loss(pred_velocity.float(), path_sample.dx_t.float(), reduction="none")
    loss = pointwise_mse.mean(dim=(1, 2)).mean()
    return loss, {
        "pred_velocity": pred_velocity,
        "target_velocity": path_sample.dx_t,
        "query_points": path_sample.x_t,
        "timesteps": t,
    }


def write_point_cloud_ply(path: str | Path, points: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points.detach().cpu().float().reshape(-1, 3)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {pts.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("end_header\n")
        for x, y, z in pts.tolist():
            handle.write(f"{x} {y} {z}\n")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def trainable_parameter_names(modules: dict[str, torch.nn.Module]) -> list[str]:
    names = []
    for prefix, module in modules.items():
        for name, param in module.named_parameters():
            if param.requires_grad:
                names.append(f"{prefix}.{name}")
    return names


def assert_only_adapter_trainable(adapter, vggt, decoder) -> None:
    bad = []
    for prefix, module in (("vggt", vggt), ("decoder", decoder)):
        for name, param in module.named_parameters():
            if param.requires_grad:
                bad.append(f"{prefix}.{name}")
    if bad:
        raise AssertionError(f"Non-adapter parameters are trainable: {bad[:20]}")
    adapter_trainable = [name for name, p in adapter.named_parameters() if p.requires_grad]
    if not adapter_trainable:
        raise AssertionError("Adapter has no trainable parameters.")


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_feature_shapes(features: Iterable[torch.Tensor]) -> None:
    for i, feat in enumerate(features):
        print(f"VGGT feature {i}: shape={tuple(feat.shape)} dtype={feat.dtype} device={feat.device}")
