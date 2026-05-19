#!/usr/bin/env python3
"""Train NOVA adapter on precomputed WAN-T2V video-context features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None

from probe.adapter import (
    VGGTToNovaAdapter,
    VGGTToNovaCrossAttentionAdapter,
    VGGTToNovaSelfAttentionAdapter,
    WanHiddenCrossAttentionResamplerAdapter,
    WanHiddenGrid2DConvAdapter,
    WanHiddenGrid2DPoolAdapter,
    WanHiddenMultiLayerCrossAttentionResamplerAdapter,
    WanHiddenMultiSourceCrossAttentionResamplerAdapter,
    WanLatentCrossAttentionResamplerAdapter,
    WanPredX0LatentConvAdapter,
)
from vggt_nova_adapter_common_raw import (
    amp_context,
    assert_only_adapter_trainable,
    barrier_if_distributed,
    build_decoder,
    build_loader,
    chamfer_l2,
    cleanup_distributed,
    count_parameters,
    get_targets,
    images_from_batch,
    init_distributed_mode,
    move_batch_to_device,
    reduce_scalar,
    sample_decoder,
    sampler_set_epoch,
    save_json,
    scene_ids_from_batch,
    set_seed,
    trainable_parameter_names,
    write_point_cloud_ply,
    resolve_device,
)
from train_vggt_nova_adapter import (
    AdapterVelocityProbe,
    QUALITY_METRIC_KEYS,
    maybe_init_swanlab,
    maybe_init_wandb,
    parse_image_root_map,
    pointcloud_quality_metrics,
    resolve_val_preview_root,
    save_checkpoint,
)

FEATURE_KIND_HIDDEN = "hidden"
FEATURE_KIND_PRED_X0_LATENT = "pred_x0_latent"
FEATURE_KIND_MODEL_OUTPUT_LATENT = "model_output_latent"
ADAPTER_TYPE_CONV2D_MLP = "conv2d_mlp"
ADAPTER_TYPE_GRID2D_POOL = "grid2d_pool"
ADAPTER_TYPE_GRID2D_CONV = "grid2d_conv"
ADAPTER_TYPE_WAN_CROSS_ATTN_RESAMPLER = "wan_cross_attn_resampler"
ADAPTER_TYPE_WAN_CROSS_ATTN_MULTILAYER_RESAMPLER = "wan_cross_attn_multilayer_resampler"
ADAPTER_TYPE_WAN_CROSS_ATTN_MULTITIME_RESAMPLER = "wan_cross_attn_multitime_resampler"
ADAPTER_TYPE_WAN_LATENT_CROSS_ATTN_RESAMPLER = "wan_latent_cross_attn_resampler"


def safe_cache_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(sample_id)).strip("_")


def is_latent_feature_kind(feature_kind: str) -> bool:
    return feature_kind in {FEATURE_KIND_PRED_X0_LATENT, FEATURE_KIND_MODEL_OUTPUT_LATENT}


def wan_cache_path(cache_root: Path, timestep: int, layer: int, sample_id: str, feature_kind: str = FEATURE_KIND_HIDDEN) -> Path:
    if is_latent_feature_kind(feature_kind):
        return cache_root / f"t{int(timestep):03d}" / feature_kind / f"{safe_cache_sample_id(sample_id)}.pt"
    if feature_kind != FEATURE_KIND_HIDDEN:
        raise ValueError(f"Unsupported WAN feature kind {feature_kind!r}")
    return cache_root / f"t{int(timestep):03d}" / f"layer{int(layer):02d}" / f"{safe_cache_sample_id(sample_id)}.pt"


def parse_wan_layers(value: str | None, fallback_layer: int) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return (int(fallback_layer),)
    layers: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(int(item))
    if not layers:
        raise ValueError("--wan_layers was provided but no valid layer ids were parsed")
    return tuple(layers)


def parse_wan_timesteps(value: str | None, fallback_timestep: int) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return (int(fallback_timestep),)
    timesteps: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        timesteps.append(int(item))
    if not timesteps:
        raise ValueError("--wan_timesteps was provided but no valid timestep ids were parsed")
    return tuple(timesteps)


@lru_cache(maxsize=128)
def cached_wan_sample_ids(feature_cache_dir: str, wan_timestep: int, wan_layer: int, feature_kind: str = FEATURE_KIND_HIDDEN) -> tuple[str, ...]:
    cache_root = Path(feature_cache_dir)
    cache_dir = wan_cache_path(cache_root, int(wan_timestep), int(wan_layer), "__dummy__", feature_kind).parent
    sample_ids = tuple(sorted(path.stem for path in cache_dir.glob("*.pt")))
    if not sample_ids:
        raise FileNotFoundError(f"No WAN cache files found under {cache_dir}")
    return sample_ids


def shuffled_wan_sample_id(feature_cache_dir: str, wan_timestep: int, wan_layer: int, sample_id: str, seed: int, feature_kind: str = FEATURE_KIND_HIDDEN) -> str:
    safe_id = safe_cache_sample_id(sample_id)
    sample_ids = cached_wan_sample_ids(feature_cache_dir, int(wan_timestep), int(wan_layer), feature_kind)
    digest = hashlib.sha1(f"{safe_id}:{int(seed)}:{int(wan_timestep)}:{int(wan_layer)}:{feature_kind}".encode("utf-8")).hexdigest()
    idx = int(digest[:16], 16) % len(sample_ids)
    if len(sample_ids) > 1 and sample_ids[idx] == safe_id:
        idx = (idx + 1) % len(sample_ids)
    return sample_ids[idx]


def normalize_wan_features(features: torch.Tensor, mode: str = "none", eps: float = 1e-6) -> torch.Tensor:
    if mode == "none":
        return features
    eps = float(eps)
    if mode == "token_layernorm":
        mean = features.mean(dim=-1, keepdim=True)
        var = features.var(dim=-1, unbiased=False, keepdim=True)
        return (features - mean) * torch.rsqrt(var + eps)
    if mode == "sample_standardize":
        dims = tuple(range(1, features.ndim))
        mean = features.mean(dim=dims, keepdim=True)
        var = features.var(dim=dims, unbiased=False, keepdim=True)
        return (features - mean) * torch.rsqrt(var + eps)
    if mode == "token_l2":
        return torch.nn.functional.normalize(features, p=2.0, dim=-1, eps=eps)
    raise ValueError(f"Unsupported WAN feature normalization mode {mode!r}")


def get_wan_t2v_cached_features(
    batch,
    device,
    feature_cache_dir: str,
    wan_timestep: int,
    wan_layer: int,
    feature_mode: str = "cache",
    shuffle_seed: int = 17,
    feature_norm: str = "none",
    feature_norm_eps: float = 1e-6,
    feature_kind: str = FEATURE_KIND_HIDDEN,
) -> torch.Tensor:
    cache_root = Path(feature_cache_dir)
    if feature_mode not in {"cache", "zero", "sample_shuffle"}:
        raise ValueError(f"Unsupported WAN feature mode {feature_mode!r}")
    selected_list = []
    for sample_id in batch["scene_ids"]:
        load_sample_id = str(sample_id)
        if feature_mode == "sample_shuffle":
            load_sample_id = shuffled_wan_sample_id(
                str(cache_root),
                int(wan_timestep),
                int(wan_layer),
                str(sample_id),
                int(shuffle_seed),
                feature_kind,
            )
        path = wan_cache_path(cache_root, int(wan_timestep), int(wan_layer), load_sample_id, feature_kind)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing WAN T2V cache for sample={load_sample_id} timestep={wan_timestep} "
                f"layer={wan_layer} feature_kind={feature_kind}: {path}"
            )
        payload = torch.load(path, map_location="cpu")
        features = payload["features"] if isinstance(payload, dict) else payload
        if features.ndim != 2:
            raise ValueError(f"Expected WAN cache tensor [tokens, dim], got {tuple(features.shape)} from {path}")
        if feature_mode == "zero":
            features = torch.zeros_like(features)
        selected_list.append(features.to(device=device, dtype=torch.float32))
    selected = torch.stack(selected_list, dim=0).contiguous()
    return normalize_wan_features(selected, mode=feature_norm, eps=feature_norm_eps).contiguous()


def get_wan_t2v_multilayer_cached_features(
    batch,
    device,
    feature_cache_dir: str,
    wan_timestep: int,
    wan_layers: tuple[int, ...],
    feature_mode: str = "cache",
    shuffle_seed: int = 17,
    feature_norm: str = "none",
    feature_norm_eps: float = 1e-6,
) -> torch.Tensor:
    if len(wan_layers) < 2:
        raise ValueError(f"Multi-layer WAN features require at least two layers, got {wan_layers}")
    per_layer = [
        get_wan_t2v_cached_features(
            batch,
            device,
            feature_cache_dir,
            wan_timestep,
            int(layer),
            feature_mode,
            shuffle_seed,
            feature_norm,
            feature_norm_eps,
            FEATURE_KIND_HIDDEN,
        )
        for layer in wan_layers
    ]
    shapes = {tuple(features.shape) for features in per_layer}
    if len(shapes) != 1:
        raise ValueError(f"WAN multi-layer cache feature shapes must match, got {sorted(shapes)}")
    return torch.cat(per_layer, dim=1).contiguous()


def get_wan_t2v_multitime_cached_features(
    batch,
    device,
    feature_cache_dir: str,
    wan_timesteps: tuple[int, ...],
    wan_layer: int,
    feature_mode: str = "cache",
    shuffle_seed: int = 17,
    feature_norm: str = "none",
    feature_norm_eps: float = 1e-6,
) -> torch.Tensor:
    if len(wan_timesteps) < 2:
        raise ValueError(f"Multi-timestep WAN features require at least two timesteps, got {wan_timesteps}")
    per_timestep = [
        get_wan_t2v_cached_features(
            batch,
            device,
            feature_cache_dir,
            int(timestep),
            int(wan_layer),
            feature_mode,
            shuffle_seed,
            feature_norm,
            feature_norm_eps,
            FEATURE_KIND_HIDDEN,
        )
        for timestep in wan_timesteps
    ]
    shapes = {tuple(features.shape) for features in per_timestep}
    if len(shapes) != 1:
        raise ValueError(f"WAN multi-timestep cache feature shapes must match, got {sorted(shapes)}")
    return torch.cat(per_timestep, dim=1).contiguous()


def read_wan_cache_metadata(feature_cache_dir: str, wan_timestep: int, wan_layer: int, sample_id: str, feature_kind: str = FEATURE_KIND_HIDDEN) -> dict:
    path = wan_cache_path(Path(feature_cache_dir), int(wan_timestep), int(wan_layer), str(sample_id), feature_kind)
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        return dict(payload["metadata"])
    return {}


def infer_wan_hidden_grid_shape(metadata: dict, selected: torch.Tensor, num_sources: int = 1) -> tuple[int, int, int]:
    if selected.ndim != 3:
        raise ValueError(f"Expected loaded WAN features [B,tokens,dim], got {tuple(selected.shape)}")
    token_count = int(selected.shape[1])
    num_sources = max(1, int(num_sources))
    if token_count % num_sources != 0:
        raise ValueError(f"WAN token count {token_count} is not divisible by num_sources={num_sources}")
    tokens_per_source = token_count // num_sources
    if isinstance(metadata, dict):
        grid_shape = metadata.get("grid_shape")
        if isinstance(grid_shape, (list, tuple)) and len(grid_shape) >= 3:
            temporal, height, width = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
            if temporal * height * width == tokens_per_source:
                return temporal, height, width
        full_feature_shape = metadata.get("full_feature_shape")
        if isinstance(full_feature_shape, (list, tuple)) and len(full_feature_shape) >= 4:
            temporal = len(metadata.get("temporal_indices", [])) or 2
            height, width = int(full_feature_shape[1]), int(full_feature_shape[2])
            if temporal * height * width == tokens_per_source:
                return temporal, height, width
        feature_shape = metadata.get("feature_shape")
        if isinstance(feature_shape, (list, tuple)) and len(feature_shape) >= 2:
            meta_tokens = int(feature_shape[0])
            if meta_tokens != tokens_per_source:
                raise ValueError(
                    f"WAN cache metadata feature_shape token count {meta_tokens} does not match "
                    f"loaded per-source tensor {tokens_per_source}"
                )
    if tokens_per_source == 2 * 30 * 52:
        return 2, 30, 52
    if tokens_per_source == 2 * 45 * 80:
        return 2, 45, 80
    if tokens_per_source % 2 == 0:
        spatial = tokens_per_source // 2
        for height in (30, 45, 24, 60):
            if spatial % height == 0:
                return 2, height, spatial // height
    raise ValueError(
        "Could not infer WAN hidden grid shape from metadata/tensor. "
        f"metadata grid_shape={metadata.get('grid_shape') if isinstance(metadata, dict) else None}, "
        f"selected_shape={tuple(selected.shape)}, num_sources={num_sources}"
    )


def infer_wan_latent_grid_shape(metadata: dict, selected: torch.Tensor) -> tuple[int, int, int]:
    if selected.ndim != 3:
        raise ValueError(f"Expected loaded WAN latent features [B,tokens,dim], got {tuple(selected.shape)}")
    token_count = int(selected.shape[1])
    if isinstance(metadata, dict):
        full_feature_shape = metadata.get("full_feature_shape")
        if isinstance(full_feature_shape, (list, tuple)) and len(full_feature_shape) >= 4:
            temporal = len(metadata.get("temporal_indices", [])) or 2
            height, width = int(full_feature_shape[1]), int(full_feature_shape[2])
            if temporal * height * width == token_count:
                return temporal, height, width
        feature_shape = metadata.get("feature_shape")
        if isinstance(feature_shape, (list, tuple)) and len(feature_shape) >= 2:
            meta_tokens = int(feature_shape[0])
            if meta_tokens != token_count:
                raise ValueError(
                    f"WAN latent cache metadata feature_shape token count {meta_tokens} does not match "
                    f"loaded tensor {token_count}"
                )
    if token_count == 2 * 60 * 104:
        return 2, 60, 104
    if token_count % 2 == 0:
        spatial = token_count // 2
        for height in (60, 45, 30, 24):
            if spatial % height == 0:
                return 2, height, spatial // height
    raise ValueError(
        "Could not infer WAN latent grid shape from metadata/tensor. "
        f"metadata full_feature_shape={metadata.get('full_feature_shape') if isinstance(metadata, dict) else None}, "
        f"selected_shape={tuple(selected.shape)}"
    )


def read_wan_multilayer_cache_metadata(feature_cache_dir: str, wan_timestep: int, wan_layers: tuple[int, ...], sample_id: str) -> list[dict]:
    return [
        {
            "layer": int(layer),
            "metadata": read_wan_cache_metadata(
                feature_cache_dir,
                int(wan_timestep),
                int(layer),
                sample_id,
                FEATURE_KIND_HIDDEN,
            ),
        }
        for layer in wan_layers
    ]


def read_wan_multitime_cache_metadata(feature_cache_dir: str, wan_timesteps: tuple[int, ...], wan_layer: int, sample_id: str) -> list[dict]:
    return [
        {
            "timestep": int(timestep),
            "metadata": read_wan_cache_metadata(
                feature_cache_dir,
                int(timestep),
                int(wan_layer),
                sample_id,
                FEATURE_KIND_HIDDEN,
            ),
        }
        for timestep in wan_timesteps
    ]


def load_wan_features_for_batch(batch, device, args, wan_layers: tuple[int, ...], wan_timesteps: tuple[int, ...]) -> torch.Tensor:
    if len(wan_layers) > 1 and len(wan_timesteps) > 1:
        raise ValueError("Combining multiple WAN layers and multiple timesteps in one run is not supported")
    if len(wan_timesteps) > 1:
        return get_wan_t2v_multitime_cached_features(
            batch,
            device,
            args.wan_feature_cache_dir,
            wan_timesteps,
            int(wan_layers[0]),
            args.wan_feature_mode,
            args.wan_feature_shuffle_seed,
            args.wan_feature_norm,
            args.wan_feature_norm_eps,
        )
    if len(wan_layers) > 1:
        return get_wan_t2v_multilayer_cached_features(
            batch,
            device,
            args.wan_feature_cache_dir,
            args.wan_timestep,
            wan_layers,
            args.wan_feature_mode,
            args.wan_feature_shuffle_seed,
            args.wan_feature_norm,
            args.wan_feature_norm_eps,
        )
    return get_wan_t2v_cached_features(
        batch,
        device,
        args.wan_feature_cache_dir,
        int(wan_timesteps[0]),
        int(wan_layers[0]),
        args.wan_feature_mode,
        args.wan_feature_shuffle_seed,
        args.wan_feature_norm,
        args.wan_feature_norm_eps,
        args.wan_feature_kind,
    )


def run_eval(adapter, decoder, loader, device, meta, args, max_batches=None, output_dir: Path | None = None, global_step: int | None = None, save_previews: bool = False):
    from train_vggt_nova_adapter import (
        QUALITY_METRIC_KEYS,
        _sum_metrics_into,
        flow_matching_diagnostics,
        save_val_preview_sample,
    )

    adapter_module = unwrap_adapter(adapter)
    adapter_module.eval()
    total_chamfer = 0.0
    chamfer_count = 0
    total_velocity = 0.0
    velocity_count = 0
    bin_count = max(0, int(args.val_flow_t_bins))
    bin_totals = [0.0 for _ in range(bin_count)]
    bin_counts = [0 for _ in range(bin_count)]
    quality_totals = {key: 0.0 for key in QUALITY_METRIC_KEYS}
    quality_counts = {key: 0 for key in QUALITY_METRIC_KEYS}
    preview_records = []
    preview_limit = max(0, int(args.val_preview_samples))
    wan_layers = tuple(int(layer) for layer in getattr(args, "wan_layers_parsed", (int(args.wan_layer),)))
    wan_timesteps = tuple(int(timestep) for timestep in getattr(args, "wan_timesteps_parsed", (int(args.wan_timestep),)))
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            images = images_from_batch(batch)
            selected = load_wan_features_for_batch(batch, device, args, wan_layers, wan_timesteps)
            tokens = adapter_module(selected)
            pred = sample_decoder(decoder, tokens, args.num_queries, meta["fm_step_size"], args.seed + batch_idx, images.shape[1])
            target = get_targets(
                batch,
                meta["query_source"],
                max_points=args.num_queries,
                norm_mode=meta.get("norm_mode", "none"),
            )
            chamfer_loss = chamfer_l2(pred, target)
            velocity_mse, timesteps = flow_matching_diagnostics(
                decoder,
                tokens,
                target,
                seed=args.seed + 100000 + batch_idx,
                num_views=images.shape[1],
                t_bins=bin_count,
            )
            batch_quality_for_preview = None
            for sample_idx in range(pred.shape[0]):
                sample_quality = pointcloud_quality_metrics(
                    pred[sample_idx],
                    target[sample_idx],
                    max_points=int(args.val_metric_max_points),
                    seed=int(args.seed + 400000 + batch_idx * 100 + sample_idx),
                )
                _sum_metrics_into(quality_totals, quality_counts, sample_quality)
                if sample_idx == 0:
                    batch_quality_for_preview = sample_quality
            total_chamfer += float(chamfer_loss.item())
            chamfer_count += 1
            total_velocity += float(velocity_mse.sum().item())
            velocity_count += int(velocity_mse.numel())
            if bin_count > 0:
                indices = torch.clamp((timesteps * bin_count).long(), min=0, max=bin_count - 1)
                for bin_idx in range(bin_count):
                    mask = indices == bin_idx
                    if mask.any():
                        values = velocity_mse[mask]
                        bin_totals[bin_idx] += float(values.sum().item())
                        bin_counts[bin_idx] += int(values.numel())
            if save_previews and output_dir is not None and global_step is not None and len(preview_records) < preview_limit:
                preview_records.append(
                    save_val_preview_sample(
                        decoder,
                        tokens,
                        images,
                        batch,
                        meta,
                        args,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        output_dir=output_dir,
                        sample_metrics=batch_quality_for_preview or {},
                    )
                )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats = torch.tensor(
            [total_chamfer, float(chamfer_count), total_velocity, float(velocity_count)],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_chamfer, chamfer_count, total_velocity, velocity_count = stats.tolist()
        if bin_count > 0:
            bin_stats = torch.tensor(
                [value for pair in zip(bin_totals, bin_counts) for value in pair],
                dtype=torch.float64,
                device=device,
            )
            torch.distributed.all_reduce(bin_stats, op=torch.distributed.ReduceOp.SUM)
            vals = bin_stats.tolist()
            bin_totals = vals[0::2]
            bin_counts = [int(x) for x in vals[1::2]]
        quality_stats = torch.tensor(
            [value for key in QUALITY_METRIC_KEYS for value in (quality_totals[key], quality_counts[key])],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(quality_stats, op=torch.distributed.ReduceOp.SUM)
        vals = quality_stats.tolist()
        for idx, key in enumerate(QUALITY_METRIC_KEYS):
            quality_totals[key] = vals[idx * 2]
            quality_counts[key] = int(vals[idx * 2 + 1])
    metrics = {
        "chamfer_l2": total_chamfer / max(1, int(chamfer_count)),
        "velocity_mse": total_velocity / max(1, int(velocity_count)),
        "preview_records": preview_records,
    }
    loss_per_t_bin = []
    for bin_idx in range(bin_count):
        value = bin_totals[bin_idx] / max(1, bin_counts[bin_idx])
        metrics[f"loss_t_bin_{bin_idx:02d}"] = value
        loss_per_t_bin.append({"bin": bin_idx, "count": int(bin_counts[bin_idx]), "velocity_mse": value})
    metrics["loss_per_t_bin"] = loss_per_t_bin
    for key in QUALITY_METRIC_KEYS:
        metrics[key] = quality_totals[key] / max(1, quality_counts[key])
    adapter_module.train()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter_layers", type=int, default=4)
    parser.add_argument("--adapter_hidden_dim", type=int, default=1024)
    parser.add_argument(
        "--adapter_type",
        default="mlp",
        choices=(
            "mlp",
            "cross_attention",
            "self_attention",
            ADAPTER_TYPE_CONV2D_MLP,
            ADAPTER_TYPE_GRID2D_POOL,
            ADAPTER_TYPE_GRID2D_CONV,
            ADAPTER_TYPE_WAN_CROSS_ATTN_RESAMPLER,
            ADAPTER_TYPE_WAN_CROSS_ATTN_MULTILAYER_RESAMPLER,
            ADAPTER_TYPE_WAN_CROSS_ATTN_MULTITIME_RESAMPLER,
            ADAPTER_TYPE_WAN_LATENT_CROSS_ATTN_RESAMPLER,
        ),
    )
    parser.add_argument("--adapter_heads", type=int, default=8)
    parser.add_argument("--adapter_mlp_ratio", type=float, default=2.0)
    parser.add_argument(
        "--adapter_gated",
        action="store_true",
        help="Use zero-initialized residual gates in WAN cross-attention readout blocks.",
    )
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/probe3d/result/wan_t2v_nova_adapter_seed17")
    parser.add_argument("--debug_one_batch", action="store_true")
    parser.add_argument("--nova_ckpt", default=None)
    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter", "scannet"))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--max_val_scenes", type=int, default=0)
    parser.add_argument("--max_test_scenes", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--query_source", default=None)
    parser.add_argument("--scannet_target_mode", default="complete_zpos")
    parser.add_argument("--scannet_frustum_margin", type=float, default=1.0)
    parser.add_argument("--scannet_min_views", type=int, default=2)
    parser.add_argument("--scannet_complete_points", type=int, default=10000)
    parser.add_argument("--scannet_max_interval", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=20000)
    parser.add_argument("--save_ply_queries", type=int, default=40960)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--loss_type", default="nova_flow", choices=("nova_flow", "chamfer_sample", "flow_chamfer_hybrid"))
    parser.add_argument("--chamfer_weight", type=float, default=0.1)
    parser.add_argument("--image_root_map", default=None)
    parser.add_argument("--wan_feature_cache_dir", required=True)
    parser.add_argument("--wan_timestep", type=int, default=749)
    parser.add_argument(
        "--wan_timesteps",
        default="",
        help="Optional comma-separated hidden timestep list for multi-timestep WAN adapters; empty preserves --wan_timestep.",
    )
    parser.add_argument("--wan_layer", type=int, default=20)
    parser.add_argument(
        "--wan_layers",
        default="",
        help="Optional comma-separated hidden layer list for multi-layer WAN adapters; empty preserves --wan_layer.",
    )
    parser.add_argument(
        "--wan_feature_kind",
        default=FEATURE_KIND_HIDDEN,
        choices=(FEATURE_KIND_HIDDEN, FEATURE_KIND_PRED_X0_LATENT, FEATURE_KIND_MODEL_OUTPUT_LATENT),
    )
    parser.add_argument("--wan_feature_mode", default="cache", choices=("cache", "zero", "sample_shuffle"))
    parser.add_argument("--wan_feature_shuffle_seed", type=int, default=17001)
    parser.add_argument(
        "--wan_feature_norm",
        default="none",
        choices=("none", "token_layernorm", "sample_standardize", "token_l2"),
        help="Optional normalization applied after loading cached WAN features; default preserves historical runs.",
    )
    parser.add_argument("--wan_feature_norm_eps", type=float, default=1e-6)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="PSUVPSC3DD")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="PSUVPSC3DD")
    parser.add_argument("--swanlab_workspace", default=None)
    parser.add_argument("--swanlab_experiment", default=None)
    parser.add_argument("--final_test", action="store_true")
    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument("--test_eval_batches", type=int, default=0)
    parser.add_argument("--val_flow_t_bins", type=int, default=10)
    parser.add_argument("--val_metric_max_points", type=int, default=20000)
    parser.add_argument("--val_preview_samples", type=int, default=12)
    parser.add_argument("--val_preview_queries", type=int, default=40960)
    parser.add_argument("--val_preview_dir", default=None)
    return parser.parse_args()


def unwrap_adapter(adapter):
    if isinstance(adapter, (nn.DataParallel, DDP)):
        return unwrap_adapter(adapter.module)
    if isinstance(adapter, AdapterVelocityProbe):
        return adapter.adapter
    return adapter


def main():
    dist_ctx = init_distributed_mode()
    try:
        args = parse_args()
        if args.adapter_type == ADAPTER_TYPE_CONV2D_MLP and not is_latent_feature_kind(args.wan_feature_kind):
            raise ValueError(
                "--adapter_type conv2d_mlp is only supported with latent WAN feature kinds: "
                "pred_x0_latent or model_output_latent"
            )
        if args.adapter_type == ADAPTER_TYPE_WAN_LATENT_CROSS_ATTN_RESAMPLER and not is_latent_feature_kind(args.wan_feature_kind):
            raise ValueError(
                "--adapter_type wan_latent_cross_attn_resampler is only supported with latent WAN feature kinds: "
                "pred_x0_latent or model_output_latent"
            )
        if args.adapter_type == ADAPTER_TYPE_GRID2D_POOL and args.wan_feature_kind != FEATURE_KIND_HIDDEN:
            raise ValueError("--adapter_type grid2d_pool is only supported with --wan_feature_kind hidden")
        if args.adapter_type == ADAPTER_TYPE_GRID2D_CONV and args.wan_feature_kind != FEATURE_KIND_HIDDEN:
            raise ValueError("--adapter_type grid2d_conv is only supported with --wan_feature_kind hidden")
        if args.adapter_type == ADAPTER_TYPE_WAN_CROSS_ATTN_RESAMPLER and args.wan_feature_kind != FEATURE_KIND_HIDDEN:
            raise ValueError("--adapter_type wan_cross_attn_resampler is only supported with --wan_feature_kind hidden")
        if args.adapter_type == ADAPTER_TYPE_WAN_CROSS_ATTN_MULTILAYER_RESAMPLER and args.wan_feature_kind != FEATURE_KIND_HIDDEN:
            raise ValueError("--adapter_type wan_cross_attn_multilayer_resampler is only supported with --wan_feature_kind hidden")
        if args.adapter_type == ADAPTER_TYPE_WAN_CROSS_ATTN_MULTITIME_RESAMPLER and args.wan_feature_kind != FEATURE_KIND_HIDDEN:
            raise ValueError("--adapter_type wan_cross_attn_multitime_resampler is only supported with --wan_feature_kind hidden")
        args.wan_layers_parsed = parse_wan_layers(args.wan_layers, args.wan_layer)
        args.wan_timesteps_parsed = parse_wan_timesteps(args.wan_timesteps, args.wan_timestep)
        if len(args.wan_layers_parsed) > 1 and len(args.wan_timesteps_parsed) > 1:
            raise ValueError("--wan_layers and --wan_timesteps cannot both contain multiple values")
        if len(args.wan_layers_parsed) > 1 and args.wan_feature_kind != FEATURE_KIND_HIDDEN:
            raise ValueError("--wan_layers multi-layer mode is only supported with --wan_feature_kind hidden")
        if len(args.wan_timesteps_parsed) > 1 and args.wan_feature_kind != FEATURE_KIND_HIDDEN:
            raise ValueError("--wan_timesteps multi-timestep mode is only supported with --wan_feature_kind hidden")
        if args.adapter_type == ADAPTER_TYPE_WAN_CROSS_ATTN_MULTILAYER_RESAMPLER and len(args.wan_layers_parsed) < 2:
            raise ValueError("--adapter_type wan_cross_attn_multilayer_resampler requires --wan_layers with at least two layers")
        if args.adapter_type == ADAPTER_TYPE_WAN_CROSS_ATTN_MULTITIME_RESAMPLER and len(args.wan_timesteps_parsed) < 2:
            raise ValueError("--adapter_type wan_cross_attn_multitime_resampler requires --wan_timesteps with at least two timesteps")
        if len(args.wan_layers_parsed) > 1 and args.adapter_type != ADAPTER_TYPE_WAN_CROSS_ATTN_MULTILAYER_RESAMPLER:
            raise ValueError("--wan_layers with multiple layers requires --adapter_type wan_cross_attn_multilayer_resampler")
        if len(args.wan_timesteps_parsed) > 1 and args.adapter_type != ADAPTER_TYPE_WAN_CROSS_ATTN_MULTITIME_RESAMPLER:
            raise ValueError("--wan_timesteps with multiple timesteps requires --adapter_type wan_cross_attn_multitime_resampler")
        if args.debug_one_batch:
            args.max_steps = 1
            args.save_every = 1
            args.val_every = 1
        set_seed(args.seed)
        device = dist_ctx["device"] if dist_ctx["enabled"] else resolve_device(args.device)
        is_main = dist_ctx["is_main"]
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "training.log"

        decoder, meta, cfg = build_decoder(device, args.nova_ckpt)
        meta = dict(meta)
        if args.query_source is not None:
            meta["query_source"] = args.query_source
        elif args.scannet_target_mode == "src_view":
            meta["query_source"] = "src_view"
        image_root_map = parse_image_root_map(args.image_root_map)
        train_loader, data_args = build_loader(
            cfg, args.batch_size, args.num_workers, test=False, image_root_map=image_root_map,
            dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
            split_override=args.train_split, distributed=dist_ctx["enabled"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"],
            scannet_target_mode=args.scannet_target_mode, scannet_frustum_margin=args.scannet_frustum_margin, scannet_min_views=args.scannet_min_views,
            scannet_complete_points=args.scannet_complete_points, scannet_max_interval=args.scannet_max_interval,
        )
        val_loader = None
        test_loader = None
        if dist_ctx["enabled"] or is_main:
            val_loader, _ = build_loader(
                cfg, args.batch_size, max(0, min(args.num_workers, 2)), test=True, image_root_map=image_root_map,
                dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
                split_override=args.val_split, max_scenes=args.max_val_scenes,
                distributed=dist_ctx["enabled"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"],
                scannet_target_mode=args.scannet_target_mode, scannet_frustum_margin=args.scannet_frustum_margin, scannet_min_views=args.scannet_min_views,
                scannet_complete_points=args.scannet_complete_points, scannet_max_interval=args.scannet_max_interval,
            )
            if args.final_test:
                test_loader, _ = build_loader(
                    cfg, args.batch_size, max(0, min(args.num_workers, 2)), test=True, image_root_map=image_root_map,
                    dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
                    split_override=args.test_split, max_scenes=args.max_test_scenes,
                    distributed=dist_ctx["enabled"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"],
                    scannet_target_mode=args.scannet_target_mode, scannet_frustum_margin=args.scannet_frustum_margin, scannet_min_views=args.scannet_min_views,
                    scannet_complete_points=args.scannet_complete_points, scannet_max_interval=args.scannet_max_interval,
                )

        sampler_set_epoch(train_loader, 0)
        first_batch = next(iter(train_loader))
        first_batch = move_batch_to_device(first_batch, device)
        first_images = images_from_batch(first_batch)
        selected = load_wan_features_for_batch(first_batch, device, args, args.wan_layers_parsed, args.wan_timesteps_parsed)
        if is_main:
            print("Feature backbone: wan_t2v_cache")
            print(f"Requested WAN T2V timestep/layer: {args.wan_timestep}/{args.wan_layer}")
            print(f"Requested WAN T2V timesteps: {list(args.wan_timesteps_parsed)}")
            print(f"Requested WAN T2V layers: {list(args.wan_layers_parsed)}")
            print(f"WAN feature kind: {args.wan_feature_kind}")
            print(f"WAN feature mode: {args.wan_feature_mode}")
            print(f"WAN feature norm: {args.wan_feature_norm} eps={args.wan_feature_norm_eps}")
            print(f"Selected WAN feature shape: {tuple(selected.shape)}")
        if len(args.wan_timesteps_parsed) > 1:
            wan_cache_metadata = {}
            wan_layer_metadata = []
            wan_timestep_metadata = read_wan_multitime_cache_metadata(
                args.wan_feature_cache_dir,
                args.wan_timesteps_parsed,
                int(args.wan_layers_parsed[0]),
                first_batch["scene_ids"][0],
            )
        elif len(args.wan_layers_parsed) > 1:
            wan_cache_metadata = {}
            wan_layer_metadata = read_wan_multilayer_cache_metadata(
                args.wan_feature_cache_dir,
                args.wan_timestep,
                args.wan_layers_parsed,
                first_batch["scene_ids"][0],
            )
            wan_timestep_metadata = []
        else:
            wan_cache_metadata = read_wan_cache_metadata(
                args.wan_feature_cache_dir,
                int(args.wan_timesteps_parsed[0]),
                int(args.wan_layers_parsed[0]),
                first_batch["scene_ids"][0],
                args.wan_feature_kind,
            )
            wan_layer_metadata = []
            wan_timestep_metadata = []
        wan_num_sources = max(1, len(args.wan_layers_parsed), len(args.wan_timesteps_parsed))
        wan_hidden_grid_shape = (
            infer_wan_hidden_grid_shape(wan_cache_metadata, selected, num_sources=wan_num_sources)
            if args.wan_feature_kind == FEATURE_KIND_HIDDEN
            else ()
        )
        wan_latent_grid_shape = (
            infer_wan_latent_grid_shape(wan_cache_metadata, selected)
            if is_latent_feature_kind(args.wan_feature_kind)
            else ()
        )

        if args.adapter_type == "mlp":
            adapter = VGGTToNovaAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
            )
        elif args.adapter_type == "cross_attention":
            adapter = VGGTToNovaCrossAttentionAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                mlp_ratio=args.adapter_mlp_ratio,
            )
        elif args.adapter_type == "self_attention":
            adapter = VGGTToNovaSelfAttentionAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                mlp_ratio=args.adapter_mlp_ratio,
            )
        elif args.adapter_type == ADAPTER_TYPE_CONV2D_MLP:
            adapter = WanPredX0LatentConvAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
            )
        elif args.adapter_type == ADAPTER_TYPE_WAN_LATENT_CROSS_ATTN_RESAMPLER:
            adapter = WanLatentCrossAttentionResamplerAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                mlp_ratio=args.adapter_mlp_ratio,
                gated=args.adapter_gated,
                temporal_tokens=int(wan_latent_grid_shape[0]),
                latent_height=int(wan_latent_grid_shape[1]),
                latent_width=int(wan_latent_grid_shape[2]),
            )
        elif args.adapter_type == ADAPTER_TYPE_GRID2D_POOL:
            adapter = WanHiddenGrid2DPoolAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                temporal_tokens=int(wan_hidden_grid_shape[0]),
                grid_height=int(wan_hidden_grid_shape[1]),
                grid_width=int(wan_hidden_grid_shape[2]),
            )
        elif args.adapter_type == ADAPTER_TYPE_GRID2D_CONV:
            adapter = WanHiddenGrid2DConvAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                temporal_tokens=int(wan_hidden_grid_shape[0]),
                grid_height=int(wan_hidden_grid_shape[1]),
                grid_width=int(wan_hidden_grid_shape[2]),
            )
        elif args.adapter_type == ADAPTER_TYPE_WAN_CROSS_ATTN_RESAMPLER:
            adapter = WanHiddenCrossAttentionResamplerAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                mlp_ratio=args.adapter_mlp_ratio,
                gated=args.adapter_gated,
                temporal_tokens=int(wan_hidden_grid_shape[0]),
                grid_height=int(wan_hidden_grid_shape[1]),
                grid_width=int(wan_hidden_grid_shape[2]),
            )
        elif args.adapter_type == ADAPTER_TYPE_WAN_CROSS_ATTN_MULTILAYER_RESAMPLER:
            adapter = WanHiddenMultiLayerCrossAttentionResamplerAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                num_layers=len(args.wan_layers_parsed),
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                mlp_ratio=args.adapter_mlp_ratio,
                gated=args.adapter_gated,
                temporal_tokens=int(wan_hidden_grid_shape[0]),
                grid_height=int(wan_hidden_grid_shape[1]),
                grid_width=int(wan_hidden_grid_shape[2]),
            )
        elif args.adapter_type == ADAPTER_TYPE_WAN_CROSS_ATTN_MULTITIME_RESAMPLER:
            adapter = WanHiddenMultiSourceCrossAttentionResamplerAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                num_sources=len(args.wan_timesteps_parsed),
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                mlp_ratio=args.adapter_mlp_ratio,
                gated=args.adapter_gated,
                temporal_tokens=int(wan_hidden_grid_shape[0]),
                grid_height=int(wan_hidden_grid_shape[1]),
                grid_width=int(wan_hidden_grid_shape[2]),
            )
        else:
            raise ValueError(f"Unsupported adapter_type={args.adapter_type!r}")
        adapter = adapter.to(device)
        probe_model = AdapterVelocityProbe(adapter, decoder).to(device)
        use_parallel = False
        if dist_ctx["enabled"]:
            ddp_device_ids = [dist_ctx["local_rank"]] if device.type == "cuda" else None
            probe_model = DDP(probe_model, device_ids=ddp_device_ids, output_device=dist_ctx["local_rank"] if device.type == "cuda" else None)
            if is_main:
                print(f"Using DDP on world_size={dist_ctx['world_size']} (rank={dist_ctx['rank']}, local_rank={dist_ctx['local_rank']}).")
        else:
            use_parallel = bool(args.parallel and device.type == "cuda" and torch.cuda.device_count() > 1)
            if use_parallel:
                probe_model = nn.DataParallel(probe_model)
                if is_main:
                    print(f"Using DataParallel on {torch.cuda.device_count()} visible CUDA devices.")
            else:
                if is_main:
                    print(f"Using single-device training on {device}.")
        optimizer = torch.optim.AdamW(unwrap_adapter(probe_model).parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
        assert_only_adapter_trainable(unwrap_adapter(probe_model), None, decoder)

        config = vars(args).copy()
        config.update(
            {
                "feature_backbone": "wan_t2v_cache",
                "selected_feature_shape": list(selected.shape),
                "wan_hidden_grid_shape": list(wan_hidden_grid_shape),
                "wan_latent_grid_shape": list(wan_latent_grid_shape),
                "wan_timesteps": [int(timestep) for timestep in args.wan_timesteps_parsed],
                "num_wan_timesteps": int(len(args.wan_timesteps_parsed)),
                "wan_layers": [int(layer) for layer in args.wan_layers_parsed],
                "num_wan_layers": int(len(args.wan_layers_parsed)),
                "wan_cache_metadata": {
                    key: wan_cache_metadata.get(key)
                    for key in (
                        "noise_mode",
                        "timestep_index",
                        "requested_timestep_index",
                        "scheduler_timestep",
                        "sigma",
                        "scheduler_class",
                        "scheduler_prediction_type",
                        "scheduler_predict_x0",
                        "x0_formula",
                        "latent_noise_applied",
                        "low_noise_index",
                        "feature_kind",
                        "feature_source",
                        "source",
                        "window_mode",
                        "condition_frames",
                        "model_id",
                        "model_local_dir",
                        "height",
                        "width",
                        "grid_shape",
                        "full_feature_shape",
                        "source_latent_shape",
                        "transformer_latent_shape",
                        "model_output_shape",
                        "pred_x0_latent_shape",
                        "model_output_latent_shape",
                        "model_output_formula",
                        "feature_shape",
                    )
                    if key in wan_cache_metadata
                },
                "wan_layer_metadata": wan_layer_metadata,
                "wan_timestep_metadata": wan_timestep_metadata,
                "adapter_type": args.adapter_type,
                "adapter_heads": args.adapter_heads,
                "adapter_mlp_ratio": args.adapter_mlp_ratio,
                "adapter_gated": bool(args.adapter_gated),
                "adapter_attention_gated": getattr(unwrap_adapter(probe_model), "gated", None),
                "adapter_param_count": count_parameters(unwrap_adapter(probe_model)),
                "adapter_conv_readout_shape": list(getattr(unwrap_adapter(probe_model), "conv_readout_shape", ())),
                "adapter_grid_input_shape": list(getattr(unwrap_adapter(probe_model), "grid_input_shape", ())),
                "adapter_grid_readout_shape": list(getattr(unwrap_adapter(probe_model), "grid_readout_shape", ())),
                "adapter_resampler_shape": list(getattr(unwrap_adapter(probe_model), "resampler_shape", ())),
                "loss_type": args.loss_type,
                "chamfer_weight": args.chamfer_weight,
                "scannet_complete_points": args.scannet_complete_points,
                "scannet_max_interval": args.scannet_max_interval,
                "parallel": use_parallel,
                "ddp": dist_ctx["enabled"],
                "rank": dist_ctx["rank"],
                "world_size": dist_ctx["world_size"],
                "visible_cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
                "nova_decoder_meta": meta,
                "nova_ckpt": str(args.nova_ckpt) if args.nova_ckpt else None,
                "dataset": {"data_root": data_args.data_root, "test_dataset_name": data_args.test_dataset_name},
                "image_root_map": args.image_root_map,
            }
        )
        if is_main:
            save_json(output_dir / "config.json", config)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(config, indent=2) + "\n")
                log.write("Trainable parameter names:\n")
                for name in trainable_parameter_names({"adapter": unwrap_adapter(probe_model), "decoder": decoder}):
                    log.write(f"{name}\n")
                    print(f"trainable: {name}")

        wandb_run = maybe_init_wandb(args, output_dir, config) if is_main else None
        swanlab_run = maybe_init_swanlab(args, output_dir, config) if is_main else False

        best_loss = math.inf
        best_val_chamfer_l2 = math.inf
        best_val_fscore_tau_010 = -math.inf
        best_val_pred_to_gt_p90 = math.inf
        last_val_metrics = None
        first_loss = None
        final_loss = None
        global_step = 0
        train_epoch = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            unwrap_adapter(probe_model).load_state_dict(checkpoint["adapter"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            global_step = int(checkpoint.get("global_step", 0))
            best_loss = float(checkpoint.get("best_loss", best_loss))
            first_loss = checkpoint.get("first_loss", first_loss)
            final_loss = checkpoint.get("final_loss", final_loss)
            best_val_chamfer_l2 = float(checkpoint.get("best_val_chamfer_l2", best_val_chamfer_l2))
            best_val_fscore_tau_010 = float(checkpoint.get("best_val_fscore_tau_0.10", best_val_fscore_tau_010))
            best_val_pred_to_gt_p90 = float(checkpoint.get("best_val_pred_to_gt_p90", best_val_pred_to_gt_p90))
            if is_main:
                print(f"Resumed from {args.resume} at global_step={global_step}")

        sampler_set_epoch(train_loader, train_epoch)
        data_iter = iter(train_loader)
        while global_step < args.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                train_epoch += 1
                sampler_set_epoch(train_loader, train_epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = move_batch_to_device(batch, device)
            images = images_from_batch(batch)
            selected = load_wan_features_for_batch(batch, device, args, args.wan_layers_parsed, args.wan_timesteps_parsed)
            target = get_targets(batch, meta["query_source"], max_points=args.num_queries, norm_mode=meta.get("norm_mode", "none"))

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                loss, tokens, pred_velocity, target_velocity = probe_model(
                    selected,
                    target,
                    args.seed + global_step,
                    images.shape[1],
                    loss_type=args.loss_type,
                    num_queries=args.num_queries,
                    fm_step_size=meta["fm_step_size"],
                    chamfer_weight=args.chamfer_weight,
                )
                if loss.ndim > 0:
                    loss = loss.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step + 1}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grads_ok = all(p.grad is not None for p in unwrap_adapter(probe_model).parameters() if p.requires_grad)
            if global_step == 0 and not grads_ok:
                raise AssertionError("Adapter parameters did not receive gradients after first backward pass.")
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            reduced_loss = reduce_scalar(loss, dist_ctx["enabled"])
            final_loss = reduced_loss
            if first_loss is None:
                first_loss = final_loss
            if final_loss < best_loss:
                best_loss = final_loss
            if is_main and (global_step % 10 == 0 or global_step == 1 or args.debug_one_batch):
                print(f"step={global_step} loss={final_loss:.6f} best={best_loss:.6f}")
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"step={global_step} loss={final_loss:.8f} best={best_loss:.8f}\n")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": final_loss, "train/best_loss": best_loss}, step=global_step)
                if swanlab_run:
                    swanlab.log({"train/loss": final_loss, "train/best_loss": best_loss}, step=global_step)

            if (global_step % args.val_every == 0 or args.debug_one_batch) and val_loader is not None:
                val_metrics = run_eval(
                    probe_model,
                    decoder,
                    val_loader,
                    device,
                    meta,
                    args,
                    max_batches=None if args.eval_batches <= 0 else args.eval_batches,
                    output_dir=output_dir,
                    global_step=global_step,
                    save_previews=is_main and args.val_preview_samples > 0,
                )
                if is_main:
                    last_val_metrics = val_metrics
                    val_chamfer_l2 = float(val_metrics["chamfer_l2"])
                    val_velocity_mse = float(val_metrics["velocity_mse"])
                    val_fscore_tau_010 = float(val_metrics["fscore_tau_0.10"])
                    val_pred_to_gt_p90 = float(val_metrics["pred_to_gt_p90"])
                    validation_payload = {
                        "step": global_step,
                        "val_chamfer_l2": val_chamfer_l2,
                        "val_velocity_mse": val_velocity_mse,
                        "loss_per_t_bin": val_metrics.get("loss_per_t_bin", []),
                        "preview_records": val_metrics.get("preview_records", []),
                    }
                    for key, value in val_metrics.items():
                        if key.startswith("loss_t_bin_"):
                            validation_payload[key] = float(value)
                        elif key in QUALITY_METRIC_KEYS:
                            validation_payload[f"val_{key}"] = float(value)
                    save_json(output_dir / "validation_metrics.json", validation_payload)
                    for record in validation_payload.get("loss_per_t_bin", []):
                        validation_payload[f"loss_t_bin_{int(record['bin']):02d}"] = float(record["velocity_mse"])
                    if args.val_preview_samples > 0:
                        preview_root = resolve_val_preview_root(args, output_dir)
                        save_json(
                            preview_root / "latest_index.json",
                            {
                                "step": global_step,
                                "preview_root": str(preview_root),
                                "records": val_metrics.get("preview_records", []),
                            },
                        )
                    print(
                        f"val step={global_step} "
                        f"val_chamfer_l2={val_chamfer_l2:.8f} "
                        f"val_velocity_mse={val_velocity_mse:.8f} "
                        f"val_pred_to_gt_p90={val_pred_to_gt_p90:.8f} "
                        f"val_gt_to_pred_p90={float(val_metrics['gt_to_pred_p90']):.8f} "
                        f"val_fscore_tau_0.05={float(val_metrics['fscore_tau_0.05']):.8f} "
                        f"val_fscore_tau_0.10={val_fscore_tau_010:.8f}"
                    )
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(
                            f"val step={global_step} "
                            f"val_chamfer_l2={val_chamfer_l2:.8f} "
                            f"val_velocity_mse={val_velocity_mse:.8f} "
                            f"val_pred_to_gt_p90={val_pred_to_gt_p90:.8f} "
                            f"val_gt_to_pred_p90={float(val_metrics['gt_to_pred_p90']):.8f} "
                            f"val_fscore_tau_0.05={float(val_metrics['fscore_tau_0.05']):.8f} "
                            f"val_fscore_tau_0.10={val_fscore_tau_010:.8f}\n"
                        )
                    next_best_val_chamfer_l2 = min(best_val_chamfer_l2, val_chamfer_l2)
                    next_best_val_fscore_tau_010 = max(best_val_fscore_tau_010, val_fscore_tau_010)
                    next_best_val_pred_to_gt_p90 = min(best_val_pred_to_gt_p90, val_pred_to_gt_p90)
                    if val_chamfer_l2 < best_val_chamfer_l2:
                        best_val_chamfer_l2 = val_chamfer_l2
                        save_checkpoint(output_dir / "best.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": best_val_chamfer_l2, "best_val_fscore_tau_0.10": next_best_val_fscore_tau_010, "best_val_pred_to_gt_p90": next_best_val_pred_to_gt_p90})
                    if val_fscore_tau_010 > best_val_fscore_tau_010:
                        best_val_fscore_tau_010 = val_fscore_tau_010
                        save_checkpoint(output_dir / "best_fscore_010.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": next_best_val_chamfer_l2, "best_val_fscore_tau_0.10": best_val_fscore_tau_010, "best_val_pred_to_gt_p90": next_best_val_pred_to_gt_p90})
                    if val_pred_to_gt_p90 < best_val_pred_to_gt_p90:
                        best_val_pred_to_gt_p90 = val_pred_to_gt_p90
                        save_checkpoint(output_dir / "best_pred_to_gt_p90.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": next_best_val_chamfer_l2, "best_val_fscore_tau_0.10": next_best_val_fscore_tau_010, "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90})
                    tracking_metrics = {
                        "val/chamfer_l2": val_chamfer_l2,
                        "val/velocity_mse": val_velocity_mse,
                        "val/best_chamfer_l2": float(best_val_chamfer_l2),
                        "val/pred_to_gt_p90": val_pred_to_gt_p90,
                        "val/gt_to_pred_p90": float(val_metrics["gt_to_pred_p90"]),
                        "val/fscore_tau_0.05": float(val_metrics["fscore_tau_0.05"]),
                        "val/fscore_tau_0.10": val_fscore_tau_010,
                        "val/precision_tau_0.10": float(val_metrics["precision_tau_0.10"]),
                        "val/recall_tau_0.10": float(val_metrics["recall_tau_0.10"]),
                        "val/trimmed_cd_l2_95": float(val_metrics["trimmed_cd_l2_95"]),
                        "val/best_fscore_tau_0.10": float(best_val_fscore_tau_010),
                        "val/best_pred_to_gt_p90": float(best_val_pred_to_gt_p90),
                    }
                    for key, value in validation_payload.items():
                        if key.startswith("loss_t_bin_"):
                            tracking_metrics[f"val/{key}"] = float(value)
                    if wandb_run is not None:
                        wandb_run.log(tracking_metrics, step=global_step)
                    if swanlab_run:
                        swanlab.log(tracking_metrics, step=global_step)
                barrier_if_distributed(dist_ctx["enabled"])
            if global_step % args.save_every == 0 or args.debug_one_batch:
                if is_main:
                    checkpoint_extra = {
                        "best_val_chamfer_l2": best_val_chamfer_l2,
                        "best_val_fscore_tau_0.10": best_val_fscore_tau_010,
                        "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
                    }
                    save_checkpoint(output_dir / f"step_{global_step:06d}.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra=checkpoint_extra)
                    save_checkpoint(output_dir / "latest.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra=checkpoint_extra)
                    try:
                        scene_ids = scene_ids_from_batch(batch, global_step)
                        ply_dir = output_dir / "ply"
                        pred_path = ply_dir / f"{scene_ids[0]}_step{global_step:06d}_pred.ply"
                        gt_path = ply_dir / f"{scene_ids[0]}_pseudo_gt.ply"
                        ply_pred = sample_decoder(decoder, tokens.detach(), args.save_ply_queries, meta["fm_step_size"], args.seed + 999 + global_step, images.shape[1])
                        write_point_cloud_ply(pred_path, ply_pred[0])
                        if not gt_path.exists():
                            ply_target = get_targets(batch, meta["query_source"], max_points=args.save_ply_queries, norm_mode=meta.get("norm_mode", "none"))
                            write_point_cloud_ply(gt_path, ply_target[0])
                    except Exception as exc:
                        warn = f"export warning step={global_step}: {type(exc).__name__}: {exc}"
                        print(warn)
                        with log_path.open("a", encoding="utf-8") as log:
                            log.write(warn + "\n")
                        if wandb_run is not None:
                            wandb_run.log({"export/failed": 1.0}, step=global_step)
                        if swanlab_run:
                            swanlab.log({"export/failed": 1.0}, step=global_step)
                barrier_if_distributed(dist_ctx["enabled"])

        test_metrics = None
        if test_loader is not None:
            test_metrics = run_eval(
                probe_model,
                decoder,
                test_loader,
                device,
                meta,
                args,
                max_batches=None if args.test_eval_batches <= 0 else args.test_eval_batches,
            )
        if is_main:
            final_metrics = {
                "feature_backbone": "wan_t2v_cache",
                "wan_timestep": int(args.wan_timestep),
                "wan_timesteps": [int(timestep) for timestep in args.wan_timesteps_parsed],
                "num_wan_timesteps": int(len(args.wan_timesteps_parsed)),
                "wan_layer": int(args.wan_layer),
                "wan_layers": [int(layer) for layer in args.wan_layers_parsed],
                "num_wan_layers": int(len(args.wan_layers_parsed)),
                "wan_feature_kind": args.wan_feature_kind,
                "wan_feature_mode": args.wan_feature_mode,
                "wan_feature_shuffle_seed": int(args.wan_feature_shuffle_seed),
                "wan_feature_norm": args.wan_feature_norm,
                "wan_feature_norm_eps": float(args.wan_feature_norm_eps),
                "adapter_type": args.adapter_type,
                "adapter_gated": bool(args.adapter_gated),
                "wan_hidden_grid_shape": config.get("wan_hidden_grid_shape", []),
                "wan_latent_grid_shape": config.get("wan_latent_grid_shape", []),
                "wan_cache_metadata": config.get("wan_cache_metadata", {}),
                "wan_layer_metadata": config.get("wan_layer_metadata", []),
                "wan_timestep_metadata": config.get("wan_timestep_metadata", []),
                "first_loss": first_loss,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "best_val_chamfer_l2": best_val_chamfer_l2,
                "best_val_fscore_tau_0.10": best_val_fscore_tau_010,
                "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
            }
            if last_val_metrics is not None:
                final_metrics["last_val_chamfer_l2"] = float(last_val_metrics["chamfer_l2"])
                final_metrics["last_val_velocity_mse"] = float(last_val_metrics["velocity_mse"])
                final_metrics["last_val_loss_per_t_bin"] = last_val_metrics.get("loss_per_t_bin", [])
                for key in QUALITY_METRIC_KEYS:
                    final_metrics[f"last_val_{key}"] = float(last_val_metrics[key])
            if test_metrics is not None:
                test_chamfer_l2 = float(test_metrics["chamfer_l2"])
                test_velocity_mse = float(test_metrics["velocity_mse"])
                final_metrics["test_chamfer_l2"] = test_chamfer_l2
                final_metrics["test_velocity_mse"] = test_velocity_mse
                for key in QUALITY_METRIC_KEYS:
                    final_metrics[f"test_{key}"] = float(test_metrics[key])
            save_json(output_dir / "final_metrics.json", final_metrics)
            if wandb_run is not None:
                wandb_run.summary["first_loss"] = first_loss
                wandb_run.summary["final_loss"] = final_loss
                wandb_run.summary["best_loss"] = best_loss
                wandb_run.summary["best_val_chamfer_l2"] = best_val_chamfer_l2
                wandb_run.summary["best_val_fscore_tau_0.10"] = best_val_fscore_tau_010
                wandb_run.summary["best_val_pred_to_gt_p90"] = best_val_pred_to_gt_p90
                wandb_run.finish()
            if swanlab_run:
                swanlab.log(
                    {
                        "summary/first_loss": first_loss,
                        "summary/final_loss": final_loss,
                        "summary/best_loss": best_loss,
                        "summary/best_val_chamfer_l2": best_val_chamfer_l2,
                        "summary/best_val_fscore_tau_0.10": best_val_fscore_tau_010,
                        "summary/best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
                    },
                    step=global_step,
                )
                try:
                    swanlab.finish()
                except Exception as exc:
                    print(f"SwanLab finish warning: {type(exc).__name__}: {exc}")
            print(f"First loss: {first_loss}")
            print(f"Final loss: {final_loss}")
            print(f"Output directory: {output_dir}")
        barrier_if_distributed(dist_ctx["enabled"])
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
