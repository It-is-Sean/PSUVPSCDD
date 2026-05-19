#!/usr/bin/env python3
"""Precompute SCRREAM pair-conditioned frozen V-JEPA 2.1 encoder features."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.backends\.cuda\.sdp_kernel\(\)` is deprecated",
    category=FutureWarning,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
VJEPA2_ROOT = REPO_ROOT / "third_party" / "vjepa2"
if str(VJEPA2_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA2_ROOT))

from evals.hub.preprocessor import vjepa2_preprocessor
from src.hub.backbones import (  # type: ignore
    _clean_backbone_key,
    vjepa2_1_vit_base_384,
    vjepa2_1_vit_giant_384,
    vjepa2_1_vit_gigantic_384,
    vjepa2_1_vit_large_384,
)


DEFAULT_ADAPTER_DATA = (
    "experiments/probe3d/adapter_data/"
    "scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt"
)
DEFAULT_CACHE_ROOT = "experiments/probe3d/feature_cache/scrream_vjepa21"
DEFAULT_CHECKPOINT = "checkpoints/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
DEFAULT_MODEL_NAME = "vjepa2_1_vit_large_384"
DEFAULT_CROP_SIZE = 384
DEFAULT_PATCH_SIZE = 16
DEFAULT_TUBELET_SIZE = 2
MODE_PAIR_EXACT16 = "pair_exact16"
MODE_CTX_ANCHOR16 = "ctx_anchor16"
MODE_CTX_SHUFFLE16 = "ctx_shuffle16"
MODE_CTX_ANCHOR32 = "ctx_anchor32"
MODE_CTX_ANCHOR64 = "ctx_anchor64"
MODE_CTX_CONTIG81 = "ctx_contig81"
MODE_CTX_WAN16 = "ctx_wan16"
MODE_CTX_WAN64 = "ctx_wan64"
WINDOW_MODES = (
    MODE_PAIR_EXACT16,
    MODE_CTX_ANCHOR16,
    MODE_CTX_SHUFFLE16,
    MODE_CTX_ANCHOR32,
    MODE_CTX_ANCHOR64,
    MODE_CTX_CONTIG81,
    MODE_CTX_WAN16,
    MODE_CTX_WAN64,
)


@dataclass(frozen=True)
class WindowSpec:
    sample_id: str
    scene_id: str
    sequence_id: str
    frame_ids: tuple[int, int]
    frame_paths: tuple[str, str]
    sequence_dir: str
    min_frame: int
    max_frame: int


def safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(sample_id)).strip("_")


def cache_path(cache_root: Path, clip_mode: str, sample_id: str) -> Path:
    return cache_root / clip_mode / f"{safe_sample_id(sample_id)}.pt"


def load_rgb_paths(sequence_dir: Path) -> dict[int, Path]:
    rgb_dir = sequence_dir / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing SCRREAM rgb directory: {rgb_dir}")
    mapping: dict[int, Path] = {}
    for path in sorted(rgb_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        mapping[frame_id] = path
    if not mapping:
        raise FileNotFoundError(f"No numbered RGB frames found under {rgb_dir}")
    return mapping


def build_window_spec(meta: dict[str, Any]) -> WindowSpec:
    frame_ids_raw = meta.get("frame_ids")
    frame_paths_raw = meta.get("frame_paths")
    if not frame_ids_raw or len(frame_ids_raw) != 2:
        raise ValueError(f"Sample {meta.get('sample_id', '<unknown>')} has invalid frame_ids={frame_ids_raw!r}")
    if not frame_paths_raw or len(frame_paths_raw) != 2:
        raise ValueError(f"Sample {meta.get('sample_id', '<unknown>')} has invalid frame_paths={frame_paths_raw!r}")

    frame_ids = (int(frame_ids_raw[0]), int(frame_ids_raw[1]))
    frame_paths = (str(frame_paths_raw[0]), str(frame_paths_raw[1]))
    sequence_dir = Path(frame_paths[0]).parent.parent
    sample_id = str(
        meta.get("sample_id")
        or f"{meta.get('scene_id')}/{meta.get('sequence_id')}_{frame_ids[0]:06d}_{frame_ids[1]:06d}"
        )
    scene_id = str(meta.get("scene_id", ""))
    sequence_id = str(meta.get("sequence_id", sequence_dir.name))
    rgb_map = load_rgb_paths(sequence_dir)
    min_frame = min(rgb_map)
    max_frame = max(rgb_map)
    missing_pair_paths = [path for path in frame_paths if not Path(path).is_file()]
    if missing_pair_paths:
        raise FileNotFoundError(f"Missing pair RGB frames for {sample_id}: {missing_pair_paths}")
    return WindowSpec(
        sample_id=sample_id,
        scene_id=scene_id,
        sequence_id=sequence_id,
        frame_ids=frame_ids,
        frame_paths=frame_paths,
        sequence_dir=str(sequence_dir),
        min_frame=min_frame,
        max_frame=max_frame,
    )


def _linspace_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 1:
        return [int(end)]
    values = torch.linspace(float(start), float(end), steps=count)
    return [int(round(float(v.item()))) for v in values]


def _contiguous_block_indices(anchor_frame: int, block_size: int, min_frame: int, max_frame: int, align: str) -> list[int]:
    if align == "end":
        start = anchor_frame - (block_size - 1)
        start = max(min_frame, min(start, max_frame - block_size + 1))
    elif align == "start":
        start = anchor_frame
        start = max(min_frame, min(start, max_frame - block_size + 1))
    else:
        raise ValueError(f"Unsupported align={align!r}")
    return list(range(start, start + block_size))


def _find_anchor_positions(raw_indices: list[int], frame_ids: tuple[int, int]) -> tuple[int, int]:
    f0, f1 = int(frame_ids[0]), int(frame_ids[1])
    try:
        f0_pos = max(idx for idx, frame_id in enumerate(raw_indices) if int(frame_id) == f0)
    except ValueError as exc:
        raise ValueError(f"Could not locate f0={f0} in clip indices {raw_indices}") from exc
    try:
        f1_pos = min(idx for idx, frame_id in enumerate(raw_indices) if int(frame_id) == f1 and idx >= f0_pos)
    except ValueError:
        try:
            f1_pos = min(idx for idx, frame_id in enumerate(raw_indices) if int(frame_id) == f1)
        except ValueError as exc:
            raise ValueError(f"Could not locate f1={f1} in clip indices {raw_indices}") from exc
    return int(f0_pos), int(f1_pos)


def _build_ctx81_indices(spec: WindowSpec) -> list[int]:
    rgb_map = load_rgb_paths(Path(spec.sequence_dir))
    min_frame = min(rgb_map)
    max_frame = max(rgb_map)
    if max_frame - min_frame + 1 < 81:
        raise ValueError(f"Sequence {spec.sequence_dir} has fewer than 81 numbered frames")
    mid = (spec.frame_ids[0] + spec.frame_ids[1]) // 2
    start = mid - 40
    start = max(min_frame, min(start, max_frame - 80))
    end = start + 80
    if not (start <= spec.frame_ids[0] <= end and start <= spec.frame_ids[1] <= end):
        raise ValueError(
            f"Pair {spec.frame_ids} cannot fit in 81-frame window [{start}, {end}] for {spec.sequence_dir}"
        )
    missing = [idx for idx in range(start, end + 1) if idx not in rgb_map]
    if missing:
        raise FileNotFoundError(f"Missing RGB frames in {spec.sequence_dir}: first missing ids {missing[:8]}")
    return list(range(start, end + 1))


def _sample_ctx81_to_wan16(raw81_indices: list[int], frame_ids: tuple[int, int]) -> list[int]:
    start = int(raw81_indices[0])
    end = int(raw81_indices[-1])
    f0 = int(frame_ids[0])
    f1 = int(frame_ids[1])
    left = _linspace_indices(start, f0, 5)
    middle = _linspace_indices(f0, f1, 6)
    right = _linspace_indices(f1, end, 7)
    merged = left + middle[1:] + right[1:]
    if len(merged) != 16:
        raise ValueError(f"ctx_wan16 expected 16 frames, got {len(merged)}")
    if merged[0] != start:
        merged[0] = start
    if merged[-1] != end:
        merged[-1] = end
    f0_pos, f1_pos = _find_anchor_positions(merged, frame_ids)
    merged[f0_pos] = f0
    merged[f1_pos] = f1
    for idx in range(1, len(merged)):
        if merged[idx] < merged[idx - 1]:
            merged[idx] = merged[idx - 1]
    for idx in range(len(merged) - 2, -1, -1):
        if merged[idx] > merged[idx + 1]:
            merged[idx] = merged[idx + 1]
    merged[0] = start
    merged[-1] = end
    if f0 not in merged or f1 not in merged:
        raise ValueError(f"ctx_wan16 failed to retain pair frames {frame_ids} in sampled indices {merged}")
    return [int(x) for x in merged]


def _sample_ctx81_uniform(raw81_indices: list[int], frame_ids: tuple[int, int], count: int) -> list[int]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count > len(raw81_indices):
        raise ValueError(f"Cannot sample {count} frames from ctx81 window of size {len(raw81_indices)}")
    base_positions = _linspace_indices(0, len(raw81_indices) - 1, count)
    selected_positions = sorted({int(x) for x in base_positions})
    while len(selected_positions) < count:
        for pos in range(len(raw81_indices)):
            if pos not in selected_positions:
                selected_positions.append(pos)
                if len(selected_positions) == count:
                    break
    f0_pos = raw81_indices.index(int(frame_ids[0]))
    f1_pos = raw81_indices.index(int(frame_ids[1]))
    if f0_pos not in selected_positions:
        worst = min(selected_positions, key=lambda pos: min(abs(pos - f0_pos), abs(pos - f1_pos)))
        selected_positions.remove(worst)
        selected_positions.append(f0_pos)
    if f1_pos not in selected_positions:
        worst = min(selected_positions, key=lambda pos: abs(pos - f1_pos))
        if worst == f0_pos and len(selected_positions) > 1:
            worst = min([pos for pos in selected_positions if pos != f0_pos], key=lambda pos: abs(pos - f1_pos))
        selected_positions.remove(worst)
        selected_positions.append(f1_pos)
    selected_positions = sorted(selected_positions)
    sampled = [int(raw81_indices[pos]) for pos in selected_positions]
    if int(frame_ids[0]) not in sampled or int(frame_ids[1]) not in sampled:
        raise ValueError(f"ctx_uniform sampling failed to retain pair frames {frame_ids} in sampled indices {sampled}")
    return sampled


def build_clip_from_window(spec: WindowSpec, clip_mode: str, shuffle_seed: int) -> tuple[list[str], dict[str, Any]]:
    if clip_mode == MODE_PAIR_EXACT16:
        clip_paths = [spec.frame_paths[0]] * 8 + [spec.frame_paths[1]] * 8
        raw_indices = [spec.frame_ids[0]] * 8 + [spec.frame_ids[1]] * 8
        anchor_positions = _find_anchor_positions(raw_indices, spec.frame_ids)
        window_paths_raw = clip_paths[:]
        window_size_raw = len(window_paths_raw)
        pair_temporal_indices_raw = [int(anchor_positions[0]), int(anchor_positions[1])]
    elif clip_mode in {MODE_CTX_WAN16, MODE_CTX_WAN64, MODE_CTX_CONTIG81}:
        rgb_map = load_rgb_paths(Path(spec.sequence_dir))
        raw81_indices = _build_ctx81_indices(spec)
        if clip_mode == MODE_CTX_WAN16:
            raw_indices = _sample_ctx81_to_wan16(raw81_indices, spec.frame_ids)
        elif clip_mode == MODE_CTX_WAN64:
            raw_indices = _sample_ctx81_uniform(raw81_indices, spec.frame_ids, count=64)
        else:
            raw_indices = list(raw81_indices)
        clip_paths = [str(rgb_map[idx]) for idx in raw_indices]
        anchor_positions = _find_anchor_positions(raw_indices, spec.frame_ids)
        window_paths_raw = [str(rgb_map[idx]) for idx in raw81_indices]
        window_size_raw = len(window_paths_raw)
        pair_temporal_indices_raw = [raw81_indices.index(int(spec.frame_ids[0])), raw81_indices.index(int(spec.frame_ids[1]))]
    else:
        rgb_map = load_rgb_paths(Path(spec.sequence_dir))
        if clip_mode in {MODE_CTX_ANCHOR16, MODE_CTX_SHUFFLE16}:
            total_frames = 16
        elif clip_mode == MODE_CTX_ANCHOR32:
            total_frames = 32
        elif clip_mode == MODE_CTX_ANCHOR64:
            total_frames = 64
        else:
            raise ValueError(f"Unsupported V-JEPA clip_mode={clip_mode!r}")
        left = total_frames // 2
        right = total_frames - left
        left_ids = _contiguous_block_indices(spec.frame_ids[0], left, spec.min_frame, spec.max_frame, align="end")
        right_ids = _contiguous_block_indices(spec.frame_ids[1], right, spec.min_frame, spec.max_frame, align="start")
        raw_indices = left_ids + right_ids
        clip_paths = [str(rgb_map[idx]) for idx in raw_indices]
        anchor_positions = _find_anchor_positions(raw_indices, spec.frame_ids)
        window_paths_raw = clip_paths[:]
        window_size_raw = len(window_paths_raw)
        pair_temporal_indices_raw = [int(anchor_positions[0]), int(anchor_positions[1])]
        if clip_mode == MODE_CTX_SHUFFLE16:
            generator = random.Random(int(shuffle_seed))
            keep = {anchor_positions[0], anchor_positions[1]}
            movable = [idx for idx in range(len(clip_paths)) if idx not in keep]
            shuffled = movable[:]
            generator.shuffle(shuffled)
            shuffled_paths = list(clip_paths)
            shuffled_indices = list(raw_indices)
            for src, dst in zip(movable, shuffled):
                shuffled_paths[src] = clip_paths[dst]
                shuffled_indices[src] = raw_indices[dst]
            clip_paths = shuffled_paths
            raw_indices = shuffled_indices
    meta = {
        "window_size_raw": int(window_size_raw),
        "window_paths_raw": list(window_paths_raw),
        "resampled_num_frames": len(clip_paths),
        "resampled_frame_indices": raw_indices,
        "pair_temporal_indices_raw": pair_temporal_indices_raw,
        "pair_temporal_indices_resampled": [int(anchor_positions[0]), int(anchor_positions[1])],
    }
    return clip_paths, meta


def load_frames(paths: list[str]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            frames.append(img.convert("RGB").copy())
    return frames


def load_vjepa21_encoder(model_name: str, checkpoint_path: str, num_frames: int, device: torch.device):
    builders = {
        "vjepa2_1_vit_base_384": vjepa2_1_vit_base_384,
        "vjepa2_1_vit_large_384": vjepa2_1_vit_large_384,
        "vjepa2_1_vit_giant_384": vjepa2_1_vit_giant_384,
        "vjepa2_1_vit_gigantic_384": vjepa2_1_vit_gigantic_384,
    }
    if model_name not in builders:
        raise ValueError(f"Unsupported V-JEPA 2.1 model_name={model_name!r}")
    encoder, _ = builders[model_name](pretrained=False, num_frames=int(num_frames))
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "ema_encoder" not in payload:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain ema_encoder")
    state_dict = _clean_backbone_key(dict(payload["ema_encoder"]))
    msg = encoder.load_state_dict(state_dict, strict=True)
    logging.info("Loaded %s from %s with msg=%s", model_name, checkpoint_path, msg)
    encoder = encoder.to(device).eval()
    encoder.return_hierarchical = False
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def preprocess_frames(frames: list[Image.Image], crop_size: int) -> torch.Tensor:
    transform = vjepa2_preprocessor(crop_size=int(crop_size))
    clip = transform(frames)[0]
    return clip.unsqueeze(0)


def encode_clip(
    encoder,
    clip_tensor: torch.Tensor,
    pair_temporal_indices_resampled: tuple[int, int],
    patch_size: int,
    tubelet_size: int,
    crop_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        tokens = encoder(clip_tensor)
    if tokens.ndim != 3:
        raise ValueError(f"Expected encoder output [B,N,C], got {tuple(tokens.shape)}")
    bsz, num_tokens, dim = tokens.shape
    if bsz != 1:
        raise ValueError(f"Expected batch size 1 during precompute, got {bsz}")
    h_tokens = int(crop_size) // int(patch_size)
    w_tokens = int(crop_size) // int(patch_size)
    t_tokens = int(clip_tensor.shape[2]) // int(tubelet_size)
    expected_tokens = t_tokens * h_tokens * w_tokens
    if num_tokens != expected_tokens:
        raise ValueError(
            f"Token shape mismatch: expected {expected_tokens} = {t_tokens}*{h_tokens}*{w_tokens}, got {num_tokens}"
        )
    grid = tokens.reshape(1, t_tokens, h_tokens, w_tokens, dim)
    pair_token_indices = sorted(
        {
            int(pair_temporal_indices_resampled[0]) // int(tubelet_size),
            int(pair_temporal_indices_resampled[1]) // int(tubelet_size),
        }
    )
    selected = grid[:, pair_token_indices].reshape(1, len(pair_token_indices) * h_tokens * w_tokens, dim)
    meta = {
        "full_encoder_token_shape": [int(t_tokens), int(h_tokens), int(w_tokens), int(dim)],
        "selected_temporal_token_indices": [int(x) for x in pair_token_indices],
        "selected_feature_shape": list(selected.shape[1:]),
    }
    return selected.squeeze(0).detach().cpu().to(torch.float16), meta


def write_cache(path: Path, features: torch.Tensor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save({"features": features, "metadata": metadata}, tmp_path)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_data", default=DEFAULT_ADAPTER_DATA)
    parser.add_argument("--output_dir", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--window_mode", default=MODE_CTX_ANCHOR16, choices=WINDOW_MODES)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--crop_size", type=int, default=DEFAULT_CROP_SIZE)
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--tubelet_size", type=int, default=DEFAULT_TUBELET_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = torch.load(args.adapter_data, map_location="cpu")
    metadata = payload.get("metadata")
    splits = payload.get("splits", ["train"] * len(metadata))
    if metadata is None:
        raise ValueError(f"{args.adapter_data} does not contain metadata")
    selected_indices = [idx for idx in range(len(metadata)) if args.split is None or splits[idx] == args.split]
    if args.max_samples > 0:
        selected_indices = selected_indices[: int(args.max_samples)]
    if not selected_indices:
        raise ValueError("No SCRREAM samples matched the requested split/max_samples filter")

    if args.window_mode == MODE_CTX_ANCHOR32:
        num_frames = 32
    elif args.window_mode == MODE_CTX_ANCHOR64:
        num_frames = 64
    elif args.window_mode == MODE_CTX_CONTIG81:
        num_frames = 81
    elif args.window_mode == MODE_CTX_WAN64:
        num_frames = 64
    else:
        num_frames = 16
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    encoder = load_vjepa21_encoder(args.model_name, args.checkpoint_path, num_frames=num_frames, device=device)
    cache_root = Path(args.output_dir)

    for item_idx, sample_idx in enumerate(selected_indices):
        meta = metadata[sample_idx]
        spec = build_window_spec(meta)
        clip_paths, clip_meta = build_clip_from_window(
            spec,
            clip_mode=args.window_mode,
            shuffle_seed=int(args.seed + sample_idx * 9973),
        )
        frames = load_frames(clip_paths)
        clip_tensor = preprocess_frames(frames, crop_size=int(args.crop_size)).to(device)
        features, feature_meta = encode_clip(
            encoder,
            clip_tensor,
            tuple(int(x) for x in clip_meta["pair_temporal_indices_resampled"]),
            patch_size=int(args.patch_size),
            tubelet_size=int(args.tubelet_size),
            crop_size=int(args.crop_size),
        )
        metadata_out = {
            "feature_backbone": "vjepa21_encoder",
            "model_name": args.model_name,
            "checkpoint_path": str(args.checkpoint_path),
            "sample_id": spec.sample_id,
            "scene_id": spec.scene_id,
            "sequence_id": spec.sequence_id,
            "window_mode": args.window_mode,
            "window_size_raw": int(clip_meta["window_size_raw"]),
            "window_paths_raw": list(clip_meta["window_paths_raw"]),
            "resampled_num_frames": int(clip_meta["resampled_num_frames"]),
            "resampled_frame_indices": list(clip_meta["resampled_frame_indices"]),
            "pair_frame_ids": list(spec.frame_ids),
            "pair_frame_paths": list(spec.frame_paths),
            "pair_temporal_indices_raw": list(clip_meta["pair_temporal_indices_raw"]),
            "pair_temporal_indices_resampled": list(clip_meta["pair_temporal_indices_resampled"]),
            "tubelet_size": int(args.tubelet_size),
            "crop_size": int(args.crop_size),
            "patch_size": int(args.patch_size),
            "num_frames": int(num_frames),
            **feature_meta,
        }
        out_path = cache_path(cache_root, args.window_mode, spec.sample_id)
        write_cache(out_path, features, metadata_out)
        logging.info(
            "[%d/%d] wrote %s shape=%s selected_temporal_token_indices=%s",
            item_idx + 1,
            len(selected_indices),
            out_path,
            tuple(features.shape),
            metadata_out["selected_temporal_token_indices"],
        )

    manifest = {
        "feature_backbone": "vjepa21_encoder",
        "window_mode": args.window_mode,
        "model_name": args.model_name,
        "checkpoint_path": str(args.checkpoint_path),
        "num_frames": int(num_frames),
        "tubelet_size": int(args.tubelet_size),
        "patch_size": int(args.patch_size),
        "crop_size": int(args.crop_size),
        "sample_count": len(selected_indices),
    }
    manifest_path = Path(args.output_dir) / args.window_mode / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
