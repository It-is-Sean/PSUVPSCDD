from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE3D_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, PROBE3D_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SCRREAM adapter .pt inputs before Slurm training.")
    parser.add_argument("--adapter_data", required=True)
    parser.add_argument("--nova_ckpt", default=None)
    parser.add_argument("--num_queries", type=int, default=10000)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_data).expanduser().resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)

    payload = torch.load(adapter_path, map_location="cpu")
    scene_ids = payload["scene_ids"]
    targets = payload["target_points"].float()
    splits = payload.get("splits", ["train"] * len(scene_ids))
    metadata = payload.get("metadata", [{} for _ in scene_ids])
    meta = payload.get("meta", {})

    if targets.ndim != 3 or targets.shape[-1] != 3:
        raise ValueError(f"Expected target_points [N,P,3], got {tuple(targets.shape)}")
    if targets.shape[1] < args.num_queries:
        raise ValueError(f"target_points has only {targets.shape[1]} points, requested {args.num_queries}")
    if not torch.isfinite(targets).all():
        raise ValueError("target_points contains non-finite values")
    if len(scene_ids) != targets.shape[0] or len(splits) != targets.shape[0] or len(metadata) != targets.shape[0]:
        raise ValueError("scene_ids / splits / metadata lengths do not match target_points")

    split_counts = {split: splits.count(split) for split in sorted(set(splits))}
    if args.split not in split_counts:
        raise ValueError(f"Requested split={args.split!r} not found; available={split_counts}")

    first_meta = metadata[0]
    missing_frames = []
    for item in metadata[: min(8, len(metadata))]:
        for frame_path in item.get("frame_paths", []):
            if not Path(frame_path).is_file():
                missing_frames.append(frame_path)
    if missing_frames:
        raise FileNotFoundError(f"Missing frame_paths in metadata: {missing_frames[:4]}")

    decoder_meta = {}
    if args.nova_ckpt:
        from vggt_nova_adapter_common_raw import build_decoder, resolve_device

        device = resolve_device("cpu")
        _decoder, decoder_meta, _cfg = build_decoder(device, args.nova_ckpt)

    summary = {
        "adapter_data": str(adapter_path),
        "num_samples": int(targets.shape[0]),
        "target_points_shape": list(targets.shape),
        "split_counts": split_counts,
        "first_scene_id": scene_ids[0],
        "first_sample_metadata": {
            "sample_id": first_meta.get("sample_id"),
            "scene_id": first_meta.get("scene_id"),
            "sequence_id": first_meta.get("sequence_id"),
            "frame_ids": first_meta.get("frame_ids"),
            "target_source": first_meta.get("target_source"),
            "num_points_after_frustum_crop": first_meta.get("num_points_after_frustum_crop"),
        },
        "target_raw_bounds": {
            "min": targets.amin(dim=(0, 1)).tolist(),
            "max": targets.amax(dim=(0, 1)).tolist(),
            "mean_norm": float(targets.norm(dim=-1).mean().item()),
            "median_norm": float(targets.norm(dim=-1).median().item()),
        },
        "adapter_meta": meta,
        "decoder_meta": decoder_meta,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
