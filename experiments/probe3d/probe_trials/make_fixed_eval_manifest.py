#!/usr/bin/env python3
"""Create a fixed ScanNet eval manifest for visual-first VGGT->NOVA probes.

This is intentionally metadata-only: it does not instantiate models, allocate GPU,
or load RGB/depth arrays.  The manifest freezes scene/start-frame choices so later
oracle/adapter evaluation can report sample-stable median/p75/failure-rate metrics
instead of two-sample means.

For the current corrected ScanNet protocol we use max_interval=1.  With processed
ScanNet frame_skip=20, that means adjacent processed frames (~20 raw-frame spacing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _split_root(root: Path, split: str) -> Path:
    preferred = root / f"scans_{split}"
    if preferred.is_dir():
        return preferred
    fallback = root / ("scans_train" if split == "train" else "scans_test")
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"Could not find scans_{split} or fallback split under {root}")


def _even_offsets(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    k = min(k, n)
    if k == 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def build_manifest(
    data_root: Path,
    split: str,
    num_views: int,
    max_interval: int,
    scene_count: int,
    samples_per_scene: int,
    start_offset: int,
) -> dict[str, Any]:
    if max_interval != 1:
        raise ValueError(
            "This lightweight manifest freezes explicit labels only for max_interval=1. "
            "For wider/random intervals, extend the script to call the dataset RNG path."
        )
    split_dir = _split_root(data_root, split)
    scenes = sorted(p.name for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("scene"))
    rows: list[dict[str, Any]] = []
    global_image_offset = 0
    dataset_index_offset = 0
    used_scenes = 0

    for scene in scenes:
        scene_dir = split_dir / scene
        meta_path = scene_dir / "new_scene_metadata.npz"
        if not meta_path.is_file():
            continue
        with np.load(meta_path, allow_pickle=True) as data:
            images = [str(x) for x in data["images"].tolist()]
        num_imgs = len(images)
        cut_off = int(num_views)
        num_starts = max(0, num_imgs - cut_off + 1)
        if num_starts <= start_offset:
            global_image_offset += num_imgs
            dataset_index_offset += num_starts
            continue
        if used_scenes >= scene_count:
            break
        local_candidates = list(range(start_offset, num_starts))
        local_positions = _even_offsets(len(local_candidates), samples_per_scene)
        for pos in local_positions:
            local_start = local_candidates[pos]
            labels = [f"{scene}_{images[local_start + view_i]}" for view_i in range(num_views)]
            global_view_ids = [global_image_offset + local_start + view_i for view_i in range(num_views)]
            dataset_index = dataset_index_offset + local_start
            rows.append(
                {
                    "manifest_row": len(rows),
                    "dataset_index": dataset_index,
                    "scene": scene,
                    "local_start": local_start,
                    "global_start_id": global_image_offset + local_start,
                    "global_view_ids": global_view_ids,
                    "labels": labels,
                    "processed_gap": 1 if num_views > 1 else 0,
                    "raw_gap_equiv_note": "~20 raw frames if source preprocessing used frame_skip=20",
                }
            )
        used_scenes += 1
        global_image_offset += num_imgs
        dataset_index_offset += num_starts

    return {
        "schema": "psuvpsc3dd.scannet_fixed_eval_manifest.v1",
        "purpose": "fixed visual-first eval sample list for robust oracle/adapter comparison",
        "data_root": str(data_root),
        "split": split,
        "num_views": int(num_views),
        "max_interval": int(max_interval),
        "scene_count_requested": int(scene_count),
        "samples_per_scene_requested": int(samples_per_scene),
        "start_offset": int(start_offset),
        "sample_count": len(rows),
        "selection": "first N sorted scenes, evenly spaced legal start frames per scene; explicit labels assume max_interval=1",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--max_interval", type=int, default=1)
    parser.add_argument("--scene_count", type=int, default=10)
    parser.add_argument("--samples_per_scene", type=int, default=3)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(
        data_root=args.data_root,
        split=args.split,
        num_views=args.num_views,
        max_interval=args.max_interval,
        scene_count=args.scene_count,
        samples_per_scene=args.samples_per_scene,
        start_offset=args.start_offset,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest['sample_count']} samples to {args.out}")
    if manifest["sample_count"]:
        print("first:", manifest["rows"][0])
        print("last:", manifest["rows"][-1])


if __name__ == "__main__":
    main()
