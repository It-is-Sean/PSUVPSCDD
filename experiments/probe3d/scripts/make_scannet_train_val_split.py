from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a train/val/test ScanNet split root using symlinks.")
    parser.add_argument("--source_root", default="/data1/jcd_data/scannet_processed_large")
    parser.add_argument("--output_root", default="/data1/jcd_data/scannet_processed_large_split_seed17")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_count", type=int, default=None, help="Override val scene count; otherwise derived from val_ratio.")
    parser.add_argument("--link_mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--force", action="store_true", help="Clear existing split dirs under output_root before recreating them.")
    return parser.parse_args()


def list_scenes(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("scene"))


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
    if mode == "symlink":
        os.symlink(src, dst, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    train_src = source_root / "scans_train"
    test_src = source_root / "scans_test"
    if not train_src.is_dir():
        raise FileNotFoundError(f"Missing source train dir: {train_src}")
    if not test_src.is_dir():
        raise FileNotFoundError(f"Missing source test dir: {test_src}")

    train_scenes = list_scenes(train_src)
    test_scenes = list_scenes(test_src)
    if len(train_scenes) < 2:
        raise ValueError(f"Need at least 2 training scenes, got {len(train_scenes)}")

    shuffled = list(train_scenes)
    random.Random(args.seed).shuffle(shuffled)
    val_count = args.val_count if args.val_count is not None else max(1, int(round(len(shuffled) * args.val_ratio)))
    val_count = min(max(1, val_count), len(shuffled) - 1)
    val_scenes = sorted(shuffled[:val_count])
    train_keep = sorted(shuffled[val_count:])

    scans_train_out = output_root / "scans_train"
    scans_val_out = output_root / "scans_val"
    scans_test_out = output_root / "scans_test"
    for out_dir in (scans_train_out, scans_val_out, scans_test_out):
        ensure_clean_dir(out_dir, args.force)

    for scene in train_keep:
        materialize(train_src / scene, scans_train_out / scene, args.link_mode)
    for scene in val_scenes:
        materialize(train_src / scene, scans_val_out / scene, args.link_mode)
    for scene in test_scenes:
        materialize(test_src / scene, scans_test_out / scene, args.link_mode)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "val_count": val_count,
        "link_mode": args.link_mode,
        "scene_counts": {
            "train": len(train_keep),
            "val": len(val_scenes),
            "test": len(test_scenes),
        },
        "scene_splits": {
            "train": train_keep,
            "val": val_scenes,
            "test": test_scenes,
        },
    }
    (output_root / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
