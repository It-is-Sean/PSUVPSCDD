from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from nova3r.probe.backbones.vggt_extractor import VGGTFeatureExtractor
from nova3r.probe.bridges import ZeroShotSceneTokenBridge
from nova3r.probe.canonical_decoder import FrozenCanonicalPointDecoder


DEFAULT_IMAGES = [
    "demo/examples/scrream_scene09_200.png",
    "demo/examples/scrream_scene09_275.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a VGGT -> NOVA3R decoder sanity pass.")
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES, help="Input image paths relative to repo root or absolute paths.")
    parser.add_argument("--nova3r-ckpt", default="checkpoints/scene_ae/checkpoint-last.pth")
    parser.add_argument("--vggt-repo", default="third_party/vggt", help="Path to the vendored/cloned VGGT repo (relative to repo root or absolute).")
    parser.add_argument("--vggt-model-id", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-weights", default=None, help="Optional local VGGT weights path (e.g. /data1/.../model.pt).")
    parser.add_argument("--layer", default="final", help="VGGT layer selector: early|mid|final|<int>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-queries", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/reports/vggt_to_nova3r")
    return parser.parse_args()


def _resolve_paths(repo_root: Path, paths: Sequence[str]) -> list[str]:
    resolved = []
    for path in paths:
        p = Path(path)
        if not p.is_absolute():
            p = repo_root / p
        resolved.append(str(p.resolve()))
    return resolved


def _write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{float(x)} {float(y)} {float(z)}\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    image_paths = _resolve_paths(repo_root, args.images)
    output_root = (repo_root / args.output_dir).resolve()
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.nova3r_ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path

    vggt_repo = Path(args.vggt_repo)
    if not vggt_repo.is_absolute():
        vggt_repo = (repo_root / args.vggt_repo).resolve()

    layer = int(args.layer) if str(args.layer).isdigit() else args.layer

    extractor = VGGTFeatureExtractor(
        device=args.device,
        model_id=args.vggt_model_id,
        repo_path=vggt_repo,
        weights_path=args.vggt_weights,
        layer=layer,
    )
    representation = extractor.extract(image_paths)

    decoder = FrozenCanonicalPointDecoder(ckpt_path=ckpt_path, device=args.device)
    bridge = ZeroShotSceneTokenBridge(
        target_tokens=decoder.num_scene_tokens,
        target_dim=decoder.token_dim,
    ).to(decoder.device)

    source_tokens = representation.tokens.to(decoder.device)
    scene_tokens = bridge(source_tokens)
    points = decoder.sample_from_tokens(
        tokens=scene_tokens,
        num_queries=args.num_queries,
        seed=args.seed,
        num_views=len(image_paths),
    )

    points_np = points[0].detach().cpu().numpy()
    np.save(run_dir / "pointcloud.npy", points_np)
    _write_ascii_ply(run_dir / "pointcloud.ply", points_np)

    summary = {
        "images": image_paths,
        "vggt_repo": str(vggt_repo),
        "vggt_model_id": args.vggt_model_id,
        "layer": args.layer,
        "representation_meta": representation.meta,
        "source_token_shape": list(source_tokens.shape),
        "bridged_token_shape": list(scene_tokens.shape),
        "decoder": decoder.extra_repr(),
        "num_queries": args.num_queries,
        "seed": args.seed,
        "output_ply": str((run_dir / "pointcloud.ply").resolve()),
        "output_npy": str((run_dir / "pointcloud.npy").resolve()),
        "point_stats": {
            "min": points_np.min(axis=0).tolist(),
            "max": points_np.max(axis=0).tolist(),
            "mean": points_np.mean(axis=0).tolist(),
            "std": points_np.std(axis=0).tolist(),
        },
        "note": "This is a training-free sanity bridge (adaptive pooling), not the final learned adapter from the proposal.",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
