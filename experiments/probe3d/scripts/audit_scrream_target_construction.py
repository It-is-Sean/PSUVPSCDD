from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SCRREAM adapter target construction metadata and optional target distances.")
    parser.add_argument("--adapter_data", required=True)
    parser.add_argument("--baseline_adapter_data", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--max_compare_samples", type=int, default=8)
    parser.add_argument("--max_compare_points", type=int, default=4096)
    return parser.parse_args()


def summarize_metadata(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    keys = [
        "num_points_after_source_voxel",
        "num_dense_points_after_voxel",
        "num_mesh_points_after_voxel",
        "num_points_after_frustum_crop",
        "num_depth_points_after_voxel",
        "num_mesh_points_after_frustum_crop_pre_mix",
        "num_mesh_clean_visible",
        "num_mesh_uncertain_visible",
        "num_mesh_invisible_complete",
        "num_mesh_conflict",
        "sampled_visible_clean",
        "sampled_invisible_complete",
        "sampled_uncertain_visible",
        "sampled_fallback",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        values = [item.get(key) for item in metadata if item.get(key) is not None]
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        out[key] = {
            "mean": float(tensor.mean().item()),
            "median": float(tensor.median().item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }
    return out


def chamfer_l2(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    import torch

    dists = torch.cdist(a.float(), b.float(), p=2).pow(2)
    a_to_b = dists.min(dim=1).values
    b_to_a = dists.min(dim=0).values
    return {
        "a_to_b": float(a_to_b.mean().item()),
        "b_to_a": float(b_to_a.mean().item()),
        "symmetric": float((a_to_b.mean() + b_to_a.mean()).item()),
    }


def maybe_compare_targets(payload: dict[str, Any], baseline_path: str | None, max_samples: int, max_points: int) -> dict[str, Any]:
    import torch

    if not baseline_path:
        return {}
    baseline = torch.load(Path(baseline_path).expanduser().resolve(), map_location="cpu")
    scene_ids = payload["scene_ids"]
    baseline_ids = baseline["scene_ids"]
    baseline_index = {scene_id: idx for idx, scene_id in enumerate(baseline_ids)}
    target_points = payload["target_points"].float()
    baseline_points = baseline["target_points"].float()
    rows = []
    for idx, scene_id in enumerate(scene_ids[:max_samples]):
        if scene_id not in baseline_index:
            continue
        base_idx = baseline_index[scene_id]
        a = target_points[idx, :max_points]
        b = baseline_points[base_idx, :max_points]
        rows.append({"scene_id": scene_id, **chamfer_l2(a, b)})
    if not rows:
        return {"target_vs_baseline": {"sample_count": 0, "rows": []}}
    symmetric = torch.tensor([row["symmetric"] for row in rows], dtype=torch.float32)
    return {
        "target_vs_baseline": {
            "sample_count": len(rows),
            "mean_symmetric": float(symmetric.mean().item()),
            "max_symmetric": float(symmetric.max().item()),
            "rows": rows,
        }
    }


def main() -> None:
    args = parse_args()
    import torch

    adapter_path = Path(args.adapter_data).expanduser().resolve()
    payload = torch.load(adapter_path, map_location="cpu")
    targets = payload["target_points"].float()
    metadata = payload.get("metadata", [{} for _ in payload["scene_ids"]])
    meta = payload.get("meta", {})
    target_sources = {}
    for item in metadata:
        source = str(item.get("target_source", "unknown"))
        target_sources[source] = target_sources.get(source, 0) + 1

    report: dict[str, Any] = {
        "adapter_data": str(adapter_path),
        "num_samples": int(targets.shape[0]),
        "target_points_shape": list(targets.shape),
        "target_sources": target_sources,
        "adapter_meta": meta,
        "metadata_summary": summarize_metadata(metadata),
        "target_bounds": {
            "min": targets.amin(dim=(0, 1)).tolist(),
            "max": targets.amax(dim=(0, 1)).tolist(),
            "mean_norm": float(targets.norm(dim=-1).mean().item()),
            "median_norm": float(targets.norm(dim=-1).median().item()),
        },
    }
    report.update(
        maybe_compare_targets(
            payload,
            args.baseline_adapter_data,
            max_samples=args.max_compare_samples,
            max_points=args.max_compare_points,
        )
    )

    text = json.dumps(report, indent=2)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
