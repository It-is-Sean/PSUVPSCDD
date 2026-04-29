#!/usr/bin/env python3
"""Robust point-cloud diagnostics for autoresearch_probe artifacts.

This intentionally complements, rather than replaces, the historical symmetric
Chamfer rows in results.tsv. The goal is to avoid over-interpreting one noisy CD
number by logging precision/recall-style geometry diagnostics from saved PLYs.

Typical use:

  python experiments/probe3d/autoresearch_probe/robust_ply_metrics.py \
    --pred path/to/pred.ply --gt path/to/target.ply \
    --out path/to/robust_metrics.json

or scan a directory containing *_pred.ply / *_target.ply pairs:

  python experiments/probe3d/autoresearch_probe/robust_ply_metrics.py \
    --dir experiments/probe3d/result/autoresearch_probe/<run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def read_ply_xyz(path: str | Path) -> torch.Tensor:
    path = Path(path)
    pts: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        for line in handle:
            vals = line.split()
            if len(vals) < 3:
                continue
            try:
                pts.append([float(vals[0]), float(vals[1]), float(vals[2])])
            except ValueError:
                continue
    if not pts:
        raise ValueError(f"No xyz points found in {path}")
    x = torch.tensor(pts, dtype=torch.float32)
    finite = torch.isfinite(x).all(dim=1)
    nonzero = x.abs().sum(dim=1) > 0
    return x[finite & nonzero]


def deterministic_subsample(x: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    if max_points <= 0 or x.shape[0] <= max_points:
        return x
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(x.shape[0], generator=gen)[:max_points]
    return x[idx]


def directed_nn_stats(src: torch.Tensor, dst: torch.Tensor, chunk: int = 4096) -> dict:
    mins = []
    for start in range(0, src.shape[0], chunk):
        d = torch.cdist(src[start : start + chunk][None], dst[None])[0]
        mins.append(d.min(dim=1).values.cpu())
    dist = torch.cat(mins, dim=0)
    sq = dist.square()
    out = {
        "mean": float(dist.mean()),
        "median": float(dist.median()),
        "p75": float(torch.quantile(dist, 0.75)),
        "p90": float(torch.quantile(dist, 0.90)),
        "p95": float(torch.quantile(dist, 0.95)),
        "p99": float(torch.quantile(dist, 0.99)),
        "rmse": float(torch.sqrt(sq.mean())),
        "mse": float(sq.mean()),
    }
    for q in [0.90, 0.95, 0.99]:
        k = max(1, int(round(q * sq.numel())))
        trimmed = torch.topk(sq, k, largest=False).values
        out[f"trimmed_mse_{int(q*100)}"] = float(trimmed.mean())
    return out


def fscore(pred_to_gt: torch.Tensor, gt_to_pred: torch.Tensor, thresholds: Iterable[float]) -> dict:
    out = {}
    for th in thresholds:
        precision = float((pred_to_gt <= th).float().mean())
        recall = float((gt_to_pred <= th).float().mean())
        f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        key = f"tau_{th:g}"
        out[key] = {"precision": precision, "recall": recall, "fscore": f}
    return out


def nearest_dist(src: torch.Tensor, dst: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    mins = []
    for start in range(0, src.shape[0], chunk):
        d = torch.cdist(src[start : start + chunk][None], dst[None])[0]
        mins.append(d.min(dim=1).values.cpu())
    return torch.cat(mins, dim=0)


def metrics_for_pair(pred_path: str | Path, gt_path: str | Path, max_points: int, seed: int) -> dict:
    pred = deterministic_subsample(read_ply_xyz(pred_path), max_points, seed)
    gt = deterministic_subsample(read_ply_xyz(gt_path), max_points, seed + 1)
    p2g_dist = nearest_dist(pred, gt)
    g2p_dist = nearest_dist(gt, pred)
    pred_to_gt = directed_nn_stats(pred, gt)
    gt_to_pred = directed_nn_stats(gt, pred)
    symmetric_cd_l2 = pred_to_gt["mse"] + gt_to_pred["mse"]
    trimmed_cd_l2_95 = pred_to_gt["trimmed_mse_95"] + gt_to_pred["trimmed_mse_95"]
    trimmed_cd_l2_99 = pred_to_gt["trimmed_mse_99"] + gt_to_pred["trimmed_mse_99"]
    return {
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "points": {"pred": int(pred.shape[0]), "gt": int(gt.shape[0]), "max_points": int(max_points)},
        "symmetric_cd_l2": float(symmetric_cd_l2),
        "trimmed_cd_l2_95": float(trimmed_cd_l2_95),
        "trimmed_cd_l2_99": float(trimmed_cd_l2_99),
        "pred_to_gt_precision_side": pred_to_gt,
        "gt_to_pred_recall_side": gt_to_pred,
        "fscore": fscore(p2g_dist, g2p_dist, thresholds=[0.02, 0.05, 0.10, 0.20]),
    }


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for gt in sorted(root.rglob("*target*.ply")):
        candidates = []
        stem = gt.name.replace("target", "pred")
        candidates.append(gt.with_name(stem))
        candidates.extend(sorted(gt.parent.glob("*pred*.ply")))
        for pred in candidates:
            if pred.exists() and pred != gt:
                pairs.append((pred, gt))
                break
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for pred, gt in pairs:
        key = (pred.resolve(), gt.resolve())
        if key not in seen:
            seen.add(key)
            unique.append((pred, gt))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", help="Prediction PLY path")
    parser.add_argument("--gt", help="Target/GT PLY path")
    parser.add_argument("--dir", help="Directory to scan for pred/target PLY pairs")
    parser.add_argument("--out", help="Output JSON path")
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.dir:
        root = Path(args.dir)
        pairs = find_pairs(root)
        if not pairs:
            raise SystemExit(f"No pred/target PLY pairs found under {root}")
        results = [metrics_for_pair(pred, gt, args.max_points, args.seed + i * 1000) for i, (pred, gt) in enumerate(pairs)]
        payload = {"root": str(root), "pair_count": len(results), "pairs": results}
        out = Path(args.out) if args.out else root / "robust_ply_metrics.json"
    else:
        if not args.pred or not args.gt:
            raise SystemExit("Provide either --dir or both --pred and --gt")
        payload = metrics_for_pair(args.pred, args.gt, args.max_points, args.seed)
        out = Path(args.out) if args.out else Path(args.pred).with_suffix(".robust_metrics.json")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)
    if "symmetric_cd_l2" in payload:
        print(json.dumps({k: payload[k] for k in ["symmetric_cd_l2", "trimmed_cd_l2_95", "trimmed_cd_l2_99", "fscore"]}, indent=2))
    else:
        for item in payload["pairs"]:
            print(Path(item["pred_path"]).name, "sym", f"{item['symmetric_cd_l2']:.6f}", "trim95", f"{item['trimmed_cd_l2_95']:.6f}")


if __name__ == "__main__":
    main()
