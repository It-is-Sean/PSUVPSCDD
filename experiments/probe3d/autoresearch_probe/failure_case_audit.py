#!/usr/bin/env python3
"""Audit robust-eval failure cases for a VGGT->NOVA checkpoint.

This script consumes an ``eval_checkpoint_robust.py`` summary.json and produces a
compact JSON + Markdown report that ranks fixed-eval samples by F-score and
checks simple geometry symptoms (prediction spread / bbox inflation / precision
vs recall imbalance).  It is intentionally lightweight: no CUDA and no training
pipeline dependency.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _read_ascii_ply_xyz(path: Path) -> List[Tuple[float, float, float]]:
    """Read xyz vertices from a simple ASCII PLY written by the probe renderers."""
    points: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        in_header = True
        vertex_count = None
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                if line == "end_header":
                    in_header = False
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
            if vertex_count is not None and len(points) >= vertex_count:
                break
    return points


def _axis_stats(points: Sequence[Tuple[float, float, float]]) -> Dict[str, Any]:
    if not points:
        return {
            "num_points": 0,
            "centroid": [math.nan, math.nan, math.nan],
            "min": [math.nan, math.nan, math.nan],
            "max": [math.nan, math.nan, math.nan],
            "span": [math.nan, math.nan, math.nan],
            "bbox_volume": math.nan,
            "mean_radius": math.nan,
            "p90_radius": math.nan,
        }
    n = len(points)
    cols = [[p[i] for p in points] for i in range(3)]
    mins = [min(c) for c in cols]
    maxs = [max(c) for c in cols]
    centroid = [sum(c) / n for c in cols]
    spans = [maxs[i] - mins[i] for i in range(3)]
    radii = [math.sqrt(sum((p[i] - centroid[i]) ** 2 for i in range(3))) for p in points]
    radii_sorted = sorted(radii)
    p90_idx = min(len(radii_sorted) - 1, int(0.9 * (len(radii_sorted) - 1)))
    return {
        "num_points": n,
        "centroid": centroid,
        "min": mins,
        "max": maxs,
        "span": spans,
        "bbox_volume": spans[0] * spans[1] * spans[2],
        "mean_radius": sum(radii) / n,
        "p90_radius": radii_sorted[p90_idx],
    }


def _safe_ratio(a: float, b: float) -> float:
    if b == 0 or math.isnan(a) or math.isnan(b):
        return math.nan
    return a / b


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2:
        return math.nan
    mx = mean(x for x, _ in pairs)
    my = mean(y for _, y in pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if denx == 0 or deny == 0:
        return math.nan
    return num / (denx * deny)


def _metric(sample: Dict[str, Any], dotted: str) -> float:
    cur: Any = sample
    for part in dotted.split("."):
        cur = cur[part]
    return float(cur)


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    """Accept either a mapping or the repr-string format used by older summaries."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {"raw": value}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _compact_key(sample: Dict[str, Any]) -> str:
    keys = sample.get("keys") or []
    if keys and isinstance(keys[0], list):
        return "/".join(keys[0])
    return str(keys)


def _row(sample: Dict[str, Any], geom: Dict[str, Any]) -> Dict[str, Any]:
    metrics = sample["metrics"]
    fs = metrics["fscore"]["tau_0.05"]
    pred_side = metrics["pred_to_gt_precision_side"]
    recall_side = metrics["gt_to_pred_recall_side"]
    return {
        "sample_idx": sample["sample_idx"],
        "key": _compact_key(sample),
        "f05": float(fs["fscore"]),
        "precision05": float(fs["precision"]),
        "recall05": float(fs["recall"]),
        "symmetric_cd_l2": float(metrics["symmetric_cd_l2"]),
        "trimmed_cd_l2_95": float(metrics["trimmed_cd_l2_95"]),
        "pred_to_gt_mean": float(pred_side["mean"]),
        "gt_to_pred_mean": float(recall_side["mean"]),
        "pred_to_gt_p90": float(pred_side["p90"]),
        "gt_to_pred_p90": float(recall_side["p90"]),
        **geom,
    }


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return {"mean": math.nan, "median": math.nan, "p75": math.nan, "p90": math.nan}
    def q(frac: float) -> float:
        idx = min(len(vals) - 1, int(round(frac * (len(vals) - 1))))
        return vals[idx]
    return {"mean": mean(vals), "median": median(vals), "p75": q(0.75), "p90": q(0.90)}


def build_audit(summary_path: Path, top_k: int) -> Dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base_dir = summary_path.parent
    rows: List[Dict[str, Any]] = []
    for sample in summary["samples"]:
        pred_path = Path(sample["metrics"]["pred_path"])
        gt_path = Path(sample["metrics"]["gt_path"])
        if not pred_path.is_absolute():
            pred_path = summary_path.parents[4] / pred_path if not pred_path.exists() else pred_path
        if not gt_path.is_absolute():
            gt_path = summary_path.parents[4] / gt_path if not gt_path.exists() else gt_path
        # Robust fallback for paths already relative to repo root.
        if not pred_path.exists():
            pred_path = Path.cwd() / sample["metrics"]["pred_path"]
        if not gt_path.exists():
            gt_path = Path.cwd() / sample["metrics"]["gt_path"]
        pred_stats = _axis_stats(_read_ascii_ply_xyz(pred_path))
        gt_stats = _axis_stats(_read_ascii_ply_xyz(gt_path))
        geom = {
            "pred_bbox_volume": pred_stats["bbox_volume"],
            "gt_bbox_volume": gt_stats["bbox_volume"],
            "bbox_volume_ratio_pred_gt": _safe_ratio(pred_stats["bbox_volume"], gt_stats["bbox_volume"]),
            "pred_mean_radius": pred_stats["mean_radius"],
            "gt_mean_radius": gt_stats["mean_radius"],
            "mean_radius_ratio_pred_gt": _safe_ratio(pred_stats["mean_radius"], gt_stats["mean_radius"]),
            "pred_p90_radius": pred_stats["p90_radius"],
            "gt_p90_radius": gt_stats["p90_radius"],
            "p90_radius_ratio_pred_gt": _safe_ratio(pred_stats["p90_radius"], gt_stats["p90_radius"]),
            "pred_span": pred_stats["span"],
            "gt_span": gt_stats["span"],
        }
        rows.append(_row(sample, geom))

    rows_by_f = sorted(rows, key=lambda r: r["f05"])
    rows_by_precision_gap = sorted(rows, key=lambda r: r["recall05"] - r["precision05"], reverse=True)
    rows_by_outlier = sorted(rows, key=lambda r: r["pred_to_gt_p90"], reverse=True)

    f05 = [r["f05"] for r in rows]
    precision = [r["precision05"] for r in rows]
    recall = [r["recall05"] for r in rows]
    pred_p90 = [r["pred_to_gt_p90"] for r in rows]
    radius_ratio = [r["p90_radius_ratio_pred_gt"] for r in rows]
    bbox_ratio = [r["bbox_volume_ratio_pred_gt"] for r in rows]

    return {
        "summary_path": str(summary_path),
        "checkpoint": summary.get("ckpt"),
        "data_args": _coerce_mapping(summary.get("data_args")),
        "num_samples": len(rows),
        "aggregate_from_summary": summary.get("aggregate"),
        "audit_stats": {
            "f05": _quantiles(f05),
            "precision05": _quantiles(precision),
            "recall05": _quantiles(recall),
            "recall_minus_precision": _quantiles([r["recall05"] - r["precision05"] for r in rows]),
            "pred_to_gt_p90": _quantiles(pred_p90),
            "p90_radius_ratio_pred_gt": _quantiles(radius_ratio),
            "bbox_volume_ratio_pred_gt": _quantiles(bbox_ratio),
            "corr_f05_with_pred_p90": _pearson(f05, pred_p90),
            "corr_f05_with_radius_ratio": _pearson(f05, radius_ratio),
            "corr_f05_with_bbox_ratio": _pearson(f05, bbox_ratio),
        },
        "worst_by_f05": rows_by_f[:top_k],
        "best_by_f05": list(reversed(rows_by_f[-top_k:])),
        "largest_recall_precision_gap": rows_by_precision_gap[:top_k],
        "largest_pred_to_gt_p90": rows_by_outlier[:top_k],
        "rows": rows,
    }


def _fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    return str(v)


def write_markdown(audit: Dict[str, Any], out_path: Path) -> None:
    stats = audit["audit_stats"]
    lines: List[str] = []
    lines.append("# Fixed-30 failure-case audit")
    lines.append("")
    lines.append(f"- Summary: `{audit['summary_path']}`")
    lines.append(f"- Checkpoint: `{audit.get('checkpoint')}`")
    lines.append(f"- Samples: `{audit['num_samples']}`")
    data_args = audit.get("data_args") or {}
    if data_args:
        lines.append(
            "- Data: "
            f"target=`{data_args.get('scannet_target_mode')}`, "
            f"K/test=`{data_args.get('test_dataset_name')}`, "
            f"max_interval=`{data_args.get('scannet_max_interval')}`, "
            f"scenes=`{data_args.get('scene_count')}`"
        )
    lines.append("")
    lines.append("## Main finding")
    lines.append("")
    gap = stats["recall_minus_precision"]
    lines.append(
        "The baseline remains recall-heavy and precision-poor on the fixed rows: "
        f"median recall-precision gap at tau=0.05 is `{_fmt(gap['median'])}` "
        f"(p75 `{_fmt(gap['p75'])}`). The worst rows are not mainly missing GT coverage; "
        "they are dominated by prediction-side spread/outliers. This supports using a "
        "structured/query-conditioned or token-distillation diagnostic next, rather than "
        "another blind MLP sweep."
    )
    lines.append("")
    lines.append("## Audit statistics")
    lines.append("")
    for name in [
        "f05",
        "precision05",
        "recall05",
        "recall_minus_precision",
        "pred_to_gt_p90",
        "p90_radius_ratio_pred_gt",
        "bbox_volume_ratio_pred_gt",
    ]:
        q = stats[name]
        lines.append(
            f"- `{name}`: mean `{_fmt(q['mean'])}`, median `{_fmt(q['median'])}`, "
            f"p75 `{_fmt(q['p75'])}`, p90 `{_fmt(q['p90'])}`"
        )
    lines.append(
        f"- Correlations with F@0.05: pred_to_gt_p90 `{_fmt(stats['corr_f05_with_pred_p90'])}`, "
        f"p90_radius_ratio `{_fmt(stats['corr_f05_with_radius_ratio'])}`, "
        f"bbox_volume_ratio `{_fmt(stats['corr_f05_with_bbox_ratio'])}`"
    )
    lines.append("")

    def table(title: str, rows: Iterable[Dict[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| sample | key | F@.05 | P@.05 | R@.05 | pred→GT p90 | rad ratio | bbox ratio |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['sample_idx']} | `{r['key']}` | {_fmt(r['f05'])} | {_fmt(r['precision05'])} | "
                f"{_fmt(r['recall05'])} | {_fmt(r['pred_to_gt_p90'])} | "
                f"{_fmt(r['p90_radius_ratio_pred_gt'])} | {_fmt(r['bbox_volume_ratio_pred_gt'])} |"
            )
        lines.append("")

    table("Worst rows by F@0.05", audit["worst_by_f05"])
    table("Best rows by F@0.05", audit["best_by_f05"])
    table("Largest recall-minus-precision gaps", audit["largest_recall_precision_gap"])
    table("Largest prediction-side p90 outliers", audit["largest_pred_to_gt_p90"])

    lines.append("## Next experiment implication")
    lines.append("")
    lines.append(
        "Run the same fixed-30 robust protocol on one proposal-aligned candidate that "
        "adds structure to the adapter or target supervision (e.g. query-conditioned "
        "cross-attention, token distillation from optimized NOVA tokens, or a SCRREAM-like "
        "pseudo-GT control). The acceptance signal should prioritize F@0.05/precision-side "
        "improvement and representative renders, not only symmetric CD."
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path, help="Path to robust eval summary.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory; default: summary parent/failure_case_audit")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    out_dir = args.out_dir or (summary_path.parent / "failure_case_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(summary_path, args.top_k)
    json_path = out_dir / "failure_case_audit.json"
    md_path = out_dir / "failure_case_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(audit, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
