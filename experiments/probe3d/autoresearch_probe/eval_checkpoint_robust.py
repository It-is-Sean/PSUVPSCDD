#!/usr/bin/env python3
"""Run fixed-sample robust evaluation for a saved VGGT->NOVA adapter checkpoint.

This is intentionally artifact-first: it saves pred/GT PLYs, input contact sheets,
per-sample robust metrics, and a compact summary. It is meant to replace single
sample / single symmetric-CD interpretations in the autoresearch loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROBE_ROOT = ROOT / "experiments" / "probe3d"
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))

from probe.adapter import (  # noqa: E402
    VGGTToNovaAdapter,
    VGGTToNovaCrossAttentionAdapter,
    VGGTToNovaSelfAttentionAdapter,
)
from vggt_nova_adapter_common_raw import (  # noqa: E402
    build_decoder,
    build_loader,
    extract_vggt_features,
    get_targets,
    images_from_batch,
    load_vggt,
    move_batch_to_device,
    sample_decoder,
    sample_keys_from_batch,
    select_vggt_layer23,
    set_seed,
    write_point_cloud_ply,
)
from experiments.probe3d.autoresearch_probe.robust_ply_metrics import metrics_for_pair  # noqa: E402


def save_input_contact(images: torch.Tensor, path: Path, title: str) -> None:
    imgs = images[0].detach().cpu().float().clamp(0, 1)
    width, height = 320, 280
    canvas = Image.new("RGB", (width * len(imgs), height + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title[:180], fill=(0, 0, 0))
    for i, img in enumerate(imgs):
        arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        im = Image.fromarray(arr).resize((width, height))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 120, 24], fill=(255, 255, 255))
        d.text((5, 5), f"view{i}", fill=(0, 0, 0))
        canvas.paste(im, (width * i, 34))
    canvas.save(path)


def build_adapter(adapter_type: str, cfg_train: dict, input_dim: int, meta: dict) -> torch.nn.Module:
    common = dict(
        input_dim=input_dim,
        output_dim=int(meta["token_dim"]),
        output_tokens=int(meta["num_scene_tokens"]),
        hidden_dim=int(cfg_train.get("adapter_hidden_dim", 1024)),
        adapter_layers=int(cfg_train.get("adapter_layers", 4)),
    )
    if adapter_type == "mlp":
        return VGGTToNovaAdapter(**common)
    if adapter_type == "cross_attention":
        return VGGTToNovaCrossAttentionAdapter(
            **common,
            num_heads=int(cfg_train.get("adapter_heads", cfg_train.get("attention_heads", 8))),
            mlp_ratio=float(cfg_train.get("adapter_mlp_ratio", cfg_train.get("attention_mlp_ratio", 2.0))),
        )
    if adapter_type == "self_attention":
        return VGGTToNovaSelfAttentionAdapter(
            **common,
            num_heads=int(cfg_train.get("adapter_heads", cfg_train.get("attention_heads", 8))),
            mlp_ratio=float(cfg_train.get("adapter_mlp_ratio", cfg_train.get("attention_mlp_ratio", 2.0))),
        )
    raise ValueError(f"Unsupported adapter_type={adapter_type!r}")


def fetch_samples(loader, sample_indices: list[int]):
    wanted = set(sample_indices)
    out = {}
    for idx, batch in enumerate(loader):
        if idx in wanted:
            out[idx] = batch
            if len(out) == len(wanted):
                break
    missing = sorted(wanted - set(out))
    if missing:
        raise RuntimeError(f"Missing requested sample indices: {missing}")
    return out


def summarize_metrics(items: list[dict]) -> dict:
    def collect(path: list[str]) -> np.ndarray:
        vals = []
        for item in items:
            x = item
            for key in path:
                x = x[key]
            vals.append(float(x))
        return np.asarray(vals, dtype=np.float64)

    fields = {
        "symmetric_cd_l2": ["symmetric_cd_l2"],
        "trimmed_cd_l2_95": ["trimmed_cd_l2_95"],
        "pred_to_gt_mean": ["pred_to_gt_precision_side", "mean"],
        "gt_to_pred_mean": ["gt_to_pred_recall_side", "mean"],
        "fscore_tau_0.05": ["fscore", "tau_0.05", "fscore"],
        "precision_tau_0.05": ["fscore", "tau_0.05", "precision"],
        "recall_tau_0.05": ["fscore", "tau_0.05", "recall"],
        "fscore_tau_0.10": ["fscore", "tau_0.1", "fscore"],
    }
    summary = {"count": len(items)}
    for name, path in fields.items():
        arr = collect(path)
        summary[name] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_indices", default="0,30,60,90,120")
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional fixed eval manifest JSON produced by make_fixed_eval_manifest.py. "
            "When set, dataset_index values from manifest rows override --sample_indices "
            "so repeated checkpoint evals use the exact same scene/frame rows."
        ),
    )
    parser.add_argument("--num_queries", type=int, default=20000)
    parser.add_argument("--metric_max_points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = manifest.get("rows", [])
        if not rows:
            raise ValueError(f"Manifest has no rows: {manifest_path}")
        sample_indices = [int(row["dataset_index"]) for row in rows]
    else:
        sample_indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_train = ckpt["config"]
    meta = dict(ckpt.get("meta", ckpt.get("decoder_meta", cfg_train.get("nova_decoder_meta", {}))))
    decoder, decoder_meta, cfg = build_decoder(device, cfg_train.get("nova_ckpt"))
    for key, val in dict(decoder_meta).items():
        meta.setdefault(key, val)
    decoder.eval().requires_grad_(False)

    dataset_name = cfg_train.get("dataset") if isinstance(cfg_train.get("dataset"), str) else "scannet"
    loader, data_args = build_loader(
        cfg,
        batch_size=1,
        num_workers=0,
        test=True,
        dataset_name=dataset_name,
        data_root=cfg_train.get("data_root"),
        seed=args.seed,
        num_views=int(cfg_train.get("num_views", 2)),
        split_override=cfg_train.get("val_split", "val"),
        max_scenes=int(cfg_train.get("max_val_scenes", 10)),
        scannet_target_mode=cfg_train.get("scannet_target_mode", "complete_zpos"),
        scannet_frustum_margin=float(cfg_train.get("scannet_frustum_margin", 1.0)),
        scannet_min_views=int(cfg_train.get("scannet_min_views", 2)),
        scannet_complete_points=int(cfg_train.get("scannet_complete_points", 10000)),
        scannet_max_interval=int(cfg_train.get("scannet_max_interval", 1)),
    )
    batches = fetch_samples(loader, sample_indices)

    vggt = load_vggt(device)
    vggt.eval().requires_grad_(False)

    metrics_items = []
    sample_records = []
    adapter = None
    selected_idx = None
    selection_reason = None

    for ordinal, sample_idx in enumerate(sample_indices):
        sample_dir = out_dir / f"sample_{sample_idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        batch = move_batch_to_device(batches[sample_idx], device)
        images = images_from_batch(batch)
        keys = [list(k) for k in sample_keys_from_batch(batch)]
        with torch.no_grad():
            features, _ = extract_vggt_features(vggt, images, amp=True)
            selected, selected_idx, selection_reason = select_vggt_layer23(features)
        if adapter is None:
            adapter_type = str(cfg_train.get("adapter_type", "mlp"))
            adapter = build_adapter(adapter_type, cfg_train, selected.shape[-1], meta).to(device)
            adapter.load_state_dict(ckpt["adapter"])
            adapter.eval()
        with torch.no_grad():
            tokens = adapter(selected)
            pred = sample_decoder(
                decoder,
                tokens,
                num_queries=args.num_queries,
                step_size=float(meta["fm_step_size"]),
                seed=args.seed + 10000 + sample_idx,
                num_views=images.shape[1],
            )
            target = get_targets(
                batch,
                str(meta.get("query_source", "src_complete")),
                max_points=args.num_queries,
                norm_mode=str(meta.get("norm_mode", "none")),
            )
        pred_path = sample_dir / "pred.ply"
        gt_path = sample_dir / "target.ply"
        write_point_cloud_ply(pred_path, pred[0])
        write_point_cloud_ply(gt_path, target[0])
        save_input_contact(images, sample_dir / "inputs.png", f"sample_idx={sample_idx} keys={keys}")
        m = metrics_for_pair(pred_path, gt_path, max_points=args.metric_max_points, seed=args.seed + ordinal * 1000)
        (sample_dir / "robust_metrics.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
        metrics_items.append(m)
        record = {"sample_idx": sample_idx, "keys": keys, "dir": str(sample_dir), "metrics": m}
        if manifest is not None:
            record["manifest_row"] = manifest["rows"][ordinal]
        sample_records.append(record)
        print(
            f"sample={sample_idx} sym={m['symmetric_cd_l2']:.6f} "
            f"p2g={m['pred_to_gt_precision_side']['mean']:.4f} "
            f"g2p={m['gt_to_pred_recall_side']['mean']:.4f} "
            f"F05={m['fscore']['tau_0.05']['fscore']:.4f}",
            flush=True,
        )

    summary = {
        "ckpt": str(args.ckpt),
        "sample_indices": sample_indices,
        "manifest": {
            "path": str(args.manifest) if args.manifest else None,
            "schema": manifest.get("schema") if manifest else None,
            "sample_count": manifest.get("sample_count") if manifest else None,
            "data_root": manifest.get("data_root") if manifest else None,
            "split": manifest.get("split") if manifest else None,
            "num_views": manifest.get("num_views") if manifest else None,
            "max_interval": manifest.get("max_interval") if manifest else None,
        },
        "num_queries": args.num_queries,
        "metric_max_points": args.metric_max_points,
        "data_args": str(data_args),
        "selected_idx": selected_idx,
        "selection_reason": selection_reason,
        "config_subset": {
            k: cfg_train.get(k)
            for k in [
                "dataset",
                "num_views",
                "scannet_target_mode",
                "scannet_max_interval",
                "loss_type",
                "adapter_type",
                "adapter_layers",
                "adapter_hidden_dim",
            ]
        },
        "aggregate": summarize_metrics(metrics_items),
        "samples": sample_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "aggregate": summary["aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
