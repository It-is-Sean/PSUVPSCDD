#!/usr/bin/env python3
"""Oracle-token baseline for ScanNet -> frozen NOVA generator.

This bypasses VGGT and the MLP adapter. It directly optimizes the generator input
scene tokens for a small ScanNet subset. If this cannot fit the target points,
the bottleneck is the frozen generator / target domain / coordinate convention,
not the VGGT->MLP adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vggt_nova_adapter_common_raw import (  # noqa: E402
    build_decoder,
    build_loader,
    chamfer_l2,
    get_targets,
    move_batch_to_device,
    sample_decoder,
    sample_keys_from_batch,
    set_seed,
    write_point_cloud_ply,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cuda")
    p.add_argument("--nova_ckpt", default="checkpoints/scene_ae/checkpoint-last.pth")
    p.add_argument("--data_root", default="/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17")
    p.add_argument("--split", default="val")
    p.add_argument("--max_scenes", type=int, default=1)
    p.add_argument("--num_views", type=int, default=4)
    p.add_argument("--scannet_target_mode", default="complete_zpos", help="ScanNet complete-GT target support: complete_zpos, anchor_frustum, anchor_frustum_margin, covered_by_ge2, ...")
    p.add_argument("--scannet_frustum_margin", type=float, default=1.0, help="Projection margin for anchor_frustum_margin target mode.")
    p.add_argument("--scannet_min_views", type=int, default=2, help="Minimum covering views for covered_by_ge* target modes.")
    p.add_argument("--scannet_complete_points", type=int, default=10000, help="Maximum complete GT points kept per ScanNet sample before query-source sampling/FPS.")
    p.add_argument("--scannet_max_interval", type=int, default=1, help="Maximum processed-frame interval between selected ScanNet views.")
    p.add_argument("--num_queries", type=int, default=2048)
    p.add_argument("--query_source", default=None, help="Override decoder meta query_source, e.g. src_view or src_complete.")
    p.add_argument("--num_samples", type=int, default=2)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--out_dir", default="experiments/probe3d/result/scannet_oracle_tokens_val1")
    p.add_argument("--init_std", type=float, default=0.02)
    p.add_argument("--sample_seed", type=int, default=17000)
    return p.parse_args()


@torch.no_grad()
def eval_cd(decoder: nn.Module, tokens: torch.Tensor, target: torch.Tensor, step_size: float, seed: int, num_views: int):
    pred = sample_decoder(decoder, tokens, num_queries=target.shape[1], step_size=step_size, seed=seed, num_views=num_views)
    return float(chamfer_l2(pred, target).item()), pred


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder, meta, cfg = build_decoder(device, args.nova_ckpt)
    meta = dict(meta)
    if args.query_source is not None:
        meta["query_source"] = args.query_source
    elif args.scannet_target_mode == "src_view":
        meta["query_source"] = "src_view"
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    loader, data_args = build_loader(
        cfg,
        batch_size=1,
        num_workers=0,
        test=True,
        dataset_name="scannet",
        data_root=args.data_root,
        seed=args.seed,
        num_views=args.num_views,
        split_override=args.split,
        max_scenes=args.max_scenes,
        distributed=False,
        rank=0,
        world_size=1,
        scannet_target_mode=args.scannet_target_mode,
        scannet_frustum_margin=args.scannet_frustum_margin,
        scannet_min_views=args.scannet_min_views,
        scannet_complete_points=args.scannet_complete_points,
        scannet_max_interval=args.scannet_max_interval,
    )

    token_count = int(meta.get("num_scene_tokens", meta.get("latent_shape", [768, 128])[0]))
    token_dim = int(meta.get("token_dim", meta.get("latent_shape", [768, 128])[1]))
    step_size = float(meta.get("fm_step_size", 0.04))
    query_source = str(args.query_source or meta.get("query_source", "src_complete"))

    summary = {
        "args": vars(args),
        "meta": meta,
        "data_args": str(data_args),
        "samples": [],
    }

    for sample_idx, batch in zip(range(args.num_samples), loader):
        batch = move_batch_to_device(batch, device)
        keys = sample_keys_from_batch(batch)
        target = get_targets(batch, query_source, max_points=args.num_queries, norm_mode=meta.get("norm_mode", "none")).to(device).float()
        tokens = nn.Parameter(torch.randn(target.shape[0], token_count, token_dim, device=device) * args.init_std)
        opt = torch.optim.AdamW([tokens], lr=args.lr, weight_decay=args.weight_decay)

        init_cd, init_pred = eval_cd(decoder, tokens, target, step_size, args.sample_seed + sample_idx, args.num_views)
        sample_record = {
            "sample_idx": sample_idx,
            "keys": [list(k) for k in keys],
            "target_shape": list(target.shape),
            "init_cd": init_cd,
            "history": [],
        }
        print(f"sample={sample_idx} keys={keys} init_cd={init_cd:.6f}", flush=True)

        best_cd = init_cd
        best_step = 0
        best_tokens = tokens.detach().cpu()
        for step in range(1, args.steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, _aux = __import__("vggt_nova_adapter_common_raw").nova_flow_matching_loss(
                decoder,
                tokens,
                target,
                seed=args.seed * 100000 + sample_idx * 10000 + step,
                num_views=args.num_views,
            )
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_([tokens], max_norm=10.0).item())
            opt.step()

            if step % args.log_every == 0 or step == 1:
                print(f"sample={sample_idx} step={step} flow_loss={float(loss.item()):.8f} grad_norm={grad_norm:.4f} best_cd={best_cd:.6f}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                cd, pred = eval_cd(decoder, tokens, target, step_size, args.sample_seed + sample_idx, args.num_views)
                rec = {"step": step, "flow_loss": float(loss.item()), "cd": cd, "grad_norm": grad_norm}
                sample_record["history"].append(rec)
                print(f"EVAL sample={sample_idx} step={step} cd={cd:.6f} flow_loss={float(loss.item()):.8f}", flush=True)
                if cd < best_cd:
                    best_cd = cd
                    best_step = step
                    best_tokens = tokens.detach().cpu()
                    torch.save({"tokens": best_tokens, "sample_record": sample_record, "meta": meta}, out_dir / f"sample{sample_idx:02d}_best_tokens.pt")
                    write_point_cloud_ply(out_dir / f"sample{sample_idx:02d}_best_pred.ply", pred[0])
                    write_point_cloud_ply(out_dir / f"sample{sample_idx:02d}_target.ply", target[0])
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        sample_record["best_cd"] = best_cd
        sample_record["best_step"] = best_step
        sample_record["final_cd"] = sample_record["history"][-1]["cd"] if sample_record["history"] else init_cd
        summary["samples"].append(sample_record)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"DONE sample={sample_idx} best_cd={best_cd:.6f} best_step={best_step}", flush=True)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
