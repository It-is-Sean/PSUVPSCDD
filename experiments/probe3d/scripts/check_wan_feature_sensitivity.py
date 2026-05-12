#!/usr/bin/env python3
"""Check whether WAN feature choices change adapter tokens and initial loss."""

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

from probe.adapter import VGGTToNovaAdapter
from train_wan_t2v_nova_adapter import AdapterVelocityProbe, get_wan_t2v_cached_features
from train_vggt_nova_adapter import parse_image_root_map
from vggt_nova_adapter_common_raw import (
    build_decoder,
    build_loader,
    get_targets,
    images_from_batch,
    move_batch_to_device,
    resolve_device,
    set_seed,
)


def parse_configs(value: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected config item TIMESTEP:LAYER, got {item!r}")
        timestep, layer = item.split(":", 1)
        configs.append((int(timestep), int(layer)))
    if not configs:
        raise ValueError("Expected at least one WAN config")
    return configs


def tensor_stats(x: torch.Tensor) -> dict[str, float | list[int]]:
    x = x.detach().float()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "abs_mean": float(x.abs().mean().item()),
        "token_mean_std": float(x.mean(dim=-1).std().item()) if x.ndim >= 2 else 0.0,
        "dim_mean_std": float(x.mean(dim=-2).std().item()) if x.ndim >= 2 else 0.0,
    }


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    x = a.detach().float().flatten()
    y = b.detach().float().flatten()
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0).item()),
        "relative_l2": float((x - y).norm().item() / max(1e-8, x.norm().item())),
        "mean_abs": float((x - y).abs().mean().item()),
        "max_abs": float((x - y).abs().max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter_data", required=True)
    parser.add_argument("--nova_ckpt", default=None)
    parser.add_argument("--wan_feature_cache_dir", required=True)
    parser.add_argument("--configs", default="249:9,249:29,499:9,749:9")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--num_queries", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--adapter_layers", type=int, default=4)
    parser.add_argument("--adapter_hidden_dim", type=int, default=1024)
    parser.add_argument("--loss_type", default="nova_flow", choices=("nova_flow", "chamfer_sample", "flow_chamfer_hybrid"))
    parser.add_argument("--chamfer_weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_root_map", default=None)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = parse_configs(args.configs)
    set_seed(args.seed)
    device = resolve_device(args.device)
    decoder, meta, cfg = build_decoder(device, args.nova_ckpt)
    meta = dict(meta)
    image_root_map = parse_image_root_map(args.image_root_map)
    loader, data_args = build_loader(
        cfg,
        args.batch_size,
        args.num_workers,
        test=False,
        image_root_map=image_root_map,
        dataset_name="scrream_adapter",
        data_root=args.adapter_data,
        seed=args.seed,
        num_views=args.num_views,
        split_override=args.split,
    )

    results: dict[str, object] = {
        "adapter_data": args.adapter_data,
        "wan_feature_cache_dir": args.wan_feature_cache_dir,
        "configs": [{"timestep": t, "layer": l} for t, l in configs],
        "num_samples": int(args.num_samples),
        "loss_type": args.loss_type,
        "samples": [],
        "aggregate": {},
    }
    aggregate_losses = {f"t{t:03d}_layer{l:02d}": [] for t, l in configs}
    aggregate_feature_rel_l2 = []
    aggregate_token_rel_l2 = []

    for sample_idx, batch in enumerate(loader):
        if sample_idx >= args.num_samples:
            break
        batch = move_batch_to_device(batch, device)
        images = images_from_batch(batch)
        target = get_targets(batch, meta["query_source"], max_points=args.num_queries, norm_mode=meta.get("norm_mode", "none"))
        sample_id = str(batch["scene_ids"][0])
        per_config = {}
        for timestep, layer in configs:
            key = f"t{timestep:03d}_layer{layer:02d}"
            set_seed(args.seed)
            selected = get_wan_t2v_cached_features(
                batch,
                device,
                args.wan_feature_cache_dir,
                timestep,
                layer,
            )
            adapter = VGGTToNovaAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
            ).to(device)
            probe = AdapterVelocityProbe(adapter, decoder).to(device)
            with torch.no_grad():
                loss, tokens, _pred_velocity, _target_velocity = probe(
                    selected,
                    target,
                    args.seed + sample_idx,
                    images.shape[1],
                    loss_type=args.loss_type,
                    num_queries=args.num_queries,
                    fm_step_size=meta["fm_step_size"],
                    chamfer_weight=args.chamfer_weight,
                )
            loss_value = float(loss.mean().item() if loss.ndim > 0 else loss.item())
            aggregate_losses[key].append(loss_value)
            per_config[key] = {
                "timestep": int(timestep),
                "layer": int(layer),
                "initial_loss": loss_value,
                "feature_stats": tensor_stats(selected),
                "token_stats": tensor_stats(tokens),
                "features": selected.detach().cpu(),
                "tokens": tokens.detach().cpu(),
            }

        reference_key = f"t{configs[0][0]:03d}_layer{configs[0][1]:02d}"
        comparisons = {}
        for key, item in per_config.items():
            if key == reference_key:
                continue
            feature_cmp = compare_tensors(per_config[reference_key]["features"], item["features"])
            token_cmp = compare_tensors(per_config[reference_key]["tokens"], item["tokens"])
            aggregate_feature_rel_l2.append(feature_cmp["relative_l2"])
            aggregate_token_rel_l2.append(token_cmp["relative_l2"])
            comparisons[f"{reference_key}_vs_{key}"] = {
                "feature": feature_cmp,
                "adapter_tokens": token_cmp,
            }

        clean_per_config = {}
        for key, item in per_config.items():
            clean_per_config[key] = {
                k: v for k, v in item.items()
                if k not in {"features", "tokens"}
            }
        results["samples"].append({
            "sample_id": sample_id,
            "per_config": clean_per_config,
            "comparisons_to_reference": comparisons,
        })

    results["aggregate"] = {
        "mean_initial_loss": {
            key: (sum(values) / max(1, len(values)))
            for key, values in aggregate_losses.items()
        },
        "mean_feature_relative_l2_to_reference": (
            sum(aggregate_feature_rel_l2) / max(1, len(aggregate_feature_rel_l2))
        ),
        "mean_adapter_token_relative_l2_to_reference": (
            sum(aggregate_token_rel_l2) / max(1, len(aggregate_token_rel_l2))
        ),
        "dataset": {"data_root": data_args.data_root, "test_dataset_name": data_args.test_dataset_name},
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results["aggregate"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
