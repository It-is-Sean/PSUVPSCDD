#!/usr/bin/env python3
"""Train NOVA adapter on precomputed V-JEPA 2.1 encoder features."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None

from probe.adapter import VGGTToNovaAdapter
from vggt_nova_adapter_common_raw import (
    amp_context,
    assert_only_adapter_trainable,
    barrier_if_distributed,
    build_decoder,
    build_loader,
    chamfer_l2,
    cleanup_distributed,
    count_parameters,
    get_targets,
    images_from_batch,
    init_distributed_mode,
    move_batch_to_device,
    reduce_scalar,
    sample_decoder,
    sampler_set_epoch,
    save_json,
    scene_ids_from_batch,
    set_seed,
    trainable_parameter_names,
    write_point_cloud_ply,
    resolve_device,
)
from train_vggt_nova_adapter import (
    AdapterVelocityProbe,
    QUALITY_METRIC_KEYS,
    flow_matching_diagnostics,
    maybe_init_swanlab,
    maybe_init_wandb,
    parse_image_root_map,
    pointcloud_quality_metrics,
    resolve_val_preview_root,
    save_checkpoint,
    save_val_preview_sample,
    _sum_metrics_into,
)


def safe_cache_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(sample_id)).strip("_")


def vjepa_cache_path(cache_root: Path, window_mode: str, sample_id: str) -> Path:
    return cache_root / window_mode / f"{safe_cache_sample_id(sample_id)}.pt"


def get_vjepa_cached_features(batch, device, feature_cache_dir: str, window_mode: str) -> torch.Tensor:
    cache_root = Path(feature_cache_dir)
    selected_list = []
    for sample_id in batch["scene_ids"]:
        path = vjepa_cache_path(cache_root, window_mode, str(sample_id))
        if not path.exists():
            raise FileNotFoundError(f"Missing V-JEPA 2.1 cache for sample={sample_id} window_mode={window_mode}: {path}")
        payload = torch.load(path, map_location="cpu")
        features = payload["features"] if isinstance(payload, dict) else payload
        if features.ndim != 2:
            raise ValueError(f"Expected V-JEPA cache tensor [tokens, dim], got {tuple(features.shape)} from {path}")
        selected_list.append(features.to(device=device, dtype=torch.float32))
    return torch.stack(selected_list, dim=0).contiguous()


def read_vjepa_cache_metadata(feature_cache_dir: str, window_mode: str, sample_id: str) -> dict:
    path = vjepa_cache_path(Path(feature_cache_dir), window_mode, str(sample_id))
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        return dict(payload["metadata"])
    return {}


def run_eval(adapter, decoder, loader, device, meta, args, max_batches=None, output_dir: Path | None = None, global_step: int | None = None, save_previews: bool = False):
    adapter_module = unwrap_adapter(adapter)
    adapter_module.eval()
    total_chamfer = 0.0
    chamfer_count = 0
    total_velocity = 0.0
    velocity_count = 0
    bin_count = max(0, int(args.val_flow_t_bins))
    bin_totals = [0.0 for _ in range(bin_count)]
    bin_counts = [0 for _ in range(bin_count)]
    quality_totals = {key: 0.0 for key in QUALITY_METRIC_KEYS}
    quality_counts = {key: 0 for key in QUALITY_METRIC_KEYS}
    preview_records = []
    preview_limit = max(0, int(args.val_preview_samples))
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            images = images_from_batch(batch)
            selected = get_vjepa_cached_features(
                batch,
                device,
                args.vjepa_feature_cache_dir,
                args.vjepa_window_mode,
            )
            tokens = adapter_module(selected)
            pred = sample_decoder(decoder, tokens, args.num_queries, meta["fm_step_size"], args.seed + batch_idx, images.shape[1])
            target = get_targets(batch, meta["query_source"], max_points=args.num_queries, norm_mode=meta.get("norm_mode", "none"))
            chamfer_loss = chamfer_l2(pred, target)
            velocity_mse, timesteps = flow_matching_diagnostics(
                decoder,
                tokens,
                target,
                seed=args.seed + 100000 + batch_idx,
                num_views=images.shape[1],
                t_bins=bin_count,
            )
            batch_quality_for_preview = None
            for sample_idx in range(pred.shape[0]):
                sample_quality = pointcloud_quality_metrics(
                    pred[sample_idx],
                    target[sample_idx],
                    max_points=int(args.val_metric_max_points),
                    seed=int(args.seed + 400000 + batch_idx * 100 + sample_idx),
                )
                _sum_metrics_into(quality_totals, quality_counts, sample_quality)
                if sample_idx == 0:
                    batch_quality_for_preview = sample_quality
            total_chamfer += float(chamfer_loss.item())
            chamfer_count += 1
            total_velocity += float(velocity_mse.sum().item())
            velocity_count += int(velocity_mse.numel())
            if bin_count > 0:
                indices = torch.clamp((timesteps * bin_count).long(), min=0, max=bin_count - 1)
                for bin_idx in range(bin_count):
                    mask = indices == bin_idx
                    if mask.any():
                        values = velocity_mse[mask]
                        bin_totals[bin_idx] += float(values.sum().item())
                        bin_counts[bin_idx] += int(values.numel())
            if save_previews and output_dir is not None and global_step is not None and len(preview_records) < preview_limit:
                preview_records.append(
                    save_val_preview_sample(
                        decoder,
                        tokens,
                        images,
                        batch,
                        meta,
                        args,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        output_dir=output_dir,
                        sample_metrics=batch_quality_for_preview or {},
                    )
                )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats = torch.tensor([total_chamfer, float(chamfer_count), total_velocity, float(velocity_count)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_chamfer, chamfer_count, total_velocity, velocity_count = stats.tolist()
        if bin_count > 0:
            bin_stats = torch.tensor([value for pair in zip(bin_totals, bin_counts) for value in pair], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(bin_stats, op=torch.distributed.ReduceOp.SUM)
            vals = bin_stats.tolist()
            bin_totals = vals[0::2]
            bin_counts = [int(x) for x in vals[1::2]]
        quality_stats = torch.tensor(
            [value for key in QUALITY_METRIC_KEYS for value in (quality_totals[key], quality_counts[key])],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(quality_stats, op=torch.distributed.ReduceOp.SUM)
        vals = quality_stats.tolist()
        for idx, key in enumerate(QUALITY_METRIC_KEYS):
            quality_totals[key] = vals[idx * 2]
            quality_counts[key] = int(vals[idx * 2 + 1])
    metrics = {
        "chamfer_l2": total_chamfer / max(1, int(chamfer_count)),
        "velocity_mse": total_velocity / max(1, int(velocity_count)),
        "preview_records": preview_records,
    }
    loss_per_t_bin = []
    for bin_idx in range(bin_count):
        value = bin_totals[bin_idx] / max(1, bin_counts[bin_idx])
        metrics[f"loss_t_bin_{bin_idx:02d}"] = value
        loss_per_t_bin.append({"bin": bin_idx, "count": int(bin_counts[bin_idx]), "velocity_mse": value})
    metrics["loss_per_t_bin"] = loss_per_t_bin
    for key in QUALITY_METRIC_KEYS:
        metrics[key] = quality_totals[key] / max(1, quality_counts[key])
    adapter_module.train()
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter_layers", type=int, default=4)
    parser.add_argument("--adapter_hidden_dim", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/probe3d/result/vjepa21_nova_adapter_seed17")
    parser.add_argument("--debug_one_batch", action="store_true")
    parser.add_argument("--nova_ckpt", default=None)
    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter", "scannet"))
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--max_val_scenes", type=int, default=0)
    parser.add_argument("--max_test_scenes", type=int, default=0)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--query_source", default=None)
    parser.add_argument("--scannet_target_mode", default="complete_zpos")
    parser.add_argument("--scannet_frustum_margin", type=float, default=1.0)
    parser.add_argument("--scannet_min_views", type=int, default=2)
    parser.add_argument("--scannet_complete_points", type=int, default=10000)
    parser.add_argument("--scannet_max_interval", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=20000)
    parser.add_argument("--save_ply_queries", type=int, default=40960)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--loss_type", default="nova_flow", choices=("nova_flow", "chamfer_sample", "flow_chamfer_hybrid"))
    parser.add_argument("--chamfer_weight", type=float, default=0.1)
    parser.add_argument("--image_root_map", default=None)
    parser.add_argument("--vjepa_feature_cache_dir", required=True)
    parser.add_argument("--vjepa_window_mode", required=True)
    parser.add_argument("--vjepa_model_name", default="vjepa2_1_vit_large_384")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="PSUVPSC3DD")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="PSUVPSC3DD")
    parser.add_argument("--swanlab_workspace", default=None)
    parser.add_argument("--swanlab_experiment", default=None)
    parser.add_argument("--final_test", action="store_true")
    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument("--test_eval_batches", type=int, default=0)
    parser.add_argument("--val_flow_t_bins", type=int, default=10)
    parser.add_argument("--val_metric_max_points", type=int, default=20000)
    parser.add_argument("--val_preview_samples", type=int, default=12)
    parser.add_argument("--val_preview_queries", type=int, default=40960)
    parser.add_argument("--val_preview_dir", default=None)
    return parser.parse_args()


def unwrap_adapter(adapter):
    if isinstance(adapter, (nn.DataParallel, DDP)):
        return unwrap_adapter(adapter.module)
    if isinstance(adapter, AdapterVelocityProbe):
        return adapter.adapter
    return adapter


def main():
    dist_ctx = init_distributed_mode()
    try:
        args = parse_args()
        if args.debug_one_batch:
            args.max_steps = 1
            args.save_every = 1
            args.val_every = 1
        set_seed(args.seed)
        device = dist_ctx["device"] if dist_ctx["enabled"] else resolve_device(args.device)
        is_main = dist_ctx["is_main"]
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "training.log"

        decoder, meta, cfg = build_decoder(device, args.nova_ckpt)
        meta = dict(meta)
        if args.query_source is not None:
            meta["query_source"] = args.query_source
        elif args.scannet_target_mode == "src_view":
            meta["query_source"] = "src_view"
        image_root_map = parse_image_root_map(args.image_root_map)
        train_loader, data_args = build_loader(
            cfg, args.batch_size, args.num_workers, test=False, image_root_map=image_root_map,
            dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
            split_override=args.train_split, distributed=dist_ctx["enabled"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"],
            scannet_target_mode=args.scannet_target_mode, scannet_frustum_margin=args.scannet_frustum_margin, scannet_min_views=args.scannet_min_views,
            scannet_complete_points=args.scannet_complete_points, scannet_max_interval=args.scannet_max_interval,
        )
        val_loader = None
        test_loader = None
        if dist_ctx["enabled"] or is_main:
            val_loader, _ = build_loader(
                cfg, args.batch_size, max(0, min(args.num_workers, 2)), test=True, image_root_map=image_root_map,
                dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
                split_override=args.val_split, max_scenes=args.max_val_scenes,
                distributed=dist_ctx["enabled"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"],
                scannet_target_mode=args.scannet_target_mode, scannet_frustum_margin=args.scannet_frustum_margin, scannet_min_views=args.scannet_min_views,
                scannet_complete_points=args.scannet_complete_points, scannet_max_interval=args.scannet_max_interval,
            )
            if args.final_test:
                test_loader, _ = build_loader(
                    cfg, args.batch_size, max(0, min(args.num_workers, 2)), test=True, image_root_map=image_root_map,
                    dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
                    split_override=args.test_split, max_scenes=args.max_test_scenes,
                    distributed=dist_ctx["enabled"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"],
                    scannet_target_mode=args.scannet_target_mode, scannet_frustum_margin=args.scannet_frustum_margin, scannet_min_views=args.scannet_min_views,
                    scannet_complete_points=args.scannet_complete_points, scannet_max_interval=args.scannet_max_interval,
                )

        sampler_set_epoch(train_loader, 0)
        first_batch = next(iter(train_loader))
        first_batch = move_batch_to_device(first_batch, device)
        selected = get_vjepa_cached_features(first_batch, device, args.vjepa_feature_cache_dir, args.vjepa_window_mode)
        if is_main:
            print("Feature backbone: vjepa21_encoder_cache")
            print(f"V-JEPA window mode: {args.vjepa_window_mode}")
            print(f"V-JEPA model: {args.vjepa_model_name}")
            print(f"Selected V-JEPA feature shape: {tuple(selected.shape)}")
        cache_metadata = read_vjepa_cache_metadata(args.vjepa_feature_cache_dir, args.vjepa_window_mode, first_batch["scene_ids"][0])

        adapter = VGGTToNovaAdapter(
            input_dim=selected.shape[-1],
            output_dim=meta["token_dim"],
            output_tokens=meta["num_scene_tokens"],
            hidden_dim=args.adapter_hidden_dim,
            adapter_layers=args.adapter_layers,
        ).to(device)
        probe_model = AdapterVelocityProbe(adapter, decoder).to(device)
        use_parallel = False
        if dist_ctx["enabled"]:
            ddp_device_ids = [dist_ctx["local_rank"]] if device.type == "cuda" else None
            probe_model = DDP(probe_model, device_ids=ddp_device_ids, output_device=dist_ctx["local_rank"] if device.type == "cuda" else None)
        else:
            use_parallel = bool(args.parallel and device.type == "cuda" and torch.cuda.device_count() > 1)
            if use_parallel:
                probe_model = nn.DataParallel(probe_model)
        optimizer = torch.optim.AdamW(unwrap_adapter(probe_model).parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
        assert_only_adapter_trainable(unwrap_adapter(probe_model), None, decoder)

        config = vars(args).copy()
        config.update(
            {
                "feature_backbone": "vjepa21_encoder_cache",
                "selected_feature_shape": list(selected.shape),
                "adapter_type": "mlp",
                "adapter_param_count": count_parameters(unwrap_adapter(probe_model)),
                "loss_type": args.loss_type,
                "chamfer_weight": args.chamfer_weight,
                "parallel": use_parallel,
                "ddp": dist_ctx["enabled"],
                "rank": dist_ctx["rank"],
                "world_size": dist_ctx["world_size"],
                "visible_cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
                "nova_decoder_meta": meta,
                "nova_ckpt": str(args.nova_ckpt) if args.nova_ckpt else None,
                "dataset": {"data_root": data_args.data_root, "test_dataset_name": data_args.test_dataset_name},
                "image_root_map": args.image_root_map,
                "vjepa_cache_metadata": cache_metadata,
            }
        )
        if is_main:
            save_json(output_dir / "config.json", config)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(config, indent=2) + "\n")
                log.write("Trainable parameter names:\n")
                for name in trainable_parameter_names({"adapter": unwrap_adapter(probe_model), "decoder": decoder}):
                    log.write(f"{name}\n")

        wandb_run = maybe_init_wandb(args, output_dir, config) if is_main else None
        swanlab_run = maybe_init_swanlab(args, output_dir, config) if is_main else False

        best_loss = math.inf
        best_val_chamfer_l2 = math.inf
        best_val_fscore_tau_010 = -math.inf
        best_val_pred_to_gt_p90 = math.inf
        last_val_metrics = None
        first_loss = None
        final_loss = None
        global_step = 0
        train_epoch = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            unwrap_adapter(probe_model).load_state_dict(checkpoint["adapter"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))
            best_loss = float(checkpoint.get("best_loss", best_loss))
            first_loss = checkpoint.get("first_loss", first_loss)
            final_loss = checkpoint.get("final_loss", final_loss)
            best_val_chamfer_l2 = float(checkpoint.get("best_val_chamfer_l2", best_val_chamfer_l2))
            best_val_fscore_tau_010 = float(checkpoint.get("best_val_fscore_tau_0.10", best_val_fscore_tau_010))
            best_val_pred_to_gt_p90 = float(checkpoint.get("best_val_pred_to_gt_p90", best_val_pred_to_gt_p90))

        sampler_set_epoch(train_loader, train_epoch)
        data_iter = iter(train_loader)
        while global_step < args.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                train_epoch += 1
                sampler_set_epoch(train_loader, train_epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = move_batch_to_device(batch, device)
            images = images_from_batch(batch)
            selected = get_vjepa_cached_features(batch, device, args.vjepa_feature_cache_dir, args.vjepa_window_mode)
            target = get_targets(batch, meta["query_source"], max_points=args.num_queries, norm_mode=meta.get("norm_mode", "none"))

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                loss, tokens, _, _ = probe_model(
                    selected,
                    target,
                    args.seed + global_step,
                    images.shape[1],
                    loss_type=args.loss_type,
                    num_queries=args.num_queries,
                    fm_step_size=meta["fm_step_size"],
                    chamfer_weight=args.chamfer_weight,
                )
                if loss.ndim > 0:
                    loss = loss.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step + 1}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grads_ok = all(p.grad is not None for p in unwrap_adapter(probe_model).parameters() if p.requires_grad)
            if global_step == 0 and not grads_ok:
                raise AssertionError("Adapter parameters did not receive gradients after first backward pass.")
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            reduced_loss = reduce_scalar(loss, dist_ctx["enabled"])
            final_loss = reduced_loss
            if first_loss is None:
                first_loss = final_loss
            if final_loss < best_loss:
                best_loss = final_loss
            if is_main and (global_step % 10 == 0 or global_step == 1 or args.debug_one_batch):
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"step={global_step} loss={final_loss:.8f} best={best_loss:.8f}\n")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": final_loss, "train/best_loss": best_loss}, step=global_step)
                if swanlab_run:
                    swanlab.log({"train/loss": final_loss, "train/best_loss": best_loss}, step=global_step)

            if (global_step % args.val_every == 0 or args.debug_one_batch) and val_loader is not None:
                val_metrics = run_eval(
                    probe_model,
                    decoder,
                    val_loader,
                    device,
                    meta,
                    args,
                    max_batches=None if args.eval_batches <= 0 else args.eval_batches,
                    output_dir=output_dir,
                    global_step=global_step,
                    save_previews=is_main and args.val_preview_samples > 0,
                )
                if is_main:
                    last_val_metrics = val_metrics
                    val_chamfer_l2 = float(val_metrics["chamfer_l2"])
                    val_velocity_mse = float(val_metrics["velocity_mse"])
                    val_fscore_tau_010 = float(val_metrics["fscore_tau_0.10"])
                    val_pred_to_gt_p90 = float(val_metrics["pred_to_gt_p90"])
                    validation_payload = {
                        "step": global_step,
                        "val_chamfer_l2": val_chamfer_l2,
                        "val_velocity_mse": val_velocity_mse,
                        "loss_per_t_bin": val_metrics.get("loss_per_t_bin", []),
                        "preview_records": val_metrics.get("preview_records", []),
                    }
                    for key, value in val_metrics.items():
                        if key.startswith("loss_t_bin_"):
                            validation_payload[key] = float(value)
                        elif key in QUALITY_METRIC_KEYS:
                            validation_payload[f"val_{key}"] = float(value)
                    save_json(output_dir / "validation_metrics.json", validation_payload)
                    if args.val_preview_samples > 0:
                        preview_root = resolve_val_preview_root(args, output_dir)
                        save_json(preview_root / "latest_index.json", {"step": global_step, "preview_root": str(preview_root), "records": val_metrics.get("preview_records", [])})
                    next_best_val_chamfer_l2 = min(best_val_chamfer_l2, val_chamfer_l2)
                    next_best_val_fscore_tau_010 = max(best_val_fscore_tau_010, val_fscore_tau_010)
                    next_best_val_pred_to_gt_p90 = min(best_val_pred_to_gt_p90, val_pred_to_gt_p90)
                    if val_chamfer_l2 < best_val_chamfer_l2:
                        best_val_chamfer_l2 = val_chamfer_l2
                        save_checkpoint(output_dir / "best.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": best_val_chamfer_l2, "best_val_fscore_tau_0.10": next_best_val_fscore_tau_010, "best_val_pred_to_gt_p90": next_best_val_pred_to_gt_p90})
                    if val_fscore_tau_010 > best_val_fscore_tau_010:
                        best_val_fscore_tau_010 = val_fscore_tau_010
                        save_checkpoint(output_dir / "best_fscore_010.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": next_best_val_chamfer_l2, "best_val_fscore_tau_0.10": best_val_fscore_tau_010, "best_val_pred_to_gt_p90": next_best_val_pred_to_gt_p90})
                    if val_pred_to_gt_p90 < best_val_pred_to_gt_p90:
                        best_val_pred_to_gt_p90 = val_pred_to_gt_p90
                        save_checkpoint(output_dir / "best_pred_to_gt_p90.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": next_best_val_chamfer_l2, "best_val_fscore_tau_0.10": next_best_val_fscore_tau_010, "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90})
                barrier_if_distributed(dist_ctx["enabled"])
            if global_step % args.save_every == 0 or args.debug_one_batch:
                if is_main:
                    checkpoint_extra = {
                        "best_val_chamfer_l2": best_val_chamfer_l2,
                        "best_val_fscore_tau_0.10": best_val_fscore_tau_010,
                        "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
                    }
                    save_checkpoint(output_dir / f"step_{global_step:06d}.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra=checkpoint_extra)
                    save_checkpoint(output_dir / "latest.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra=checkpoint_extra)
                    try:
                        scene_ids = scene_ids_from_batch(batch, global_step)
                        ply_dir = output_dir / "ply"
                        pred_path = ply_dir / f"{scene_ids[0]}_step{global_step:06d}_pred.ply"
                        gt_path = ply_dir / f"{scene_ids[0]}_pseudo_gt.ply"
                        ply_pred = sample_decoder(decoder, tokens.detach(), args.save_ply_queries, meta["fm_step_size"], args.seed + 999 + global_step, images.shape[1])
                        write_point_cloud_ply(pred_path, ply_pred[0])
                        if not gt_path.exists():
                            ply_target = get_targets(batch, meta["query_source"], max_points=args.save_ply_queries, norm_mode=meta.get("norm_mode", "none"))
                            write_point_cloud_ply(gt_path, ply_target[0])
                    except Exception:
                        pass
                barrier_if_distributed(dist_ctx["enabled"])

        test_metrics = None
        if test_loader is not None:
            test_metrics = run_eval(
                probe_model,
                decoder,
                test_loader,
                device,
                meta,
                args,
                max_batches=None if args.test_eval_batches <= 0 else args.test_eval_batches,
            )
        if is_main:
            final_metrics = {
                "feature_backbone": "vjepa21_encoder_cache",
                "vjepa_window_mode": args.vjepa_window_mode,
                "vjepa_model_name": args.vjepa_model_name,
                "vjepa_cache_metadata": config.get("vjepa_cache_metadata", {}),
                "first_loss": first_loss,
                "final_loss": final_loss,
                "best_loss": best_loss,
                "best_val_chamfer_l2": best_val_chamfer_l2,
                "best_val_fscore_tau_0.10": best_val_fscore_tau_010,
                "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
            }
            if last_val_metrics is not None:
                final_metrics["last_val_chamfer_l2"] = float(last_val_metrics["chamfer_l2"])
                final_metrics["last_val_velocity_mse"] = float(last_val_metrics["velocity_mse"])
                final_metrics["last_val_loss_per_t_bin"] = last_val_metrics.get("loss_per_t_bin", [])
                for key in QUALITY_METRIC_KEYS:
                    final_metrics[f"last_val_{key}"] = float(last_val_metrics[key])
            if test_metrics is not None:
                final_metrics["test_chamfer_l2"] = float(test_metrics["chamfer_l2"])
                final_metrics["test_velocity_mse"] = float(test_metrics["velocity_mse"])
                for key in QUALITY_METRIC_KEYS:
                    final_metrics[f"test_{key}"] = float(test_metrics[key])
            save_json(output_dir / "final_metrics.json", final_metrics)
            if wandb_run is not None:
                wandb_run.finish()
            if swanlab_run:
                swanlab.finish()
        barrier_if_distributed(dist_ctx["enabled"])
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
