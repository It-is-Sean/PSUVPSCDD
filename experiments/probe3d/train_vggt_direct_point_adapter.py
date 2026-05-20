from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

try:
    import swanlab
except ImportError:
    swanlab = None

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for path in (THIS_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from probe.adapter import DirectPointCrossAttentionReadout
from vggt_nova_adapter_common_raw import (
    amp_context,
    build_loader,
    chamfer_l2,
    collate_adapter_samples,
    count_parameters,
    get_targets,
    images_from_batch,
    load_visual_backbone,
    move_batch_to_device,
    resolve_device,
    sample_keys_from_batch,
    save_json,
    set_seed,
    write_point_cloud_ply,
)
from train_vggt_nova_adapter import (
    QUALITY_METRIC_KEYS,
    get_selected_features,
    pad_or_subsample_visual_points,
    pointcloud_quality_metrics,
    sanitize_sample_id,
    save_input_contact_sheet,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=15850)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--debug_one_batch", action="store_true")

    parser.add_argument("--backbone", default="vggt", choices=("vggt", "vggt_omega"))
    parser.add_argument("--vggt_weights", default=None)
    parser.add_argument("--vggt_omega_weights", default=None)
    parser.add_argument("--vggt_omega_image_resolution", type=int, default=512)
    parser.add_argument("--vggt_layer", type=int, default=16)
    parser.add_argument("--vggt_token_mode", default="full", choices=("full", "registers"))
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--feature_control", default="none", choices=("none", "zero", "sample_shuffle"))

    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter",))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--image_root_map", default=None)

    parser.add_argument("--readout_type", default="point_ca", choices=("point_ca",))
    parser.add_argument("--point_queries", type=int, default=4096)
    parser.add_argument("--adapter_layers", type=int, default=4)
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--adapter_heads", type=int, default=8)
    parser.add_argument("--adapter_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--target_train_points", type=int, default=4096)
    parser.add_argument("--target_norm_mode", default="median_3")

    parser.add_argument("--eval_batches", type=int, default=0)
    parser.add_argument("--val_metric_max_points", type=int, default=20000)
    parser.add_argument("--val_preview_samples", type=int, default=12)
    parser.add_argument("--val_preview_queries", type=int, default=4096)
    parser.add_argument("--val_preview_dir", default=None)
    parser.add_argument("--save_ply_queries", type=int, default=4096)

    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab_project", default="PSUVPSC3DD")
    parser.add_argument("--swanlab_workspace", default=None)
    parser.add_argument("--swanlab_experiment", default=None)
    return parser.parse_args()


def parse_image_root_map(raw: str | None):
    if not raw:
        return None
    if "=" not in raw:
        raise ValueError("--image_root_map must be OLD=NEW")
    old, new = raw.split("=", 1)
    return old, new


def maybe_init_swanlab(args, output_dir: Path, config: dict):
    if not args.swanlab:
        return False
    if swanlab is None:
        raise ImportError("swanlab is not installed. Install it in the active environment before using --swanlab.")
    try:
        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key:
            swanlab.login(api_key=api_key, save=False)
        swanlab.init(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            experiment_name=args.swanlab_experiment or output_dir.name,
            config=config,
            logdir=str(output_dir),
            mode="cloud",
            reinit=True,
        )
        return True
    except Exception as exc:
        print(f"swanlab warning: init failed, continuing without swanlab: {type(exc).__name__}: {exc}")
        return False


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: Path, model, optimizer, step: int, config: dict, best_loss, first_loss, final_loss, extra: dict | None = None):
    payload = {
        "readout": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": config,
        "best_loss": best_loss,
        "first_loss": first_loss,
        "final_loss": final_loss,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def make_key_to_dataset_position(dataset) -> dict[tuple[str, ...], int]:
    mapping = {}
    if not hasattr(dataset, "indices") or not hasattr(dataset, "metadata"):
        raise ValueError("sample_shuffle control requires AdapterImagePointDataset with indices and metadata")
    for pos, src_idx in enumerate(dataset.indices):
        paths = dataset._resolve_frame_paths(dataset.metadata[src_idx])
        mapping[tuple(str(p) for p in paths)] = pos
    return mapping


class FeatureShuffleResolver:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.key_to_position = make_key_to_dataset_position(dataset)
        positions = list(range(len(dataset)))
        if len(positions) < 2:
            raise ValueError("sample_shuffle control requires at least two samples")
        self.shuffle_position = {
            position: positions[(idx + 1) % len(positions)]
            for idx, position in enumerate(positions)
        }

    def shuffled_batch_for(self, batch, device: torch.device):
        samples = []
        for key in sample_keys_from_batch(batch):
            position = self.key_to_position.get(tuple(key))
            if position is None:
                raise KeyError(f"Cannot sample-shuffle unknown sample key: {key}")
            samples.append(self.dataset[self.shuffle_position[position]])
        return move_batch_to_device(collate_adapter_samples(samples), device)


def apply_feature_control(
    selected: torch.Tensor,
    *,
    args,
    batch,
    device,
    vggt,
    feature_cache,
    shuffle_resolver: FeatureShuffleResolver | None,
):
    if args.feature_control == "none":
        return selected
    if args.feature_control == "zero":
        return torch.zeros_like(selected)
    if args.feature_control == "sample_shuffle":
        if shuffle_resolver is None:
            raise ValueError("sample_shuffle control requires a shuffle resolver")
        shuffled_batch = shuffle_resolver.shuffled_batch_for(batch, device)
        shuffled_images = images_from_batch(shuffled_batch)
        return get_selected_features(
            vggt,
            shuffled_images,
            shuffled_batch,
            feature_cache,
            device,
            args.amp,
            args.feature_cache_dir,
            vggt_layer=args.vggt_layer,
            backbone=args.backbone,
            token_mode=args.vggt_token_mode,
        )
    raise ValueError(f"Unsupported feature_control={args.feature_control!r}")


def resolve_val_preview_root(args, output_dir: Path) -> Path:
    if args.val_preview_dir:
        return Path(args.val_preview_dir)
    return output_dir / f"val_visual_{int(args.val_preview_queries)}"


def save_val_preview_sample(readout, pred: torch.Tensor, images: torch.Tensor, batch, args, batch_idx: int, global_step: int, output_dir: Path, sample_metrics: dict[str, float]):
    scene_id = str(batch["scene_ids"][0])
    sample_id = sanitize_sample_id(scene_id)
    preview_root = resolve_val_preview_root(args, output_dir)
    step_dir = preview_root / f"step_{global_step:06d}" / sample_id
    step_dir.mkdir(parents=True, exist_ok=True)

    target_raw = get_targets(batch, "src_complete", max_points=None, norm_mode=args.target_norm_mode)[:1]
    visual_target = pad_or_subsample_visual_points(
        target_raw[0],
        target_count=int(args.val_preview_queries),
        seed=int(args.seed + 300000 + global_step * 1000 + batch_idx),
    )
    visual_pred = pad_or_subsample_visual_points(
        pred[0],
        target_count=int(args.val_preview_queries),
        seed=int(args.seed + 310000 + global_step * 1000 + batch_idx),
    )

    pred_path = step_dir / f"pred_{int(args.val_preview_queries)}.ply"
    gt_path = step_dir / f"pseudo_gt_{int(args.val_preview_queries)}.ply"
    inputs_path = step_dir / "inputs.png"
    metrics_path = step_dir / "metrics.json"
    write_point_cloud_ply(pred_path, visual_pred)
    write_point_cloud_ply(gt_path, visual_target)
    save_input_contact_sheet(images[:1], inputs_path, f"step={global_step} {scene_id}")
    payload = {
        "step": int(global_step),
        "scene_id": scene_id,
        "sample_id": sample_id,
        "pred_path": str(pred_path),
        "pseudo_gt_path": str(gt_path),
        "inputs_path": str(inputs_path),
        "source_gt_points": int(target_raw.shape[1]),
        "visual_gt_points": int(visual_target.shape[0]),
        "visual_pred_points": int(visual_pred.shape[0]),
    }
    payload.update({f"val_{key}": float(value) for key, value in sample_metrics.items() if key in QUALITY_METRIC_KEYS})
    save_json(metrics_path, payload)
    return payload


def sum_metrics_into(totals: dict[str, float], counts: dict[str, int], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if key in QUALITY_METRIC_KEYS:
            totals[key] += float(value)
            counts[key] += 1


def run_eval(readout, loader, device, args, vggt, feature_cache, shuffle_resolver=None, max_batches=None, output_dir: Path | None = None, global_step: int | None = None, save_previews: bool = False):
    readout.eval()
    total_chamfer = 0.0
    chamfer_count = 0
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
            selected = get_selected_features(
                vggt,
                images,
                batch,
                feature_cache,
                device,
                args.amp,
                args.feature_cache_dir,
                vggt_layer=args.vggt_layer,
                backbone=args.backbone,
                token_mode=args.vggt_token_mode,
            )
            selected = apply_feature_control(
                selected,
                args=args,
                batch=batch,
                device=device,
                vggt=vggt,
                feature_cache=feature_cache,
                shuffle_resolver=shuffle_resolver,
            )
            pred = readout(selected)
            target = get_targets(batch, "src_complete", max_points=None, norm_mode=args.target_norm_mode)
            chamfer_loss = chamfer_l2(pred, target)
            batch_quality_for_preview = None
            for sample_idx in range(pred.shape[0]):
                sample_quality = pointcloud_quality_metrics(
                    pred[sample_idx],
                    target[sample_idx],
                    max_points=int(args.val_metric_max_points),
                    seed=int(args.seed + 400000 + batch_idx * 100 + sample_idx),
                )
                sum_metrics_into(quality_totals, quality_counts, sample_quality)
                if sample_idx == 0:
                    batch_quality_for_preview = sample_quality
            total_chamfer += float(chamfer_loss.item())
            chamfer_count += 1
            if save_previews and output_dir is not None and global_step is not None and len(preview_records) < preview_limit:
                preview_records.append(
                    save_val_preview_sample(
                        readout,
                        pred.detach(),
                        images,
                        batch,
                        args,
                        batch_idx=batch_idx,
                        global_step=global_step,
                        output_dir=output_dir,
                        sample_metrics=batch_quality_for_preview or {},
                    )
                )
    readout.train()
    metrics = {
        "chamfer_l2": total_chamfer / max(chamfer_count, 1),
        "preview_records": preview_records,
    }
    for key in QUALITY_METRIC_KEYS:
        metrics[key] = quality_totals[key] / max(quality_counts[key], 1)
    return metrics


def main():
    args = parse_args()
    if args.debug_one_batch:
        args.max_steps = 1
        args.save_every = 1
        args.val_every = 1
        args.val_preview_samples = min(args.val_preview_samples, 2)
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    image_root_map = parse_image_root_map(args.image_root_map)
    train_loader, data_args = build_loader(
        None,
        args.batch_size,
        args.num_workers,
        test=False,
        image_root_map=image_root_map,
        dataset_name=args.dataset,
        data_root=args.data_root,
        seed=args.seed,
        num_views=args.num_views,
        split_override=args.train_split,
        image_loader=args.backbone,
        image_resolution=args.vggt_omega_image_resolution,
    )
    val_loader, _ = build_loader(
        None,
        args.batch_size,
        max(0, min(args.num_workers, 2)),
        test=True,
        image_root_map=image_root_map,
        dataset_name=args.dataset,
        data_root=args.data_root,
        seed=args.seed,
        num_views=args.num_views,
        split_override=args.val_split,
        image_loader=args.backbone,
        image_resolution=args.vggt_omega_image_resolution,
    )
    train_shuffle = FeatureShuffleResolver(train_loader.dataset) if args.feature_control == "sample_shuffle" else None
    val_shuffle = FeatureShuffleResolver(val_loader.dataset) if args.feature_control == "sample_shuffle" else None

    vggt = load_visual_backbone(
        device,
        backbone=args.backbone,
        vggt_weights=args.vggt_weights,
        vggt_omega_weights=args.vggt_omega_weights,
    )
    vggt.eval()
    for param in vggt.parameters():
        param.requires_grad_(False)
    feature_cache = {}

    first_batch = move_batch_to_device(next(iter(train_loader)), device)
    first_images = images_from_batch(first_batch)
    first_selected = get_selected_features(
        vggt,
        first_images,
        first_batch,
        feature_cache,
        device,
        args.amp,
        args.feature_cache_dir,
        vggt_layer=args.vggt_layer,
        backbone=args.backbone,
        token_mode=args.vggt_token_mode,
    )
    first_selected = apply_feature_control(
        first_selected,
        args=args,
        batch=first_batch,
        device=device,
        vggt=vggt,
        feature_cache=feature_cache,
        shuffle_resolver=train_shuffle,
    )
    if args.readout_type != "point_ca":
        raise ValueError(f"Unsupported readout_type={args.readout_type!r}")
    readout = DirectPointCrossAttentionReadout(
        input_dim=int(first_selected.shape[-1]),
        point_queries=int(args.point_queries),
        hidden_dim=int(args.adapter_hidden_dim),
        adapter_layers=int(args.adapter_layers),
        num_heads=int(args.adapter_heads),
        mlp_ratio=float(args.adapter_mlp_ratio),
    ).to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    config = vars(args).copy()
    config.update(
        {
            "selected_feature_shape": list(first_selected.shape),
            "input_dim": int(first_selected.shape[-1]),
            "readout_param_count": count_parameters(readout),
            "dataset": {"data_root": data_args.data_root, "test_dataset_name": data_args.test_dataset_name},
            "target_norm_mode": args.target_norm_mode,
        }
    )
    save_json(output_dir / "config.json", config)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(config, indent=2) + "\n")
        for name, param in readout.named_parameters():
            if param.requires_grad:
                log.write(f"trainable: {name}\n")
                print(f"trainable: {name}")

    swanlab_run = maybe_init_swanlab(args, output_dir, config)
    best_loss = math.inf
    best_val_chamfer_l2 = math.inf
    best_val_fscore_tau_010 = -math.inf
    best_val_pred_to_gt_p90 = math.inf
    first_loss = None
    final_loss = None
    last_val_metrics = None
    global_step = 0

    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None and resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu")
        readout.load_state_dict(resume["readout"])
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        global_step = int(resume.get("step", 0))
        best_loss = float(resume.get("best_loss", best_loss))
        best_val_chamfer_l2 = float(resume.get("best_val_chamfer_l2", best_val_chamfer_l2))
        best_val_fscore_tau_010 = float(resume.get("best_val_fscore_tau_0.10", best_val_fscore_tau_010))
        best_val_pred_to_gt_p90 = float(resume.get("best_val_pred_to_gt_p90", best_val_pred_to_gt_p90))
        first_loss = resume.get("first_loss", first_loss)
        final_loss = resume.get("final_loss", final_loss)

    data_iter = iter(train_loader)
    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = move_batch_to_device(batch, device)
        images = images_from_batch(batch)
        selected = get_selected_features(
            vggt,
            images,
            batch,
            feature_cache,
            device,
            args.amp,
            args.feature_cache_dir,
            vggt_layer=args.vggt_layer,
            backbone=args.backbone,
            token_mode=args.vggt_token_mode,
        )
        selected = apply_feature_control(
            selected,
            args=args,
            batch=batch,
            device=device,
            vggt=vggt,
            feature_cache=feature_cache,
            shuffle_resolver=train_shuffle,
        )
        target = get_targets(
            batch,
            "src_complete",
            max_points=int(args.target_train_points),
            norm_mode=args.target_norm_mode,
        )

        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.amp):
            pred = readout(selected)
            loss = chamfer_l2(pred, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {global_step + 1}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads_ok = all(p.grad is not None for p in readout.parameters() if p.requires_grad)
        if global_step == 0 and not grads_ok:
            raise AssertionError("Direct readout parameters did not receive gradients after first backward pass.")
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        final_loss = float(loss.detach().item())
        if first_loss is None:
            first_loss = final_loss
        best_loss = min(best_loss, final_loss)
        line = f"step={global_step} chamfer_loss={final_loss:.8f} best={best_loss:.8f} grads_ok={grads_ok}"
        print(line)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")
        if swanlab_run:
            swanlab.log({"train/loss": final_loss, "train/best_loss": best_loss, "train/grads_ok": float(grads_ok)}, step=global_step)

        if global_step % args.val_every == 0:
            if global_step % args.save_every == 0 or args.debug_one_batch:
                extra = {
                    "best_val_chamfer_l2": best_val_chamfer_l2,
                    "best_val_fscore_tau_0.10": best_val_fscore_tau_010,
                    "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
                }
                save_checkpoint(output_dir / f"step_{global_step:06d}.pth", readout, optimizer, global_step, config, best_loss, first_loss, final_loss, extra=extra)
                save_checkpoint(output_dir / "latest.pth", readout, optimizer, global_step, config, best_loss, first_loss, final_loss, extra=extra)
            val_metrics = run_eval(
                readout,
                val_loader,
                device,
                args,
                vggt,
                feature_cache,
                shuffle_resolver=val_shuffle,
                max_batches=None if args.eval_batches <= 0 else args.eval_batches,
                output_dir=output_dir,
                global_step=global_step,
                save_previews=args.val_preview_samples > 0,
            )
            last_val_metrics = val_metrics
            val_chamfer_l2 = float(val_metrics["chamfer_l2"])
            val_fscore_tau_010 = float(val_metrics["fscore_tau_0.10"])
            val_pred_to_gt_p90 = float(val_metrics["pred_to_gt_p90"])
            validation_payload = {
                "step": global_step,
                "val_chamfer_l2": val_chamfer_l2,
                "preview_records": val_metrics.get("preview_records", []),
            }
            for key, value in val_metrics.items():
                if key in QUALITY_METRIC_KEYS:
                    validation_payload[f"val_{key}"] = float(value)
            if args.val_preview_samples > 0:
                preview_root = resolve_val_preview_root(args, output_dir)
                save_json(
                    preview_root / "latest_index.json",
                    {
                        "step": global_step,
                        "step_dir": str(preview_root / f"step_{global_step:06d}"),
                        "preview_records": validation_payload["preview_records"],
                    },
                )
            save_json(output_dir / "validation_metrics.json", validation_payload)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"validation step={global_step} "
                    f"val_chamfer_l2={val_chamfer_l2:.8f} "
                    f"val_pred_to_gt_p90={val_pred_to_gt_p90:.8f} "
                    f"val_gt_to_pred_p90={float(val_metrics['gt_to_pred_p90']):.8f} "
                    f"val_fscore_tau_0.05={float(val_metrics['fscore_tau_0.05']):.8f} "
                    f"val_fscore_tau_0.10={val_fscore_tau_010:.8f}\n"
                )
            next_best_val_chamfer_l2 = min(best_val_chamfer_l2, val_chamfer_l2)
            next_best_val_fscore_tau_010 = max(best_val_fscore_tau_010, val_fscore_tau_010)
            next_best_val_pred_to_gt_p90 = min(best_val_pred_to_gt_p90, val_pred_to_gt_p90)
            if val_chamfer_l2 < best_val_chamfer_l2:
                best_val_chamfer_l2 = val_chamfer_l2
                save_checkpoint(
                    output_dir / "best.pth",
                    readout,
                    optimizer,
                    global_step,
                    config,
                    best_loss,
                    first_loss,
                    final_loss,
                    extra={
                        "best_val_chamfer_l2": next_best_val_chamfer_l2,
                        "best_val_fscore_tau_0.10": next_best_val_fscore_tau_010,
                        "best_val_pred_to_gt_p90": next_best_val_pred_to_gt_p90,
                    },
                )
            if val_fscore_tau_010 > best_val_fscore_tau_010:
                best_val_fscore_tau_010 = val_fscore_tau_010
                save_checkpoint(
                    output_dir / "best_fscore_010.pth",
                    readout,
                    optimizer,
                    global_step,
                    config,
                    best_loss,
                    first_loss,
                    final_loss,
                    extra={
                        "best_val_chamfer_l2": next_best_val_chamfer_l2,
                        "best_val_fscore_tau_0.10": next_best_val_fscore_tau_010,
                        "best_val_pred_to_gt_p90": next_best_val_pred_to_gt_p90,
                    },
                )
            if val_pred_to_gt_p90 < best_val_pred_to_gt_p90:
                best_val_pred_to_gt_p90 = val_pred_to_gt_p90
                save_checkpoint(
                    output_dir / "best_pred_to_gt_p90.pth",
                    readout,
                    optimizer,
                    global_step,
                    config,
                    best_loss,
                    first_loss,
                    final_loss,
                    extra={
                        "best_val_chamfer_l2": next_best_val_chamfer_l2,
                        "best_val_fscore_tau_0.10": next_best_val_fscore_tau_010,
                        "best_val_pred_to_gt_p90": next_best_val_pred_to_gt_p90,
                    },
                )
            tracking = {
                "val/chamfer_l2": val_chamfer_l2,
                "val/pred_to_gt_p90": val_pred_to_gt_p90,
                "val/gt_to_pred_p90": float(val_metrics["gt_to_pred_p90"]),
                "val/fscore_tau_0.10": val_fscore_tau_010,
                "val/precision_tau_0.10": float(val_metrics["precision_tau_0.10"]),
                "val/recall_tau_0.10": float(val_metrics["recall_tau_0.10"]),
                "val/best_fscore_tau_0.10": float(best_val_fscore_tau_010),
                "val/best_pred_to_gt_p90": float(best_val_pred_to_gt_p90),
            }
            if swanlab_run:
                swanlab.log(tracking, step=global_step)

        if global_step % args.save_every == 0 or args.debug_one_batch:
            extra = {
                "best_val_chamfer_l2": best_val_chamfer_l2,
                "best_val_fscore_tau_0.10": best_val_fscore_tau_010,
                "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
            }
            save_checkpoint(output_dir / f"step_{global_step:06d}.pth", readout, optimizer, global_step, config, best_loss, first_loss, final_loss, extra=extra)
            save_checkpoint(output_dir / "latest.pth", readout, optimizer, global_step, config, best_loss, first_loss, final_loss, extra=extra)
            try:
                ply_dir = output_dir / "ply"
                scene_id = sanitize_sample_id(str(batch["scene_ids"][0]))
                pred_path = ply_dir / f"{scene_id}_step{global_step:06d}_pred.ply"
                gt_path = ply_dir / f"{scene_id}_pseudo_gt.ply"
                visual_pred = pad_or_subsample_visual_points(pred.detach()[0], int(args.save_ply_queries), args.seed + 999 + global_step)
                visual_gt = get_targets(batch, "src_complete", max_points=int(args.save_ply_queries), norm_mode=args.target_norm_mode)[0]
                write_point_cloud_ply(pred_path, visual_pred)
                if not gt_path.exists():
                    write_point_cloud_ply(gt_path, visual_gt)
            except Exception as exc:
                warn = f"export warning step={global_step}: {type(exc).__name__}: {exc}"
                print(warn)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(warn + "\n")

    final_metrics = {
        "first_loss": first_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_val_chamfer_l2": best_val_chamfer_l2,
        "best_val_fscore_tau_0.10": best_val_fscore_tau_010,
        "best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
    }
    if last_val_metrics is not None:
        final_metrics["last_val_chamfer_l2"] = float(last_val_metrics["chamfer_l2"])
        for key in QUALITY_METRIC_KEYS:
            final_metrics[f"last_val_{key}"] = float(last_val_metrics[key])
    save_json(output_dir / "final_metrics.json", final_metrics)
    if swanlab_run:
        try:
            swanlab.log(
                {
                    "summary/first_loss": first_loss,
                    "summary/final_loss": final_loss,
                    "summary/best_loss": best_loss,
                    "summary/best_val_chamfer_l2": best_val_chamfer_l2,
                    "summary/best_val_fscore_tau_0.10": best_val_fscore_tau_010,
                    "summary/best_val_pred_to_gt_p90": best_val_pred_to_gt_p90,
                },
                step=global_step,
            )
            swanlab.finish()
        except Exception as exc:
            print(f"SwanLab finish warning: {type(exc).__name__}: {exc}")
    print(f"First loss: {first_loss}")
    print(f"Final loss: {final_loss}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
