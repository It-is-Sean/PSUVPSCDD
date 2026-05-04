from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb
except ImportError:
    wandb = None

from probe.adapter import VGGTToNovaSelfAttentionAdapter
from vggt_nova_adapter_common_raw import (
    DEFAULT_NOVA_CKPT,
    amp_context,
    assert_only_adapter_trainable,
    barrier_if_distributed,
    build_decoder,
    build_loader,
    chamfer_l2,
    cleanup_distributed,
    count_parameters,
    extract_vggt_features,
    get_targets,
    get_targets_cached,
    images_from_batch,
    init_distributed_mode,
    move_batch_to_device,
    nova_flow_matching_loss,
    reduce_scalar,
    sample_decoder,
    sample_keys_from_batch,
    sampler_set_epoch,
    save_json,
    scene_ids_from_batch,
    select_vggt_layer23,
    set_seed,
    trainable_parameter_names,
    write_point_cloud_ply,
    load_vggt,
    resolve_device,
)


class AdapterVelocityProbe(nn.Module):
    def __init__(self, adapter, decoder):
        super().__init__()
        self.adapter = adapter
        self.decoder = decoder

    def forward(self, selected_features, target_points, seed_offset: int, num_views: int):
        tokens = self.adapter(selected_features)
        loss, aux = nova_flow_matching_loss(
            self.decoder,
            tokens,
            target_points,
            seed=int(seed_offset),
            num_views=int(num_views),
        )
        return loss, tokens, aux["pred_velocity"], aux["target_velocity"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter_layers", type=int, default=2, )
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--attention_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/probe3d/result/vggt23_nova_self_attention_l4_scrream_raw_lr2e-5_seed17")
    parser.add_argument("--debug_one_batch", action="store_true")
    parser.add_argument("--nova_ckpt", default=None)
    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter", "scannet"))
    parser.add_argument("--data_root", default=None, help="Dataset root for --dataset scannet; default /data1/jcd_data/scannet_processed_large")
    parser.add_argument("--train_split", default="train", help="Training split name for the selected dataset.")
    parser.add_argument("--val_split", default="test", help="Validation split name used during training.")
    parser.add_argument("--test_split", default="test", help="Final held-out evaluation split name.")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--num_queries", type=int, default=2048)
    parser.add_argument("--save_ply_queries", type=int, default=40960)
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint, typically output_dir/latest.pth.")
    parser.add_argument("--parallel", action="store_true", help="Use torch.nn.DataParallel across visible CUDA devices.")
    parser.add_argument("--loss_type", default="nova_flow", choices=("nova_flow", "chamfer_sample"))
    parser.add_argument(
        "--image_root_map",
        default=None,
        help="Optional OLD=NEW path remap for SCRREAM frame_paths stored in the adapter dataset.",
    )
    parser.add_argument(
        "--feature_cache_dir",
        default=None,
        help="Optional directory for persistent VGGT layer-23 feature cache keyed by frame paths.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases tracking.")
    parser.add_argument("--wandb_project", default="PSUVPSC3DD")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--final_test", action="store_true", help="Run held-out test evaluation at the end of training.")
    parser.add_argument("--eval_batches", type=int, default=2, help="Validation batches per evaluation; <=0 means full split.")
    parser.add_argument("--test_eval_batches", type=int, default=0, help="Test batches for final evaluation; <=0 means full split.")
    return parser.parse_args()


def parse_image_root_map(value):
    if value is None:
        return None
    if "=" not in value:
        raise ValueError("--image_root_map must use OLD=NEW format")
    old, new = value.split("=", 1)
    if not old or not new:
        raise ValueError("--image_root_map must use non-empty OLD=NEW paths")
    return old.rstrip("/"), new.rstrip("/")


def unwrap_adapter(adapter):
    if isinstance(adapter, (nn.DataParallel, DDP)):
        return unwrap_adapter(adapter.module)
    if isinstance(adapter, AdapterVelocityProbe):
        return adapter.adapter
    return adapter


def save_checkpoint(path, adapter, optimizer, step, config, meta, best_loss, first_loss, final_loss):
    adapter_to_save = unwrap_adapter(adapter)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": adapter_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "decoder_meta": meta,
            "best_loss": best_loss,
            "first_loss": first_loss,
            "final_loss": final_loss,
        },
        path,
    )


def run_eval(adapter, decoder, loader, device, meta, args, max_batches=None):
    adapter_module = unwrap_adapter(adapter)
    adapter_module.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            images = images_from_batch(batch)
            selected = get_selected_features(
                run_eval.vggt, images, batch, run_eval.feature_cache, device, args.amp, args.feature_cache_dir
            )
            tokens = adapter_module(selected)
            pred = sample_decoder(decoder, tokens, args.num_queries, meta["fm_step_size"], args.seed + batch_idx, images.shape[1])
            target = get_targets_cached(
                batch,
                meta["query_source"],
                max_points=args.num_queries,
                cache=run_eval.target_cache,
            )
            loss = chamfer_l2(pred, target)
            total += loss.item()
            count += 1
    adapter_module.train()
    return total / max(count, 1)


def cache_key_from_paths(paths):
    return hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()


def get_selected_features(vggt, images, batch, feature_cache, device, amp, feature_cache_dir=None):
    batch_size = images.shape[0]
    path_tuples = sample_keys_from_batch(batch)
    if len(path_tuples) != batch_size:
        raise ValueError(f"Feature-cache key count mismatch: got {len(path_tuples)} keys for batch_size={batch_size}")

    cache_root = Path(feature_cache_dir) if feature_cache_dir else None
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    selected_list = []
    for i, paths in enumerate(path_tuples):
        key = cache_key_from_paths(paths)
        selected_cpu = feature_cache.get(key)
        cache_path = cache_root / f"{key}.pt" if cache_root is not None else None
        if selected_cpu is None and cache_path is not None and cache_path.exists():
            selected_cpu = torch.load(cache_path, map_location="cpu")
            feature_cache[key] = selected_cpu
        if selected_cpu is None:
            with torch.no_grad():
                features, _ = extract_vggt_features(vggt, images[i : i + 1], amp=amp)
                selected, _, _ = select_vggt_layer23(features)
            selected_cpu = selected.detach().cpu().to(torch.float16)
            feature_cache[key] = selected_cpu
            if cache_path is not None:
                torch.save(selected_cpu, cache_path)
        selected_list.append(selected_cpu.to(device))

    return torch.cat(selected_list, dim=0).contiguous()



def maybe_init_wandb(args, output_dir: Path, config: dict):
    if not args.wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install it in the active environment before using --wandb.")
    try:
        return wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or output_dir.name,
            dir=str(output_dir),
            config=config,
            reinit=True,
        )
    except Exception as exc:
        print(f"wandb warning: init failed, continuing without wandb: {type(exc).__name__}: {exc}")
        return None


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
        image_root_map = parse_image_root_map(args.image_root_map)
        train_loader, data_args = build_loader(
            cfg, args.batch_size, args.num_workers, test=False, image_root_map=image_root_map,
            dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
            split_override=args.train_split, distributed=dist_ctx["enabled"], rank=dist_ctx["rank"], world_size=dist_ctx["world_size"],
        )
        val_loader = None
        test_loader = None
        if is_main:
            val_loader, _ = build_loader(
                cfg, args.batch_size, max(0, min(args.num_workers, 2)), test=True, image_root_map=image_root_map,
                dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
                split_override=args.val_split,
            )
            if args.final_test:
                test_loader, _ = build_loader(
                    cfg, args.batch_size, max(0, min(args.num_workers, 2)), test=True, image_root_map=image_root_map,
                    dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views,
                    split_override=args.test_split,
                )
        vggt = load_vggt(device)
        run_eval.vggt = vggt
        feature_cache = {}
        run_eval.feature_cache = feature_cache
        run_eval.target_cache = {}

        sampler_set_epoch(train_loader, 0)
        first_batch = next(iter(train_loader))
        first_batch = move_batch_to_device(first_batch, device)
        first_images = images_from_batch(first_batch)
        first_features, patch_start_idx = extract_vggt_features(vggt, first_images[:1], amp=args.amp)
        if is_main:
            for i, feat in enumerate(first_features):
                print(f"VGGT feature {i}: shape={tuple(feat.shape)}")
        selected, selected_idx, reason = select_vggt_layer23(first_features)
        if is_main:
            print(f"Selected VGGT feature index: {selected_idx}")
            print(f"Selection reason: {reason}")
            print(f"Selected VGGT feature shape: {tuple(selected.shape)}")

        adapter = VGGTToNovaSelfAttentionAdapter(
            input_dim=selected.shape[-1],
            output_dim=meta["token_dim"],
            output_tokens=meta["num_scene_tokens"],
            hidden_dim=args.adapter_hidden_dim,
            adapter_layers=args.adapter_layers,
            num_heads=args.attention_heads,
            mlp_ratio=args.attention_mlp_ratio,
        ).to(device)
        probe_model = AdapterVelocityProbe(adapter, decoder).to(device)
        use_parallel = False
        if dist_ctx["enabled"]:
            ddp_device_ids = [dist_ctx["local_rank"]] if device.type == "cuda" else None
            probe_model = DDP(probe_model, device_ids=ddp_device_ids, output_device=dist_ctx["local_rank"] if device.type == "cuda" else None)
            if is_main:
                print(f"Using DDP on world_size={dist_ctx['world_size']} (rank={dist_ctx['rank']}, local_rank={dist_ctx['local_rank']}).")
        else:
            use_parallel = bool(args.parallel and device.type == "cuda" and torch.cuda.device_count() > 1)
            if use_parallel:
                probe_model = nn.DataParallel(probe_model)
                if is_main:
                    print(f"Using DataParallel on {torch.cuda.device_count()} visible CUDA devices.")
            else:
                if is_main:
                    print(f"Using single-device training on {device}.")
        optimizer = torch.optim.AdamW(unwrap_adapter(probe_model).parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
        assert_only_adapter_trainable(unwrap_adapter(probe_model), vggt, decoder)

        config = vars(args).copy()
        config.update(
            {
                "selected_vggt_feature_index": selected_idx,
                "selected_vggt_feature_shape": list(selected.shape),
                "selection_reason": reason,
                "patch_start_idx": patch_start_idx,
                "adapter_type": "self_attention",
                "attention_heads": args.attention_heads,
                "attention_mlp_ratio": args.attention_mlp_ratio,
                "adapter_param_count": count_parameters(unwrap_adapter(probe_model)),
                "loss_type": args.loss_type,
                "parallel": use_parallel,
                "ddp": dist_ctx["enabled"],
                "rank": dist_ctx["rank"],
                "world_size": dist_ctx["world_size"],
                "visible_cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
                "nova_decoder_meta": meta,
                "dataset": {"data_root": data_args.data_root, "test_dataset_name": data_args.test_dataset_name},
                "image_root_map": args.image_root_map,
            }
        )
        if is_main:
            save_json(output_dir / "config.json", config)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(config, indent=2) + "\n")
                log.write("Trainable parameter names:\n")
                for name in trainable_parameter_names({"adapter": unwrap_adapter(probe_model), "vggt": vggt, "decoder": decoder}):
                    log.write(f"{name}\n")
                    print(f"trainable: {name}")

        wandb_run = maybe_init_wandb(args, output_dir, config) if is_main else None

        best_loss = math.inf
        first_loss = None
        final_loss = None
        global_step = 0
        resume_path = Path(args.resume) if args.resume else None
        if resume_path is not None and resume_path.exists():
            resume = torch.load(resume_path, map_location="cpu")
            unwrap_adapter(probe_model).load_state_dict(resume["adapter"])
            if "optimizer" in resume:
                optimizer.load_state_dict(resume["optimizer"])
            global_step = int(resume.get("step", 0))
            best_loss = float(resume.get("best_loss", best_loss))
            first_loss = resume.get("first_loss", first_loss)
            final_loss = resume.get("final_loss", final_loss)
            if is_main:
                print(f"Resumed from {resume_path} at step={global_step} best={best_loss}")
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"resumed_from={resume_path} step={global_step} best={best_loss}\n")
        train_epoch = 0
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
            selected = get_selected_features(vggt, images, batch, feature_cache, device, args.amp, args.feature_cache_dir)
            target = get_targets(batch, meta["query_source"], max_points=args.num_queries)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                if args.loss_type == "nova_flow":
                    loss, tokens, pred_velocity, target_velocity = probe_model(
                        selected,
                        target,
                        args.seed + global_step,
                        images.shape[1],
                    )
                    if loss.ndim > 0:
                        loss = loss.mean()
                else:
                    tokens = unwrap_adapter(probe_model)(selected)
                    pred = sample_decoder(
                        decoder, tokens, args.num_queries, meta["fm_step_size"], args.seed + global_step, images.shape[1]
                    )
                    loss = chamfer_l2(pred, target)
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
                if is_main:
                    save_checkpoint(output_dir / "best.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss)
            if is_main:
                save_checkpoint(output_dir / "latest.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss)

            if is_main:
                line = f"step={global_step} {args.loss_type}_loss={final_loss:.8f} best={best_loss:.8f} grads_ok={grads_ok}"
                print(line)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(line + "\n")
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": final_loss,
                            "train/best_loss": best_loss,
                            "train/grads_ok": float(grads_ok),
                        },
                        step=global_step,
                    )

            if global_step % args.val_every == 0:
                if is_main and val_loader is not None:
                    val_loss = run_eval(
                        probe_model,
                        decoder,
                        val_loader,
                        device,
                        meta,
                        args,
                        max_batches=None if args.eval_batches <= 0 else args.eval_batches,
                    )
                    save_json(output_dir / "validation_metrics.json", {"step": global_step, "val_chamfer_l2": val_loss})
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(f"validation step={global_step} val_chamfer_l2={val_loss:.8f}\n")
                    if wandb_run is not None:
                        wandb_run.log({"val/chamfer_l2": float(val_loss)}, step=global_step)
                barrier_if_distributed(dist_ctx["enabled"])
            if global_step % args.save_every == 0 or args.debug_one_batch:
                if is_main:
                    save_checkpoint(output_dir / f"step_{global_step:06d}.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss)
                    try:
                        scene_ids = scene_ids_from_batch(batch, global_step)
                        ply_dir = output_dir / "ply"
                        pred_path = ply_dir / f"{scene_ids[0]}_step{global_step:06d}_pred.ply"
                        gt_path = ply_dir / f"{scene_ids[0]}_pseudo_gt.ply"
                        ply_pred = sample_decoder(decoder, tokens.detach(), args.save_ply_queries, meta["fm_step_size"], args.seed + 999 + global_step, images.shape[1])
                        write_point_cloud_ply(pred_path, ply_pred[0])
                        if not gt_path.exists():
                            ply_target = get_targets(batch, meta["query_source"], max_points=args.save_ply_queries)
                            write_point_cloud_ply(gt_path, ply_target[0])
                    except Exception as exc:
                        warn = f"export warning step={global_step}: {type(exc).__name__}: {exc}"
                        print(warn)
                        with log_path.open("a", encoding="utf-8") as log:
                            log.write(warn + "\n")
                        if wandb_run is not None:
                            wandb_run.log({"export/failed": 1.0}, step=global_step)
                barrier_if_distributed(dist_ctx["enabled"])

        if is_main:
            final_metrics = {"first_loss": first_loss, "final_loss": final_loss, "best_loss": best_loss}
            if test_loader is not None:
                test_loss = run_eval(
                    probe_model,
                    decoder,
                    test_loader,
                    device,
                    meta,
                    args,
                    max_batches=None if args.test_eval_batches <= 0 else args.test_eval_batches,
                )
                final_metrics["test_chamfer_l2"] = float(test_loss)
                save_json(output_dir / "test_metrics.json", {"step": global_step, "test_chamfer_l2": float(test_loss)})
                print(f"Final test chamfer_l2: {test_loss}")
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"final_test step={global_step} test_chamfer_l2={test_loss:.8f}\n")
                if wandb_run is not None:
                    wandb_run.summary["test_chamfer_l2"] = float(test_loss)
                    wandb_run.log({"test/chamfer_l2": float(test_loss)}, step=global_step)
            save_json(output_dir / "final_metrics.json", final_metrics)
            if wandb_run is not None:
                wandb_run.summary["first_loss"] = first_loss
                wandb_run.summary["final_loss"] = final_loss
                wandb_run.summary["best_loss"] = best_loss
                wandb_run.finish()
            print(f"First loss: {first_loss}")
            print(f"Final loss: {final_loss}")
            print(f"Output directory: {output_dir}")
        barrier_if_distributed(dist_ctx["enabled"])
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
