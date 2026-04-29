from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

try:
    import wandb
except ImportError:
    wandb = None

from probe.adapter import VGGTToNovaAttentionAdapter, VGGTToNovaSelfAttentionAdapter
from vggt_nova_adapter_common import (
    DEFAULT_NOVA_CKPT,
    amp_context,
    assert_only_adapter_trainable,
    build_decoder,
    build_loader,
    chamfer_l2,
    count_parameters,
    extract_vggt_features,
    get_targets,
    images_from_batch,
    load_vggt,
    move_batch_to_device,
    nova_flow_matching_loss,
    resolve_device,
    sample_decoder,
    save_json,
    scene_ids_from_batch,
    select_vggt_layer23,
    set_seed,
    trainable_parameter_names,
    write_point_cloud_ply,
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
    parser.add_argument("--adapter_type", default="self_attention", choices=("cross_attention", "self_attention"))
    parser.add_argument("--adapter_layers", type=int, default=4)
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--attention_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/probe3d/result/vggt23_nova_self_attention_l4_scrream_seed17")
    parser.add_argument("--debug_one_batch", action="store_true")
    parser.add_argument("--nova_ckpt", default=None)
    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter", "scannet"))
    parser.add_argument("--data_root", default=None, help="Dataset root for --dataset scannet; default /data1/jcd_data/scannet_processed_large")
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
    if isinstance(adapter, nn.DataParallel):
        return adapter.module.adapter
    if isinstance(adapter, AdapterVelocityProbe):
        return adapter.adapter
    return adapter


def save_checkpoint(path, adapter, optimizer, step, config, meta, best_loss, best_val_loss, first_loss, final_loss):
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
            "best_val_loss": best_val_loss,
            "first_loss": first_loss,
            "final_loss": final_loss,
        },
        path,
    )


def run_eval(adapter, decoder, loader, device, meta, args, max_batches=2):
    adapter_module = unwrap_adapter(adapter)
    adapter_module.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = move_batch_to_device(batch, device)
            images = images_from_batch(batch) if "images" in batch else None
            batch_num_views = num_views_from_batch(batch, args.num_views)
            selected = get_selected_features(
                run_eval.vggt,
                images,
                batch,
                run_eval.feature_cache,
                device,
                args.amp,
                args.feature_cache_dir,
                prefer_precomputed_features=getattr(run_eval, "prefer_precomputed_features", True),
            )
            tokens = adapter_module(selected)
            pred = sample_decoder(decoder, tokens, args.num_queries, meta["fm_step_size"], args.seed + batch_idx, batch_num_views)
            target = get_targets(batch, meta["query_source"], max_points=args.num_queries)
            loss = chamfer_l2(pred, target)
            total += loss.item()
            count += 1
    adapter_module.train()
    return total / max(count, 1)


def cache_key_from_paths(paths):
    return hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()


def num_views_from_batch(batch, default_num_views: int) -> int:
    if isinstance(batch, dict):
        if "images" in batch and batch["images"].ndim >= 2:
            return int(batch["images"].shape[1])
        if "frame_paths" in batch and batch["frame_paths"]:
            return len(batch["frame_paths"][0])
    return int(default_num_views)


def get_selected_features(vggt, images, batch, feature_cache, device, amp, feature_cache_dir=None, prefer_precomputed_features=True):
    if prefer_precomputed_features and isinstance(batch, dict) and "features" in batch:
        return batch["features"].to(device).contiguous()

    batch_size = images.shape[0]
    if isinstance(batch, dict) and "frame_paths" in batch:
        path_tuples = [tuple(paths) for paths in batch["frame_paths"]]
    else:
        path_tuples = [(str(i),) for i in range(batch_size)]

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


def inspect_resume_checkpoint(resume_path: Path | None):
    if resume_path is None or not resume_path.exists():
        return None, {"input_dim": None, "gated_cross_attention": None}
    payload = torch.load(resume_path, map_location="cpu")
    adapter_state = payload.get("adapter", {}) if isinstance(payload, dict) else {}
    input_proj_weight = adapter_state.get("input_proj.weight")
    input_dim = int(input_proj_weight.shape[1]) if input_proj_weight is not None else None
    gated_cross_attention = any("attn_gate" in key for key in adapter_state.keys())
    return payload, {"input_dim": input_dim, "gated_cross_attention": gated_cross_attention}



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
    args = parse_args()
    if args.debug_one_batch:
        args.max_steps = 1
        args.save_every = 1
        args.val_every = 1
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    decoder, meta, cfg = build_decoder(device, args.nova_ckpt)
    image_root_map = parse_image_root_map(args.image_root_map)
    resume_path = Path(args.resume) if args.resume else None
    resume_payload, resume_info = inspect_resume_checkpoint(resume_path)
    train_loader, data_args = build_loader(
        cfg, args.batch_size, args.num_workers, test=False, image_root_map=image_root_map,
        dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views
    )
    val_loader, _ = build_loader(
        cfg, args.batch_size, max(0, min(args.num_workers, 2)), test=True, image_root_map=image_root_map,
        dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views
    )
    vggt = load_vggt(device)
    run_eval.vggt = vggt
    feature_cache = {}
    run_eval.feature_cache = feature_cache

    first_batch = next(iter(train_loader))
    first_batch = move_batch_to_device(first_batch, device)
    use_precomputed_features = bool("features" in first_batch)
    if use_precomputed_features and resume_info["input_dim"] is not None:
        use_precomputed_features = int(first_batch["features"].shape[-1]) == int(resume_info["input_dim"])
    run_eval.prefer_precomputed_features = use_precomputed_features

    if use_precomputed_features:
        selected = first_batch["features"][:1]
        selected_idx = 22
        reason = "Using precomputed adapter-dataset features directly; treating them as the selected VGGT feature stream."
        patch_start_idx = None
    else:
        first_images = images_from_batch(first_batch)
        first_features, patch_start_idx = extract_vggt_features(vggt, first_images[:1], amp=args.amp)
        for i, feat in enumerate(first_features):
            print(f"VGGT feature {i}: shape={tuple(feat.shape)}")
        selected, selected_idx, reason = select_vggt_layer23(first_features)
        if resume_info["input_dim"] is not None:
            reason += f" Resume checkpoint expects adapter input dim {resume_info['input_dim']}, so raw VGGT features are used instead of precomputed 128-d features."
    print(f"Selected VGGT feature index: {selected_idx}")
    print(f"Selection reason: {reason}")
    print(f"Selected VGGT feature shape: {tuple(selected.shape)}")

    adapter_cls = {
        "cross_attention": VGGTToNovaAttentionAdapter,
        "self_attention": VGGTToNovaSelfAttentionAdapter,
    }[args.adapter_type]
    adapter_kwargs = dict(
        input_dim=selected.shape[-1],
        output_dim=meta["token_dim"],
        output_tokens=meta["num_scene_tokens"],
        hidden_dim=args.adapter_hidden_dim,
        adapter_layers=args.adapter_layers,
        num_heads=args.attention_heads,
        mlp_ratio=args.attention_mlp_ratio,
    )
    if args.adapter_type == "cross_attention":
        adapter_kwargs["gated"] = resume_info["gated_cross_attention"] if resume_info["input_dim"] is not None else True
    adapter = adapter_cls(**adapter_kwargs).to(device)
    probe_model = AdapterVelocityProbe(adapter, decoder).to(device)
    use_parallel = bool(args.parallel and device.type == "cuda" and torch.cuda.device_count() > 1)
    if use_parallel:
        probe_model = nn.DataParallel(probe_model)
        print(f"Using DataParallel on {torch.cuda.device_count()} visible CUDA devices.")
    else:
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
            "adapter_type": args.adapter_type,
            "attention_heads": args.attention_heads,
            "attention_mlp_ratio": args.attention_mlp_ratio,
            "attention_gated": getattr(unwrap_adapter(probe_model), "gated", None),
            "use_precomputed_features": use_precomputed_features,
            "resume_expected_input_dim": resume_info["input_dim"],
            "adapter_param_count": count_parameters(unwrap_adapter(probe_model)),
            "loss_type": args.loss_type,
            "parallel": use_parallel,
            "visible_cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            "nova_decoder_meta": meta,
            "dataset": {"data_root": data_args.data_root, "test_dataset_name": data_args.test_dataset_name},
            "image_root_map": args.image_root_map,
        }
    )
    save_json(output_dir / "config.json", config)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(config, indent=2) + "\n")
        log.write("Trainable parameter names:\n")
        for name in trainable_parameter_names({"adapter": unwrap_adapter(probe_model), "vggt": vggt, "decoder": decoder}):
            log.write(f"{name}\n")
            print(f"trainable: {name}")

    wandb_run = maybe_init_wandb(args, output_dir, config)

    best_loss = math.inf
    best_val_loss = math.inf
    first_loss = None
    final_loss = None
    global_step = 0
    if resume_path is not None and resume_path.exists():
        resume = resume_payload if resume_payload is not None else torch.load(resume_path, map_location="cpu")
        unwrap_adapter(probe_model).load_state_dict(resume["adapter"])
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        global_step = int(resume.get("step", 0))
        best_loss = float(resume.get("best_loss", best_loss))
        best_val_loss = float(resume.get("best_val_loss", best_val_loss))
        first_loss = resume.get("first_loss", first_loss)
        final_loss = resume.get("final_loss", final_loss)
        print(f"Resumed from {resume_path} at step={global_step} best={best_loss}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"resumed_from={resume_path} step={global_step} best={best_loss}\n")
    data_iter = iter(train_loader)

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = move_batch_to_device(batch, device)
        images = images_from_batch(batch) if "images" in batch else None
        batch_num_views = num_views_from_batch(batch, args.num_views)
        selected = get_selected_features(
            vggt,
            images,
            batch,
            feature_cache,
            device,
            args.amp,
            args.feature_cache_dir,
            prefer_precomputed_features=use_precomputed_features,
        )
        target = get_targets(batch, meta["query_source"], max_points=args.num_queries)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, args.amp):
            if args.loss_type == "nova_flow":
                loss, tokens, pred_velocity, target_velocity = probe_model(
                    selected,
                    target,
                    args.seed + global_step,
                    batch_num_views,
                )
                if loss.ndim > 0:
                    loss = loss.mean()
            else:
                tokens = unwrap_adapter(probe_model)(selected)
                pred = sample_decoder(
                    decoder, tokens, args.num_queries, meta["fm_step_size"], args.seed + global_step, batch_num_views
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
        final_loss = float(loss.item())
        if first_loss is None:
            first_loss = final_loss
        if final_loss < best_loss:
            best_loss = final_loss
            save_checkpoint(output_dir / "best.pth", probe_model, optimizer, global_step, config, meta, best_loss, best_val_loss, first_loss, final_loss)
        save_checkpoint(output_dir / "latest.pth", probe_model, optimizer, global_step, config, meta, best_loss, best_val_loss, first_loss, final_loss)

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
            val_loss = run_eval(probe_model, decoder, val_loader, device, meta, args)
            best_val_loss = min(best_val_loss, float(val_loss))
            save_json(output_dir / "validation_metrics.json", {"step": global_step, "val_chamfer_l2": val_loss, "best_val_chamfer_l2": best_val_loss})
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"validation step={global_step} val_chamfer_l2={val_loss:.8f}\n")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/chamfer_l2": float(val_loss),
                        "val/best_chamfer_l2": best_val_loss,
                    },
                    step=global_step,
                )
        if global_step % args.save_every == 0 or args.debug_one_batch:
            save_checkpoint(output_dir / f"step_{global_step:06d}.pth", probe_model, optimizer, global_step, config, meta, best_loss, best_val_loss, first_loss, final_loss)
            try:
                ply_pred = sample_decoder(decoder, tokens.detach(), args.save_ply_queries, meta["fm_step_size"], args.seed + 999 + global_step, batch_num_views)
                ply_target = get_targets(batch, meta["query_source"], max_points=args.save_ply_queries)
                scene_ids = scene_ids_from_batch(batch, global_step)
                ply_dir = output_dir / "ply"
                pred_path = ply_dir / f"{scene_ids[0]}_step{global_step:06d}_pred.ply"
                gt_path = ply_dir / f"{scene_ids[0]}_pseudo_gt.ply"
                write_point_cloud_ply(pred_path, ply_pred[0])
                if not gt_path.exists():
                    write_point_cloud_ply(gt_path, ply_target[0])
            except Exception as exc:
                warn = f"export warning step={global_step}: {type(exc).__name__}: {exc}"
                print(warn)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(warn + "\n")
                if wandb_run is not None:
                    wandb_run.log({"export/failed": 1.0}, step=global_step)

    save_json(
        output_dir / "final_metrics.json",
        {
            "first_loss": first_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "best_val_chamfer_l2": best_val_loss if math.isfinite(best_val_loss) else None,
        },
    )
    if wandb_run is not None:
        wandb_run.summary["first_loss"] = first_loss
        wandb_run.summary["final_loss"] = final_loss
        wandb_run.summary["best_loss"] = best_loss
        if math.isfinite(best_val_loss):
            wandb_run.summary["best_val_chamfer_l2"] = best_val_loss
        wandb_run.finish()
    print(f"First loss: {first_loss}")
    print(f"Final loss: {final_loss}")
    if math.isfinite(best_val_loss):
        print(f"Best val Chamfer L2: {best_val_loss}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
