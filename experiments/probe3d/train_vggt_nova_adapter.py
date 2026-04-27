from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from probe.adapter import VGGTToNovaAdapter
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
    move_batch_to_device,
    nova_flow_matching_loss,
    sample_decoder,
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
    parser.add_argument("--adapter_hidden_dim", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/probe3d/result/vggt23_nova_adapter_long_seed17")
    parser.add_argument("--debug_one_batch", action="store_true")
    parser.add_argument("--nova_ckpt", default=str(DEFAULT_NOVA_CKPT))
    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter", "scannet"))
    parser.add_argument("--data_root", default=None, help="Dataset root for --dataset scannet; default /data1/jcd_data/scannet_processed_large")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--num_queries", type=int, default=2048)
    parser.add_argument("--save_ply_queries", type=int, default=4096)
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
            images = images_from_batch(batch)
            selected = get_selected_features(
                run_eval.vggt, images, batch, run_eval.feature_cache, device, args.amp, args.feature_cache_dir
            )
            tokens = adapter_module(selected)
            pred = sample_decoder(decoder, tokens, args.num_queries, meta["fm_step_size"], args.seed + batch_idx, images.shape[1])
            target = get_targets(batch, meta["query_source"], max_points=args.num_queries)
            loss = chamfer_l2(pred, target)
            total += loss.item()
            count += 1
    adapter_module.train()
    return total / max(count, 1)


def cache_key_from_paths(paths):
    return hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()


def get_selected_features(vggt, images, batch, feature_cache, device, amp, feature_cache_dir=None):
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
    first_images = images_from_batch(first_batch)
    first_features, patch_start_idx = extract_vggt_features(vggt, first_images[:1], amp=args.amp)
    for i, feat in enumerate(first_features):
        print(f"VGGT feature {i}: shape={tuple(feat.shape)}")
    selected, selected_idx, reason = select_vggt_layer23(first_features)
    print(f"Selected VGGT feature index: {selected_idx}")
    print(f"Selection reason: {reason}")
    print(f"Selected VGGT feature shape: {tuple(selected.shape)}")

    adapter = VGGTToNovaAdapter(
        input_dim=selected.shape[-1],
        output_dim=meta["token_dim"],
        output_tokens=meta["num_scene_tokens"],
        hidden_dim=args.adapter_hidden_dim,
        adapter_layers=args.adapter_layers,
    ).to(device)
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
        final_loss = float(loss.item())
        if first_loss is None:
            first_loss = final_loss
        if final_loss < best_loss:
            best_loss = final_loss
            save_checkpoint(output_dir / "best.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss)
        save_checkpoint(output_dir / "latest.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss)

        line = f"step={global_step} {args.loss_type}_loss={final_loss:.8f} best={best_loss:.8f} grads_ok={grads_ok}"
        print(line)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line + "\n")

        if global_step % args.val_every == 0:
            val_loss = run_eval(probe_model, decoder, val_loader, device, meta, args)
            save_json(output_dir / "validation_metrics.json", {"step": global_step, "val_chamfer_l2": val_loss})
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"validation step={global_step} val_chamfer_l2={val_loss:.8f}\n")
        if global_step % args.save_every == 0 or args.debug_one_batch:
            save_checkpoint(output_dir / f"step_{global_step:06d}.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss)
            ply_pred = sample_decoder(decoder, tokens.detach(), args.save_ply_queries, meta["fm_step_size"], args.seed + 999 + global_step, images.shape[1])
            ply_target = get_targets(batch, meta["query_source"], max_points=args.save_ply_queries)
            scene_ids = scene_ids_from_batch(batch, global_step)
            write_point_cloud_ply(output_dir / "ply" / f"{scene_ids[0]}_step{global_step:06d}_pred.ply", ply_pred[0])
            write_point_cloud_ply(output_dir / "ply" / f"{scene_ids[0]}_step{global_step:06d}_gt.ply", ply_target[0])

    save_json(output_dir / "final_metrics.json", {"first_loss": first_loss, "final_loss": final_loss, "best_loss": best_loss})
    print(f"First loss: {first_loss}")
    print(f"Final loss: {final_loss}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
