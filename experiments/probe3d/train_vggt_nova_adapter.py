from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

from probe.adapter import (
    VGGTToNovaAdapter,
    VGGTToNovaCrossAttentionAdapter,
    VGGTToNovaSelfAttentionAdapter,
)
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

    def forward(
        self,
        selected_features,
        target_points,
        seed_offset: int,
        num_views: int,
        loss_type: str = "nova_flow",
        num_queries: int | None = None,
        fm_step_size: float | None = None,
        chamfer_weight: float = 0.1,
    ):
        tokens = self.adapter(selected_features)
        if loss_type == "nova_flow":
            loss, aux = nova_flow_matching_loss(
                self.decoder,
                tokens,
                target_points,
                seed=int(seed_offset),
                num_views=int(num_views),
            )
            return loss, tokens, aux["pred_velocity"], aux["target_velocity"]
        if loss_type == "chamfer_sample":
            if num_queries is None or fm_step_size is None:
                raise ValueError("chamfer_sample requires num_queries and fm_step_size")
            pred = sample_decoder(
                self.decoder,
                tokens,
                int(num_queries),
                float(fm_step_size),
                int(seed_offset),
                int(num_views),
            )
            loss = chamfer_l2(pred, target_points)
            return loss, tokens, pred, target_points
        if loss_type == "flow_chamfer_hybrid":
            if num_queries is None or fm_step_size is None:
                raise ValueError("flow_chamfer_hybrid requires num_queries and fm_step_size")
            flow_loss, aux = nova_flow_matching_loss(
                self.decoder,
                tokens,
                target_points,
                seed=int(seed_offset),
                num_views=int(num_views),
            )
            pred = sample_decoder(
                self.decoder,
                tokens,
                int(num_queries),
                float(fm_step_size),
                int(seed_offset),
                int(num_views),
            )
            cd_loss = chamfer_l2(pred, target_points)
            loss = flow_loss + float(chamfer_weight) * cd_loss
            return loss, tokens, pred, target_points
        raise ValueError(f"Unsupported loss_type={loss_type!r}")


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
    parser.add_argument("--adapter_type", default="mlp", choices=("mlp", "cross_attention", "self_attention"))
    parser.add_argument("--adapter_heads", type=int, default=8)
    parser.add_argument("--adapter_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--output_dir", default="experiments/probe3d/result/vggt23_nova_adapter_long_seed17")
    parser.add_argument("--debug_one_batch", action="store_true")
    parser.add_argument("--nova_ckpt", default=None)
    parser.add_argument("--vggt_weights", default=None, help="Optional local VGGT-1B model.pt path; avoids network fallback on Slurm nodes.")
    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter", "scannet"))
    parser.add_argument("--data_root", default=None, help="Dataset root for --dataset scannet, or adapter .pt path for --dataset scrream_adapter.")
    parser.add_argument("--train_split", default="train", help="Training split name for the selected dataset.")
    parser.add_argument("--val_split", default="test", help="Validation split name used during training.")
    parser.add_argument("--test_split", default="test", help="Final held-out evaluation split name.")
    parser.add_argument("--max_val_scenes", type=int, default=0, help="For ScanNet validation, restrict eval to the first N scenes after deterministic scene sorting; <=0 uses the full split.")
    parser.add_argument("--max_test_scenes", type=int, default=0, help="For ScanNet final test, restrict eval to the first N scenes after deterministic scene sorting; <=0 uses the full split.")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--query_source", default=None, help="Override NOVA decoder query_source for targets, e.g. src_view or src_complete.")
    parser.add_argument("--scannet_target_mode", default="complete_zpos", help="ScanNet complete-GT target support: complete_zpos, anchor_frustum, anchor_frustum_margin, covered_by_ge2, ...")
    parser.add_argument("--scannet_frustum_margin", type=float, default=1.0, help="Projection margin for anchor_frustum_margin target mode.")
    parser.add_argument("--scannet_min_views", type=int, default=2, help="Minimum covering views for covered_by_ge* target modes.")
    parser.add_argument("--scannet_complete_points", type=int, default=10000, help="Maximum complete GT points kept per ScanNet sample before query-source sampling/FPS.")
    parser.add_argument("--scannet_max_interval", type=int, default=1, help="Maximum processed-frame interval between selected ScanNet views; processed data already uses frame_skip=20, so keep this small for overlap-sensitive probes.")
    parser.add_argument("--num_queries", type=int, default=2048)
    parser.add_argument("--save_ply_queries", type=int, default=40960)
    parser.add_argument("--resume", default=None, help="Resume from a checkpoint, typically output_dir/latest.pth.")
    parser.add_argument("--parallel", action="store_true", help="Use torch.nn.DataParallel across visible CUDA devices.")
    parser.add_argument("--loss_type", default="nova_flow", choices=("nova_flow", "chamfer_sample", "flow_chamfer_hybrid"))
    parser.add_argument("--chamfer_weight", type=float, default=0.1, help="Chamfer coefficient for flow_chamfer_hybrid.")
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
    parser.add_argument("--swanlab", action="store_true", help="Enable SwanLab tracking.")
    parser.add_argument("--swanlab_project", default="PSUVPSC3DD")
    parser.add_argument("--swanlab_workspace", default=None)
    parser.add_argument("--swanlab_experiment", default=None)
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


def save_checkpoint(path, adapter, optimizer, step, config, meta, best_loss, first_loss, final_loss, extra=None):
    adapter_to_save = unwrap_adapter(adapter)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter": adapter_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "decoder_meta": meta,
        "best_loss": best_loss,
        "first_loss": first_loss,
        "final_loss": final_loss,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


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
            target = get_targets(
                batch,
                meta["query_source"],
                max_points=args.num_queries,
            )
            loss = chamfer_l2(pred, target)
            total += loss.item()
            count += 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total = float(stats[0].item())
        count = int(stats[1].item())
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
    # A single ScanNet VGGT layer-23 feature can be tens of MB. Keeping every
    # unique train/val sample in RAM will OOM long runs. Default to no in-memory
    # feature cache; opt in with FEATURE_MEMORY_CACHE_MAX=N, or -1 for unlimited.
    memory_cache_max = int(os.environ.get("FEATURE_MEMORY_CACHE_MAX", "0"))
    use_memory_cache = feature_cache is not None and memory_cache_max != 0

    selected_list = []
    for i, paths in enumerate(path_tuples):
        key = cache_key_from_paths(paths)
        selected_cpu = feature_cache.get(key) if use_memory_cache else None
        cache_path = cache_root / f"{key}.pt" if cache_root is not None else None
        if selected_cpu is None and cache_path is not None and cache_path.exists():
            selected_cpu = torch.load(cache_path, map_location="cpu")
        if selected_cpu is None:
            with torch.no_grad():
                features, _ = extract_vggt_features(vggt, images[i : i + 1], amp=amp)
                selected, _, _ = select_vggt_layer23(features)
            selected_cpu = selected.detach().cpu().to(torch.float16)
            if cache_path is not None:
                torch.save(selected_cpu, cache_path)
        if use_memory_cache:
            if memory_cache_max > 0:
                while len(feature_cache) >= memory_cache_max:
                    feature_cache.pop(next(iter(feature_cache)))
            feature_cache[key] = selected_cpu
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
        vggt = load_vggt(device, weights_path=args.vggt_weights)
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

        if args.adapter_type == "mlp":
            adapter = VGGTToNovaAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
            )
        elif args.adapter_type == "cross_attention":
            adapter = VGGTToNovaCrossAttentionAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                mlp_ratio=args.adapter_mlp_ratio,
            )
        elif args.adapter_type == "self_attention":
            adapter = VGGTToNovaSelfAttentionAdapter(
                input_dim=selected.shape[-1],
                output_dim=meta["token_dim"],
                output_tokens=meta["num_scene_tokens"],
                hidden_dim=args.adapter_hidden_dim,
                adapter_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                mlp_ratio=args.adapter_mlp_ratio,
            )
        else:
            raise ValueError(f"Unsupported adapter_type={args.adapter_type!r}")
        adapter = adapter.to(device)
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
                "adapter_type": args.adapter_type,
                "adapter_heads": args.adapter_heads,
                "adapter_mlp_ratio": args.adapter_mlp_ratio,
                "adapter_param_count": count_parameters(unwrap_adapter(probe_model)),
                "loss_type": args.loss_type,
                "chamfer_weight": args.chamfer_weight,
                "scannet_complete_points": args.scannet_complete_points,
                "scannet_max_interval": args.scannet_max_interval,
                "parallel": use_parallel,
                "ddp": dist_ctx["enabled"],
                "rank": dist_ctx["rank"],
                "world_size": dist_ctx["world_size"],
                "visible_cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
                "nova_decoder_meta": meta,
                "nova_ckpt": str(args.nova_ckpt) if args.nova_ckpt else None,
                "vggt_weights": str(args.vggt_weights) if args.vggt_weights else None,
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
        swanlab_run = maybe_init_swanlab(args, output_dir, config) if is_main else False

        best_loss = math.inf
        best_val_chamfer_l2 = math.inf
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
            best_val_chamfer_l2 = float(resume.get("best_val_chamfer_l2", best_val_chamfer_l2))
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
            target = get_targets(batch, meta["query_source"], max_points=args.num_queries, norm_mode=meta.get("norm_mode", "none"))

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                loss, tokens, pred_velocity, target_velocity = probe_model(
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
                if swanlab_run:
                    swanlab.log(
                        {
                            "train/loss": final_loss,
                            "train/best_loss": best_loss,
                            "train/grads_ok": float(grads_ok),
                        },
                        step=global_step,
                    )

            if global_step % args.val_every == 0:
                # Save before full validation as well: ScanNet validation can be
                # memory/driver fragile, and losing the just-finished step makes
                # debugging unnecessarily expensive.
                if is_main and (global_step % args.save_every == 0 or args.debug_one_batch):
                    save_checkpoint(output_dir / f"step_{global_step:06d}.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": best_val_chamfer_l2})
                    save_checkpoint(output_dir / "latest.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": best_val_chamfer_l2})
                barrier_if_distributed(dist_ctx["enabled"])
                val_loss = None
                if val_loader is not None:
                    val_loss = run_eval(
                        probe_model,
                        decoder,
                        val_loader,
                        device,
                        meta,
                        args,
                        max_batches=None if args.eval_batches <= 0 else args.eval_batches,
                    )
                if is_main and val_loss is not None:
                    save_json(output_dir / "validation_metrics.json", {"step": global_step, "val_chamfer_l2": val_loss})
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(f"validation step={global_step} val_chamfer_l2={val_loss:.8f}\n")
                    if val_loss < best_val_chamfer_l2:
                        best_val_chamfer_l2 = float(val_loss)
                        save_checkpoint(output_dir / "best.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": best_val_chamfer_l2})
                    if wandb_run is not None:
                        wandb_run.log({"val/chamfer_l2": float(val_loss), "val/best_chamfer_l2": float(best_val_chamfer_l2)}, step=global_step)
                    if swanlab_run:
                        swanlab.log({"val/chamfer_l2": float(val_loss), "val/best_chamfer_l2": float(best_val_chamfer_l2)}, step=global_step)
                barrier_if_distributed(dist_ctx["enabled"])
            if global_step % args.save_every == 0 or args.debug_one_batch:
                if is_main:
                    save_checkpoint(output_dir / f"step_{global_step:06d}.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": best_val_chamfer_l2})
                    save_checkpoint(output_dir / "latest.pth", probe_model, optimizer, global_step, config, meta, best_loss, first_loss, final_loss, extra={"best_val_chamfer_l2": best_val_chamfer_l2})
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
                    except Exception as exc:
                        warn = f"export warning step={global_step}: {type(exc).__name__}: {exc}"
                        print(warn)
                        with log_path.open("a", encoding="utf-8") as log:
                            log.write(warn + "\n")
                        if wandb_run is not None:
                            wandb_run.log({"export/failed": 1.0}, step=global_step)
                        if swanlab_run:
                            swanlab.log({"export/failed": 1.0}, step=global_step)
                barrier_if_distributed(dist_ctx["enabled"])

        test_loss = None
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
        if is_main:
            final_metrics = {"first_loss": first_loss, "final_loss": final_loss, "best_loss": best_loss, "best_val_chamfer_l2": best_val_chamfer_l2}
            if test_loss is not None:
                final_metrics["test_chamfer_l2"] = float(test_loss)
                save_json(output_dir / "test_metrics.json", {"step": global_step, "test_chamfer_l2": float(test_loss)})
                print(f"Final test chamfer_l2: {test_loss}")
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"final_test step={global_step} test_chamfer_l2={test_loss:.8f}\n")
                if wandb_run is not None:
                    wandb_run.summary["test_chamfer_l2"] = float(test_loss)
                    wandb_run.log({"test/chamfer_l2": float(test_loss)}, step=global_step)
                if swanlab_run:
                    swanlab.log({"test/chamfer_l2": float(test_loss)}, step=global_step)
            save_json(output_dir / "final_metrics.json", final_metrics)
            if wandb_run is not None:
                wandb_run.summary["first_loss"] = first_loss
                wandb_run.summary["final_loss"] = final_loss
                wandb_run.summary["best_loss"] = best_loss
                wandb_run.summary["best_val_chamfer_l2"] = best_val_chamfer_l2
                wandb_run.finish()
            if swanlab_run:
                swanlab.log(
                    {
                        "summary/first_loss": first_loss,
                        "summary/final_loss": final_loss,
                        "summary/best_loss": best_loss,
                        "summary/best_val_chamfer_l2": best_val_chamfer_l2,
                    },
                    step=global_step,
                )
                swanlab.finish()
            print(f"First loss: {first_loss}")
            print(f"Final loss: {final_loss}")
            print(f"Output directory: {output_dir}")
        barrier_if_distributed(dist_ctx["enabled"])
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
