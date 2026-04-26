import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", default="experiments/probe3d/adapter_data")
    parser.add_argument("--manifest_name", default="scrream_adapter_manifest.json")
    parser.add_argument("--dataset_name", default="scrream_adapter_dataset.pt")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--pad_short_scenes", action="store_true")
    parser.add_argument("--pseudo_gt_views", type=int, default=2)
    parser.add_argument("--pseudo_gt_queries", type=int, default=20000)
    parser.add_argument("--max_target_points", type=int, default=20000)
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--feature_ckpt", default="/home/wdh/nova3r/checkpoints/scene_n2/checkpoint-last.pth")
    parser.add_argument("--pseudo_gt_ckpt", default="/home/wdh/nova3r/checkpoints/scene_n2/checkpoint-last.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_failures", action="store_true")
    return parser.parse_args()


def discover_scenes(data_root: str):
    scenes = []
    root = Path(data_root)
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        seq_dirs = [item for item in sorted(scene_dir.iterdir()) if item.is_dir()]
        if not seq_dirs:
            continue
        seq_dir = seq_dirs[0]
        rgb_dir = seq_dir / "rgb"
        ldi_dir = seq_dir / "ldi"
        if not rgb_dir.is_dir() or not ldi_dir.is_dir():
            continue

        rgb_ids = {path.stem for path in rgb_dir.glob("*.png")}
        ldi_ids = {path.name.replace("_ldi.npz", "") for path in ldi_dir.glob("*_ldi.npz")}
        shared_ids = sorted(rgb_ids.intersection(ldi_ids))
        frame_ids = [int(frame_id) for frame_id in shared_ids]
        scenes.append(
            {
                "scene_id": scene_dir.name,
                "sequence_id": seq_dir.name,
                "sequence_dir": str(seq_dir),
                "rgb_dir": str(rgb_dir),
                "ldi_dir": str(ldi_dir),
                "frame_ids": frame_ids,
            }
        )
    if not scenes:
        raise RuntimeError(f"No SCRREAM scenes found under {data_root}")
    return scenes


def split_scenes(scene_ids, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be positive")

    shuffled = list(scene_ids)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count < 3:
        raise ValueError("Need at least 3 scenes to create train/val/test splits")

    val_count = max(1, int(round(count * (val_ratio / total_ratio)))) if val_ratio > 0 else 0
    test_count = max(1, int(round(count * (test_ratio / total_ratio)))) if test_ratio > 0 else 0
    train_count = count - val_count - test_count
    if train_count <= 0:
        train_count = 1
        overflow = val_count + test_count - (count - train_count)
        while overflow > 0 and test_count > 1:
            test_count -= 1
            overflow -= 1
        while overflow > 0 and val_count > 1:
            val_count -= 1
            overflow -= 1
        if overflow > 0:
            raise ValueError("Not enough scenes to satisfy the requested split ratios")

    train_ids = sorted(shuffled[:train_count])
    val_ids = sorted(shuffled[train_count : train_count + val_count])
    test_ids = sorted(shuffled[train_count + val_count : train_count + val_count + test_count])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def build_frame_groups(frame_ids, group_size: int, stride: int, pad_short_scenes: bool):
    if not frame_ids:
        return []
    if len(frame_ids) < group_size:
        if not pad_short_scenes:
            return []
        padded = list(frame_ids) + [frame_ids[-1]] * (group_size - len(frame_ids))
        return [padded]
    return [frame_ids[idx : idx + group_size] for idx in range(0, len(frame_ids) - group_size + 1, stride)]


def build_manifest(scenes, args):
    split_map = split_scenes(
        [scene["scene_id"] for scene in scenes],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    split_by_scene = {}
    for split_name, scene_ids in split_map.items():
        for scene_id in scene_ids:
            split_by_scene[scene_id] = split_name

    samples = []
    for scene in scenes:
        groups = build_frame_groups(
            scene["frame_ids"],
            group_size=args.group_size,
            stride=args.sample_stride,
            pad_short_scenes=args.pad_short_scenes,
        )
        split_name = split_by_scene[scene["scene_id"]]
        for sample_idx, group in enumerate(groups):
            frame_paths = [str(Path(scene["rgb_dir"]) / f"{frame_id:06d}.png") for frame_id in group]
            samples.append(
                {
                    "sample_id": f"{scene['scene_id']}__{sample_idx:04d}",
                    "split": split_name,
                    "scene_id": scene["scene_id"],
                    "sequence_id": scene["sequence_id"],
                    "frame_ids": group,
                    "frame_paths": frame_paths,
                    "pseudo_gt_frame_ids": group[: args.pseudo_gt_views],
                    "pseudo_gt_frame_paths": frame_paths[: args.pseudo_gt_views],
                }
            )

    split_counts = {}
    for split_name in ("train", "val", "test"):
        split_counts[split_name] = sum(1 for sample in samples if sample["split"] == split_name)

    manifest = {
        "version": 1,
        "dataset": "SCRREAM",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "group_size": args.group_size,
            "sample_stride": args.sample_stride,
            "split_seed": args.split_seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "pad_short_scenes": args.pad_short_scenes,
            "pseudo_gt_views": args.pseudo_gt_views,
            "resolution": args.resolution,
        },
        "scene_splits": split_map,
        "sample_counts": split_counts,
        "samples": samples,
    }
    return manifest


def build_image_transform():
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_nova3r_model(ckpt_path, device):
    from demo_nova3r import load_model

    return load_model(ckpt_path, device)


def load_views(frame_paths, resolution, image_transform):
    import numpy as np
    import PIL.Image

    width, height = resolution
    views = []
    for view_idx, frame_path in enumerate(frame_paths):
        image = PIL.Image.open(frame_path).convert("RGB")
        image = image.resize((width, height), PIL.Image.LANCZOS)
        views.append(
            {
                "img": image_transform(image)[None],
                "true_shape": np.int32([height, width]),
                "idx": view_idx,
                "instance": os.path.basename(frame_path),
                "view_label": f"input_{view_idx}",
            }
        )
    return views


def stack_sample_images(views, device):
    image_tensors = [view["img"].squeeze(0) for view in views]
    return torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)


def extract_features(model, device, frame_paths, resolution, image_transform):
    import torch

    views = load_views(frame_paths, resolution, image_transform)
    images = stack_sample_images(views, device)
    with torch.no_grad():
        encoded = model._encode(images=images, test=True)
    return encoded["tokens"][0].detach().float().cpu()


def build_inference_cfg(cfg):
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    if "fm_step_size" not in cfg:
        cfg.fm_step_size = 0.04
    if "fm_sampling" not in cfg:
        cfg.fm_sampling = "euler"
    return cfg


def generate_pseudo_gt(model, cfg, device, frame_paths, resolution, image_transform, num_queries, n_views):
    from dust3r.image_pairs import make_pairs
    from nova3r.inference import inference_nova3r

    views = load_views(frame_paths[:n_views], resolution, image_transform)
    symmetrize = len(views) == 1
    pairs = make_pairs(views, scene_graph="complete", prefilter=None, symmetrize=symmetrize)
    with torch.no_grad():
        output = inference_nova3r(
            cfg,
            pairs,
            model,
            device,
            batch_size=1,
            verbose=False,
            num_queries=num_queries,
            n_views=min(n_views, len(views)),
            method=cfg.get("fm_sampling", "euler"),
        )
    return output["pred"]["pts3d_xyz"][0].detach().float().cpu()


def save_manifest(path: str, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved manifest to {path}")


def main():
    args = parse_args()
    scenes = discover_scenes(args.data_root)
    manifest = build_manifest(scenes, args)

    output_root = os.path.abspath(args.output_root)
    manifest_path = os.path.join(output_root, args.manifest_name)
    dataset_path = os.path.join(output_root, args.dataset_name)
    save_manifest(manifest_path, manifest)

    if args.manifest_only:
        return

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    image_transform = build_image_transform()
    feature_model, feature_cfg = load_nova3r_model(args.feature_ckpt, args.device)
    pseudo_gt_model = feature_model
    pseudo_gt_cfg = None
    if os.path.abspath(args.pseudo_gt_ckpt) != os.path.abspath(args.feature_ckpt):
        pseudo_gt_model, pseudo_gt_cfg = load_nova3r_model(args.pseudo_gt_ckpt, args.device)
    feature_cfg = build_inference_cfg(feature_cfg)
    if pseudo_gt_cfg is None:
        pseudo_gt_cfg = feature_cfg
    else:
        pseudo_gt_cfg = build_inference_cfg(pseudo_gt_cfg)

    features = []
    targets = []
    sample_ids = []
    split_labels = []
    sample_metadata = []
    failures = []

    samples = manifest["samples"]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    for sample in samples:
        try:
            feature_tensor = extract_features(
                feature_model,
                args.device,
                sample["frame_paths"],
                args.resolution,
                image_transform,
            )
            target_tensor = generate_pseudo_gt(
                pseudo_gt_model,
                pseudo_gt_cfg,
                args.device,
                sample["pseudo_gt_frame_paths"],
                args.resolution,
                image_transform,
                args.pseudo_gt_queries,
                args.pseudo_gt_views,
            )
            if args.max_target_points and target_tensor.shape[0] > args.max_target_points:
                target_tensor = target_tensor[: args.max_target_points]
        except Exception as exc:
            sample["status"] = "failed"
            sample["error"] = str(exc)
            failures.append({"sample_id": sample["sample_id"], "error": str(exc)})
            if not args.skip_failures:
                raise
            print(f"Skipping failed sample {sample['sample_id']}: {exc}")
            continue

        sample["status"] = "ok"
        sample["feature_shape"] = list(feature_tensor.shape)
        sample["target_shape"] = list(target_tensor.shape)
        features.append(feature_tensor)
        targets.append(target_tensor)
        sample_ids.append(sample["sample_id"])
        split_labels.append(sample["split"])
        sample_metadata.append(sample)
        print(
            f"Prepared {sample['sample_id']} split={sample['split']} "
            f"features={tuple(feature_tensor.shape)} target={tuple(target_tensor.shape)}"
        )

    if not features:
        raise RuntimeError("No samples were prepared successfully")

    dataset = {
        "scene_ids": sample_ids,
        "features": torch.stack(features, dim=0),
        "target_points": torch.stack(targets, dim=0),
        "splits": split_labels,
        "metadata": sample_metadata,
        "meta": {
            "feature_ckpt": args.feature_ckpt,
            "pseudo_gt_ckpt": args.pseudo_gt_ckpt,
            "pseudo_gt_queries": args.pseudo_gt_queries,
            "resolution": args.resolution,
            "failures": failures,
        },
    }

    os.makedirs(output_root, exist_ok=True)
    torch.save(dataset, dataset_path)
    manifest["prepared_dataset_path"] = dataset_path
    manifest["prepared_sample_count"] = len(sample_ids)
    manifest["failures"] = failures
    save_manifest(manifest_path, manifest)
    print(f"Saved adapter dataset to {dataset_path}")


if __name__ == "__main__":
    main()
