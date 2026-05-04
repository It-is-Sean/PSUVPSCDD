from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str


@dataclass(frozen=True)
class PairSample:
    sample_id: str
    scene_id: str
    sequence_id: str
    sequence_dir: Path
    frame_ids: tuple[int, int]
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an adapter-training .pt dataset from the full SCRREAM layout. "
            "Targets are approximated from dense depth_gt views, voxel-filtered, "
            "cropped to the selected input-view frustums, and stored in the first "
            "input camera coordinate frame."
        )
    )
    parser.add_argument("--data_root", default="~/datasets/SCRREAM")
    parser.add_argument("--pair_list", default="data/scrream/scrream_n2_list.json")
    parser.add_argument(
        "--output_path",
        default="experiments/probe3d/adapter_data/scrream_full_n2_adapter_seed17.pt",
    )
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--target_points", type=int, default=10000)
    parser.add_argument(
        "--target_source",
        default="depth_gt_dense",
        choices=("depth_gt_dense", "mesh_complete"),
        help="Use dense depth_gt aggregation or registered scene meshes as complete target source.",
    )
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--dense_stride", type=int, default=5)
    parser.add_argument("--dense_context", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=17)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--frustum_margin", type=float, default=1.0)
    parser.add_argument("--fps_pool_size", type=int, default=50000)
    parser.add_argument(
        "--mesh_sample_points",
        type=int,
        default=250000,
        help="Reservoir size for scene-level mesh surface samples before frustum crop.",
    )
    parser.add_argument(
        "--mesh_cache_dir",
        default=None,
        help="Optional cache for per-scene mesh reservoirs. Defaults to output_path.parent / mesh_cache.",
    )
    parser.add_argument(
        "--fps_device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for farthest-point sampling; auto uses CUDA when available.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_points_after_crop", type=int, default=100)
    parser.add_argument("--save_preview_dir", default=None)
    parser.add_argument("--preview_count", type=int, default=4)
    parser.add_argument(
        "--pose_convention",
        default="camera_to_world",
        choices=("camera_to_world", "world_to_camera"),
        help="Interpretation of camera_pose/*.txt. SCRREAM-full is assumed camera_to_world.",
    )
    parser.add_argument("--skip_failures", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_pair_rows(pair_list_path: Path) -> list[str]:
    with pair_list_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for value in payload.values():
            if isinstance(value, list):
                rows.extend(value)
            else:
                rows.append(value)
    else:
        raise TypeError(f"Unsupported pair list payload type: {type(payload).__name__}")
    return [str(row).strip() for row in rows if str(row).strip()]


def parse_pair_row(row: str, data_root: Path) -> tuple[str, str, Path, int, int]:
    parts = row.split()
    if len(parts) != 3:
        raise ValueError(f"Expected '<scene>/<sequence> frame_a frame_b', got: {row!r}")
    sequence_ref, frame_a_raw, frame_b_raw = parts
    ref_parts = sequence_ref.split("/")
    if len(ref_parts) != 2:
        raise ValueError(f"Expected sequence reference like scene09/scene09_full_00, got: {sequence_ref!r}")
    scene_id, sequence_id = ref_parts
    sequence_dir = data_root / scene_id / sequence_id
    return scene_id, sequence_id, sequence_dir, int(frame_a_raw), int(frame_b_raw)


def split_scene_ids(
    scene_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    unique_scene_ids = sorted(set(scene_ids))
    if not unique_scene_ids:
        raise ValueError("No scene ids found in pair list")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be positive")

    shuffled = list(unique_scene_ids)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)

    if count == 1:
        return {"train": sorted(shuffled), "val": [], "test": []}
    if count == 2:
        return {"train": sorted(shuffled[:1]), "val": [], "test": sorted(shuffled[1:])}

    val_count = max(1, int(round(count * (val_ratio / total_ratio)))) if val_ratio > 0 else 0
    test_count = max(1, int(round(count * (test_ratio / total_ratio)))) if test_ratio > 0 else 0
    train_count = count - val_count - test_count
    if train_count <= 0:
        train_count = 1
        while train_count + val_count + test_count > count and test_count > 1:
            test_count -= 1
        while train_count + val_count + test_count > count and val_count > 1:
            val_count -= 1
        if train_count + val_count + test_count > count:
            raise ValueError(f"Cannot split {count} scenes with the requested ratios")

    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count : train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count : train_count + val_count + test_count]),
    }


def build_samples(pair_rows: list[str], data_root: Path, args: argparse.Namespace) -> tuple[list[PairSample], dict[str, list[str]]]:
    parsed_rows = [parse_pair_row(row, data_root) for row in pair_rows]
    split_map = split_scene_ids(
        [scene_id for scene_id, _, _, _, _ in parsed_rows],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    split_by_scene = {
        scene_id: split_name
        for split_name, split_scene_ids_ in split_map.items()
        for scene_id in split_scene_ids_
    }

    samples = []
    for scene_id, sequence_id, sequence_dir, frame_a, frame_b in parsed_rows:
        sample_id = f"{scene_id}/{sequence_id}_{frame_a:06d}_{frame_b:06d}"
        samples.append(
            PairSample(
                sample_id=sample_id,
                scene_id=scene_id,
                sequence_id=sequence_id,
                sequence_dir=sequence_dir,
                frame_ids=(frame_a, frame_b),
                split=split_by_scene[scene_id],
            )
        )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    return samples, split_map


def parse_colmap_camera(camera_path: Path) -> CameraIntrinsics:
    with camera_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            model = parts[1].upper()
            width = int(parts[2])
            height = int(parts[3])
            params = [float(value) for value in parts[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
                f, cx, cy = params[:3]
                fx = fy = f
            elif model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
                fx, fy, cx, cy = params[:4]
            else:
                raise ValueError(f"Unsupported COLMAP camera model {model!r} in {camera_path}")
            return CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, model=model)
    raise ValueError(f"No camera row found in {camera_path}")


def load_pose_c2w(pose_path: Path, pose_convention: str) -> np.ndarray:
    pose = np.loadtxt(pose_path, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose in {pose_path}, got shape {pose.shape}")
    if pose_convention == "camera_to_world":
        return pose
    if pose_convention == "world_to_camera":
        return np.linalg.inv(pose)
    raise ValueError(f"Unsupported pose convention: {pose_convention}")


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_grid_filter(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3).astype(np.float32)
    if voxel_size <= 0:
        return points.astype(np.float32, copy=False)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices].astype(np.float32, copy=False)


def frame_path(sequence_dir: Path, subdir: str, frame_id: int, suffix: str) -> Path:
    return sequence_dir / subdir / f"{frame_id:06d}{suffix}"


def load_depth_points_world(
    sequence_dir: Path,
    frame_id: int,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
    voxel_size: float,
    pose_convention: str,
) -> np.ndarray:
    depth_path = frame_path(sequence_dir, "depth_gt", frame_id, ".png")
    pose_path = frame_path(sequence_dir, "camera_pose", frame_id, ".txt")
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    depth = np.asarray(Image.open(depth_path), dtype=np.float32) / float(depth_scale)
    if depth.ndim != 2:
        raise ValueError(f"Expected single-channel depth in {depth_path}, got shape {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth[ys, xs]
    x = (xs.astype(np.float32) - float(intrinsics.cx)) / float(intrinsics.fx) * z
    y = (ys.astype(np.float32) - float(intrinsics.cy)) / float(intrinsics.fy) * z
    points_cam = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    c2w = load_pose_c2w(pose_path, pose_convention=pose_convention)
    points_world = transform_points(points_cam, c2w)
    return voxel_grid_filter(points_world, voxel_size=voxel_size)


def require_trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Mesh-complete target generation requires trimesh. "
            "Activate an environment that contains trimesh or install it before running --target_source mesh_complete."
        ) from exc
    return trimesh


def sample_mesh_surface_points(mesh_path: Path, sample_count: int, seed: int) -> np.ndarray:
    trimesh = require_trimesh()
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if getattr(geom, "vertices", None) is not None]
        if not meshes:
            return np.empty((0, 3), dtype=np.float32)
        loaded = trimesh.util.concatenate(meshes)
    if loaded.vertices is None or len(loaded.vertices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if loaded.faces is None or len(loaded.faces) == 0:
        return np.asarray(loaded.vertices, dtype=np.float32)

    rng = np.random.default_rng(seed)
    areas = np.asarray(loaded.area_faces, dtype=np.float64)
    valid = np.isfinite(areas) & (areas > 0)
    if not np.any(valid):
        return np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64)[valid]
    areas = areas[valid]
    probabilities = areas / areas.sum()
    face_indices = rng.choice(len(faces), size=max(1, int(sample_count)), replace=True, p=probabilities)
    triangles = np.asarray(loaded.vertices, dtype=np.float64)[faces[face_indices]]
    uv = rng.random((len(face_indices), 2), dtype=np.float64)
    swap = uv.sum(axis=1) > 1.0
    uv[swap] = 1.0 - uv[swap]
    points = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    return points.astype(np.float32, copy=False)


def estimate_mesh_surface_area(mesh_path: Path) -> float:
    trimesh = require_trimesh()
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        areas = [float(getattr(geom, "area", 0.0)) for geom in loaded.geometry.values()]
        return float(np.sum([area for area in areas if np.isfinite(area) and area > 0]))
    area = float(getattr(loaded, "area", 0.0))
    return area if np.isfinite(area) and area > 0 else 0.0


def load_or_build_mesh_points_world(
    scene_dir: Path,
    cache_dir: Path,
    sample_count: int,
    voxel_size: float,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    mesh_dir = scene_dir / "meshes"
    mesh_paths = sorted(mesh_dir.glob("*.obj"))
    if not mesh_paths:
        raise FileNotFoundError(f"No OBJ meshes found in {mesh_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{scene_dir.name}_mesh_complete_area_s{sample_count}_v{voxel_size:.5f}_seed{seed}.npz"
    cache_path = cache_dir / cache_name
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=True) as data:
            points = data["points_world"].astype(np.float32)
            cached_mesh_files = [str(x) for x in data["mesh_files"].tolist()]
        return points, cached_mesh_files

    mesh_areas = np.asarray([estimate_mesh_surface_area(mesh_path) for mesh_path in mesh_paths], dtype=np.float64)
    if np.isfinite(mesh_areas).all() and mesh_areas.sum() > 0:
        weights = mesh_areas / mesh_areas.sum()
        budgets = np.maximum(256, np.ceil(weights * float(sample_count)).astype(np.int64))
    else:
        budgets = np.full(len(mesh_paths), max(1000, int(np.ceil(float(sample_count) / float(len(mesh_paths))))), dtype=np.int64)
    all_points = []
    for mesh_idx, mesh_path in enumerate(mesh_paths):
        mesh_seed = stable_seed(f"{scene_dir.name}/{mesh_path.name}", seed)
        points = sample_mesh_surface_points(mesh_path, sample_count=int(budgets[mesh_idx]), seed=mesh_seed)
        if points.size > 0:
            all_points.append(points)
    if not all_points:
        raise ValueError(f"No surface points could be sampled from {mesh_dir}")
    points_world = np.concatenate(all_points, axis=0)
    points_world = voxel_grid_filter(points_world, voxel_size=voxel_size)
    if points_world.shape[0] > sample_count:
        rng = np.random.default_rng(seed)
        choice = rng.choice(points_world.shape[0], size=sample_count, replace=False)
        points_world = points_world[choice]
    np.savez_compressed(
        cache_path,
        points_world=points_world.astype(np.float32),
        mesh_files=np.asarray([str(path) for path in mesh_paths], dtype=object),
        mesh_areas=mesh_areas.astype(np.float32),
        mesh_sample_budgets=budgets.astype(np.int64),
    )
    return points_world.astype(np.float32, copy=False), [str(path) for path in mesh_paths]


def dense_frame_ids(frame_ids: tuple[int, int], sequence_dir: Path, dense_stride: int, dense_context: int) -> list[int]:
    if dense_stride <= 0:
        raise ValueError("--dense_stride must be positive")
    start = max(0, min(frame_ids) - dense_context)
    end = max(frame_ids) + dense_context
    depth_dir = sequence_dir / "depth_gt"
    available = {int(path.stem) for path in depth_dir.glob("*.png") if path.stem.isdigit()}
    ids = [frame_id for frame_id in range(start, end + 1, dense_stride) if frame_id in available]
    for frame_id in frame_ids:
        if frame_id in available and frame_id not in ids:
            ids.append(frame_id)
    return sorted(ids)


def crop_to_input_frustums(
    points_world: np.ndarray,
    input_poses_c2w: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    margin: float,
) -> np.ndarray:
    if points_world.size == 0:
        return points_world.reshape(0, 3).astype(np.float32)
    keep = np.zeros(points_world.shape[0], dtype=bool)
    for c2w in input_poses_c2w:
        w2c = np.linalg.inv(c2w)
        points_cam = transform_points(points_world, w2c)
        z = points_cam[:, 2]
        positive = z > 1e-6
        u = np.full(points_world.shape[0], np.nan, dtype=np.float64)
        v = np.full(points_world.shape[0], np.nan, dtype=np.float64)
        u[positive] = intrinsics.fx * points_cam[positive, 0] / z[positive] + intrinsics.cx
        v[positive] = intrinsics.fy * points_cam[positive, 1] / z[positive] + intrinsics.cy
        in_view = (
            positive
            & (u >= -margin)
            & (u <= (intrinsics.width - 1 + margin))
            & (v >= -margin)
            & (v <= (intrinsics.height - 1 + margin))
        )
        keep |= in_view
    return points_world[keep].astype(np.float32, copy=False)


def stable_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2**32)


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "prepare_scrream_full_adapter_data.py requires torch to generate the .pt dataset. "
            "Activate the training environment that contains PyTorch before running generation."
        ) from exc
    return torch


def resolve_fps_device(requested: str):
    torch = require_torch()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--fps_device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def farthest_point_sample(points: np.ndarray, target_points: int, seed: int, device) -> np.ndarray:
    torch = require_torch()
    if points.shape[0] == target_points:
        return points.astype(np.float32, copy=False)
    if points.shape[0] < target_points:
        raise ValueError("farthest_point_sample requires at least target_points input points")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    count = pts.shape[0]
    selected = torch.empty((target_points,), dtype=torch.long, device=device)
    distances = torch.full((count,), float("inf"), dtype=torch.float32, device=device)
    farthest = torch.randint(0, count, (1,), generator=generator, device=device, dtype=torch.long).squeeze(0)

    for idx in range(target_points):
        selected[idx] = farthest
        centroid = pts[farthest].view(1, 3)
        dist = torch.sum((pts - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)

    return pts[selected].detach().cpu().numpy().astype(np.float32, copy=False)


def sample_target_points(
    points: np.ndarray,
    target_points: int,
    fps_pool_size: int,
    seed: int,
    fps_device,
) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError("Cannot sample target points from an empty point cloud")
    rng = np.random.default_rng(seed)
    if points.shape[0] < target_points:
        extra_indices = rng.choice(points.shape[0], size=target_points - points.shape[0], replace=True)
        return np.concatenate([points, points[extra_indices]], axis=0).astype(np.float32, copy=False)
    if points.shape[0] > fps_pool_size >= target_points:
        pool_indices = rng.choice(points.shape[0], size=fps_pool_size, replace=False)
        points = points[pool_indices]
    return farthest_point_sample(points, target_points=target_points, seed=seed, device=fps_device)


def write_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {pts.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("end_header\n")
        for x, y, z in pts.tolist():
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def process_sample(sample: PairSample, args: argparse.Namespace, fps_device: torch.device, mesh_cache_dir: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    sequence_dir = sample.sequence_dir
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)

    intrinsics = parse_colmap_camera(sequence_dir / "sparse" / "0" / "cameras.txt")
    input_frame_paths = [frame_path(sequence_dir, "rgb", frame_id, ".png") for frame_id in sample.frame_ids]
    missing_rgb = [path for path in input_frame_paths if not path.is_file()]
    if missing_rgb:
        raise FileNotFoundError(f"Missing input RGB files: {missing_rgb[:2]}")

    input_poses = [
        load_pose_c2w(frame_path(sequence_dir, "camera_pose", frame_id, ".txt"), pose_convention=args.pose_convention)
        for frame_id in sample.frame_ids
    ]
    dense_ids: list[int] = []
    mesh_files: list[str] = []
    points_after_source = 0
    scene_dir = sequence_dir.parent
    if args.target_source == "depth_gt_dense":
        dense_ids = dense_frame_ids(
            sample.frame_ids,
            sequence_dir=sequence_dir,
            dense_stride=args.dense_stride,
            dense_context=args.dense_context,
        )
        if not dense_ids:
            raise ValueError(f"No dense depth_gt frames found for {sample.sample_id}")

        per_frame_points = [
            load_depth_points_world(
                sequence_dir,
                frame_id,
                intrinsics=intrinsics,
                depth_scale=args.depth_scale,
                voxel_size=args.voxel_size,
                pose_convention=args.pose_convention,
            )
            for frame_id in dense_ids
        ]
        nonempty_points = [points for points in per_frame_points if points.size > 0]
        if not nonempty_points:
            raise ValueError(f"No valid depth_gt points found in dense frames for {sample.sample_id}")
        points_world = np.concatenate(nonempty_points, axis=0)
        points_world = voxel_grid_filter(points_world, voxel_size=args.voxel_size)
        points_after_source = int(points_world.shape[0])
        target_source_label = "scrream_depth_gt_dense"
    elif args.target_source == "mesh_complete":
        if mesh_cache_dir is None:
            raise ValueError("mesh_cache_dir must be provided for mesh_complete target generation")
        scene_seed = stable_seed(f"{scene_dir.name}/mesh_complete", args.split_seed)
        points_world, mesh_files = load_or_build_mesh_points_world(
            scene_dir=scene_dir,
            cache_dir=mesh_cache_dir,
            sample_count=args.mesh_sample_points,
            voxel_size=args.voxel_size,
            seed=scene_seed,
        )
        points_after_source = int(points_world.shape[0])
        target_source_label = "scrream_registered_mesh_complete"
    else:
        raise ValueError(f"Unsupported target_source={args.target_source!r}")

    points_world = crop_to_input_frustums(
        points_world,
        input_poses_c2w=input_poses,
        intrinsics=intrinsics,
        margin=args.frustum_margin,
    )
    if points_world.shape[0] < args.min_points_after_crop:
        raise ValueError(
            f"Only {points_world.shape[0]} points remain after frustum crop for {sample.sample_id}; "
            f"check pose convention, intrinsics, and depth_scale"
        )

    first_w2c = np.linalg.inv(input_poses[0])
    points_first_cam = transform_points(points_world, first_w2c).astype(np.float32, copy=False)
    target_seed = stable_seed(sample.sample_id, args.split_seed)
    sampled = sample_target_points(
        points_first_cam,
        target_points=args.target_points,
        fps_pool_size=args.fps_pool_size,
        seed=target_seed,
        fps_device=fps_device,
    )

    metadata = {
        "sample_id": sample.sample_id,
        "scene_id": sample.scene_id,
        "sequence_id": sample.sequence_id,
        "sequence_dir": str(sequence_dir),
        "frame_ids": list(sample.frame_ids),
        "frame_paths": [str(path) for path in input_frame_paths],
        "target_source": target_source_label,
        "target_frame_ids": dense_ids,
        "dense_frame_ids": dense_ids,
        "mesh_files": mesh_files,
        "mesh_sample_points": int(args.mesh_sample_points),
        "depth_scale": float(args.depth_scale),
        "voxel_size": float(args.voxel_size),
        "dense_stride": int(args.dense_stride),
        "dense_context": int(args.dense_context),
        "frustum_margin": float(args.frustum_margin),
        "pose_convention": args.pose_convention,
        "camera": {
            "model": intrinsics.model,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
        },
        "num_points_after_source_voxel": points_after_source,
        "num_dense_points_after_voxel": points_after_source if args.target_source == "depth_gt_dense" else 0,
        "num_mesh_points_after_voxel": points_after_source if args.target_source == "mesh_complete" else 0,
        "num_points_after_frustum_crop": int(points_world.shape[0]),
        "target_seed": int(target_seed),
    }
    return sampled, metadata


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.target_points <= 0:
        raise ValueError("--target_points must be positive")
    if args.depth_scale <= 0:
        raise ValueError("--depth_scale must be positive")
    if args.voxel_size <= 0:
        raise ValueError("--voxel_size must be positive")
    if args.fps_pool_size <= args.target_points:
        args.fps_pool_size = args.target_points

    data_root = resolve_path(args.data_root)
    pair_list_path = resolve_path(args.pair_list)
    output_path = resolve_path(args.output_path)
    manifest_path = resolve_path(args.manifest_path) if args.manifest_path else output_path.with_suffix(".manifest.json")
    preview_dir = resolve_path(args.save_preview_dir) if args.save_preview_dir else None
    mesh_cache_dir = resolve_path(args.mesh_cache_dir) if args.mesh_cache_dir else output_path.parent / "mesh_cache"
    fps_device = resolve_fps_device(args.fps_device)
    torch = require_torch()

    pair_rows = load_pair_rows(pair_list_path)
    samples, scene_splits = build_samples(pair_rows, data_root=data_root, args=args)
    if not samples:
        raise ValueError("No samples selected from pair list")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    scene_ids: list[str] = []
    splits: list[str] = []
    metadata: list[dict[str, Any]] = []
    target_tensors = []
    failures: list[dict[str, str]] = []

    for sample_idx, sample in enumerate(samples):
        print(f"[{sample_idx + 1}/{len(samples)}] {sample.sample_id} split={sample.split}")
        try:
            target_points, sample_meta = process_sample(sample, args=args, fps_device=fps_device, mesh_cache_dir=mesh_cache_dir)
        except Exception as exc:
            failure = {"sample_id": sample.sample_id, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            if args.skip_failures:
                print(f"  skip: {failure['error']}")
                continue
            raise

        scene_ids.append(sample.sample_id)
        splits.append(sample.split)
        metadata.append(sample_meta)
        target_tensors.append(torch.from_numpy(target_points).float())

        if preview_dir is not None and len(target_tensors) <= args.preview_count:
            preview_name = sample.sample_id.replace("/", "__")
            write_point_cloud_ply(preview_dir / f"{preview_name}_target_first_view.ply", target_points)

    if not target_tensors:
        raise RuntimeError("No adapter samples were generated")

    target_points_tensor = torch.stack(target_tensors, dim=0).contiguous()
    payload = {
        "scene_ids": scene_ids,
        "target_points": target_points_tensor,
        "splits": splits,
        "metadata": metadata,
        "meta": {
            "version": 1,
            "dataset": "SCRREAM-full",
            "target_source": args.target_source,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root),
            "pair_list": str(pair_list_path),
            "output_path": str(output_path),
            "target_points": int(args.target_points),
            "depth_scale": float(args.depth_scale),
            "voxel_size": float(args.voxel_size),
            "mesh_sample_points": int(args.mesh_sample_points),
            "mesh_cache_dir": str(mesh_cache_dir),
            "dense_stride": int(args.dense_stride),
            "dense_context": int(args.dense_context),
            "split_seed": int(args.split_seed),
            "scene_splits": scene_splits,
            "pose_convention": args.pose_convention,
            "features": "omitted; VGGT features are extracted online by train_vggt_nova_adapter.py",
        },
    }
    torch.save(payload, output_path)

    split_counts = {split: splits.count(split) for split in sorted(set(splits))}
    manifest = {
        "version": 1,
        "dataset": "SCRREAM-full",
        "generated_at": payload["meta"]["generated_at"],
        "data_root": str(data_root),
        "pair_list": str(pair_list_path),
        "output_path": str(output_path),
        "target_points_shape": list(target_points_tensor.shape),
        "scene_splits": scene_splits,
        "split_counts": split_counts,
        "sample_count": len(scene_ids),
        "failure_count": len(failures),
        "failures": failures,
        "samples": metadata,
    }
    save_manifest(manifest_path, manifest)
    print(f"Wrote adapter dataset: {output_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"target_points shape: {tuple(target_points_tensor.shape)}")
    print(f"split counts: {split_counts}")


if __name__ == "__main__":
    main()
