from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Build scene-level ScanNet complete-GT reservoirs from vh_clean meshes.")
    parser.add_argument("--source_root", required=True, help="Raw ScanNet root, e.g. /data1/jcd_data/scannerv2_paraell_w48")
    parser.add_argument("--processed_root", required=True, help="Processed ScanNet root containing scans_train/scans_test")
    parser.add_argument("--splits", nargs="+", default=["scans_train", "scans_test"], help="Processed splits to populate")
    parser.add_argument("--points_per_scene", type=int, default=200000, help="Uniform surface samples stored per scene")
    parser.add_argument("--output_name", default="mesh_complete_reservoir_vh_clean.npz")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit_scenes", type=int, default=None)
    parser.add_argument("--scene", action="append", default=None, help="Optional explicit scene id(s) to build")
    return parser.parse_args()


def scene_mesh_path(source_root: Path, split: str, scene: str) -> Path:
    if split == "scans_train":
        return source_root / "scans" / scene / f"{scene}_vh_clean.ply"
    if split == "scans_test":
        return source_root / "scans_test" / scene / f"{scene}_vh_clean.ply"
    raise ValueError(f"Unsupported split {split!r}")


def iter_scene_jobs(args) -> Iterable[tuple[str, str, str, int, str, bool]]:
    processed_root = Path(args.processed_root)
    requested = set(args.scene or [])
    for split in args.splits:
        split_root = processed_root / split
        if not split_root.is_dir():
            continue
        scene_dirs = sorted([p for p in split_root.iterdir() if p.is_dir() and p.name.startswith("scene")])
        if requested:
            scene_dirs = [p for p in scene_dirs if p.name in requested]
        if args.limit_scenes is not None:
            scene_dirs = scene_dirs[: args.limit_scenes]
        for scene_dir in scene_dirs:
            yield (
                str(Path(args.source_root)),
                str(scene_dir),
                split,
                int(args.points_per_scene),
                str(args.output_name),
                bool(args.overwrite),
            )


def sample_scene_mesh(source_root_str: str, scene_dir_str: str, split: str, points_per_scene: int, output_name: str, overwrite: bool):
    source_root = Path(source_root_str)
    scene_dir = Path(scene_dir_str)
    scene = scene_dir.name
    output_path = scene_dir / output_name
    if output_path.exists() and not overwrite:
        with np.load(output_path) as data:
            count = int(data["points_world"].shape[0])
        return {"scene": scene, "split": split, "status": "exists", "count": count, "path": str(output_path)}

    mesh_path = scene_mesh_path(source_root, split, scene)
    if not mesh_path.is_file():
        return {"scene": scene, "split": split, "status": "missing_mesh", "path": str(mesh_path)}

    mesh = trimesh.load(mesh_path, process=False)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    if len(mesh.faces) == 0:
        points_world = np.asarray(mesh.vertices, dtype=np.float32)
        if points_world.shape[0] > points_per_scene:
            rng = np.random.default_rng(0)
            choice = rng.choice(points_world.shape[0], size=points_per_scene, replace=False)
            points_world = points_world[choice]
    else:
        points_world, _ = trimesh.sample.sample_surface(mesh, points_per_scene)
        points_world = points_world.astype(np.float32)

    scene_txt = mesh_path.with_name(f"{scene}.txt")
    axis_alignment = None
    if scene_txt.exists():
        for line in scene_txt.read_text().splitlines():
            if line.startswith("axisAlignment"):
                vals = [float(x) for x in line.split("=")[1].strip().split()]
                axis_alignment = np.asarray(vals, dtype=np.float32).reshape(4, 4)
                break

    meta = {
        "scene": scene,
        "split": split,
        "mesh_path": str(mesh_path),
        "points_per_scene": int(points_world.shape[0]),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
    }
    np.savez_compressed(
        output_path,
        points_world=points_world,
        axis_alignment=axis_alignment if axis_alignment is not None else np.eye(4, dtype=np.float32),
        metadata=json.dumps(meta),
    )
    return {"scene": scene, "split": split, "status": "built", "count": int(points_world.shape[0]), "path": str(output_path)}


def main():
    args = parse_args()
    jobs = list(iter_scene_jobs(args))
    if not jobs:
        print("No scene jobs found.")
        return

    print(f"Building ScanNet complete-GT reservoirs for {len(jobs)} scenes")
    built = 0
    existed = 0
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(sample_scene_mesh, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            status = result["status"]
            if status == "built":
                built += 1
                print(f"[built] {result['split']} {result['scene']} -> {result['count']} pts")
            elif status == "exists":
                existed += 1
                print(f"[exists] {result['split']} {result['scene']} -> {result['count']} pts")
            else:
                failed.append(result)
                print(f"[fail]  {result['split']} {result['scene']} -> {status}: {result['path']}")

    print(f"Done. built={built} existed={existed} failed={len(failed)}")
    if failed:
        for item in failed[:20]:
            print(item)


if __name__ == "__main__":
    main()
