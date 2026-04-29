from __future__ import annotations

import argparse
import multiprocessing as mp
import struct
import subprocess
import zlib
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def read_matrix(handle) -> np.ndarray:
    return np.asarray(struct.unpack("f" * 16, handle.read(16 * 4)), dtype=np.float32).reshape(4, 4)


def save_mat(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in matrix:
            np.savetxt(handle, line[np.newaxis], fmt="%f")


def export_sens_sampled(sens_path: Path, output_path: Path, frame_skip: int) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    color_dir = output_path / "color"
    depth_dir = output_path / "depth"
    pose_dir = output_path / "pose"
    intrinsic_dir = output_path / "intrinsic"
    for d in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        d.mkdir(parents=True, exist_ok=True)

    with sens_path.open("rb") as handle:
        version = struct.unpack("I", handle.read(4))[0]
        if version != 4:
            raise ValueError(f"Unsupported .sens version {version}: {sens_path}")
        strlen = struct.unpack("Q", handle.read(8))[0]
        handle.read(strlen)
        intrinsic_color = read_matrix(handle)
        extrinsic_color = read_matrix(handle)
        intrinsic_depth = read_matrix(handle)
        extrinsic_depth = read_matrix(handle)
        color_compression = COMPRESSION_TYPE_COLOR[struct.unpack("i", handle.read(4))[0]]
        depth_compression = COMPRESSION_TYPE_DEPTH[struct.unpack("i", handle.read(4))[0]]
        handle.read(4)  # color_width
        handle.read(4)  # color_height
        depth_width = struct.unpack("I", handle.read(4))[0]
        depth_height = struct.unpack("I", handle.read(4))[0]
        handle.read(4)  # depth_shift
        num_frames = struct.unpack("Q", handle.read(8))[0]

        save_mat(intrinsic_dir / "intrinsic_color.txt", intrinsic_color)
        save_mat(intrinsic_dir / "extrinsic_color.txt", extrinsic_color)
        save_mat(intrinsic_dir / "intrinsic_depth.txt", intrinsic_depth)
        save_mat(intrinsic_dir / "extrinsic_depth.txt", extrinsic_depth)

        exported = 0
        for frame_idx in range(num_frames):
            camera_to_world = read_matrix(handle)
            handle.read(8)  # timestamp_color
            handle.read(8)  # timestamp_depth
            color_size = struct.unpack("Q", handle.read(8))[0]
            depth_size = struct.unpack("Q", handle.read(8))[0]
            color_data = handle.read(color_size)
            depth_data = handle.read(depth_size)
            if frame_idx % frame_skip != 0:
                continue
            if not np.isfinite(camera_to_world).all():
                continue
            if color_compression != "jpeg" or depth_compression != "zlib_ushort":
                raise ValueError(f"Unsupported compression color={color_compression}, depth={depth_compression}")
            color = imageio.imread(color_data)
            depth = np.frombuffer(zlib.decompress(depth_data), dtype=np.uint16).reshape(depth_height, depth_width)
            imageio.imwrite(color_dir / f"{frame_idx}.jpg", color)
            cv2.imwrite(str(depth_dir / f"{frame_idx}.png"), depth)
            save_mat(pose_dir / f"{frame_idx}.txt", camera_to_world)
            exported += 1
        if exported < 4:
            raise RuntimeError(f"Too few exported frames from {sens_path}: {exported}")


def list_scenes(split_root: Path, limit: int | None) -> list[str]:
    scenes = sorted(p.name for p in split_root.iterdir() if p.is_dir() and p.name.startswith("scene"))
    return scenes if limit is None else scenes[:limit]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def extract_one(task: tuple[str, Path, Path, int]) -> None:
    split, src_scene, raw_root, frame_skip = task
    scene = src_scene.name
    out_scene = raw_root / split / scene
    done = out_scene / ".extract_done"
    if done.exists():
        return
    sens = src_scene / f"{scene}.sens"
    if not sens.exists():
        raise FileNotFoundError(sens)
    print(f"extract {split}/{scene} frame_skip={frame_skip}", flush=True)
    export_sens_sampled(sens, out_scene, frame_skip)
    done.write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", default="/data1/jcd_data/scannerv2_paraell_w48")
    parser.add_argument("--raw_extract_root", default="/data1/jcd_data/scannet_raw_extract_large")
    parser.add_argument("--processed_root", default="/data1/jcd_data/scannet_processed_large")
    parser.add_argument("--train_scenes", type=int, default=300)
    parser.add_argument("--test_scenes", type=int, default=50)
    parser.add_argument("--frame_skip", type=int, default=20, help="Keep every Nth frame directly while reading .sens.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preprocess_script", default="datasets_preprocess/preprocess_scannet.py")
    parser.add_argument("--generate_script", default="datasets_preprocess/generate_set_scannet.py")
    parser.add_argument("--build_complete_gt", action="store_true")
    parser.add_argument("--complete_gt_script", default="experiments/probe3d/build_scannet_complete_gt.py")
    parser.add_argument("--complete_points_per_scene", type=int, default=200000)
    parser.add_argument("--complete_output_name", default="mesh_complete_reservoir_vh_clean.npz")
    parser.add_argument("--overwrite_complete_gt", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    raw_root = Path(args.raw_extract_root)
    processed_root = Path(args.processed_root)
    split_specs = [
        ("scans_train", source_root / "scans", args.train_scenes),
        ("scans_test", source_root / "scans_test", args.test_scenes),
    ]
    tasks = []
    for split, split_root, limit in split_specs:
        scenes = list_scenes(split_root, limit)
        print(f"{split}: selected {len(scenes)} scenes from {split_root}", flush=True)
        tasks.extend((split, split_root / scene, raw_root, args.frame_skip) for scene in scenes)

    raw_root.mkdir(parents=True, exist_ok=True)
    if args.workers <= 1:
        for task in tasks:
            extract_one(task)
    else:
        with mp.Pool(args.workers) as pool:
            for _ in pool.imap_unordered(extract_one, tasks):
                pass

    processed_root.mkdir(parents=True, exist_ok=True)
    run(["python", args.preprocess_script, "--scannet_dir", str(raw_root), "--output_dir", str(processed_root)])
    run(["python", args.generate_script, "--root", str(processed_root), "--splits", "scans_train", "scans_test", "--max_interval", "150", "--num_workers", str(args.workers)])

    if args.build_complete_gt:
        complete_cmd = [
            "python",
            args.complete_gt_script,
            "--source_root",
            str(source_root),
            "--processed_root",
            str(processed_root),
            "--points_per_scene",
            str(args.complete_points_per_scene),
            "--output_name",
            args.complete_output_name,
            "--workers",
            str(args.workers),
        ]
        if args.overwrite_complete_gt:
            complete_cmd.append("--overwrite")
        run(complete_cmd)

    (processed_root / "PREPARED_BY_probe3d_prepare_scannet_large.txt").write_text(
        f"source_root={source_root}\nraw_extract_root={raw_root}\nprocessed_root={processed_root}\n"
        f"train_scenes={args.train_scenes}\ntest_scenes={args.test_scenes}\nframe_skip={args.frame_skip}\nworkers={args.workers}\n"
        f"build_complete_gt={args.build_complete_gt}\ncomplete_points_per_scene={args.complete_points_per_scene}\ncomplete_output_name={args.complete_output_name}\n",
        encoding="utf-8",
    )
    print(f"Prepared ScanNet at {processed_root}", flush=True)


if __name__ == "__main__":
    main()
