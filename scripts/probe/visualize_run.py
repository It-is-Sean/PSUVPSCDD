from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDER_POINTS_PATH = REPO_ROOT / "demo" / "visualization" / "render_points.py"
DEFAULT_REPORT_ROOT = REPO_ROOT / "artifacts" / "reports" / "vggt_to_nova3r"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reusable visualization workflow for probe point-cloud runs."
    )
    parser.add_argument("--run-dir", help="Run directory containing pointcloud outputs")
    parser.add_argument("--summary", help="Path to a summary.json file")
    parser.add_argument("--ply", help="Path to a pointcloud .ply file")
    parser.add_argument(
        "--report-root",
        default=str(DEFAULT_REPORT_ROOT),
        help="Root directory used when resolving the latest run",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest run under --report-root (default if no source is given).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for visualization artifacts (default: <run-dir>/visualization)",
    )
    parser.add_argument("--preview-name", default="preview.png")
    parser.add_argument("--turntable-name", default="turntable.mp4")
    parser.add_argument("--manifest-name", default="visualization_manifest.json")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--distance", type=float, default=20.0, help="Reserved for advanced renderer compatibility")
    parser.add_argument("--elevation", type=float, default=10.0)
    parser.add_argument("--preview-azim", type=float, default=45.0)
    parser.add_argument("--azim-start", type=float, default=0.0)
    parser.add_argument("--azim-end", type=float, default=None)
    parser.add_argument("--radius", type=float, default=0.005, help="Reserved for advanced renderer compatibility")
    parser.add_argument("--points-per-pixel", type=int, default=10, help="Reserved for advanced renderer compatibility")
    parser.add_argument("--normal-neighbors", type=int, default=60, help="Reserved for advanced renderer compatibility")
    parser.add_argument("--compositor", choices=["alpha", "normweighted"], default="alpha")
    parser.add_argument("--color-type", choices=["normal", "plasma", "viridis", "xyz"], default="viridis")
    parser.add_argument("--bbox", action="store_true")
    parser.add_argument("--bbox-steps", type=int, default=100, help="Reserved for advanced renderer compatibility")
    parser.add_argument("--bbox-color", type=float, nargs=3, default=[1.0, 0.0, 0.0])
    parser.add_argument("--bbox-pca-clip", type=float, default=0.0, help="Reserved for advanced renderer compatibility")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--obb", action="store_true", help="Use Open3D OBB alignment when available")
    parser.add_argument("--floor", action="store_true")
    parser.add_argument("--flip-axis", choices=["x", "y", "z"], default=None)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--keep-outlier", action="store_true", help="Reserved for advanced renderer compatibility")
    parser.add_argument("--max-points", type=int, default=30000, help="Randomly subsample for rendering speed")
    parser.add_argument("--point-size", type=float, default=0.4)
    return parser.parse_args()



def _find_latest_run(report_root: Path) -> Path:
    if not report_root.exists():
        raise FileNotFoundError(f"Report root does not exist: {report_root}")
    candidates = [p for p in report_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {report_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]



def _resolve_sources(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    explicit = any([args.run_dir, args.summary, args.ply])
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        return run_dir, run_dir / "pointcloud.ply", run_dir / "summary.json"
    if args.summary:
        summary_path = Path(args.summary).expanduser().resolve()
        run_dir = summary_path.parent
        return run_dir, run_dir / "pointcloud.ply", summary_path
    if args.ply:
        ply_path = Path(args.ply).expanduser().resolve()
        run_dir = ply_path.parent
        summary_path = run_dir / "summary.json"
        return run_dir, ply_path, summary_path if summary_path.exists() else None

    if args.latest or not explicit:
        run_dir = _find_latest_run(Path(args.report_root).expanduser().resolve())
        return run_dir, run_dir / "pointcloud.ply", run_dir / "summary.json"

    raise ValueError("Provide one of --run-dir / --summary / --ply, or use --latest.")



def _load_summary(summary_path: Path | None) -> dict[str, Any] | None:
    if summary_path is None or not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))



def _load_ascii_ply_points(ply_path: Path) -> np.ndarray:
    with ply_path.open("r", encoding="utf-8") as f:
        vertex_count = None
        line = f.readline()
        if line.strip() != "ply":
            raise ValueError(f"Unsupported PLY format in {ply_path}: missing 'ply' header")
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Malformed PLY header in {ply_path}")
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY vertex count not found in {ply_path}")
        data = np.loadtxt(f, dtype=np.float32, max_rows=vertex_count)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, :3]



def _load_points(run_dir: Path, summary: dict[str, Any] | None, ply_path: Path | None) -> tuple[np.ndarray, str]:
    candidate_npy: Path | None = None
    if summary and summary.get("output_npy"):
        candidate_npy = Path(summary["output_npy"]).expanduser().resolve()
    elif (run_dir / "pointcloud.npy").exists():
        candidate_npy = (run_dir / "pointcloud.npy").resolve()

    if candidate_npy and candidate_npy.exists():
        return np.load(candidate_npy).astype(np.float32), str(candidate_npy)

    if ply_path and ply_path.exists():
        return _load_ascii_ply_points(ply_path), str(ply_path)

    raise FileNotFoundError("Could not resolve point cloud source (.npy or .ply).")



def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]



def _pca_rotate(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    return centered @ eigvecs



def _obb_rotate(points: np.ndarray) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - best-effort optional path
        raise RuntimeError("Open3D is required for --obb alignment") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    obb = pcd.get_oriented_bounding_box(robust=True)
    R = np.asarray(obb.R, dtype=np.float32)
    center = np.asarray(obb.center, dtype=np.float32)
    return (points - center) @ R



def _apply_transforms(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if args.pca:
        pts = _pca_rotate(pts)
    if args.obb:
        pts = _obb_rotate(pts)
    if args.flip_axis:
        axis_idx = {"x": 0, "y": 1, "z": 2}[args.flip_axis]
        pts[:, axis_idx] *= -1.0
    if args.floor:
        pts[:, 2] -= pts[:, 2].min()
    if args.center:
        pts = pts - pts.mean(axis=0, keepdims=True)
    return pts



def _color_points(points: np.ndarray, color_type: str) -> np.ndarray:
    if color_type == "xyz" or color_type == "normal":
        lo = points.min(axis=0, keepdims=True)
        hi = points.max(axis=0, keepdims=True)
        colors = (points - lo) / (hi - lo + 1e-8)
        return np.clip(colors, 0.0, 1.0)

    scalar = _pca_rotate(points)[:, 0]
    q_low, q_high = np.quantile(scalar, [0.02, 0.98])
    scalar = (scalar - q_low) / (q_high - q_low + 1e-8)
    scalar = np.clip(scalar, 0.0, 1.0)
    cmap = plt.cm.get_cmap(color_type)
    return cmap(scalar)[:, :3]



def _set_equal_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    max_range = np.max(points.max(axis=0) - points.min(axis=0)) / 2.0
    max_range = max(max_range, 1e-3)
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_box_aspect((1, 1, 1))



def _draw_bbox(ax, points: np.ndarray, color: tuple[float, float, float]) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ], dtype=np.float32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        xs = [corners[i, 0], corners[j, 0]]
        ys = [corners[i, 1], corners[j, 1]]
        zs = [corners[i, 2], corners[j, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=1.0)



def _render_matplotlib_frame(points: np.ndarray,
                             colors: np.ndarray,
                             image_size: int,
                             elevation: float,
                             azim: float,
                             point_size: float,
                             bbox: bool,
                             bbox_color: tuple[float, float, float]) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=point_size,
        depthshade=False,
        linewidths=0,
    )
    if bbox:
        _draw_bbox(ax, points, color=bbox_color)
    _set_equal_axes(ax, points)
    ax.view_init(elev=elevation, azim=azim)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return img



def _try_load_advanced_renderer():
    try:
        spec = importlib.util.spec_from_file_location("probe_render_points", RENDER_POINTS_PATH)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def _resolve_actual_turntable_path(turntable_path: Path) -> Path:
    gif_path = turntable_path.with_suffix(".gif")
    if gif_path.exists() and (not turntable_path.exists() or turntable_path.stat().st_size < 1024):
        return gif_path
    return turntable_path



def _render_with_advanced_backend(args: argparse.Namespace,
                                  ply_path: Path,
                                  output_dir: Path,
                                  preview_path: Path,
                                  turntable_path: Path) -> tuple[str | None, Path | None]:
    module = _try_load_advanced_renderer()
    if module is None:
        return None, None

    try:
        import torch
    except Exception:
        return None, None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    point_cloud = module.prepare_pointcloud_from_ply(
        ply_path=str(ply_path),
        device=device,
        remove_outlier=not args.keep_outlier,
        pca=args.pca,
        obb=args.obb,
        floor=args.floor,
        flip_axis_name=args.flip_axis,
        center=args.center,
        normal_neighbors=args.normal_neighbors,
        color_type=args.color_type,
        bbox=args.bbox,
        bbox_steps=args.bbox_steps,
        bbox_color=tuple(args.bbox_color),
        bbox_pca_clip=args.bbox_pca_clip,
    )

    azim_end = args.azim_end if args.azim_end is not None else args.azim_start + 360.0
    module.save_pointcloud_frame(
        point_cloud=point_cloud,
        outfile=str(preview_path),
        distance=args.distance,
        elevation=args.elevation,
        azim=args.preview_azim,
        image_size=args.image_size,
        radius=args.radius,
        points_per_pixel=args.points_per_pixel,
        compositor=args.compositor,
    )
    module.render_turntable_video(
        point_cloud=point_cloud,
        num_frames=args.num_frames,
        distance=args.distance,
        elevation=args.elevation,
        azim_start=args.azim_start,
        azim_end=azim_end,
        fps=args.fps,
        image_size=args.image_size,
        radius=args.radius,
        points_per_pixel=args.points_per_pixel,
        compositor=args.compositor,
        outfile=str(turntable_path),
    )
    return f"advanced:{RENDER_POINTS_PATH}", _resolve_actual_turntable_path(turntable_path)



def _render_with_matplotlib_backend(args: argparse.Namespace,
                                    points: np.ndarray,
                                    preview_path: Path,
                                    turntable_path: Path) -> tuple[str, Path]:
    points = _subsample_points(points, args.max_points)
    points = _apply_transforms(points, args)
    colors = _color_points(points, args.color_type)

    preview = _render_matplotlib_frame(
        points=points,
        colors=colors,
        image_size=args.image_size,
        elevation=args.elevation,
        azim=args.preview_azim,
        point_size=args.point_size,
        bbox=args.bbox,
        bbox_color=tuple(args.bbox_color),
    )
    imageio.imwrite(preview_path, preview)

    azim_end = args.azim_end if args.azim_end is not None else args.azim_start + 360.0
    azims = np.linspace(args.azim_start, azim_end, num=args.num_frames, endpoint=False)
    frames = [
        _render_matplotlib_frame(
            points=points,
            colors=colors,
            image_size=args.image_size,
            elevation=args.elevation,
            azim=float(azim),
            point_size=args.point_size,
            bbox=args.bbox,
            bbox_color=tuple(args.bbox_color),
        )
        for azim in azims
    ]

    try:
        imageio.mimsave(turntable_path, frames, fps=args.fps)
        actual_turntable_path = turntable_path
    except Exception:
        gif_path = turntable_path.with_suffix(".gif")
        imageio.mimsave(gif_path, frames, fps=args.fps)
        actual_turntable_path = gif_path
    return "matplotlib", actual_turntable_path



def main() -> None:
    args = parse_args()
    run_dir, ply_path, summary_path = _resolve_sources(args)
    if ply_path is None or not ply_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (run_dir / "visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_path = output_dir / args.preview_name
    turntable_path = output_dir / args.turntable_name
    manifest_path = output_dir / args.manifest_name

    summary = _load_summary(summary_path)
    points, points_source = _load_points(run_dir, summary, ply_path)

    backend, actual_turntable_path = _render_with_advanced_backend(args, ply_path, output_dir, preview_path, turntable_path)
    if backend is None:
        backend, actual_turntable_path = _render_with_matplotlib_backend(args, points, preview_path, turntable_path)

    manifest = {
        "source": {
            "run_dir": str(run_dir),
            "ply": str(ply_path),
            "summary": str(summary_path) if summary_path and summary_path.exists() else None,
            "points_source": str(points_source),
        },
        "outputs": {
            "preview": str(preview_path),
            "turntable": str(actual_turntable_path),
            "manifest": str(manifest_path),
        },
        "render": {
            "backend": backend,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "image_size": args.image_size,
            "distance": args.distance,
            "elevation": args.elevation,
            "preview_azim": args.preview_azim,
            "azim_start": args.azim_start,
            "azim_end": args.azim_end if args.azim_end is not None else args.azim_start + 360.0,
            "radius": args.radius,
            "points_per_pixel": args.points_per_pixel,
            "normal_neighbors": args.normal_neighbors,
            "compositor": args.compositor,
            "color_type": args.color_type,
            "bbox": args.bbox,
            "bbox_steps": args.bbox_steps,
            "bbox_color": args.bbox_color,
            "bbox_pca_clip": args.bbox_pca_clip,
            "pca": args.pca,
            "obb": args.obb,
            "floor": args.floor,
            "flip_axis": args.flip_axis,
            "center": args.center,
            "remove_outlier": not args.keep_outlier,
            "max_points": args.max_points,
            "point_size": args.point_size,
        },
        "upstream_summary": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
