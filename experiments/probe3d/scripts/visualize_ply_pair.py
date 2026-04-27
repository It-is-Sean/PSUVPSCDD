import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to predicted point cloud .ply")
    parser.add_argument("--gt", required=True, help="Path to ground-truth point cloud .ply")
    parser.add_argument("--output", required=True, help="Output .png path")
    parser.add_argument("--max_points", type=int, default=4096)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def load_ascii_ply(path):
    vertex_count = None
    end_header_idx = None
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("element vertex "):
            vertex_count = int(stripped.split()[-1])
        if stripped == "end_header":
            end_header_idx = idx
            break

    if vertex_count is None or end_header_idx is None:
        raise ValueError(f"Unsupported or malformed PLY file: {path}")

    data_lines = lines[end_header_idx + 1 : end_header_idx + 1 + vertex_count]
    points = np.array([[float(value) for value in line.split()[:3]] for line in data_lines], dtype=np.float32)
    return points


def maybe_subsample(points, max_points):
    if points.shape[0] <= max_points:
        return points
    stride = max(1, points.shape[0] // max_points)
    return points[::stride][:max_points]


def set_equal_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_pair(pred_points, gt_points, output_path, title=None):
    combined = np.concatenate([pred_points, gt_points], axis=0)
    fig = plt.figure(figsize=(12, 6), dpi=200)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, points, panel_title, color in [
        (axes[0], pred_points, "Prediction", "#2563eb"),
        (axes[1], gt_points, "Pseudo GT", "#dc2626"),
    ]:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, c=color, alpha=0.7, linewidths=0)
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        set_equal_axes(ax, combined)
        ax.view_init(elev=18, azim=35)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    pred_points = maybe_subsample(load_ascii_ply(args.pred), args.max_points)
    gt_points = maybe_subsample(load_ascii_ply(args.gt), args.max_points)
    render_pair(pred_points, gt_points, Path(args.output), title=args.title)
    print(args.output)


if __name__ == "__main__":
    main()
