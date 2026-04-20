# Copyright (c) 2026 Weirong Chen
#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch

from pytorch3d.io import load_ply
from pytorch3d.ops import estimate_pointcloud_normals
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.ops import knn_points
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def smooth_pointcloud_surface(point_cloud: Pointclouds, device: torch.device, 
                             num_iterations: int = 3, 
                             neighbor_size: int = 30) -> Pointclouds:
    """
    Smooth point cloud surface using iterative neighbor averaging.
    
    Args:
        point_cloud: Input point cloud
        device: Device to perform computation
        num_iterations: Number of smoothing iterations
        neighbor_size: Number of neighbors to consider for averaging
        
    Returns:
        Smoothed point cloud with same features
    """
    print(f"Smoothing point cloud with {num_iterations} iterations and {neighbor_size} neighbors...")
    points = point_cloud.unsqueeze(0).to(device)  # [1, N, 3]
    # features = point_cloud.features_packed()
    
    for _ in range(num_iterations):
        # Find k nearest neighbors
        knn_result = knn_points(points, points, K=neighbor_size + 1)
        knn_idx = knn_result.idx[0, :, 1:]  # [N, K], exclude self
        
        # Average neighbor positions
        neighbor_points = points[0, knn_idx]  # [N, K, 3]
        smoothed_points = neighbor_points.mean(dim=1)  # [N, 3]
        
        # Blend with original (Laplacian smoothing)
        points = 0.5 * points + 0.5 * smoothed_points.unsqueeze(0)
        # points = points.unsqueeze(0)
    
    return points[0]

    
    # return Pointclouds(points=[points[0].to(device)], features=[features])


def load_ply_pointcloud(ply_path: str, device: torch.device, remove_outlier: bool) -> Pointclouds:
    verts, faces = load_ply(ply_path)

    if remove_outlier:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts.cpu().numpy())
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        verts = torch.from_numpy(np.asarray(pcd.points)[ind]).to(verts.dtype)

    # verts = smooth_pointcloud_surface(verts, device=device, num_iterations=2, neighbor_size=20)
    #     from sklearn.neighbors import NearestNeighbors
    rgb = torch.full((verts.shape[0], 3), 0.7).to(verts.dtype)

    verts = verts.to(device)
    rgb = rgb.to(device)

    # Center
    center = verts.mean(dim=0, keepdim=True)
    verts = verts - center

    # Normalize to unit box (fit into [-1, 1])
    max_abs = verts.abs().max(dim=0).values.max()
    verts = verts / (max_abs + 1e-8)
    verts = verts * 0.7

    return Pointclouds(points=[verts], features=[rgb])


def pca_align_pointcloud(point_cloud: Pointclouds) -> Pointclouds:
    # PCA in point space: center then rotate into principal component basis
    pts = point_cloud.points_list()[0]
    feats = point_cloud.features_list()[0]

    mean = pts.mean(dim=0, keepdim=True)
    centered = pts - mean

    # Covariance and eigen-decomposition (symmetric)
    cov = centered.t().mm(centered) / max(centered.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    order = torch.argsort(evals, descending=True)
    evecs = evecs[:, order]

    # Rotate into PCA basis
    rotated = centered @ evecs

    return Pointclouds(points=[rotated], features=[feats])


def obb_align_pointcloud(point_cloud: Pointclouds) -> Pointclouds:
    # Align point cloud to Open3D oriented bounding box axes
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("Open3D is required for OBB alignment. Please install open3d.") from exc

    pts = point_cloud.points_list()[0]
    feats = point_cloud.features_list()[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy())
    obb = pcd.get_oriented_bounding_box(robust=True)

    R = torch.from_numpy(np.array(obb.R, copy=True)).to(pts.dtype).to(pts.device)
    center = torch.from_numpy(np.array(obb.center, copy=True)).to(pts.dtype).to(pts.device)

    centered = pts - center
    aligned = centered @ R

    return Pointclouds(points=[aligned], features=[feats])


def place_on_floor(point_cloud: Pointclouds, axis: str = "y") -> Pointclouds:
    pts = point_cloud.points_list()[0]
    feats = point_cloud.features_list()[0]

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    min_val = pts[:, axis_idx].min()
    shifted = pts.clone()
    shifted[:, axis_idx] = shifted[:, axis_idx] - min_val
    # reverse y


    return Pointclouds(points=[shifted], features=[feats])


def flip_axis(point_cloud: Pointclouds, axis: str = "y") -> Pointclouds:
    pts = point_cloud.points_list()[0]
    feats = point_cloud.features_list()[0]

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    flipped = pts.clone()
    flipped[:, axis_idx] = -flipped[:, axis_idx]
    return Pointclouds(points=[flipped], features=[feats])


def center_pointcloud(point_cloud: Pointclouds) -> Pointclouds:
    pts = point_cloud.points_list()[0]
    feats = point_cloud.features_list()[0]
    mean = pts.mean(dim=0, keepdim=True)
    centered = pts - mean
    return Pointclouds(points=[centered], features=[feats])


def build_colored_pointcloud(point_cloud: Pointclouds, device: torch.device,
                                   normal_neighbors: int = 60, color_type='normal') -> Pointclouds:
    # Estimate normals on CPU to avoid GPU OOM for large point clouds
    if color_type == 'normal':
        pc_cpu = point_cloud.to("cpu")
        normals = estimate_pointcloud_normals(pc_cpu, neighborhood_size=normal_neighbors)

        # Map normals from [-1, 1] to [0, 1] for RGB coloring
        normal_colors = (normals + 1.0) * 0.5

        # Move features back to device
        if normal_colors.dim() == 2:
            features = [normal_colors.to(device)]
        elif normal_colors.dim() == 3:
            features = [normal_colors[0].to(device)]
        else:
            features = normal_colors.to(device)
    elif color_type in ['plasma', 'viridis']:
        pts = point_cloud.points_list()[0]
        feats = point_cloud.features_list()[0]

        # Compute PCA of point cloud
        mean = pts.mean(dim=0, keepdim=True)
        centered = pts - mean
        cov = centered.t().mm(centered) / max(centered.shape[0] - 1, 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=False)
        evecs = evecs[:, order]
        
        # Project onto first principal component
        pca_coords = centered @ evecs[:, 0]
        
        # Normalize to [0, 1] using quantiles
        coord_min, coord_max = torch.quantile(pca_coords, 0.02), torch.quantile(pca_coords, 0.98)
        z_normalized = (pca_coords - coord_min) / (coord_max - coord_min + 1e-8)
        # Use plasma colormap
        plasma_cmap = plt.cm.get_cmap(color_type)
        height_colors = torch.from_numpy(
            plasma_cmap(z_normalized.cpu().numpy())[:, :3]
        ).float()
        features = [height_colors.to(device)]
    elif color_type == 'xyz':
        pts = point_cloud.points_list()[0]
        xyz_min = pts.min(dim=0).values
        xyz_max = pts.max(dim=0).values
        xyz_norm = (pts - xyz_min) / (xyz_max - xyz_min + 1e-8)
        features = [xyz_norm.clamp(0.0, 1.0).to(device)]
    else:
        raise ValueError(
            f"Unknown color_type '{color_type}'. Use 'normal', 'plasma', 'viridis', or 'xyz'."
        )

    return Pointclouds(points=point_cloud.points_list(), features=features)

def add_bounding_box_points(point_cloud: Pointclouds, device: torch.device,
                            color=(1.0, 0.0, 0.0), steps: int = 100,
                            pca_clip: float = 0.0) -> Pointclouds:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("Open3D is required for oriented bounding box drawing.") from exc

    pts = point_cloud.points_list()[0]
    feats = point_cloud.features_list()[0]

    obb_pts = pts
    if pca_clip > 0.0:
        # Drop top/bottom pca_clip quantile along the first PCA axis for OBB only.
        mean = obb_pts.mean(dim=0, keepdim=True)
        centered = obb_pts - mean
        cov = centered.t().mm(centered) / max(centered.shape[0] - 1, 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=False)
        evecs = evecs[:, order]
        pca_coords = centered @ evecs[:, 0]
        q_low = torch.quantile(pca_coords, pca_clip)
        q_high = torch.quantile(pca_coords, 1.0 - pca_clip)
        mask = (pca_coords >= q_low) & (pca_coords <= q_high)
        if mask.any():
            obb_pts = obb_pts[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obb_pts.detach().cpu().numpy())
    obb = pcd.get_oriented_bounding_box()
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    box_pts_np = np.asarray(lineset.points, dtype=np.float32)
    edges = np.asarray(lineset.lines, dtype=np.int64)

    t = torch.linspace(0.0, 1.0, steps=steps, device=pts.device, dtype=pts.dtype)
    box_pts = []
    box_pts_tensor = torch.from_numpy(box_pts_np).to(device=pts.device, dtype=pts.dtype)
    for i, j in edges:
        start = box_pts_tensor[i]
        end = box_pts_tensor[j]
        seg = start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]
        box_pts.append(seg)

    box_pts = torch.cat(box_pts, dim=0)
    box_color = torch.tensor(color, device=pts.device, dtype=feats.dtype).view(1, 3)
    box_feats = box_color.repeat(box_pts.shape[0], 1)

    merged_pts = torch.cat([pts, box_pts], dim=0)
    merged_feats = torch.cat([feats, box_feats], dim=0)

    return Pointclouds(points=[merged_pts.to(device)], features=[merged_feats.to(device)])


def render_turntable_video(point_cloud: Pointclouds,
                            num_frames: int = 120,
                            distance: float = 20,
                            elevation: float = 10,
                            azim_start: float = 0,
                            azim_end: float = 360,
                            fps: int = 30,
                            image_size: int = 512,
                            radius: float = 0.003,
                            points_per_pixel: int = 10,
                            background_color=(1.0, 1.0, 1.0),
                            compositor: str = "alpha",
                            outfile: str = "outputs/pointcloud_360_normals.mp4"):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel,
    )

    frames = []
    azims = np.linspace(azim_start, azim_end, num_frames, endpoint=False)

    for azim in azims:
        R, T = look_at_view_transform(distance, elevation, azim)
        cameras = FoVOrthographicCameras(device=point_cloud.device, R=R, T=T, znear=0.01)
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        if compositor == "alpha":
            compositor_impl = AlphaCompositor(background_color=background_color)
        elif compositor == "normweighted":
            compositor_impl = NormWeightedCompositor(background_color=background_color)
        else:
            raise ValueError(f"Unknown compositor '{compositor}'. Use 'alpha' or 'normweighted'.")

        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor_impl)
        images = renderer(point_cloud)
        img = images[0, ..., :3].detach().cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        frames.append(img)

    try:
        imageio.mimsave(outfile, frames, fps=fps)
        print(f"Saved video to {outfile}")
    except Exception as e:
        fallback = outfile.rsplit(".", 1)[0] + ".gif"
        imageio.mimsave(fallback, frames, fps=fps)
        print(f"MP4 save failed ({e}). Saved GIF to {fallback}")


def parse_args():
    p = argparse.ArgumentParser(description="Render 360 normal-colored point cloud video from a PLY file.")
    p.add_argument("--ply", help="Path to input .ply file")
    p.add_argument("--ply-folder", help="Folder containing .ply files")
    p.add_argument("--out", default="outputs/pointcloud_360_normals.mp4", help="Output video path")
    p.add_argument("--save-folder", default=None, help="Output folder for batch rendering")
    p.add_argument("--num-frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--distance", type=float, default=20)
    p.add_argument("--elevation", type=float, default=10)
    p.add_argument("--azim-start", type=float, default=0.0, help="Starting azimuth angle (degrees)")
    p.add_argument("--azim-end", type=float, default=None,
                   help="Ending azimuth angle (degrees). If unset, uses azim_start + 360.")
    p.add_argument("--radius", type=float, default=0.005)
    p.add_argument("--points-per-pixel", type=int, default=10)
    p.add_argument("--normal-neighbors", type=int, default=60)
    p.add_argument("--compositor", choices=["alpha", "normweighted"], default="alpha",
                   help="Compositor type for point blending")
    p.add_argument("--color-type", choices=["normal", "plasma", "viridis", "xyz"], default="viridis",
                   help="Point color scheme")
    p.add_argument("--bbox", action="store_true", help="Draw oriented bounding box")
    p.add_argument("--bbox-steps", type=int, default=100, help="Samples per bbox edge")
    p.add_argument("--bbox-color", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                   help="BBox color as three floats in [0,1]")
    p.add_argument("--bbox-pca-clip", type=float, default=0.0,
                   help="Drop top/bottom quantile along first PCA axis for OBB (e.g. 0.01)")
    p.add_argument("--pca", action="store_true", help="Align point cloud to PCA axes before rendering")
    p.add_argument("--obb", action="store_true", help="Align point cloud to Open3D OBB axes before rendering")
    p.add_argument("--floor", action="store_true", help="Shift point cloud so min z is at 0")
    p.add_argument("--flip-axis", choices=["x", "y", "z"], default=None,
                   help="Flip point cloud along axis (useful if upside-down)")
    p.add_argument("--center", action="store_true", help="Center point cloud at origin")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if not args.ply and not args.ply_folder:
        raise ValueError("Provide --ply or --ply-folder.")
    if args.ply and args.ply_folder:
        raise ValueError("Provide only one of --ply or --ply-folder.")
    if args.ply_folder and not args.save_folder:
        raise ValueError("When using --ply-folder, also provide --save-folder.")

    def render_one(ply_path: str, out_path: str) -> None:
        point_cloud = load_ply_pointcloud(ply_path, device=device, remove_outlier=True)
        if args.pca:
            point_cloud = pca_align_pointcloud(point_cloud)
        if args.obb:
            point_cloud = obb_align_pointcloud(point_cloud)
        if args.flip_axis:
            point_cloud = flip_axis(point_cloud, axis=args.flip_axis)
        if args.floor:
            point_cloud = place_on_floor(point_cloud, axis="z")
        if args.center:
            point_cloud = center_pointcloud(point_cloud)
        point_cloud_normals = build_colored_pointcloud(
            point_cloud, device=device, normal_neighbors=args.normal_neighbors,
            color_type=args.color_type
        )
        if args.bbox:
            point_cloud_normals = add_bounding_box_points(
                point_cloud_normals,
                device=device,
                color=tuple(args.bbox_color),
                steps=args.bbox_steps,
                pca_clip=args.bbox_pca_clip,
            )

        azim_end = args.azim_end
        if azim_end is None:
            azim_end = args.azim_start + 360.0

        render_turntable_video(
            point_cloud_normals,
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
            outfile=out_path,
        )

    if args.ply:
        render_one(args.ply, args.out)
    else:
        os.makedirs(args.save_folder, exist_ok=True)
        ply_files = [f for f in os.listdir(args.ply_folder) if f.lower().endswith(".ply")]
        ply_files.sort()
        if not ply_files:
            raise ValueError(f"No .ply files found in {args.ply_folder}")
        for name in ply_files:
            ply_path = os.path.join(args.ply_folder, name)
            out_path = os.path.join(args.save_folder, os.path.splitext(name)[0] + ".mp4")
            render_one(ply_path, out_path)


if __name__ == "__main__":
    main()
