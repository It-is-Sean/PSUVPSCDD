#!/usr/bin/env python3
# Copyright (c) 2026 Weirong Chen
"""NOVA3R AE Demo: 3D point cloud reconstruction via autoencoder."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import numpy as np
import open3d as o3d
import time

from omegaconf import OmegaConf
from nova3r.models.nova3r_pts_cond import Nova3rPtsCond
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from nova3r.inference import normalize_input, amp_dtype_mapping
from demo_nova3r import load_model, save_pointcloud, render_360_video


def parse_args():
    parser = argparse.ArgumentParser(description="NOVA3R AE: 3D point cloud reconstruction via autoencoder")
    parser.add_argument("--input_ply", required=True,
                        help="Path to input PLY file")
    parser.add_argument("--ckpt", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="demo/outputs/",
                        help="Output directory (default: demo/outputs/)")
    parser.add_argument("--num_queries", type=int, default=50000,
                        help="Number of query points (default: 50000)")
    parser.add_argument("--device", default="cuda",
                        help="Device (default: cuda)")
    return parser.parse_args()


def load_pointcloud(ply_path, max_points=50000):
    """Load a point cloud from a PLY file and subsample if needed."""
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points).astype(np.float32)

    if len(pts) > max_points:
        indices = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[indices]
        print(f"Subsampled point cloud from {len(np.asarray(pcd.points))} to {max_points} points")
    else:
        print(f"Loaded point cloud with {len(pts)} points")

    return pts


def run_inference(model, cfg, pts_np, device, num_queries=50000):
    """Run AE inference: encode point cloud, then solve ODE to generate output."""
    B = 1

    pts = torch.from_numpy(pts_np).unsqueeze(0).to(device)  # (1, N, 3)
    valid = torch.ones(B, pts.shape[1], dtype=torch.bool, device=device)

    norm_mode = cfg.model.params.cfg.pts3d_head.params.get('norm_mode', 'none')
    pts, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)

    encoder_data = model._encode(pointmaps=pts, test=True)

    # Dummy images tensor (AE doesn't use images but _decode expects the shape)
    images = torch.zeros(B, 1, 3, 1, 1, device=device)

    x_init = torch.rand(B, num_queries, 3, device=device) * 2 - 1

    wrapper = BatchModelWrapper(model=model)
    solver = ODESolver(velocity_model=wrapper)

    step_size = cfg.get("fm_step_size", 0.04)
    method = cfg.get("fm_sampling", "euler")
    num_steps = int(1 // step_size)

    amp_dtype_key = cfg.get("amp_dtype", "bf16")
    amp_dtype = amp_dtype_mapping.get(amp_dtype_key, torch.float32)

    T = torch.linspace(0, 1, num_steps).to(device)

    use_amp = (device != "cpu")
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
        sol = solver.sample(
            time_grid=T,
            x_init=x_init,
            method=method,
            step_size=step_size,
            return_intermediates=False,
            images=images,
            token_mask=None,
            encoder_data=encoder_data,
            pointmaps=pts,
        )

    pts3d_xyz = sol[-1] if isinstance(sol, list) else sol
    return pts3d_xyz


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    model, cfg = load_model(args.ckpt, args.device)

    # Set inference defaults if not in the saved config
    OmegaConf.set_struct(cfg, False)
    if "fm_step_size" not in cfg:
        cfg.fm_step_size = 0.04
    if "fm_sampling" not in cfg:
        cfg.fm_sampling = "euler"
    OmegaConf.set_struct(cfg, True)

    print(f"Loading point cloud: {args.input_ply}")
    pts_np = load_pointcloud(args.input_ply, max_points=args.num_queries)

    input_name = os.path.splitext(os.path.basename(args.input_ply))[0]
    scene_dir = os.path.join(args.output_dir, input_name)
    os.makedirs(scene_dir, exist_ok=True)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        pts3d = run_inference(model, cfg, pts_np, args.device, num_queries=args.num_queries)

    elapsed = time.time() - start_time
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Inference: {elapsed:.2f}s | Peak memory: {peak_mem:.0f} MB")
    else:
        print(f"Inference: {elapsed:.2f}s")

    ply_path = save_pointcloud(pts3d, scene_dir)
    render_360_video(ply_path, scene_dir)

    print("Done!")


if __name__ == "__main__":
    main()
