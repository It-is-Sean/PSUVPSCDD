# Copyright (c) 2026 Weirong Chen
import numpy as np
import torch

from torch_cluster import fps, knn
from pytorch3d import _C
from typing import List, Optional, Tuple, Union
from pytorch3d.ops.utils import masked_gather
from random import randint

def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("A value in lengths was too large.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Check dtypes are correct and convert if necessary
    if not (points.dtype == torch.float32):
        points = points.to(torch.float32)
    if not (lengths.dtype == torch.int64):
        lengths = lengths.to(torch.int64)
    if not (K.dtype == torch.int64):
        K = K.to(torch.int64)

    # Generate the starting indices for sampling
    start_idxs = torch.zeros_like(lengths)
    if random_start_point:
        for n in range(N):
            # pyre-fixme[6]: For 1st param expected `int` but got `Tensor`.
            start_idxs[n] = torch.randint(high=lengths[n], size=(1,)).item()

    with torch.no_grad():
        # pyre-fixme[16]: `pytorch3d_._C` has no attribute `sample_farthest_points`.
        idx = _C.sample_farthest_points(points, lengths, K, start_idxs)
    sampled_points = masked_gather(points, idx)

    return sampled_points, idx

def fps_train_gen_target(pts3d_trg, valid_trg, batch_size=8192, random_start_point=True):
    """Use FPS sampling on the target points after reordering valid points first
    """
    B, N, _ = pts3d_trg.shape
    device = pts3d_trg.device
    
    # Count valid points per batch efficiently
    valid_counts = valid_trg.sum(dim=1)  # [B]
    min_valid_count = valid_counts.min().item()
    
    # Pre-allocate result tensors
    results_pts3d = torch.zeros(B, batch_size, 3, device=device, dtype=pts3d_trg.dtype)
    results_valid = torch.ones(B, batch_size, dtype=torch.bool, device=device)
    
    if min_valid_count >= batch_size:
        # All batches have enough valid points - vectorized reordering
        
        # Use argsort for efficient reordering (valid points first)
        valid_float = valid_trg.float()
        sort_keys = valid_float + torch.rand_like(valid_float) * 0.01  # small noise for stability
        reorder_indices = torch.argsort(sort_keys, dim=1, descending=True)  # [B, N]
        
        # Gather reordered points efficiently
        reordered_pts = torch.gather(
            pts3d_trg, 
            1, 
            reorder_indices.unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, N, 3]
        
        # Use batch FPS operation
        sampled_pts, _ = sample_farthest_points(
            reordered_pts, lengths=valid_counts, K=batch_size, random_start_point=random_start_point
        )
        results_pts3d = sampled_pts
        
    else:
        # Mixed case - vectorize where possible
        sufficient_mask = valid_counts >= batch_size
        
        if sufficient_mask.any():
            # Handle sufficient batches vectorized
            suf_indices = torch.where(sufficient_mask)[0]
            suf_pts = pts3d_trg[suf_indices]
            suf_valid = valid_trg[suf_indices]
            suf_counts = valid_counts[suf_indices]
            
            # Vectorized reordering for sufficient batches
            suf_valid_float = suf_valid.float()
            sort_keys = suf_valid_float + torch.rand_like(suf_valid_float) * 0.01
            reorder_indices = torch.argsort(sort_keys, dim=1, descending=True)
            
            reordered_suf = torch.gather(
                suf_pts,
                1,
                reorder_indices.unsqueeze(-1).expand(-1, -1, 3)
            )
            
            sampled_suf, _ = sample_farthest_points(
                reordered_suf, lengths=suf_counts, K=batch_size, random_start_point=True
            )
            results_pts3d[suf_indices] = sampled_suf
        
        # Handle insufficient batches (unavoidable individual processing)
        insuf_indices = torch.where(~sufficient_mask)[0]
        for b_idx in insuf_indices:
            b = b_idx.item()
            valid_indices = torch.where(valid_trg[b])[0]
            n_valid = len(valid_indices)
            
            sampled_indices = valid_indices[torch.randint(0, n_valid, (batch_size,), device=device)]
            results_pts3d[b] = pts3d_trg[b, sampled_indices]

    return results_pts3d, results_valid


def fps_fast_v2_train_gen_target(pts3d_trg, valid_trg, batch_size=8192, oversample_ratio=4):
    """Fast FPS sampling using random oversampling first, then applying fps_train_gen_target
    
    Args:
        pts3d_trg: (B, N, 3) target 3D points
        valid_trg: (B, N) valid mask for target points
        batch_size: number of points to sample
        oversample_ratio: ratio for initial random sampling (default 4x)
    
    Returns:
        results_pts3d: (B, batch_size, 3) sampled points
        results_valid: (B, batch_size) sampled valid mask
    """
    B, N, _ = pts3d_trg.shape
    device = pts3d_trg.device
    
    # Count valid points per batch
    valid_counts = valid_trg.sum(dim=1)  # [B]
    
    # Handle batches with no valid points
    no_valid_mask = valid_counts == 0
    if no_valid_mask.any():
        results_pts3d = torch.zeros(B, batch_size, 3, device=device, dtype=pts3d_trg.dtype)
        results_valid = torch.zeros(B, batch_size, dtype=torch.bool, device=device)
        if not no_valid_mask.all():
            # Process only batches with valid points
            valid_batch_mask = ~no_valid_mask
            valid_batch_indices = torch.where(valid_batch_mask)[0]
            
            # Extract valid batches
            valid_pts3d = pts3d_trg[valid_batch_indices]
            valid_mask = valid_trg[valid_batch_indices]
            
            # Apply random sampling + FPS to valid batches
            reduced_pts3d, reduced_valid = _apply_random_then_fps(
                valid_pts3d, valid_mask, batch_size, oversample_ratio
            )
            
            # Put results back
            results_pts3d[valid_batch_indices] = reduced_pts3d
            results_valid[valid_batch_indices] = reduced_valid
        
        return results_pts3d, results_valid
    
    # All batches have valid points, process normally
    return _apply_random_then_fps(pts3d_trg, valid_trg, batch_size, oversample_ratio)


def _apply_random_then_fps(pts3d_trg, valid_trg, batch_size, oversample_ratio):
    """Helper function to apply random sampling followed by FPS"""
    B, N, _ = pts3d_trg.shape
    device = pts3d_trg.device
    
    # Calculate initial sample size
    initial_samples = min(batch_size * oversample_ratio, N)
    
    # Count valid points per batch
    valid_counts = valid_trg.sum(dim=1)  # [B]
    
    # Check if we need random pre-sampling
    max_valid = valid_counts.max().item()
    if max_valid <= initial_samples:
        # No need for random sampling, directly apply FPS
        return fps_train_gen_target(pts3d_trg, valid_trg, batch_size)
    
    # Apply random sampling first
    reduced_pts3d = torch.zeros(B, initial_samples, 3, device=device, dtype=pts3d_trg.dtype)
    reduced_valid = torch.zeros(B, initial_samples, dtype=torch.bool, device=device)
    
    for b in range(B):
        valid_indices = torch.where(valid_trg[b])[0]
        n_valid = len(valid_indices)
        
        if n_valid <= initial_samples:
            # Use all valid points + pad if necessary
            if n_valid < initial_samples:
                # Pad with repetition
                pad_indices = valid_indices[torch.randint(0, n_valid, (initial_samples - n_valid,), device=device)]
                sampled_indices = torch.cat([valid_indices, pad_indices])
            else:
                sampled_indices = valid_indices
        else:
            # Random sampling
            random_selection = torch.randperm(n_valid, device=device)[:initial_samples]
            sampled_indices = valid_indices[random_selection]
        
        reduced_pts3d[b] = pts3d_trg[b, sampled_indices]
        reduced_valid[b] = True  # All sampled points are valid
    
    # Now apply FPS on the reduced set
    return fps_train_gen_target(reduced_pts3d, reduced_valid, batch_size)


def sharp_edge_fast_train_gen_target(pts3d_trg, valid_trg, batch_size=8192, oversample_ratio=4, k=16, sharp_threshold=0.985):
    """Fast sharp edge sampling using random oversampling first, similar to fps_train_gen_target interface
    
    Args:
        pts3d_trg: (B, N, 3) target 3D points
        valid_trg: (B, N) valid mask for target points
        batch_size: number of points to sample
        oversample_ratio: ratio for initial random sampling (default 4x)
        k: number of neighbors for normal estimation and edge detection
        sharp_threshold: threshold for determining sharp edges (lower = sharper)
    
    Returns:
        results_pts3d: (B, batch_size, 3) sampled points
        results_valid: (B, batch_size) sampled valid mask
    """
    from torch_cluster import knn
    
    B, N, _ = pts3d_trg.shape
    device = pts3d_trg.device
    
    # Count valid points per batch
    valid_counts = valid_trg.sum(dim=1)  # [B]
    
    # Pre-allocate result tensors
    results_pts3d = torch.zeros(B, batch_size, 3, device=device, dtype=pts3d_trg.dtype)
    results_valid = torch.ones(B, batch_size, dtype=torch.bool, device=device)
    
    # Handle batches with no valid points
    no_valid_mask = valid_counts == 0
    if no_valid_mask.any():
        results_valid[no_valid_mask] = False
        
    # Process batches with valid points
    valid_batch_mask = ~no_valid_mask
    if not valid_batch_mask.any():
        return results_pts3d, results_valid

    for b in range(B):
        if not valid_batch_mask[b]:
            continue

        valid_mask = valid_trg[b]
        valid_indices = torch.where(valid_mask)[0]
        n_valid = len(valid_indices)

        if n_valid < batch_size:
            # Sample with replacement
            sampled_indices = valid_indices[torch.randint(0, n_valid, (batch_size,), device=device)]
            results_pts3d[b] = pts3d_trg[b, sampled_indices]
        else:
            # Extract valid points
            valid_points = pts3d_trg[b, valid_indices]  # (n_valid, 3)
            
            # First random sampling to reduce computational cost
            initial_samples = min(batch_size * oversample_ratio, n_valid)
            
            if initial_samples < n_valid:
                # Random sampling for initial selection
                random_indices = torch.randperm(n_valid, device=device)[:initial_samples]
                points_reduced = valid_points[random_indices]
            else:
                points_reduced = valid_points
            
            N_reduced = points_reduced.shape[0]
            
            if N_reduced < k + 1:
                # Not enough points for sharp edge detection, fallback to random
                sampled_indices = torch.randperm(N_reduced, device=device)[:batch_size]
                if len(sampled_indices) < batch_size:
                    # Pad with repetition
                    needed = batch_size - len(sampled_indices)
                    extra_indices = torch.randint(0, N_reduced, (needed,), device=device)
                    sampled_indices = torch.cat([sampled_indices, extra_indices])
                results_pts3d[b] = points_reduced[sampled_indices]
                continue
            
            # Find neighbors using torch_cluster knn
            batch_tensor = torch.zeros(N_reduced, dtype=torch.long, device=device)
            edge_index = knn(points_reduced, points_reduced, k=k+1, batch_x=batch_tensor, batch_y=batch_tensor)
            
            # Reshape edge indices to (N_reduced, k+1)
            edge_index = edge_index.view(2, N_reduced, k+1)
            neighbor_indices = edge_index[1, :, 1:]  # exclude self, shape (N_reduced, k)
            
            # Estimate normals using the neighbor indices
            neighbors = points_reduced[neighbor_indices]  # (N_reduced, k, 3)
            centered = neighbors - points_reduced.unsqueeze(1)  # (N_reduced, k, 3)
            
            # Compute covariance matrices for all points at once
            cov = torch.bmm(centered.transpose(-1, -2), centered)  # (N_reduced, 3, 3)
            
            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = torch.linalg.eigh(cov)
            normals = eigenvecs[:, :, 0]  # smallest eigenvalue corresponds to normal
            
            # Compute sharpness measure using the same neighbor indices
            neighbor_normals = normals[neighbor_indices]  # (N_reduced, k, 3)
            
            # Dot product with all neighbors
            dots = torch.sum(neighbor_normals * normals.unsqueeze(1), dim=-1)  # (N_reduced, k)
            sharpness = torch.min(dots, dim=1)[0]  # (N_reduced,)
            
            # Identify sharp points
            sharp_mask = sharpness < sharp_threshold
            sharp_indices = torch.where(sharp_mask)[0]
            
            if len(sharp_indices) >= batch_size:
                # Sample from sharp points
                selected_sharp = sharp_indices[torch.randperm(len(sharp_indices), device=device)[:batch_size]]
                results_pts3d[b] = points_reduced[selected_sharp]
            elif len(sharp_indices) > 0:
                # Not enough sharp points, use all sharp + random remaining
                sharp_samples = points_reduced[sharp_indices]
                remaining_needed = batch_size - len(sharp_indices)
                non_sharp_mask = ~sharp_mask
                non_sharp_indices = torch.where(non_sharp_mask)[0]
                
                if len(non_sharp_indices) >= remaining_needed:
                    random_non_sharp = non_sharp_indices[torch.randperm(len(non_sharp_indices), device=device)[:remaining_needed]]
                    random_samples = points_reduced[random_non_sharp]
                    combined_samples = torch.cat([sharp_samples, random_samples], dim=0)
                else:
                    # Still not enough, pad with repetitions
                    all_remaining = torch.cat([sharp_samples, points_reduced[non_sharp_indices]], dim=0)
                    needed = batch_size - len(all_remaining)
                    if needed > 0:
                        repeated_indices = torch.randint(0, len(all_remaining), (needed,), device=device)
                        repeated_samples = all_remaining[repeated_indices]
                        combined_samples = torch.cat([all_remaining, repeated_samples], dim=0)
                    else:
                        combined_samples = all_remaining[:batch_size]
                
                results_pts3d[b] = combined_samples
            else:
                # No sharp points found, random sampling
                sampled_indices = torch.randperm(N_reduced, device=device)[:batch_size]
                if len(sampled_indices) < batch_size:
                    # Pad with repetition
                    needed = batch_size - len(sampled_indices)
                    extra_indices = torch.randint(0, N_reduced, (needed,), device=device)
                    sampled_indices = torch.cat([sampled_indices, extra_indices])
                results_pts3d[b] = points_reduced[sampled_indices]
    
    return results_pts3d, results_valid


def inf_train_gen_target(pts3d_trg, valid_trg, batch_size=8192):
    """Generate the target points for training
    """
    results_pts3d = []
    results_valid = []
    
    for b in range(pts3d_trg.shape[0]):
        # Extract valid indices for this batch item
        valid_indices = torch.where(valid_trg[b])[0]
        

        if len(valid_indices) >= batch_size:
            # Sample without replacement
            sampled_indices = valid_indices[torch.randperm(len(valid_indices), device=pts3d_trg.device)[:batch_size]]
            sampled_pts3d = pts3d_trg[b, sampled_indices]
            sampled_valid = valid_trg[b, sampled_indices]
        else:
            # Sample with replacement
            sampled_indices = valid_indices[torch.randint(0, len(valid_indices), (batch_size,), device=pts3d_trg.device)]
            sampled_pts3d = pts3d_trg[b, sampled_indices]
            sampled_valid = valid_trg[b, sampled_indices]
        
        results_pts3d.append(sampled_pts3d)
        results_valid.append(sampled_valid)
    
    return torch.stack(results_pts3d, dim=0), torch.stack(results_valid, dim=0)

def sampling_train_gen_target(pts3d_trg, valid_trg, batch, target_sampling, batch_size=8192):
    """Args:
        pts3d_trg: (B, N, 3) target 3D points
        valid_trg: (B, N) valid mask for target points
        batch: batch data (unused, kept for API compatibility)
        target_sampling: 'none', 'random', 'fps', 'fps_fast', 'edge_fast', or 'fps_edge_fast'
        batch_size: number of points to sample

    Returns:
        tuple: (sampled_pts3d, sampled_valid)
    """

    if target_sampling == 'none':
        x_1, valid = pts3d_trg, valid_trg
    elif target_sampling == 'random':
        x_1, valid = inf_train_gen_target(pts3d_trg, valid_trg, batch_size=batch_size)
    elif target_sampling == 'fps':
        x_1, valid = fps_train_gen_target(pts3d_trg, valid_trg, batch_size=batch_size)
    elif target_sampling == 'fps_fast':
        x_1, valid = fps_fast_v2_train_gen_target(pts3d_trg, valid_trg, batch_size=batch_size, oversample_ratio=4)
    elif target_sampling == 'edge_fast':
        x_1, valid = sharp_edge_fast_train_gen_target(pts3d_trg, valid_trg, batch_size=batch_size, oversample_ratio=4, k=16, sharp_threshold=0.985)
    elif target_sampling == 'fps_edge_fast':
        x_1, valid = fps_edge_fast_train_gen_target(pts3d_trg, valid_trg, batch_size=batch_size, oversample_ratio=6, fps_ratio=0.7)
    else:
        raise ValueError(f"Unknown target sampling method: {target_sampling}")

    return x_1, valid

def fps_edge_fast_train_gen_target(pts3d_trg, valid_trg, batch_size=8192, fps_ratio=0.5, oversample_ratio=4, k=16, sharp_threshold=0.985):
    """Hybrid fast sampling combining FPS and sharp edge sampling
    
    Args:
        pts3d_trg: (B, N, 3) target 3D points
        valid_trg: (B, N) valid mask for target points
        batch_size: number of points to sample
        fps_ratio: ratio of samples from FPS (remaining from sharp edge)
        oversample_ratio: ratio for initial random sampling (default 4x)
        k: number of neighbors for sharp edge detection
        sharp_threshold: threshold for determining sharp edges (lower = sharper)
    
    Returns:
        results_pts3d: (B, batch_size, 3) sampled points
        results_valid: (B, batch_size) sampled valid mask
    """
    # Calculate number of samples for each method
    num_fps = int(batch_size * fps_ratio)
    num_sharp = batch_size - num_fps
    
    # Get FPS samples
    if num_fps > 0:
        fps_samples, fps_valid = fps_fast_v2_train_gen_target(
            pts3d_trg, valid_trg, batch_size=num_fps, oversample_ratio=oversample_ratio
        )
    else:
        B = pts3d_trg.shape[0]
        device = pts3d_trg.device
        fps_samples = torch.zeros(B, 0, 3, device=device, dtype=pts3d_trg.dtype)
        fps_valid = torch.zeros(B, 0, dtype=torch.bool, device=device)
    
    # Get sharp edge samples
    if num_sharp > 0:
        sharp_samples, sharp_valid = sharp_edge_fast_train_gen_target(
            pts3d_trg, valid_trg, num_sharp, oversample_ratio, k, sharp_threshold
        )
    else:
        B = pts3d_trg.shape[0]
        device = pts3d_trg.device
        sharp_samples = torch.zeros(B, 0, 3, device=device, dtype=pts3d_trg.dtype)
        sharp_valid = torch.zeros(B, 0, dtype=torch.bool, device=device)
    
    # Combine samples
    if num_fps > 0 and num_sharp > 0:
        combined_samples = torch.cat([fps_samples, sharp_samples], dim=1)
        combined_valid = torch.cat([fps_valid, sharp_valid], dim=1)
    elif num_fps > 0:
        combined_samples = fps_samples
        combined_valid = fps_valid
    else:
        combined_samples = sharp_samples
        combined_valid = sharp_valid
    
    return combined_samples, combined_valid

