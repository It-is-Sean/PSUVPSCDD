# Copyright (c) 2026 Weirong Chen
import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from dust3r.utils.geometry import geotrf, inv
from nova3r.utils.sampling import sampling_train_gen_target
from einops import rearrange
from pytorch3d.loss import chamfer_distance

@torch.no_grad()
def get_joint_pointcloud_center_scale(pts, valid_masks=None, z_only=False, center=True):
    # pts: [B, N, 3], valid_masks: [B, N]
    
    if valid_masks is not None:
        # Set invalid points to NaN
        _pts = pts.clone()
        _pts[~valid_masks] = float('nan')
    else:
        _pts = pts

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B, 1, 3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center, scale


def get_joint_pointcloud_depth(zs, valid_masks=None, quantile=0.5):
    # zs: [B, N, 3], valid_masks: [B, N]
    
    if valid_masks is not None:
        # Set invalid points to NaN
        _zs = zs.clone()
        _zs[~valid_masks] = float('nan')
    else:
        _zs = zs

    # Extract z coordinates and flatten
    _z_coords = _zs[..., 2].reshape(_zs.shape[0], -1)  # [B, N] -> [B, N]

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_z_coords, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z_coords, quantile, dim=-1)
    return shift_z  # (B,)



def preprocess_data(args, gts, preds, use_gt_scale=True, sampling=None, num_sample=50000, scale_inv=True, shift_inv=True, use_complete=False):
    """compute shift and scale alignment according to CUT3R mv_recon"""
    if use_complete:
        in_camera1 = inv(gts[0]['camera_pose'])
        gt_pts_list = [geotrf(in_camera1, gt['pts3d_complete']) for gt in gts]
        gt_pts = torch.stack(gt_pts_list, dim=1)
        valid_num_list = [gt['pts3d_complete_valid_num'] for gt in gts]  # B, S
        valid = torch.zeros_like(gt_pts[..., 0]).bool()  # B, S, N
        for i in range(len(gts)):
            for j in range(valid_num_list[i].shape[0]):
                valid[j, i, :valid_num_list[i][j]] = True

        gt_xyz = rearrange(gt_pts, 'b s n c -> b (s n) c')
        gt_mask = rearrange(valid, 'b s n -> b (s n)')
    else:
        gt_pts_list = [gt['pts3d'] for gt in gts]
        in_camera1 = inv(gts[0]['camera_pose'])
        gt_pts_list = [geotrf(in_camera1, gt['pts3d']) for gt in gts]
        gt_pts = torch.stack(gt_pts_list, dim=1)

        valid_list = [gt['valid_mask'].clone() for gt in gts]
        valid = torch.stack(valid_list, dim=1).float()

        gt_xyz = rearrange(gt_pts, 'b s h w c -> b (s h w) c')
        gt_mask = rearrange(valid, 'b s h w -> b (s h w)')


    images = preds['images']

    gt_mask = gt_mask > 0.5

    pred_xyz = preds['pts3d_xyz']    # [B, N, 3]

    # First shift: align depth centers
    if shift_inv:
        gt_shift_z = get_joint_pointcloud_depth(gt_xyz, gt_mask)
        pred_shift_z = get_joint_pointcloud_depth(pred_xyz)

        gt_xyz[..., 2] -= gt_shift_z.unsqueeze(-1)
        pred_xyz[..., 2] -= pred_shift_z.unsqueeze(-1)
    
    # Then scale: align scales
    if scale_inv:
        gt_center, gt_scale = get_joint_pointcloud_center_scale(gt_xyz, gt_mask)
        pred_center, pred_scale = get_joint_pointcloud_center_scale(pred_xyz)

        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        if use_gt_scale:
            pred_xyz *= gt_scale.unsqueeze(-1) / pred_scale.unsqueeze(-1)
        else:
            pred_xyz *= pred_scale.unsqueeze(-1) / gt_scale.unsqueeze(-1)
            gt_xyz *= gt_scale.unsqueeze(-1) / pred_scale.unsqueeze(-1)

    pts = pred_xyz.cpu().numpy()[0]
    pts_gt = gt_xyz.cpu().numpy()[0]

    mask = gt_mask.cpu().numpy()[0]
    return pts, pts_gt, mask



from scipy.spatial import cKDTree as KDTree
from pytorch3d.ops import iterative_closest_point



def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Compute mean and median accuracy (distance from reconstructed to nearest GT point)."""
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Compute mean and median completion (distance from GT to nearest reconstructed point)."""
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median



def outlier_filtering(pred_xyz):
    """Remove statistical outliers from predicted point clouds and return inlier mask."""
    B, N, C = pred_xyz.shape
    outlier_mask = torch.zeros((B, N), dtype=torch.bool, device=pred_xyz.device)

    for b in range(B):
        pts_masked = pred_xyz[b].cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            pts_masked.reshape(-1, 3)
        )
        # outlier filtering
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=8.0)
        outlier_mask[b, ind] = True

    return outlier_mask

def scale_shift_alignment_cd(pred_xyz, gt_xyz, gt_mask):
    """
    Align pred_xyz to gt_xyz using Iterative Closest Point (ICP) with batch operation.
    
    Args:
        pred_xyz: [B, N, 3] predicted point cloud
        gt_xyz: [B, N, 3] ground truth point cloud  
        gt_mask: [B, N] validity mask for ground truth points
    
    Returns:
        aligned_pred_xyz: [B, N, 3] aligned predicted point cloud
        mask: [B, N] validity mask for predictions
    """
    
    # First remove outliers from pred_xyz
    B, N, C = pred_xyz.shape
    outlier_mask = torch.zeros((B, N), dtype=torch.bool, device=pred_xyz.device)
    
    print("Performing outlier filtering...")
    pred_xyz_aligned = pred_xyz.clone()

    for b in range(B):
        pts_masked = pred_xyz[b].cpu().numpy()
   
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            pts_masked.reshape(-1, 3)
        )
        # outlier filtering
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=4.0)
        outlier_mask[b, ind] = True
    
        # Create filtered point clouds for ICP

        pred_xyz_filtered_b = pred_xyz[b][outlier_mask[b]].unsqueeze(0)  # [1, M, 3]
        gt_xyz_b = gt_xyz[b].unsqueeze(0)
        # Apply ICP alignment on filtered points
        icp_solution = iterative_closest_point(
            pred_xyz_filtered_b, gt_xyz_b,
            estimate_scale=True,
            max_iterations=100,
            relative_rmse_thr=1e-4,
            verbose=False,
        )
        
        # Extract transformation
        R = icp_solution.RTs.R  # [B, 3, 3]
        T = icp_solution.RTs.T  # [B, 3]
        s = icp_solution.RTs.s  # [B]
        
        # Apply transformation to original pred_xyz: s * R @ x + T
        aligned_pred_xyz_b = s.unsqueeze(-1).unsqueeze(-1) * torch.bmm(pred_xyz[b].unsqueeze(0), R.transpose(-2, -1)) + T.unsqueeze(1)
        pred_xyz_aligned[b] = aligned_pred_xyz_b.squeeze(0)

    return pred_xyz_aligned, outlier_mask



def resize_pred_to_gt(gt, pred):
    if gt.shape[1] != pred.shape[1] or gt.shape[2] != pred.shape[2]:
        B, H, W, L, C = pred.shape
        gt_H, gt_W = gt.shape[1], gt.shape[2]
        
        # Determine the scaling factor (the padded tensor was generated by scaling so that
        # the long edge becomes target size, then symmetric padding was applied)
        scale = max(gt_H / H, gt_W / W)
        new_H, new_W = int(H * scale), int(W * scale)
        
        # Permute to (B, L, C, H, W) so channels (L, C) are contiguous
        pred = pred.permute(0, 3, 4, 1, 2)
        # Flatten channel dimensions: shape (B, L * C, H, W)
        pred = pred.reshape(B, L * C, H, W)
        
        pred = F.interpolate(pred, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        # Reshape back to (B, L, C, new_H, new_W)
        pred = pred.reshape(B, L, C, new_H, new_W)
        # Permute back to (B, new_H, new_W, L, C)
        pred = pred.permute(0, 3, 4, 1, 2)

        crop_y = (new_H - gt_H) // 2
        crop_x = (new_W - gt_W) // 2
        pred = pred[:, crop_y:crop_y + gt_H, crop_x:crop_x + gt_W, :, :]

    return pred

def scale_shift_alignment_pointcloud(pred_xyz, gt_xyz, gt_mask, num_sample=None):
    '''
    Perform scale-shift alignment to <pts3d_pred> with least square's solution
    pred, gt: B N 3
    mask: B N
    '''

    gt_xyz_raw = gt_xyz.clone()

    gt_center, gt_scale = get_joint_pointcloud_center_scale(gt_xyz, gt_mask)

    pred_center, pred_scale = get_joint_pointcloud_center_scale(pred_xyz)

    pred_scale = pred_scale.clip(min=1e-3, max=1e3)
    
    pred_xyz -= pred_center

    pred_xyz *= gt_scale.view(-1, 1, 1) / pred_scale.view(-1, 1, 1)

    pred_xyz += gt_center

    pred_xyz_new = scale_shift_alignment_chamfer(pred_xyz, gt_xyz_raw, gt_mask, max_iterations=100, lr=0.01, num_sample=num_sample)
    pred_mask = outlier_filtering(pred_xyz_new)

    return pred_xyz_new, pred_mask

def scale_shift_commonlayers_alignment_inverse(prediction, target, mask):
    '''
    Perform scale-shift alignment to <pts3d_pred> with least square's solution
    using only the first layer
    pred, gt: B H W L 3
    mask: B H W L 1
    '''

    mask_init = mask.clone()
    prediction_init = prediction.clone()
    target_init = target.clone()


    common_n_layers = min(target.shape[-2], prediction.shape[-2])

    mask = mask[..., 0:common_n_layers, :] # B H W L=1 1
    prediction = prediction[..., 0:common_n_layers, :] # B H W L=1 3
    target = target[..., 0:common_n_layers, :] # B H W L=1 3


    assert mask.sum() != 0

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2, 3, 4)) # B -- sum(x1^2 + y1^2 + z1^2)
    a_01 = torch.sum(mask.squeeze(-1) * prediction[:,:,:,:,2], (1, 2, 3)) # B -- sum(z1)
    a_11 = torch.sum(mask, (1, 2, 3, 4)) # B -- valid_points of 1
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2, 3, 4)) # B -- sum(x1y1 + x2y2 + x3y3)
    b_1 = torch.sum(mask.squeeze(-1) * target[:,:,:,:,2], (1, 2, 3)) # B -- sum(z2)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0 #1e-3

    # B
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    # apply to the original data
    mask_update = torch.logical_and(mask_init.squeeze(-1), valid[:, None, None, None]) # B H W L
    
    # prediction_update = prediction.clone()
    prediction_update = x_0[...,None,None,None,None] * prediction_init.clone()
    prediction_update[..., 2] = prediction_update[..., 2] + x_1[:, None, None, None] # apply scale to all xyz and shift to z
    gt_update = mask_update[..., None] * target_init # B H W L 3

    return prediction_update, gt_update, mask_update, (x_0, x_1), valid



def hole_ratio(pred, gt, num_eval_pts=None, thres_list=[0.1, 0.05, 0.02]):
    """
    Compute Chamfer Distance and F-score between predicted and ground truth point clouds.
    """

    pred = scale_shift_alignment_chamfer(pred, gt, max_iterations=100, lr=0.01)
    
    
    dist_tuple, _ = chamfer_distance(pred, gt, batch_reduction=None, point_reduction=None, norm=2)
    dist_pred, dist_gt = dist_tuple # B, N

    # Pytorch3D returns Sqared Sum of the distance, we need to manually compute the squared-root
    dist_pred = torch.sqrt(dist_pred)
    dist_gt = torch.sqrt(dist_gt)

    # Mean Chamfer Distance
    chamfer_dist = (dist_pred.mean(dim=1) + dist_gt.mean(dim=1)) / 2

    details = {}
    details = {'CD': (float(chamfer_dist.mean()), int(chamfer_dist.shape[0]))}


    for thres in thres_list:
        hole_ratio = (dist_gt > thres).float().mean(dim=1)
        details.update({"hole_ratio_{}".format(thres): (float(hole_ratio.mean()), int(hole_ratio.shape[0]))})

    density, var_density = fast_knn_density(pred.squeeze(0), k=16)
    details['density_variance'] = (float(var_density.item()), int(density.shape[0]))
    # B
    return details

from pytorch3d.ops import knn_points
def fast_knn_density(points, k=16):
    # points: (N, 3)
    pts = points.unsqueeze(0)  # (1, N, 3)
    knn = knn_points(pts, pts, K=k+1)
    dists = knn.dists[0][:, 1:]   # remove self distance
    density = 1.0 / (dists.mean(dim=1).sqrt() + 1e-8)
    var_density = density.var(unbiased=False)
    return density, var_density

class SSI3DScore(nn.Module):
    """ 
    Compute the 3D metrics (CD and F-score) between the sampled prediction and GT points.
    """

    def __init__(self, num_eval_pts, fs_thres, pts_sampling_mode, eval_layers=None, ldi_vis_only=False, alignment='cd_adam', use_cd_align=False):
        super().__init__()
        self.num_eval_pts = num_eval_pts
        self.fs_thres = fs_thres
        self.pts_sampling_mode = pts_sampling_mode
        self.eval_layers = eval_layers
        assert self.eval_layers in [None, "visible", "unseen", "all"]

        self.ldi_vis_only = ldi_vis_only # only do alignment
        self.alignment = alignment
        self.use_cd_align = use_cd_align


    def get_all_pts3d(self, pred, data):
        return NotImplementedError()


    def chamfer_and_fscore(self, pred, gt, eval_layers, use_single_dir=False):
        """
        Compute Chamfer Distance and F-score between predicted and ground truth point clouds.
        """

        dist_tuple, _ = chamfer_distance(pred, gt, batch_reduction=None, point_reduction=None, norm=2)
        dist_pred, dist_gt = dist_tuple # B, N

        if use_single_dir:
            dist_gt = torch.sqrt(dist_gt)
            dist_pred = dist_gt
        else:
            # Pytorch3D returns Sqared Sum of the distance, we need to manually compute the squared-root
            dist_pred = torch.sqrt(dist_pred)
            dist_gt = torch.sqrt(dist_gt)

        # Mean Chamfer Distance
        chamfer_dist = (dist_pred.mean(dim=1) + dist_gt.mean(dim=1)) / 2

        details = {}
        details = {'CD_{}_{}'.format(self.num_eval_pts, eval_layers if eval_layers else "full"): (float(chamfer_dist.mean()), int(chamfer_dist.shape[0]))}

        if not isinstance(self.fs_thres, list):
            f_score = self.fscore_from_cd(dist_pred, dist_gt, self.fs_thres)
            details.update({"f_score_{}_{}".format(self.fs_thres, eval_layers if eval_layers else "full"): (float(f_score.mean()), int(f_score.shape[0]))})
        else:
            for thres in self.fs_thres:
                f_score = self.fscore_from_cd(dist_pred, dist_gt, thres)
                details.update({"f_score_{}_{}".format(thres, eval_layers if eval_layers else "full"): (float(f_score.mean()), int(f_score.shape[0]))})

        # B
        return details



    def fscore_from_cd(self, dist_pred, dist_gt, fs_thres):
        # Compute F-score
        f_pred = (dist_pred < fs_thres).float().mean(dim=1)
        f_gt = (dist_gt < fs_thres).float().mean(dim=1)
        f_score = 2 * f_pred * f_gt / (f_pred + f_gt + 1e-8)  # Avoid division by zero
        return f_score
    


    def uniform_sample_3dpts_with_interp(self, point_map, mask, num_samples):

        """
        Efficiently sample a specified number of points uniformly across the batch.
        If a sample has fewer valid points than required, it duplicates valid points.
        """
        # B, H, W, L, _ = point_map.shape
        B = point_map.shape[0]
        device = point_map.device

        # Flatten spatial dimensions
        mask_flat = mask.reshape(B, -1)  # Shape: (B, H*W*L)
        point_map_flat = point_map.reshape(B, -1, 3)  # Shape: (B, H*W*L, 3)

        # Get valid indices for each batch
        valid_indices = torch.nonzero(mask_flat, as_tuple=True)  # Shape: (valid_points,)
        
        batch_ids = valid_indices[0]  # Shape: (valid_points,)
        point_ids = valid_indices[1]  # Shape: (valid_points,)

        # Count valid points per batch
        valid_counts = mask_flat.sum(dim=1)  # Shape: (B,)

        # Compute offsets for each batch in `point_ids`
        offsets = torch.cat([torch.tensor([0], device=device), valid_counts.cumsum(0)[:-1]])  # (B,)
        
        # Generate random sampling indices within each batch

      
        rand_ids = torch.randint(low=0, high=int(valid_counts.max().item()), size=(B, num_samples), device=device) % valid_counts.unsqueeze(1)  # (B, num_samples)
        
        # Compute final sampled indices (global indices in `point_ids`)
        final_sampled_indices = point_ids[rand_ids.int() + offsets.int().unsqueeze(1)]  # (B, num_samples)

        # Gather the sampled 3D points
        sampled_points = torch.gather(point_map_flat, 1, final_sampled_indices.unsqueeze(-1).expand(-1, -1, 3))

        return sampled_points



    def forward(self, pred, data, **kw):
        return NotImplementedError()
    

def scale_shift_alignment_chamfer(pred_xyz, gt_xyz, gt_mask=None, max_iterations=100, lr=0.01, num_sample=None, return_transform=False):
    """
    Align pred_xyz to gt_xyz using gradient descent to minimize chamfer distance.
    
    Args:
        pred_xyz: [B, N, 3] predicted point cloud
        gt_xyz: [B, N, 3] ground truth point cloud  
        gt_mask: [B, N] validity mask for ground truth points (optional)
        max_iterations: maximum number of optimization iterations
        lr: learning rate for optimization
    
    Returns:
        aligned_pred_xyz: [B, N, 3] aligned predicted point cloud
        final_scale: [B] final scale values
        final_shift: [B, 3] final shift values
    """
    B, N_pred, C = pred_xyz.shape
    B, N_gt, C = gt_xyz.shape
    device = pred_xyz.device
    
    # Initialize parameters to optimize
    scale = torch.ones(B, 1, 1, device=device, requires_grad=True)
    shift = torch.zeros(B, 1, 3, device=device, requires_grad=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([scale, shift], lr=lr)
    
    best_loss = float('inf')
    best_scale = scale.clone()
    best_shift = shift.clone()
    
    target_xyz = gt_xyz.clone().detach()
    target_xyz.requires_grad = True
    source_xyz = pred_xyz.clone().detach()
    source_xyz.requires_grad = True
    if num_sample is None:
        num_sample = max(N_pred, N_gt) // 4

    for i in range(max_iterations):
        optimizer.zero_grad()
        
        # Apply transformation: aligned = scale * pred + shift
        aligned_pred = scale * source_xyz + shift

        # Compute chamfer distance
        # randomly down sample
        idx_pred = torch.randint(0, N_pred, (B, min(N_pred, num_sample)), device=device)
        idx_gt = torch.randint(0, N_gt, (B, min(N_gt, num_sample)), device=device)

        if num_sample < N_pred:
            aligned_pred_sampled = aligned_pred[:, idx_pred[0], :]
        else:
            aligned_pred_sampled = aligned_pred
        if num_sample < N_gt:
            target_xyz_sampled = target_xyz[:, idx_gt[0], :]
        else:
            target_xyz_sampled = target_xyz

        cd_loss, _ = chamfer_distance(aligned_pred_sampled, target_xyz_sampled, batch_reduction='mean')

        # Backpropagation
        cd_loss.backward()
        optimizer.step()
        
        # Clamp scale to reasonable bounds
        with torch.no_grad():
            scale.clamp_(0.01, 100.0)
        
        # Track best solution
        if cd_loss.item() < best_loss:
            best_loss = cd_loss.item()
            best_scale = scale.clone()
            best_shift = shift.clone()
        
    # Apply best transformation
    with torch.no_grad():
        aligned_pred_xyz = best_scale * pred_xyz + best_shift
    
    if return_transform:
        return aligned_pred_xyz, best_shift.detach(), best_scale.detach()
    else:
        return aligned_pred_xyz





class SSI3DScore_Scene(SSI3DScore):
    '''
    3D evaluation metric for depth models
    '''
    
    def get_all_pts3d(self, data, pred, **kw):
        pts3d_gt = data[0]['pcd_eval'].clone()  # B N 3
        pts3d_gt_vis = data[0]['pcd_eval_visible'].clone()  # B N 3
        mask_gt = torch.ones(pts3d_gt.shape[:-1], device=pts3d_gt.device)
        mask_gt = (mask_gt > 0)


        pts3d_pred = pred["pts3d_xyz"]


        if self.alignment == 'cd_adam':
            pts3d_pred_new = scale_shift_alignment_chamfer(pts3d_pred, pts3d_gt, mask_gt, max_iterations=200, lr=0.01)

            pred_mask = outlier_filtering(pts3d_pred_new)

        elif self.alignment == 'depth':
            pts3d_pred_new, pred_mask = scale_shift_alignment_pointcloud(pts3d_pred, pts3d_gt_vis, mask_gt)
        elif self.alignment == 'none':
            pred_mask = outlier_filtering(pts3d_pred)
            pts3d_pred_new = pts3d_pred


        valid_batch_mask = torch.sum(mask_gt, dim=-1) != 0

        return pts3d_pred_new, pred_mask, pts3d_gt, valid_batch_mask


    def forward(self, data, pred, **kw):
        # scale-shift alignment based on LDIs
        pts3d_pred, pts3d_pred_mask, _, valid_batch_mask = self.get_all_pts3d(data, pred, **kw)
        pts3d_pred_ori = pts3d_pred

        details_overall = {}

        pts3d_uniform_gt_vis = None

        align_shift = None
        align_scale = None

        pts3d_uniform_gt_complete = None
        for eval_layers in [None, "visible", 'visible_single']:
            # select GT
            if not eval_layers:
                pts3d_uniform_gt = data[0]["pcd_eval"]
            elif eval_layers == 'visible_single':
                pts3d_uniform_gt = data[0]["pcd_eval_visible"] # B N 3
            else:
                pts3d_uniform_gt = data[0]["pcd_eval_{}".format(eval_layers)] # B N 3

            if pts3d_pred.shape[1] > pts3d_uniform_gt.shape[1]:
                pts3d_pred_eval, _ = sampling_train_gen_target(pts3d_pred, pts3d_pred_mask, None, target_sampling='random', batch_size=self.num_eval_pts)
            else:
                pts3d_pred_eval = pts3d_pred

            if pts3d_uniform_gt_vis is None and eval_layers == "visible":
                pts3d_uniform_gt_vis = pts3d_uniform_gt
            
            if eval_layers == None:
                pts3d_uniform_gt_complete = pts3d_uniform_gt

            if self.use_cd_align:
                if align_shift is not None and align_scale is not None:
                    pts3d_pred_eval = pts3d_pred_eval * align_scale + align_shift
                else:
                    pts3d_pred_eval, shift, scale = scale_shift_alignment_chamfer(pts3d_pred_eval, pts3d_uniform_gt, max_iterations=400, lr=0.03, return_transform=True)
                    align_shift = shift
                    align_scale = scale

            if eval_layers == "visible_single":
                use_single_dir=True
            else:
                use_single_dir=False

            if valid_batch_mask is not None:
                details = self.chamfer_and_fscore(pts3d_pred_eval[valid_batch_mask], pts3d_uniform_gt[valid_batch_mask], eval_layers=eval_layers, use_single_dir=use_single_dir)
            else:
                details = self.chamfer_and_fscore(pts3d_pred_eval, pts3d_uniform_gt, eval_layers=eval_layers, use_single_dir=use_single_dir)
            
            details_overall.update(details)

        


        pts3d_uniform_gt_vis_rgb = data[0]['pcd_eval_visible_rgb'].clone() if 'pcd_eval_visible_rgb' in data[0] else None


        pts3d_uniform_gt_unseen = data[0]['pcd_eval_unseen'] 

        pts3d_pred_full = pts3d_pred[pts3d_pred_mask].reshape(-1, 3)

        result_data = {
            'pts3d_pred_eval': pts3d_pred_eval,
            'pts3d_uniform_gt': pts3d_uniform_gt_complete,
            'pts3d_uniform_gt_vis': pts3d_uniform_gt_vis,
            'pts3d_uniform_gt_vis_rgb': pts3d_uniform_gt_vis_rgb,
            'pts3d_uniform_gt_unseen': pts3d_uniform_gt_unseen,
            'pts3d_pred_full': pts3d_pred_full,
            'pts3d_pred_ori': pts3d_pred_ori
        }
        return result_data, details_overall




class SSI3DScore_Scene_Multi(SSI3DScore):
    '''
    3D evaluation metric for depth models
    '''
    
    def get_all_pts3d(self, data, pred, **kw):

        pts3d_gt_list = [x['pcd_eval'].clone() for x in data]  # list of B N 3
        # stack in point dimension
        pts3d_gt = torch.cat(pts3d_gt_list, dim=1)
        in_camera1 = inv(data[0]['camera_pose'])
        pts3d_gt = geotrf(in_camera1, pts3d_gt)

        mask_gt = torch.ones(pts3d_gt.shape[:-1], device=pts3d_gt.device)
        


        pts3d_pred = pred["pts3d_xyz"]


        if self.alignment == 'cd_adam':
            pts3d_pred_new = scale_shift_alignment_chamfer(pts3d_pred, pts3d_gt, mask_gt, max_iterations=200, lr=0.01)
            pred_mask = outlier_filtering(pts3d_pred_new)

        elif self.alignment == 'depth':
            mask_gt = (mask_gt > 0)
            pts3d_pred_new, pred_mask = scale_shift_alignment_pointcloud(pts3d_pred, pts3d_gt, mask_gt, num_sample=25000)
        elif self.alignment == 'none':
            pred_mask = outlier_filtering(pts3d_pred)
            pts3d_pred_new = pts3d_pred


        valid_batch_mask = torch.sum(mask_gt, dim=-1) != 0

        return pts3d_pred_new, pred_mask, pts3d_gt, valid_batch_mask


    def forward(self, data, pred, **kw):
        # scale-shift alignment based on LDIs
        pts3d_pred, pts3d_pred_mask, _, valid_batch_mask = self.get_all_pts3d(data, pred, **kw)
        pts3d_pred_ori = pts3d_pred

        details_overall = {}

        pts3d_uniform_gt_vis = None
        pts3d_uniform_gt_complete = None

        align_shift = None
        align_scale = None


        for eval_layers in [None, "visible"]:
            # select GT
            if not eval_layers:
                pts3d_gt_list = [x['pcd_eval'].clone() for x in data] 
                pts3d_uniform_gt = torch.cat(pts3d_gt_list, dim=1)

                in_camera1 = inv(data[0]['camera_pose'])
                pts3d_uniform_gt = geotrf(in_camera1, pts3d_uniform_gt)


            else:
                pts3d_gt_list = [x[f'pcd_eval_{eval_layers}'].clone() for x in data]
                pts3d_uniform_gt = torch.cat(pts3d_gt_list, dim=1)
                in_camera1 = inv(data[0]['camera_pose'])
                pts3d_uniform_gt = geotrf(in_camera1, pts3d_uniform_gt)

            if pts3d_pred.shape[1] > pts3d_uniform_gt.shape[1]:
                pts3d_pred_eval, _ = sampling_train_gen_target(pts3d_pred, pts3d_pred_mask, None, target_sampling='random', batch_size=self.num_eval_pts)
            else:
                pts3d_pred_eval = pts3d_pred

            
            if pts3d_uniform_gt_vis is None and eval_layers == "visible":
                pts3d_uniform_gt_vis = pts3d_uniform_gt

            if eval_layers == None:
                pts3d_uniform_gt_complete = pts3d_uniform_gt

            if self.use_cd_align:
                if align_shift is not None and align_scale is not None:
                    pts3d_pred_eval = pts3d_pred_eval * align_scale + align_shift
                else:
                    pts3d_pred_eval, align_scale, align_shift = scale_shift_alignment_chamfer(pts3d_pred_eval, pts3d_uniform_gt, max_iterations=200, lr=0.01, return_transform=True)

            if valid_batch_mask is not None:
                details = self.chamfer_and_fscore(pts3d_pred_eval[valid_batch_mask], pts3d_uniform_gt[valid_batch_mask], eval_layers=eval_layers)
            else:
                details = self.chamfer_and_fscore(pts3d_pred_eval, pts3d_uniform_gt, eval_layers=eval_layers)
            
            details_overall.update(details)


        pts3d_uniform_gt_unseen_list = [x['pcd_eval_unseen'].clone() for x in data]
        pts3d_uniform_gt_unseen = torch.cat(pts3d_uniform_gt_unseen_list, dim=1)

        pts3d_pred_full = pts3d_pred[pts3d_pred_mask].reshape(-1, 3)
    
        result_data = {
            'pts3d_pred_eval': pts3d_pred_eval,
            'pts3d_uniform_gt': pts3d_uniform_gt_complete,
            'pts3d_uniform_gt_vis': pts3d_uniform_gt_vis,
            'pts3d_uniform_gt_unseen': pts3d_uniform_gt_unseen,
            'pts3d_pred_full': pts3d_pred_full,
            'pts3d_pred_ori': pts3d_pred_ori
        }
            

        return result_data, details_overall



