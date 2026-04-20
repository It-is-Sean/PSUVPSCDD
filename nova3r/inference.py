# Copyright (c) 2026 Weirong Chen
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------

import tqdm
import torch
import torch.nn.functional as F

from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_zeros
from dust3r.utils.geometry import geotrf, inv

# flow_matching
from nova3r.flow_matching.path.scheduler import CosineScheduler
from nova3r.flow_matching.path import AffineProbPath
from nova3r.flow_matching.solver import ODESolver
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.utils.sampling import sampling_train_gen_target
from einops import rearrange

path = AffineProbPath(scheduler=CosineScheduler())

amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}



def get_all_pts3d(gt_list, mode=None, down_resolution=112):
    """Extract and optionally downsample/FPS-sample ground truth 3D points from a batch."""
    if mode == 'cube':
        pts_xyz = gt_list[0]['global_center_xyz']

        valid = torch.ones_like(pts_xyz[..., 0]).bool()  # B, N
        in_camera1 = inv(gt_list[0]['camera_pose'])
        gt_pts = geotrf(in_camera1, pts_xyz)

    elif mode == 'src_complete':
        gt_pts, valid = get_complete_pts3d(gt_list)

    elif 'src_complete_fps' in mode:
        batch_size = int(mode.split('_')[-1])
        gt_pts, valid = get_complete_pts3d(gt_list)
        # run fps_fast
        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling='fps_fast', batch_size=batch_size)

    elif 'src_complete_fps_edge' in mode:
        batch_size = int(mode.split('_')[-1])
        gt_pts, valid = get_complete_pts3d(gt_list)
        # run fps_fast
        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling='fps_edge_fast', batch_size=batch_size)

    elif mode == 'cube_global':
        pts_xyz = gt_list[0]['global_center_xyz']
        valid = torch.ones_like(pts_xyz[..., 0]).bool()  # B, N
        gt_pts = pts_xyz

    elif mode == 'src_view':
        gt_pts_list = [gt['pts3d'] for gt in gt_list]

        in_camera1 = inv(gt_list[0]['camera_pose'])
        gt_pts_list = [geotrf(in_camera1, gt['pts3d']) for gt in gt_list]

        gt_pts = torch.stack(gt_pts_list, dim=1)
        B, H, W, C = gt_pts_list[0].shape
        gt_pts = rearrange(gt_pts, 'b s h w c -> (b s) c h w')
        gt_pts = F.interpolate(gt_pts, size=down_resolution, mode='nearest')

        gt_pts = rearrange(gt_pts, '(b s) c h w -> b (s h w) c', b=B)

        valid_list = [gt['valid_mask'].clone() for gt in gt_list]
        valid = torch.stack(valid_list, dim=1).float()
        valid = rearrange(valid, 'b s h w -> (b s) 1 h w')  # Add channel dimension
        valid = F.interpolate(valid, size=down_resolution, mode='nearest')
        valid = rearrange(valid, '(b s) 1 h w -> b (s h w)', b=B).bool()

    elif 'src_view_fps' in mode:
        batch_size = int(mode.split('_')[-1])
        gt_pts_list = [gt['pts3d'] for gt in gt_list]

        in_camera1 = inv(gt_list[0]['camera_pose'])
        gt_pts_list = [geotrf(in_camera1, gt['pts3d']) for gt in gt_list]

        gt_pts = torch.stack(gt_pts_list, dim=1)
        B, H, W, C = gt_pts_list[0].shape
        gt_pts = rearrange(gt_pts, 'b s h w c -> (b s) c h w')
        gt_pts = F.interpolate(gt_pts, size=down_resolution, mode='nearest')

        gt_pts = rearrange(gt_pts, '(b s) c h w -> b (s h w) c', b=B)

        valid_list = [gt['valid_mask'].clone() for gt in gt_list]
        valid = torch.stack(valid_list, dim=1).float()
        valid = rearrange(valid, 'b s h w -> (b s) 1 h w')  # Add channel dimension
        valid = F.interpolate(valid, size=down_resolution, mode='nearest')
        valid = rearrange(valid, '(b s) 1 h w -> b (s h w)', b=B).bool()

        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling='fps_fast', batch_size=batch_size)

    else:
        raise NotImplementedError
    return gt_pts, valid


def get_complete_pts3d(gt_list, valid_front=False):
    """Get complete (amodal) 3D point clouds from all views, transformed to camera 1 coordinates."""
    pts_xyz = [gt['pts3d_complete'] for gt in gt_list]
    in_camera1 = inv(gt_list[0]['camera_pose'])
    pts_xyz = [geotrf(in_camera1, pts) for pts in pts_xyz]  # B, N, 3
    gt_pts = torch.stack(pts_xyz, dim=1)  # B, S, N, 3

    valid_num_list = [gt['pts3d_complete_valid_num'] for gt in gt_list]  # B, S
    valid = torch.zeros_like(gt_pts[..., 0]).bool()  # B, S, N
    for i in range(len(gt_list)):
        for j in range(valid_num_list[i].shape[0]):
            valid[j, i, :valid_num_list[i][j]] = True

    gt_pts = rearrange(gt_pts, 'b s n c -> b (s n) c')  # B, S*N, 3
    valid = rearrange(valid, 'b s n -> b (s n)')  # B, S*N

    if valid_front:
        reordered_pts = []
        reordered_valid = []
        valid_counts = []

        for b in range(gt_pts.shape[0]):
            valid_mask = valid[b]
            valid_indices = torch.where(valid_mask)[0]
            invalid_indices = torch.where(~valid_mask)[0]

            reorder_indices = torch.cat([valid_indices, invalid_indices])

            reordered_pts.append(gt_pts[b][reorder_indices])
            reordered_valid.append(valid[b][reorder_indices])
            valid_counts.append(len(valid_indices))

        gt_pts = torch.stack(reordered_pts, dim=0)
        valid = torch.stack(reordered_valid, dim=0)
        valid_counts = torch.tensor(valid_counts, device=gt_pts.device)

        return gt_pts, valid, valid_counts
    else:
        return gt_pts, valid


def normalize_input(pts3d_src, valid_src, pts3d_trg, valid_trg, mode='none'):
    """Normalize the input points
    """
    if mode == 'none':
        return pts3d_src, pts3d_trg

    elif 'median' in mode:
        if mode == 'median':
            target_median = 1.0
        else:
            target_median = float(mode.split('_')[-1])

        pts3d_src_new = []
        pts3d_trg_new = []

        for b in range(pts3d_src.shape[0]):
            src_xyz = pts3d_src[b]
            trg_xyz = pts3d_trg[b]
            src_valid = valid_src[b]
            trg_valid = valid_trg[b]

            nan_pts, nnz = invalid_to_zeros(trg_xyz, trg_valid, ndim=3)

            all_dis = nan_pts.norm(dim=-1)

            mean_factor = all_dis.sum() / (nnz.sum() + 1e-8)

            valid_dis = all_dis[trg_valid]
            norm_factor = valid_dis.median() if valid_dis.numel() > 0 else torch.tensor(1.0, device=all_dis.device)

            norm_factor = norm_factor.clip(min=0.01, max=100.0)

            src_xyz_norm = src_xyz / norm_factor * target_median
            trg_xyz_norm = trg_xyz / norm_factor * target_median

            src_xyz_norm = torch.clamp(src_xyz_norm, min=-1000.0, max=1000.0)
            trg_xyz_norm = torch.clamp(trg_xyz_norm, min=-1000.0, max=1000.0)

            pts3d_src_new.append(src_xyz_norm)
            pts3d_trg_new.append(trg_xyz_norm)

        pts3d_src_new = torch.stack(pts3d_src_new, dim=0)  # B, N, 3
        pts3d_trg_new = torch.stack(pts3d_trg_new, dim=0)  # B, N, 3
        return pts3d_src_new, pts3d_trg_new

    elif 'cube' in mode:
        if mode == 'cube':
            target_scale = 1.0
        else:
            target_scale = float(mode.split('_')[-1])

        pts3d_src_new = []
        pts3d_trg_new = []

        for b in range(pts3d_src.shape[0]):
            src_xyz = pts3d_src[b]
            trg_xyz = pts3d_trg[b]
            src_valid = valid_src[b]
            trg_valid = valid_trg[b]

            center_trg = trg_xyz[trg_valid].mean(dim=0)

            src_xyz_centered = src_xyz - center_trg
            trg_xyz_centered = trg_xyz - center_trg

            dist_trg = torch.norm(trg_xyz_centered[trg_valid], dim=1)
            max_dist_trg = torch.quantile(dist_trg, 0.9)

            src_xyz_norm = src_xyz_centered / max_dist_trg * target_scale
            trg_xyz_norm = trg_xyz_centered / max_dist_trg * target_scale

            pts3d_src_new.append(src_xyz_norm)
            pts3d_trg_new.append(trg_xyz_norm)

        pts3d_src = torch.stack(pts3d_src_new, dim=0)
        pts3d_trg = torch.stack(pts3d_trg_new, dim=0)

        return pts3d_src, pts3d_trg


def loss_of_one_batch_lari(args, batch, model, criterion, device, use_amp=False, ret=None, num_queries=20000, model_wrapper=None, **kwargs):
    """Compute evaluation metrics (Chamfer Distance, F-Score) for trained models."""
    ignore_keys = set(['dataset', 'label', 'instance', 'idx', 'true_shape', 'rng', 'view_label'])

    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    token_mask = None

    img_src_list = []
    for view in batch:
        if 'input' in view['view_label'][0]:
            view['img'] = view['img'] * 0.5 + 0.5
            img = view['img']
            img_src_list.append(img)

    images = torch.stack(img_src_list, dim=1)

    # prepare the diffusion data
    if 'query_source' in args.model.params.cfg.pts3d_head.params:
        query_src = args.model.params.cfg.pts3d_head.params.query_source
    else:
        query_src = 'src_complete'

    if 'down_resolution' in args.model.params.cfg.pts3d_head.params:
        down_resolution = args.model.params.cfg.pts3d_head.params.down_resolution
    else:
        down_resolution = 224

    if 'norm_mode' in args.model.params.cfg.pts3d_head.params:
        norm_mode = args.model.params.cfg.pts3d_head.params.norm_mode
    else:
        norm_mode = 'none'

    if 'fm_sampling' in args:
        fm_sampling = args.fm_sampling
    else:
        fm_sampling = 'euler'

    pts3d_src, valid_src = get_all_pts3d(batch, mode=query_src, down_resolution=down_resolution)

    pts3d_src_norm, _ = normalize_input(pts3d_src, valid_src, pts3d_src, valid_src, mode=norm_mode)


    def process_point(args, images, pts3d_src, token_mask, num_queries, device, model_wrapper=None, method='euler'):
        step_size = args.fm_step_size
        num_steps = int(1 // args.fm_step_size)
        point_size = num_queries

        cfg_scale = args.cfg_scale if 'cfg_scale' in args else 1.0
        B = images.shape[0]

        T = torch.linspace(0, 1, num_steps).to(device)
        if model_wrapper is not None:
            wrapped_vf = model_wrapper
        else:
            wrapped_vf = BatchModelWrapper(model=model)
        wrapped_vf.eval()

        model.eval()
        x_init = torch.rand((B, point_size, 3), dtype=torch.float32, device=device) * 2 - 1
        solver = ODESolver(velocity_model=wrapped_vf)

        if hasattr(model, 'module'):
            encoder_data = model.module._encode(images=images, pointmaps=pts3d_src, test=True, cfg_scale=cfg_scale)
        else:
            encoder_data = model._encode(images=images, pointmaps=pts3d_src, test=True, cfg_scale=cfg_scale)

        sol = solver.sample(
            time_grid=T,
            x_init=x_init.detach(),
            method=method,
            step_size=step_size,
            return_intermediates=True,
            images=images.detach(),
            token_mask=token_mask,
            encoder_data=encoder_data,
            pointmaps=pts3d_src.detach()
        )
        return sol

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype_mapping[args.amp_dtype]):
            pts3d_xyz_list = process_point(args, images, pts3d_src_norm, token_mask, num_queries, device, model_wrapper=model_wrapper, method=fm_sampling)

    pts3d_xyz = pts3d_xyz_list[-1] 
    pred_dict = {}
    pred_dict['pts3d_xyz'] = pts3d_xyz
    pred_dict['pts3d_xyz_list'] = pts3d_xyz_list
    pred_dict["images"] = images

    pred_dict['input_pts3d'] = pts3d_src
    pred_dict['input_valid'] = valid_src

    with torch.cuda.amp.autocast(enabled=bool(use_amp), dtype=amp_dtype_mapping[args.amp_dtype]):
        pts3d_data, loss = criterion(batch, pred_dict) if criterion is not None else None

    result = dict(view=batch, pred=pred_dict, data=pts3d_data, loss=loss)
    return result[ret] if ret else result


def loss_of_one_batch_demo(args, batch, model, criterion, device, use_amp=False, ret=None, num_queries=20000, model_wrapper=None, n_views=2, method='euler', pointmaps=None):
    """Run inference for demo/visualization -- generates 3D point clouds from images."""
    ignore_keys = set(['dataset', 'label', 'instance', 'idx', 'true_shape', 'rng', 'view_label'])

    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    token_mask = None

    img_src_list = []
    for view in batch:
        if 'input' in view['view_label'][0]:
            view['img'] = view['img'] * 0.5 + 0.5
            img = view['img']
            img_src_list.append(img)


    images = torch.stack(img_src_list, dim=1)
    images = images[:, :n_views, ...]  # Use only n_views

    def process_point(args, images, pts3d_src, token_mask, num_queries, device, model_wrapper=None, method='euler'):
        step_size = args.fm_step_size
        num_steps = int(1 // args.fm_step_size)
        point_size = num_queries

        B = images.shape[0]

        T = torch.linspace(0, 1, num_steps).to(device)
        if model_wrapper is not None:
            wrapped_vf = model_wrapper
        else:
            wrapped_vf = BatchModelWrapper(model=model)
        wrapped_vf.eval()

        model.eval()
        x_init = torch.rand((B, point_size, 3), dtype=torch.float32, device=device) * 2 - 1
        solver = ODESolver(velocity_model=wrapped_vf)

        if hasattr(model, 'module'):
            encoder_data = model.module._encode(images=images, pointmaps=pts3d_src)
        else:
            encoder_data = model._encode(images=images, pointmaps=pts3d_src)

        sol = solver.sample(
            time_grid=T,
            x_init=x_init,
            method=method,
            step_size=step_size,
            return_intermediates=False,
            images=images,
            token_mask=token_mask,
            encoder_data=encoder_data,
            pointmaps=pts3d_src
        )
        return sol[-1] if isinstance(sol, list) else sol

    if pointmaps is not None:
        norm_mode = args.model.params.cfg.pts3d_head.params.get('norm_mode', 'none')
        valid = torch.ones(pointmaps.shape[0], pointmaps.shape[1], dtype=torch.bool, device=device)
        pointmaps, _ = normalize_input(pointmaps, valid, pointmaps, valid, mode=norm_mode)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            pts3d_xyz = process_point(args, images, pointmaps, token_mask, num_queries, device, model_wrapper=model_wrapper, method=method)

    pred_dict = {}
    pred_dict['pts3d_xyz'] = pts3d_xyz
    pred_dict["images"] = images

    result = dict(view=batch, pred=pred_dict)
    return result[ret] if ret else result



@torch.no_grad()
def inference_nova3r(args, pairs, model, device, batch_size=8, verbose=True, num_queries=20000, n_views=2, method='euler', pointmaps=None):
    """Run batched Nova3r inference on image pairs, returning predicted 3D point clouds."""
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch_demo(args, collate_with_cat(pairs[i:i + batch_size]), model, None, device, num_queries=num_queries, n_views=n_views, method=method, pointmaps=pointmaps)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


