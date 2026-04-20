# Copyright (c) 2026 Weirong Chen

import torch

from dust3r.utils.misc import invalid_to_zeros, invalid_to_nans


def normalize_pointcloud(pts_list, norm_mode='avg_dis', valid_list=None, ret_factor=False):
    """ Renorm pointmaps in pts_list with norm_mode.
    """
    assert isinstance(pts_list, list) and all(pts.ndim >= 3 and pts.shape[-1] == 3 for pts in pts_list)
    if valid_list is not None:
        assert isinstance(valid_list, list) and len(valid_list) == len(pts_list)

    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # Gather all points together (joint normalization)
        nan_pts_list = []
        nnz_list = []
        for i, pts in enumerate(pts_list):
            valid = valid_list[i] if valid_list is not None else None
            nan_pts, nnz = invalid_to_zeros(pts, valid, ndim=3)
            nan_pts_list.append(nan_pts)
            nnz_list.append(nnz)
        all_pts = torch.cat(nan_pts_list, dim=1)

        # Compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass  # Do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            # Actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            offset = 0
            for i, pts in enumerate(pts_list):
                H, W = pts.shape[1:-1]
                pts_list[i] = pts * warp_factor[:, offset:offset + H * W].view(-1, H, W, 1)
                offset += H * W
            all_dis = log_dis  # This is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (sum(nnz_list) + 1e-8)
    else:
        # Gather all points together (joint normalization)
        nan_pts_list = []
        for i, pts in enumerate(pts_list):
            valid = valid_list[i] if valid_list is not None else None
            nan_pts = invalid_to_nans(pts, valid, ndim=3)
            nan_pts_list.append(nan_pts)
        all_pts = torch.cat(nan_pts_list, dim=1)

        # Compute distance to origin
        all_dis = all_pts.norm(dim=-1)

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts_list[0].ndim:
        norm_factor.unsqueeze_(-1)

    res_list = [pts / norm_factor for pts in pts_list]
    if ret_factor:
        return res_list, norm_factor
    return res_list


@torch.no_grad()
def get_joint_pointcloud_depth(z_list, valid_list, quantile=0.5):
    # set invalid points to NaN
    nan_z_list = []
    for z, valid in zip(z_list, valid_list):
        nan_z = invalid_to_nans(z, valid).reshape(len(z), -1)
        nan_z_list.append(nan_z)
    all_z = torch.cat(nan_z_list, dim=-1)

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(all_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(all_z, quantile, dim=-1)
    return shift_z  # (B,)



@torch.no_grad()
def get_joint_pointcloud_center_scale(pts_list, valid_mask_list=None, z_only=False, center=True):
    # set invalid points to NaN
    nan_pts_list = []
    for i, pts in enumerate(pts_list):
        valid_mask = valid_mask_list[i] if valid_mask_list is not None else None
        nan_pts = invalid_to_nans(pts, valid_mask).reshape(len(pts), -1, 3)
        nan_pts_list.append(nan_pts)
    all_pts = torch.cat(nan_pts_list, dim=1)

    # compute median center
    _center = torch.nanmedian(all_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((all_pts - _center) if center else all_pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]

