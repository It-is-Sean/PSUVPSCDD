# Copyright (c) 2026 Weirong Chen
#
# --------------------------------------------------------
# Evaluation losses for NOVA3R
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn

from chamferdist import ChamferDistance


def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1, f"{dist.ndim}, {a.ndim}"  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance



class CDLoss (LLoss):
    """Chamfer Distance as an L-norm style loss. Wraps chamferdist for bidirectional point cloud distance."""

    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.chamfer_dist = ChamferDistance()

    def distance(self, a, b):

        dist, forward_nn, backward_nn = self.chamfer_dist(a, b, bidirectional=True, point_reduction=None, batch_reduction=None)
        # dist.shape    (B, N_patch), K
        # forward_nn.shape (B, N_patch), K
        # backward_nn.shape (B, N_patch), K
        return dist


L21 = L21Loss()
LCD = CDLoss()

class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details




class Pts3D_Regr3D_CD_V4(Criterion, MultiLoss):
    """Chamfer Distance loss for 3D point cloud evaluation. Computes bidirectional Chamfer Distance and F-Score between predicted and ground truth point clouds."""

    def __init__(self, criterion, norm_mode='avg_dis'):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        # self.gt_scale = gt_scale
        # self.pred_scale = pred_scale
        self.chamfer_dist = ChamferDistance()


    def compute_loss(self, gt_list, pred_dict, **kw):

        gt_pts = pred_dict['target_pts3d']
        gt_valid = pred_dict['target_valid']

        pr_pts = pred_dict['pts3d_xyz']


        B = pr_pts.shape[0]
        loss_forward_list = []
        loss_backward_list = []


        for b in range(B):
            pr_pts_b = pr_pts[b]  # Keep batch dimension
            gt_pts_b = gt_pts[b]
            gt_valid_b = gt_valid[b]

            gt_pts_b_valid = gt_pts_b[gt_valid_b]  # Filter out invalid points

            dist_forward, forward_nn = self.chamfer_dist(pr_pts_b[None], gt_pts_b_valid[None], bidirectional=False, point_reduction='mean')
            dist_backward, backward_nn = self.chamfer_dist(gt_pts_b_valid[None], pr_pts_b[None], bidirectional=False, point_reduction='mean')

            # valid_forward = torch.gather(gt_valid_b, 1, forward_nn)

            # dist_backward_b = dist_backward[gt_valid_b]
            # dist_forward_b = dist_forward[valid_forward]

            # loss_forward_b = dist_forward_b.sum() / gt_valid_b.sum() if gt_valid_b.sum() > 0 else dist_forward_b.new_zeros(())
            # loss_backward_b = dist_backward_b.sum() / valid_forward.sum() if valid_forward.sum() > 0 else dist_forward_b.new_zeros(())

            loss_forward_list.append(dist_forward)
            loss_backward_list.append(dist_backward)

        loss_acc = torch.stack(loss_forward_list).mean()
        loss_com = torch.stack(loss_backward_list).mean()
        loss_cd = loss_acc + loss_com

        self_name = type(self).__name__
        details = {self_name + '_accuracy': float(loss_acc.mean()),
                    self_name + '_completeness': float(loss_com.mean()),
                    self_name + '_cd': float(loss_cd.mean())}

        return Sum((loss_acc, None), (loss_com, None), (loss_cd, None)), (details | {})
