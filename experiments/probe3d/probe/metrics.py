from .losses import chamfer_l2


def chamfer_distance(pred, gt):
    return chamfer_l2(pred, gt)

