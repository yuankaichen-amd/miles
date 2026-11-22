"""Trusted PyTorch baseline for t2h/aka/points_in_boxes_f32_all at native dtype."""
import torch

_CHUNK_ELEMENTS = 1 << 26


def _inside(points, boxes):
    """[B, P, M] bool for points [B, P, 3] and boxes [B, M, 7] (cz at the box bottom)."""
    bx = boxes[:, None, :, :]
    sx = points[:, :, None, 0] - bx[..., 0]
    sy = points[:, :, None, 1] - bx[..., 1]
    sz = points[:, :, None, 2] - bx[..., 2]
    cos_r, sin_r = torch.cos(bx[..., 6]), torch.sin(bx[..., 6])
    local_x = sx * cos_r + sy * sin_r
    local_y = -sx * sin_r + sy * cos_r
    return ((sz >= 0) & (sz <= bx[..., 5]) & (local_x.abs() < bx[..., 3] / 2)
            & (local_y.abs() < bx[..., 4] / 2))


def run(*, points, boxes, box_idx_of_points):
    batch, n_points, _ = points.shape
    n_boxes = boxes.shape[1]
    step = max(1, _CHUNK_ELEMENTS // (batch * n_boxes))
    for p0 in range(0, n_points, step):
        box_idx_of_points[:, p0:p0 + step] = _inside(points[:, p0:p0 + step], boxes)
