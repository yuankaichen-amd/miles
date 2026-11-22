"""Trusted PyTorch baseline for t2h/aka/roipoint_pool3d_f32_ns512 at native dtype."""
import torch

NUM_SAMPLED = 512
_CHUNK_ELEMENTS = 1 << 25


def _membership(points, boxes):
    """[B, M, N] bool: point n lies in box m (cz <= z <= cz + dz, |local_x| < dx/2, |local_y| < dy/2)."""
    bx = boxes[:, :, None, :]
    sx = points[:, None, :, 0] - bx[..., 0]
    sy = points[:, None, :, 1] - bx[..., 1]
    sz = points[:, None, :, 2] - bx[..., 2]
    cos_r, sin_r = torch.cos(bx[..., 6]), torch.sin(bx[..., 6])
    lx = sx * cos_r + sy * sin_r
    ly = -sx * sin_r + sy * cos_r
    return ((sz >= 0) & (sz <= bx[..., 5]) & (lx.abs() < bx[..., 3] / 2)
            & (ly.abs() < bx[..., 4] / 2))


def run(*, points, point_features, boxes3d, pooled_features, pooled_empty_flag):
    batch, n_points, _ = points.shape
    n_boxes = boxes3d.shape[1]
    device = points.device
    rows = torch.cat([points, point_features], dim=-1)
    width = min(NUM_SAMPLED, n_points)
    position = torch.arange(n_points, device=device)
    slots = torch.arange(NUM_SAMPLED, device=device)
    batch_ids = torch.arange(batch, device=device)[:, None, None]
    step = max(1, _CHUNK_ELEMENTS // (batch * n_points))
    for m0 in range(0, n_boxes, step):
        inside = _membership(points, boxes3d[:, m0:m0 + step])
        key = torch.where(inside, position, n_points)
        first = torch.topk(key, width, dim=-1, largest=False).values
        count = inside.sum(-1)
        source = first.gather(-1, (slots % count.clamp(min=1).unsqueeze(-1)).clamp(max=width - 1))
        pooled = rows[batch_ids, source.clamp(max=n_points - 1)]
        pooled = torch.where((count > 0)[..., None, None], pooled, torch.zeros_like(pooled))
        pooled_features[:, m0:m0 + step] = pooled
        pooled_empty_flag[:, m0:m0 + step] = count == 0
