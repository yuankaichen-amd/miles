"""Trusted PyTorch baseline for t2h/aka/roiaware_pool3d_f32_out4_max at native dtype."""
import torch

OUT_SIZE = 4
MAX_PTS_PER_VOXEL = 128
_CHUNK_ELEMENTS = 1 << 24


def _voxels(pts, rois):
    """[R, P] flat voxel index (x, y, z) of every point in each box, or OUT_SIZE**3 outside it."""
    bx = rois[:, None, :]
    sx = pts[None, :, 0] - bx[..., 0]
    sy = pts[None, :, 1] - bx[..., 1]
    sz = pts[None, :, 2] - bx[..., 2]
    cos_r, sin_r = torch.cos(bx[..., 6]), torch.sin(bx[..., 6])
    lx = sx * cos_r + sy * sin_r
    ly = -sx * sin_r + sy * cos_r
    dx, dy, dz = bx[..., 3], bx[..., 4], bx[..., 5]
    inside = (sz >= 0) & (sz <= dz) & (lx.abs() < dx / 2) & (ly.abs() < dy / 2)
    ix = ((lx + dx / 2) / (dx / OUT_SIZE)).floor().clamp(0, OUT_SIZE - 1).long()
    iy = ((ly + dy / 2) / (dy / OUT_SIZE)).floor().clamp(0, OUT_SIZE - 1).long()
    iz = (sz / (dz / OUT_SIZE)).floor().clamp(0, OUT_SIZE - 1).long()
    voxel = (ix * OUT_SIZE + iy) * OUT_SIZE + iz
    return torch.where(inside, voxel, OUT_SIZE ** 3)


def run(*, rois, pts, pts_feature, pooled_features):
    n_rois = rois.shape[0]
    n_points, channels = pts_feature.shape
    n_voxels = OUT_SIZE ** 3
    device = pts.device
    pooled = pooled_features.view(n_rois, n_voxels, channels)
    position = torch.arange(n_points, device=device)
    bins = torch.arange(n_voxels + 1, device=device)
    step = max(1, _CHUNK_ELEMENTS // n_points)
    for r0 in range(0, n_rois, step):
        r1 = min(n_rois, r0 + step)
        sorted_voxel, order = torch.sort(_voxels(pts, rois[r0:r1]), dim=1, stable=True)
        first = torch.searchsorted(sorted_voxel, bins.expand(r1 - r0, -1).contiguous())
        rank = position[None, :] - first.gather(1, sorted_voxel)
        keep = (sorted_voxel < n_voxels) & (rank < MAX_PTS_PER_VOXEL - 1)
        target = torch.where(keep, sorted_voxel, n_voxels)
        features = pts_feature[order]
        index = target.unsqueeze(-1).expand(-1, -1, channels)
        best = features.new_full((r1 - r0, n_voxels + 1, channels), float("-inf"))
        best.scatter_reduce_(1, index, features, "amax", include_self=True)
        best = best[:, :n_voxels]
        pooled[r0:r1] = torch.where(torch.isinf(best), torch.zeros_like(best), best)
