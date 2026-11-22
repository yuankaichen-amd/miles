"""Trusted PyTorch baseline for t2h/aka/furthest_point_sample_f32 at native dtype."""
import torch


def run(*, points_xyz, idx):
    batch, n_points, _ = points_xyz.shape
    n_samples = idx.shape[1]
    device = points_xyz.device
    rows = torch.arange(batch, device=device)
    selected = torch.zeros((batch, n_samples), dtype=torch.int64, device=device)
    min_dist = torch.full((batch, n_points), 1.0e10, dtype=points_xyz.dtype, device=device)
    last = torch.zeros(batch, dtype=torch.int64, device=device)
    for step in range(1, n_samples):
        d = points_xyz - points_xyz[rows, last].unsqueeze(1)
        dist = d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]
        min_dist = torch.minimum(min_dist, dist)
        last = min_dist.argmax(dim=1)
        selected[:, step] = last
    idx.copy_(selected)
