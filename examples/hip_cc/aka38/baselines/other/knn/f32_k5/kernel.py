"""Trusted PyTorch baseline for t2h/aka/knn_f32_k5 at native dtype."""
import torch

K = 5
GRID = 128.0 ** 2
_CHUNK_ELEMENTS = 1 << 26


def run(*, xyz, center_xyz, idx, dist2):
    batch, n_points, _ = xyz.shape
    n_query = center_xyz.shape[1]
    positions = torch.arange(n_points, dtype=torch.int64, device=xyz.device)
    step = max(1, _CHUNK_ELEMENTS // (batch * n_points))
    for q0 in range(0, n_query, step):
        d = center_xyz[:, q0:q0 + step, None, :] - xyz[:, None, :, :]
        d2 = d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]
        key = torch.round(d2 * GRID).to(torch.int64) * n_points + positions
        best = torch.topk(key, K, dim=-1, largest=False, sorted=True).values
        idx[:, q0:q0 + step] = best % n_points
        dist2[:, q0:q0 + step] = torch.div(best, n_points, rounding_mode="floor").float() / GRID
