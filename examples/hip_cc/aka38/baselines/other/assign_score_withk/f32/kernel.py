"""Trusted PyTorch baseline for t2h/aka/assign_score_withk_f32 at native dtype."""
import torch

_CHUNK_ELEMENTS = 1 << 26


def run(*, scores, point_features, center_features, knn_idx, out):
    batch, n_points, m, o = point_features.shape
    n_centers, k = scores.shape[1:3]
    valid = (knn_idx >= 0) & (knn_idx < n_points)
    nb = knn_idx.clamp(0, n_points - 1)
    ct = nb[:, :, :1].expand_as(nb)
    step = max(1, _CHUNK_ELEMENTS // (n_centers * k * m * o))
    for b0 in range(0, batch, step):
        b1 = min(batch, b0 + step)
        rows = torch.arange(b0, b1, device=scores.device)[:, None, None]
        diff = point_features[rows, nb[b0:b1]] - center_features[rows, ct[b0:b1]]
        value = torch.einsum("bnkm,bnkmo->bonk", scores[b0:b1], diff)
        out[b0:b1] = value * valid[b0:b1].unsqueeze(1)
