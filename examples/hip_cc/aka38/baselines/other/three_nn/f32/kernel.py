"""Trusted PyTorch baseline for t2h/aka/three_nn_f32 at native dtype."""
import torch

_CHUNK_ELEMENTS = 1 << 25


def run(*, unknown, known, dist, idx):
    batch, n_target, _ = unknown.shape
    n_source = known.shape[1]
    step = max(1, _CHUNK_ELEMENTS // (batch * n_source))
    for t0 in range(0, n_target, step):
        d = unknown[:, t0:t0 + step, None, :] - known[:, None, :, :]
        d2 = d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]
        values, order = torch.sort(d2, dim=-1, stable=True)
        dist[:, t0:t0 + step] = values[..., :3].sqrt()
        idx[:, t0:t0 + step] = order[..., :3]
