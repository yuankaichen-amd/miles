"""Trusted PyTorch baseline for t2h/aka/ball_query_f32_ns5 at native dtype."""
import torch

NSAMPLE = 5
_CHUNK_ELEMENTS = 1 << 26


def _squared(radius):
    r = torch.tensor(radius, dtype=torch.float32)
    return float(r * r)


def run(*, xyz, center_xyz, min_radius, max_radius, idx):
    batch, n_points, _ = xyz.shape
    n_centers = center_xyz.shape[1]
    lower2, upper2 = _squared(min_radius), _squared(max_radius)
    device = xyz.device
    position = torch.arange(n_points, device=device)
    slots = torch.arange(NSAMPLE, device=device)
    width = min(NSAMPLE, n_points)
    step = max(1, _CHUNK_ELEMENTS // (batch * n_points))
    for c0 in range(0, n_centers, step):
        c = center_xyz[:, c0:c0 + step]
        d = c[:, :, None, :] - xyz[:, None, :, :]
        d2 = d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]
        match = (d2 == 0) | ((d2 >= lower2) & (d2 < upper2))
        key = torch.where(match, position, n_points)
        first = torch.topk(key, width, dim=-1, largest=False).values
        if width < NSAMPLE:
            first = torch.cat([first, first.new_full(first.shape[:-1] + (NSAMPLE - width,), n_points)], dim=-1)
        count = match.sum(-1, keepdim=True)
        padded = torch.where(slots < count, first, first[..., :1])
        padded = torch.where(count > 0, padded, torch.zeros_like(padded))
        idx[:, c0:c0 + step] = padded
