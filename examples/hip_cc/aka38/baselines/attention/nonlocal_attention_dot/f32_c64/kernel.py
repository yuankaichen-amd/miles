"""Trusted PyTorch baseline for t2h/aka/nonlocal_attention_dot_f32_c64 at native dtype."""
import torch

_MAX_ENERGY_ELEMENTS = 1 << 28


def _project(weight, bias, x):
    return torch.einsum("oc,bcn->bon", weight, x) + bias[None, :, None]


def run(*, x, query_weight, query_bias, key_weight, key_bias, value_weight, value_bias,
        gamma_weight, gamma_bias, out):
    batch, _, n = x.shape
    q = _project(query_weight, query_bias, x)
    k = _project(key_weight, key_bias, x)
    v = _project(value_weight, value_bias, x)
    y = torch.zeros_like(v)
    row_chunk = max(16, _MAX_ENERGY_ELEMENTS // (batch * n))
    for i0 in range(0, n, row_chunk):
        energy = torch.bmm(q[:, :, i0:i0 + row_chunk].transpose(1, 2), k)
        energy = torch.where(energy > 0, energy, torch.expm1(energy)) / n
        y += torch.bmm(v[:, :, i0:i0 + row_chunk], energy)
    out.copy_(_project(gamma_weight, gamma_bias, y))
