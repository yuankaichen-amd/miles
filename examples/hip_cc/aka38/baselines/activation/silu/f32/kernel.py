"""Trusted PyTorch baseline for t2h/aka/silu_f32 at native dtype."""
import torch
import torch.nn.functional as F


def run(*, x, out):
    out.copy_(F.silu(x))
