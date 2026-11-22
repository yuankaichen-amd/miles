"""Trusted PyTorch baseline for t2h/aka/tanh_f32 at native dtype."""
import torch


def run(*, x, a, scale, out):
    out.copy_(torch.tanh(x * a) * scale)
