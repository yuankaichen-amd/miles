"""Trusted PyTorch baseline for t2h/aka/gelu_f32 at native dtype."""
import torch
import torch.nn.functional as F


def run(*, x, out):
    out.copy_(F.gelu(x))
