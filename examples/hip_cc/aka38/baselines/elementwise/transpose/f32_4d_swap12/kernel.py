"""Trusted PyTorch baseline for t2h/aka/transpose_f32_4d_swap12 at native dtype."""
import torch


def run(*, x, out):
    out.copy_(x.transpose(1, 2))
