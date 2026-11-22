"""Trusted PyTorch baseline for t2h/aka/transpose_f32_4d_swap01 at native dtype."""
import torch


def run(*, x, out):
    out.copy_(x.transpose(0, 1))
