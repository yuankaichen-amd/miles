"""Trusted PyTorch baseline for t2h/aka/skinny_gemm_a16w16_m2_k7168 at native dtype."""
import torch
import torch.nn.functional as F


def run(*, x, weight, out):
    out.copy_(F.linear(x, weight))
