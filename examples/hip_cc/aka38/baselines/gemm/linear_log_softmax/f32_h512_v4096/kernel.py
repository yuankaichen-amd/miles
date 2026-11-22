"""Trusted PyTorch baseline for t2h/aka/linear_log_softmax_f32_h512_v4096 at native dtype."""
import torch
import torch.nn.functional as F


def run(*, x, weight, bias, out):
    out.copy_(torch.log_softmax(F.linear(x, weight, bias), dim=-1))
