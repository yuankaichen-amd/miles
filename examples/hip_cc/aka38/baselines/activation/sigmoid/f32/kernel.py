"""Trusted PyTorch baseline for t2h/aka/sigmoid_f32 at native dtype."""
import torch


def run(*, x, a, max_value, out):
    out.copy_(torch.sigmoid(x * a) * max_value)
