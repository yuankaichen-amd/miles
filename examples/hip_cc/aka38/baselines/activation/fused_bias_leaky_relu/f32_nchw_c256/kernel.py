"""Trusted PyTorch baseline for t2h/aka/fused_bias_leaky_relu_f32_nchw_c256 at native dtype."""
import torch


def run(*, x, bias, negative_slope, scale, out):
    v = x + bias.view(1, -1, 1, 1)
    out.copy_(torch.where(v >= 0, v, v * negative_slope) * scale)
