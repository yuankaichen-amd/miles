"""Trusted PyTorch baseline for t2h/aka/conv1x1_ffn_gelu_f32_h512 at native dtype."""
import torch
import torch.nn.functional as F


def run(*, x, conv_weight, conv_bias, linear_weight, linear_bias, out):
    hidden = F.gelu(F.linear(x, conv_weight, conv_bias))
    out.copy_(F.linear(hidden, linear_weight, linear_bias))
