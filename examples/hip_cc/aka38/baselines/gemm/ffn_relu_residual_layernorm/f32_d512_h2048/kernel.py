"""Trusted PyTorch baseline for t2h/aka/ffn_relu_residual_layernorm_f32_d512_h2048 at native dtype."""
import torch
import torch.nn.functional as F


def run(*, x, w1_weight, w1_bias, w2_weight, w2_bias, ln_gamma, ln_beta, eps, out):
    y = F.linear(torch.relu(F.linear(x, w1_weight, w1_bias)), w2_weight, w2_bias) + x
    out.copy_(F.layer_norm(y, (y.shape[-1],), ln_gamma, ln_beta, eps))
