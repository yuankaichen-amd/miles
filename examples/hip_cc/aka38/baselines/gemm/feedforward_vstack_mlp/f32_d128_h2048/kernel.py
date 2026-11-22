"""Trusted PyTorch baseline for t2h/aka/feedforward_vstack_mlp_f32_d128_h2048 at native dtype."""
import torch
import torch.nn.functional as F


def run(*, x, y, fc1_weight, fc1_bias, fc2_weight, fc2_bias, out):
    hidden = torch.relu(F.linear(torch.vstack([x, y]), fc1_weight, fc1_bias))
    out.copy_(torch.sigmoid(F.linear(hidden, fc2_weight, fc2_bias)))
