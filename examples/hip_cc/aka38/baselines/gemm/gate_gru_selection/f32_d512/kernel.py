"""Trusted PyTorch baseline for t2h/aka/gate_gru_selection_f32_d512 at native dtype."""
import torch
import torch.nn.functional as F


def run(*, x1, x2, reset_weight, reset_bias, update_weight, update_bias, proposal_weight, proposal_bias, out):
    joined = torch.cat([x1, x2], dim=-1)
    reset = torch.sigmoid(F.linear(joined, reset_weight, reset_bias))
    update = torch.sigmoid(F.linear(joined, update_weight, update_bias))
    proposal = torch.tanh(F.linear(torch.cat([reset * x1, x2], dim=-1), proposal_weight, proposal_bias))
    out.copy_((1.0 - update) * x1 + update * proposal)
