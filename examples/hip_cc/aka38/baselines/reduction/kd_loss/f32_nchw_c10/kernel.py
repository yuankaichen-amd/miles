"""Trusted PyTorch baseline for t2h/aka/kd_loss_f32_nchw_c10 at native dtype."""
import torch


def run(*, input, target, temp_factor, pixel_kl, out):
    temperature = float(temp_factor)
    log_p = torch.log_softmax(input / temperature, dim=1)
    kl = (torch.xlogy(target, target) - target * log_p).sum(dim=1)
    pixel_kl.copy_(kl)
    out.copy_(kl.sum() * (temperature ** 2 / input.shape[0]))
