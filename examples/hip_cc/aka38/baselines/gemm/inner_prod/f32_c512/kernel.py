"""Trusted PyTorch baseline for t2h/aka/inner_prod_f32_c512 at native dtype."""
import torch


def run(*, feat_img, feat_sound, scale, bias, out):
    batch, channels = feat_sound.shape[0], feat_sound.shape[1]
    weighted = feat_img.reshape(batch, 1, channels) * scale.reshape(1, 1, channels)
    z = torch.bmm(weighted, feat_sound.reshape(batch, channels, -1))
    out.copy_(z.reshape(out.shape) + bias)
