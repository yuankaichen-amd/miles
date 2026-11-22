"""Trusted PyTorch baseline for t2h/aka/cross_entropy_label_smoothing_f32_c100 at native dtype."""
import torch


def run(*, x, target, weight, ignore_index, smooth_eps, losses, out):
    num_classes = x.shape[1]
    eps = float(smooth_eps)
    eps_sum = eps / num_classes if eps > 0 else 0.0
    eps_nll = 1.0 - eps_sum - eps
    lsm = torch.log_softmax(x, dim=-1) * weight
    likelihood = lsm.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    row = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    if ignore_index >= 0:
        ignored = target == ignore_index
        row = row.masked_fill(ignored, 0.0)
        valid = (~ignored).sum()
        mean = torch.where(valid > 0, row.sum() / valid.clamp(min=1), torch.zeros_like(row.sum()))
    else:
        mean = row.sum() / row.shape[0]
    losses.copy_(row)
    out.copy_(mean)
