"""Trusted PyTorch baseline for t2h/aka/topk_transform_paged_f32_w262208_k512_p64 at native dtype."""
import torch

TOPK = 512
_CHUNK_ELEMENTS = 1 << 26


def run(*, scores, seq_lens, page_tables, page_size, metadata, out_page_indices):
    batch, width = scores.shape
    lengths = seq_lens.long().unsqueeze(1)
    positions = torch.arange(width, device=scores.device)
    prefix = positions[:TOPK].unsqueeze(0)
    step = max(1, _CHUNK_ELEMENTS // width)
    for b0 in range(0, batch, step):
        length = lengths[b0:b0 + step]
        masked = scores[b0:b0 + step].masked_fill(positions[None, :] >= length, float("-inf"))
        largest = torch.topk(masked, TOPK, dim=1).indices
        selected = torch.where(length <= TOPK, prefix, largest)
        pages = page_tables[b0:b0 + step].long().gather(1, selected // page_size)
        slots = pages * page_size + selected % page_size
        out_page_indices[b0:b0 + step] = torch.where(selected < length, slots, -1)
