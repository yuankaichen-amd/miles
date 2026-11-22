"""Trusted PyTorch baseline for t2h/aka/paged_attention_ragged_bf16_q8_kv1_d128_p1 at native dtype."""
import torch

_CHUNK_TOKENS = 1 << 16


def run(*, query, key_cache, value_cache, kv_indptr, kv_indices, scale, max_context_len, out):
    num_seqs, num_query_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    group = num_query_heads // num_kv_heads
    total = kv_indices.shape[0]
    device = query.device
    indptr = kv_indptr.long()
    position = torch.arange(total, device=device)
    valid = position < indptr[-1]
    seq = torch.searchsorted(indptr[1:], position, right=True).clamp(max=num_seqs - 1)
    slots = kv_indices.long()
    q = query.view(num_seqs, num_kv_heads, group, head_size)
    scores = torch.empty((total, num_kv_heads, group), dtype=torch.float32, device=device)
    for p0 in range(0, total, _CHUNK_TOKENS):
        part = slice(p0, p0 + _CHUNK_TOKENS)
        k = key_cache[slots[part]].unsqueeze(-1)
        scores[part] = torch.matmul(q[seq[part]], k).squeeze(-1).float() * scale
    scores.masked_fill_(~valid[:, None, None], float("-inf"))
    index = seq[:, None, None].expand_as(scores)
    smax = torch.full((num_seqs, num_kv_heads, group), float("-inf"), dtype=torch.float32, device=device)
    smax.scatter_reduce_(0, index, scores, "amax", include_self=True)
    probs = torch.where(valid[:, None, None], torch.exp(scores - smax[seq]), torch.zeros_like(scores))
    denom = torch.zeros_like(smax).index_add_(0, seq, probs)
    numer = torch.zeros((num_seqs, num_kv_heads, group, head_size), dtype=torch.float32, device=device)
    for p0 in range(0, total, _CHUNK_TOKENS):
        part = slice(p0, p0 + _CHUNK_TOKENS)
        v = value_cache[slots[part]].float().unsqueeze(2)
        numer.index_add_(0, seq[part], probs[part].unsqueeze(-1) * v)
    out.copy_((numer / denom.unsqueeze(-1)).view(num_seqs, num_query_heads, head_size))
