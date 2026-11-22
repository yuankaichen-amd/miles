"""Trusted PyTorch baseline for t2h/aka/paged_attention_decode_bf16_q32_kv8_d128_p16 at native dtype."""
import torch

_CHUNK_ELEMENTS = 1 << 27


def run(*, query, key_cache, value_cache, block_table, seq_lens, scale, out):
    num_blocks, num_kv_heads, head_outer, block_size, x = key_cache.shape
    head_size = head_outer * x
    num_seqs, num_query_heads, _ = query.shape
    group = num_query_heads // num_kv_heads
    table = block_table.long()
    max_blocks = table.shape[1]
    tokens = max_blocks * block_size
    invalid = torch.arange(tokens, device=query.device)[None, :] >= seq_lens.long()[:, None]
    step = max(1, _CHUNK_ELEMENTS // (tokens * num_kv_heads * head_size))
    for start in range(0, num_seqs, step):
        stop = min(num_seqs, start + step)
        s = stop - start
        # [s, blocks, kv, d/x, bs, x] -> [s, kv, d, tokens]; [s, blocks, kv, d, bs] -> [s, kv, tokens, d]
        k = key_cache[table[start:stop]].permute(0, 2, 3, 5, 1, 4).reshape(s, num_kv_heads, head_size, tokens)
        v = value_cache[table[start:stop]].permute(0, 2, 1, 4, 3).reshape(s, num_kv_heads, tokens, head_size)
        q = query[start:stop].view(s, num_kv_heads, group, head_size)
        scores = torch.matmul(q, k).float() * scale
        scores = scores.masked_fill(invalid[start:stop, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(query.dtype)
        out[start:stop] = torch.matmul(probs, v).reshape(s, num_query_heads, head_size)
