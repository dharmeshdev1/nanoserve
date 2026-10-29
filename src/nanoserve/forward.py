"""Forward pass over a paged cache, reusing a trained microllm model's weights."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from microllm.model import GPT, apply_rope
from torch import Tensor

from nanoserve.paged import PagedKVCache


@torch.no_grad()
def paged_forward(
    model: GPT, cache: PagedKVCache, token_ids: Tensor, positions: Tensor, tables: Tensor
) -> Tensor:
    """Logits ``[B, T, vocab]`` for ``token_ids`` placed at absolute ``positions``.

    ``tables`` is ``[B, max_blocks]``, padded with any valid block id. Each query attends
    to keys at positions ``<= its own position``, which also hides the padding blocks and
    any stale data beyond the sequence, so sequences of different lengths share one batch.
    """
    cfg = model.cfg
    B, T = token_ids.shape
    cos = model.rope_cos[positions][:, None]  # [B, 1, T, D/2]
    sin = model.rope_sin[positions][:, None]
    span = tables.shape[1] * cache.block_size
    allowed = torch.arange(span, device=token_ids.device)[None, None, :] <= positions[:, :, None]
    mask = allowed[:, None]  # [B, 1, T, span]
    group = cfg.n_head // cfg.n_kv_head

    x = model.tok_emb(token_ids)
    for i, block in enumerate(model.blocks):
        attn = block.attn
        h = block.attn_norm(x)
        q = attn.q_proj(h).view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cache.write(i, tables, positions, k, v)
        keys, values = cache.gather(i, tables)
        if group > 1:
            keys = keys.repeat_interleave(group, dim=1)
            values = values.repeat_interleave(group, dim=1)
        y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        x = x + attn.o_proj(y.transpose(1, 2).reshape(B, T, -1))
        x = x + block.mlp(block.mlp_norm(x))
    return model.lm_head(model.final_norm(x))
