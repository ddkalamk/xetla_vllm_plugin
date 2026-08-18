"""SDPA stand-in for Maple's flash_attention_forward, for a CPU reference run.

Drop this in beside modeling_maple.py as `fa3.py`: the upstream one imports
flash_attn at module scope, so the HF path cannot load without CUDA. That makes
it possible to generate a dense reference that shares nothing with our vllm
model file or the int2 kernels, which is how we confirmed maple-preview's
failure to finish a hard proof is the checkpoint and not the port.

Inputs arrive as [B, H, T, D] -- the caller transposes before calling -- and it
reshapes our result with (bsz, q_len, -1), so we return [B, T, H, D]. Getting
that backwards silently produces a shape error deep in o_proj.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attention_forward(
    module,
    query_states: torch.Tensor,   # [B, Hq, T, D]
    key_states: torch.Tensor,     # [B, Hkv, S, D]
    value_states: torch.Tensor,
    attention_mask=None,
    dropout: float = 0.0,
    position_ids=None,
    scaling=None,
    sliding_window=None,
    **kwargs,
):
    # The caller already did transpose(1, 2), so these are [B, H, T, D], and it
    # reshapes our result with (bsz, q_len, -1) -- hand back [B, T, H, D].
    q, k, v = query_states, key_states, value_states

    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    T, S = q.shape[2], k.shape[2]
    # Keys are the last S positions, so query t sits at absolute S - T + t.
    qpos = torch.arange(S - T, S, device=q.device).unsqueeze(1)
    kpos = torch.arange(S, device=q.device).unsqueeze(0)
    allowed = kpos <= qpos
    if sliding_window is not None:
        allowed &= kpos > qpos - sliding_window

    bias = torch.zeros(T, S, dtype=q.dtype, device=q.device)
    bias.masked_fill_(~allowed, torch.finfo(q.dtype).min)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=bias[None, None], dropout_p=dropout, scale=scaling
    )
    return out.transpose(1, 2).contiguous(), None
