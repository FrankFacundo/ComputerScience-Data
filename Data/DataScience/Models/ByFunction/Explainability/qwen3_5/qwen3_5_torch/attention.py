"""Full-attention block for Qwen3.5 (GQA + per-head q/k RMSNorm + output gate).

This file implements the "full attention" branch used by Qwen3.5 in roughly
every 4th transformer layer (the others use the linear-attention branch in
:mod:`linear_attention`). The block combines several well-known techniques:

* **Scaled dot-product attention** (Vaswani et al., 2017) — "Attention Is All
  You Need": https://arxiv.org/abs/1706.03762
* **Grouped-Query Attention (GQA)** (Ainslie et al., 2023) — keys/values are
  shared across groups of query heads: https://arxiv.org/abs/2305.13245
* **Rotary position embedding (RoPE)** — see :mod:`rotary`.
* **Q/K per-head RMSNorm** — as introduced by Qwen2/Qwen3 (Dehghani et al.
  "Scaling Vision Transformers to 22B Parameters", 2023, also popularised it):
  normalising each head's q and k stabilises training and also bounds the
  attention logits.
* **Output gate** — a sigmoid gate is derived from the same input and applied
  multiplicatively to the attention output before the final projection.
  Related ideas: gated-attention-unit (GAU) in Hua et al. 2022
  (https://arxiv.org/abs/2202.10447).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import HybridCache
from .config import Qwen3_5TextConfig
from .layers import RMSNorm
from .rotary import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""Repeat KV heads to match the query-head count under GQA.

    Math / semantics
    ----------------
    In Grouped-Query Attention there are ``H_q`` query heads and ``H_kv`` key
    heads with ``H_q = H_kv * n_rep``. Each KV head is shared across
    ``n_rep`` query heads. This op tiles

    .. math::
        K \in \mathbb{R}^{B \times H_{kv} \times T \times D}
        \longrightarrow
        \hat K \in \mathbb{R}^{B \times H_q \times T \times D}

    by repeating each row ``n_rep`` times. ``n_rep == 1`` means ordinary
    multi-head attention (no sharing).

    Reference (GQA): https://arxiv.org/abs/2305.13245
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    num_kv_groups: int,
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    r"""Reference scaled-dot-product attention (no Flash, no SDPA fused kernel).

    Math
    ----
    With repeated-KV for GQA, attention is

    .. math::
        A = \operatorname{softmax}\!\Big(\tfrac{Q K^{\top}}{\sqrt{d_h}} + M\Big),
        \qquad
        \text{Attn}(Q, K, V) = A V

    where :math:`M` is an additive mask with ``0`` at allowed positions and
    ``-\infty`` (i.e. ``torch.finfo(dtype).min``) at forbidden positions.
    The softmax is computed in fp32 for numerical stability and cast back
    afterwards. ``scaling = 1/sqrt(d_h)``.

    The output is reshaped to ``(B, S, H, D)`` so the subsequent reshape /
    ``o_proj`` can treat the head axis as "just another feature block".

    Reference: Vaswani et al., 2017 — https://arxiv.org/abs/1706.03762
    """
    key = repeat_kv(key, num_kv_groups)
    value = repeat_kv(value, num_kv_groups)

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    # softmax(·) in fp32 for stability, then cast back
    scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    scores = F.dropout(scores, p=dropout, training=training)
    out = torch.matmul(scores, value)  # (b, h, s, d)
    return out.transpose(1, 2).contiguous()  # (b, s, h, d)


class Qwen3_5Attention(nn.Module):
    r"""Grouped-Query Attention with q/k RMSNorm and a per-token output gate.

    Projections
    -----------
    For input :math:`x \in \mathbb{R}^{B \times S \times d_{\text{model}}}` the
    block computes

    .. math::
        [Q \,\|\, G] = x W_Q,\qquad K = x W_K,\qquad V = x W_V

    where ``q_proj`` produces a concatenated ``(Q, G)`` with twice the normal
    width — the second half is the *gate*. Shapes after reshape:

    * :math:`Q, G \in \mathbb{R}^{B \times S \times H_q \times d_h}`
    * :math:`K, V \in \mathbb{R}^{B \times S \times H_{kv} \times d_h}`

    Per-head RMSNorm
    ----------------
    RMSNorm is applied **inside** the head axis on q and k independently:

    .. math::
        Q \leftarrow \text{RMSNorm}(Q),\qquad K \leftarrow \text{RMSNorm}(K).

    This improves stability for long-context attention and keeps the softmax
    logits in a bounded range.

    Rotary
    ------
    RoPE is applied to q and k (partial rotary, see :mod:`rotary`):

    .. math::
        Q_m^{\text{rot}} = R_m Q_m,\qquad K_m^{\text{rot}} = R_m K_m

    with rotation matrices :math:`R_m` parameterised by the token position
    :math:`m`.

    Attention + gate
    ----------------
    .. math::
        O = \text{Attn}(Q^{\text{rot}}, K^{\text{rot}}, V)

    The gate is derived from the same input, passed through a sigmoid, and
    applied elementwise to the attention output **before** ``o_proj``:

    .. math::
        \tilde O = O \odot \sigma(G),\qquad y = W_O\, \tilde O.

    KV cache
    --------
    During incremental decoding, ``past_key_values.update_full`` concatenates
    the new ``(K, V)`` along the time dim. Full-attention layers therefore
    grow their KV memory linearly with the decoded length (:math:`O(T)`), in
    contrast with the linear-attention layers whose cache is constant-size.

    References
    ----------
    * GQA: https://arxiv.org/abs/2305.13245
    * Attention: https://arxiv.org/abs/1706.03762
    * RoPE: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        # scaling factor 1/sqrt(d_h) for scaled dot-product attention
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        # q_proj outputs [Q | G] concatenated, hence "* 2"
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[HybridCache] = None,
    ) -> torch.Tensor:
        r"""Run one full-attention block.

        Algorithmic steps
        -----------------
        1. Project ``x`` through ``q_proj`` and split into ``(Q, gate)``.
        2. Project ``x`` through ``k_proj``, ``v_proj``.
        3. Apply per-head RMSNorm to ``Q`` and ``K``.
        4. Apply RoPE: ``Q', K' = apply_rotary_pos_emb(Q, K, cos, sin)``.
        5. Append ``(K', V)`` to the KV cache.
        6. Compute scaled dot-product attention :math:`A V` with GQA repeat.
        7. Multiply by the sigmoid gate and project back to ``d_model``.
        """
        input_shape = hidden_states.shape[:-1]  # (B, S)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Split q_proj output into (Q, gate) — each view has per-head layout.
        q_raw = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query, gate = torch.chunk(q_raw, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        # Per-head RMSNorm on q and k (stabilises softmax logits).
        query_states = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # RoPE: Q, K  ←  R_m · Q, R_m · K
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Append to KV cache for full-attention layers.
        if past_key_values is not None:
            key_states, value_states = past_key_values.update_full(
                self.layer_idx, key_states, value_states
            )

        # Softmax attention + GQA repeat-KV.
        attn_out = eager_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            num_kv_groups=self.num_kv_groups,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )

        # Flatten heads, apply sigmoid gate, project back to hidden_size.
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        attn_out = attn_out * torch.sigmoid(gate)          # ← per-token output gate
        return self.o_proj(attn_out)
