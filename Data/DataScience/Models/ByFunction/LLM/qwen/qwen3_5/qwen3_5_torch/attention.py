"""Full-attention block for Qwen3.5 (GQA + per-head q/k RMSNorm + output gate)."""

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
    key = repeat_kv(key, num_kv_groups)
    value = repeat_kv(value, num_kv_groups)

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    scores = F.dropout(scores, p=dropout, training=training)
    out = torch.matmul(scores, value)  # (b, h, s, d)
    return out.transpose(1, 2).contiguous()  # (b, s, h, d)


class Qwen3_5Attention(nn.Module):
    """Grouped-Query Attention with a per-token output gate.

    `q_proj` outputs `2 * num_heads * head_dim` — the second half becomes a
    sigmoid gate applied to the attention output before `o_proj`.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

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
        input_shape = hidden_states.shape[:-1]  # (B, S)
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_raw = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query, gate = torch.chunk(q_raw, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        # Per-head RMSNorm on q and k
        query_states = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update_full(
                self.layer_idx, key_states, value_states
            )

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

        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        attn_out = attn_out * torch.sigmoid(gate)
        return self.o_proj(attn_out)
