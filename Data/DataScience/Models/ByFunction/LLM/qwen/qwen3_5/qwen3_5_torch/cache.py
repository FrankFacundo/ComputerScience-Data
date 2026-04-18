"""A hybrid KV / linear-attention cache for Qwen3.5.

For each layer the cache holds one of:
  - "full_attention":   key_states (B, Hk, T, Dk), value_states (B, Hk, T, Dv)
  - "linear_attention": conv_state (B, conv_dim, K), recurrent_state (B, Hv, Dk, Dv)

`HybridCache` is a per-layer dict of these states keyed by `layer_idx`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class FullAttentionState:
    key_cache: Optional[torch.Tensor] = None  # (B, num_kv_heads, T, head_dim)
    value_cache: Optional[torch.Tensor] = None

    def update(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache is None:
            self.key_cache = key
            self.value_cache = value
        else:
            self.key_cache = torch.cat([self.key_cache, key], dim=-2)
            self.value_cache = torch.cat([self.value_cache, value], dim=-2)
        return self.key_cache, self.value_cache

    def get_seq_length(self) -> int:
        if self.key_cache is None:
            return 0
        return self.key_cache.shape[-2]


@dataclass
class LinearAttentionState:
    conv_states: Optional[torch.Tensor] = None
    recurrent_states: Optional[torch.Tensor] = None


class HybridCache:
    """A minimal hybrid cache for Qwen3.5.

    Layers are addressed by `layer_idx`. For full-attention layers, K/V tensors
    are concatenated along the time dim. For linear-attention layers, the cache
    stores a causal-conv ring buffer (last `conv_kernel_size` tokens) and the
    gated-delta-rule recurrent state.
    """

    def __init__(self, layer_types: list[str]):
        self.layer_types = layer_types
        self.full: dict[int, FullAttentionState] = {}
        self.linear: dict[int, LinearAttentionState] = {}
        # `seen_tokens` tracks the total text-side positions consumed so the
        # rotary/position-id machinery has a reference for incremental decoding.
        self._seen_tokens: int = 0

    # -- helpers ---------------------------------------------------------
    def _kind(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    # -- full attention --------------------------------------------------
    def update_full(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.full.setdefault(layer_idx, FullAttentionState())
        return state.update(key_states, value_states)

    # -- linear attention ------------------------------------------------
    def has_linear_state(self, layer_idx: int) -> bool:
        s = self.linear.get(layer_idx)
        return s is not None and s.recurrent_states is not None

    def get_linear(self, layer_idx: int) -> LinearAttentionState:
        return self.linear.setdefault(layer_idx, LinearAttentionState())

    def update_linear_conv_state(self, layer_idx: int, conv_state: torch.Tensor) -> torch.Tensor:
        s = self.get_linear(layer_idx)
        s.conv_states = conv_state
        return s.conv_states

    def update_linear_recurrent_state(self, layer_idx: int, recurrent_state: torch.Tensor) -> torch.Tensor:
        s = self.get_linear(layer_idx)
        s.recurrent_states = recurrent_state
        return s.recurrent_states

    # -- bookkeeping -----------------------------------------------------
    def get_seq_length(self, layer_idx: int = 0) -> int:
        kind = self._kind(layer_idx) if layer_idx < len(self.layer_types) else "full_attention"
        if kind == "full_attention":
            s = self.full.get(layer_idx)
            return 0 if s is None else s.get_seq_length()
        return self._seen_tokens

    def advance(self, seq_len: int) -> None:
        self._seen_tokens += seq_len

    def has_previous_state(self) -> bool:
        return self._seen_tokens > 0
