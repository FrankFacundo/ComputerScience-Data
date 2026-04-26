"""A hybrid KV / linear-attention cache for Qwen3.6.

Identical in semantics to the Qwen3.5 cache: every layer is either a
*full-attention* layer (linear-in-T concat KV cache) or a *linear-attention*
layer (constant-size Gated DeltaNet state — conv ring buffer + recurrent
state). The MoE block does not touch the cache, so the cache structure does
not need any changes.

References — see qwen3_5_torch/cache.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FullAttentionState:
    """Concatenating KV cache for one full-attention layer."""

    key_cache: Optional[torch.Tensor] = None
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
    """Constant-size state for one Gated DeltaNet layer (conv buffer + recurrent S)."""

    conv_states: Optional[torch.Tensor] = None
    recurrent_states: Optional[torch.Tensor] = None


class HybridCache:
    """Per-layer KV cache: full-attention layers grow, linear-attention layers stay constant."""

    def __init__(self, layer_types: list[str]):
        self.layer_types = layer_types
        self.full: dict[int, FullAttentionState] = {}
        self.linear: dict[int, LinearAttentionState] = {}
        self._seen_tokens: int = 0

    def _kind(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def update_full(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.full.setdefault(layer_idx, FullAttentionState())
        return state.update(key_states, value_states)

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
