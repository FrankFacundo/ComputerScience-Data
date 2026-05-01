"""Minimal KV cache for full-attention Qwen3 decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FullAttentionState:
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


class Qwen3Cache:
    """Per-layer concatenating KV cache."""

    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers
        self.layers: dict[int, FullAttentionState] = {}

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.layers.setdefault(layer_idx, FullAttentionState())
        return state.update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        state = self.layers.get(layer_idx)
        return 0 if state is None else state.get_seq_length()
