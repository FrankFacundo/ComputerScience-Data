"""A hybrid KV / linear-attention cache for Qwen3.5.

Qwen3.5 interleaves *full-attention* and *linear-attention* layers (the latter
is the Gated DeltaNet — see :mod:`linear_attention`). The two kinds of layers
have fundamentally different caches during autoregressive decoding:

* **Full attention**   — cache the complete ``K, V`` tensors and grow them by
  concatenating one column per decoded token. Space per layer: ``O(T · H_kv · d_h)``.
* **Linear attention (Gated DeltaNet)** — cache a constant-size state:

  1. A **causal-conv ring buffer** holding the last ``conv_kernel_size``
     tokens of the qkv-conv input (so the depthwise conv can continue across
     decode steps). Shape: ``(B, conv_dim, K)``.
  2. A **recurrent state** :math:`S_t \in \mathbb{R}^{H_v \times d_k \times d_v}`
     that summarises everything seen so far — see Eq. (1)–(2) in
     :mod:`linear_attention` for its update rule.

  Space per layer: constant in ``T`` — this is the whole point of
  linear-attention models (Mamba, DeltaNet, GatedDeltaNet).

For each layer index the cache stores exactly one of the two kinds, dispatched
by ``config.text_config.layer_types``.

References
----------
* Mamba:           Gu & Dao 2023  — https://arxiv.org/abs/2312.00752
* Mamba-2:         Dao & Gu 2024  — https://arxiv.org/abs/2405.21060
* DeltaNet:        Yang et al. 2024 — https://arxiv.org/abs/2406.06484
* Gated DeltaNet:  Yang et al. 2024 — https://arxiv.org/abs/2412.06464
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class FullAttentionState:
    """Concatenating KV cache for one full-attention layer.

    During incremental decoding ``key_cache`` and ``value_cache`` grow along
    the time dim: at step :math:`t`, ``K_t = [K_{<t} \| k_t]`` (likewise for
    ``V``). Softmax attention then operates over the full prefix, yielding
    exact (not approximate) attention at the cost of :math:`O(T)` memory.
    """

    key_cache: Optional[torch.Tensor] = None  # (B, num_kv_heads, T, head_dim)
    value_cache: Optional[torch.Tensor] = None

    def update(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new ``(key, value)`` along the time axis (``dim=-2``).

        Math:  :math:`K_{\le t} = \text{cat}(K_{<t}, k_t)` and analogously for V.
        Returns the updated tensors so the caller can use them directly.
        """
        if self.key_cache is None:
            self.key_cache = key
            self.value_cache = value
        else:
            self.key_cache = torch.cat([self.key_cache, key], dim=-2)
            self.value_cache = torch.cat([self.value_cache, value], dim=-2)
        return self.key_cache, self.value_cache

    def get_seq_length(self) -> int:
        """Current cached KV length ``T``."""
        if self.key_cache is None:
            return 0
        return self.key_cache.shape[-2]


@dataclass
class LinearAttentionState:
    """Constant-size state for one Gated DeltaNet layer.

    Contains:

    * ``conv_states``      — last ``conv_kernel_size`` tokens of the qkv-conv
      input, used for one-step causal-conv updates during decoding.
    * ``recurrent_states`` — the gated-delta-rule recurrence
      :math:`S \in \mathbb{R}^{B \times H_v \times d_k \times d_v}` such that
      the next output is :math:`o_t = q_t^{\top} S_t` (see Eq. in
      :mod:`linear_attention`).
    """

    conv_states: Optional[torch.Tensor] = None
    recurrent_states: Optional[torch.Tensor] = None


class HybridCache:
    r"""A minimal hybrid cache for Qwen3.5.

    Layers are addressed by ``layer_idx``. For full-attention layers, K/V
    tensors are concatenated along the time dim. For linear-attention layers,
    the cache stores a causal-conv ring buffer (last ``conv_kernel_size``
    tokens) and the gated-delta-rule recurrent state.

    The ``seen_tokens`` counter tracks the total text-side positions consumed
    so the rotary / position-id machinery has a reference for incremental
    decoding — in particular, at step :math:`t` the next token's position is
    ``past_len = seen_tokens`` and the rotary cos/sin is computed at that
    offset.
    """

    def __init__(self, layer_types: list[str]):
        self.layer_types = layer_types
        self.full: dict[int, FullAttentionState] = {}
        self.linear: dict[int, LinearAttentionState] = {}
        # Tracks the total number of text tokens consumed — analogous to the
        # position counter in a streaming RNN. Used to place the next token's
        # RoPE frequency and to switch DeltaNet paths (chunk vs recurrent).
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
        """Append ``(K, V)`` to the named full-attention layer and return it."""
        state = self.full.setdefault(layer_idx, FullAttentionState())
        return state.update(key_states, value_states)

    # -- linear attention ------------------------------------------------
    def has_linear_state(self, layer_idx: int) -> bool:
        """True iff a recurrent state has been written for this linear layer."""
        s = self.linear.get(layer_idx)
        return s is not None and s.recurrent_states is not None

    def get_linear(self, layer_idx: int) -> LinearAttentionState:
        """Return (create if missing) the linear state slot for this layer."""
        return self.linear.setdefault(layer_idx, LinearAttentionState())

    def update_linear_conv_state(self, layer_idx: int, conv_state: torch.Tensor) -> torch.Tensor:
        """Replace the causal-conv ring buffer for this layer."""
        s = self.get_linear(layer_idx)
        s.conv_states = conv_state
        return s.conv_states

    def update_linear_recurrent_state(self, layer_idx: int, recurrent_state: torch.Tensor) -> torch.Tensor:
        r"""Replace the recurrent state :math:`S \in \mathbb{R}^{B \times H_v \times d_k \times d_v}`."""
        s = self.get_linear(layer_idx)
        s.recurrent_states = recurrent_state
        return s.recurrent_states

    # -- bookkeeping -----------------------------------------------------
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the effective sequence length seen by layer ``layer_idx``.

        For full-attention this is the cached KV length; for linear-attention
        layers the true "state" is recurrent (no explicit length) — we return
        ``self._seen_tokens`` as a stand-in so the rotary / position machinery
        can still index the correct token offset.
        """
        kind = self._kind(layer_idx) if layer_idx < len(self.layer_types) else "full_attention"
        if kind == "full_attention":
            s = self.full.get(layer_idx)
            return 0 if s is None else s.get_seq_length()
        return self._seen_tokens

    def advance(self, seq_len: int) -> None:
        """Advance the global token counter by ``seq_len`` (called after each forward)."""
        self._seen_tokens += seq_len

    def has_previous_state(self) -> bool:
        """True iff any token has been consumed — used to branch prefill vs decode."""
        return self._seen_tokens > 0
