"""Rotary position embeddings for Qwen3.5.

Qwen3.5 uses interleaved multi-axis RoPE (mrope) with partial rotary. The text
rotary embedding accepts position ids of shape (3, bs, seq) where the three
rows correspond to (temporal, height, width), and projects them through a
shared `inv_freq` — then interleaves the per-axis frequencies into a single
per-token rotary vector.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import Qwen3_5TextConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial rotary embeddings to q/k.

    When `cos.shape[-1]` is smaller than the head dim (partial rotary), only
    the leading `rotary_dim` are rotated; the rest pass through unchanged.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _compute_default_inv_freq(config: Qwen3_5TextConfig, device=None) -> torch.Tensor:
    rope_params = config.rope_parameters
    base = rope_params["rope_theta"]
    partial = rope_params.get("partial_rotary_factor", 1.0)
    head_dim = config.head_dim
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq


def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Interleave the per-axis frequencies produced for (T, H, W).

    freqs:         (3, bs, seq, dim_half)
    returns:       (bs, seq, dim_half)
    """
    freqs_t = freqs[0].clone()
    for axis, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[axis] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[axis, ..., idx]
    return freqs_t


class TextRotaryEmbedding(nn.Module):
    """Qwen3.5 text-side rotary embedding with interleaved mrope."""

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])
        self.attention_scaling = 1.0

        inv_freq = _compute_default_inv_freq(config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids:
            - (bs, seq) — will be expanded to (3, bs, seq)
            - (3, bs, seq) — (T, H, W) positions
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # inv_freq: (dim_half,) -> (3, bs, dim_half, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bs, 1, seq)

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = _apply_interleaved_mrope(freqs, self.mrope_section)  # (bs, seq, dim_half)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)


class VisionRotaryEmbedding(nn.Module):
    """Vision rotary embedding (single-axis, as in Qwen3VL)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q), k_embed.to(orig_k)
