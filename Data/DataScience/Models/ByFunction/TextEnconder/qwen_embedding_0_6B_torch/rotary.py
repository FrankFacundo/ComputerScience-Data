"""Rotary position embeddings for text-only Qwen3."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import Qwen3EmbeddingConfig


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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _compute_inv_freq(config: Qwen3EmbeddingConfig, device=None) -> torch.Tensor:
    return 1.0 / (
        config.rope_theta
        ** (
            torch.arange(0, config.head_dim, 2, dtype=torch.int64)
            .to(device=device, dtype=torch.float)
            / config.head_dim
        )
    )


class Qwen3RotaryEmbedding(nn.Module):
    """Default Qwen3 RoPE over the full attention head dimension."""

    def __init__(self, config: Qwen3EmbeddingConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", _compute_inv_freq(config, device=device), persistent=False)

    def reset_inv_freq(self, device=None) -> None:
        device = self.inv_freq.device if device is None else device
        self.inv_freq = _compute_inv_freq(self.config, device=device)

    def _apply(self, fn):
        super()._apply(fn)
        self.reset_inv_freq()
        return self

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)
