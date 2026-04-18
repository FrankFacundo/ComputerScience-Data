"""Decoder layer, text model and causal-LM wrapper."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import Qwen3_5Attention
from .cache import HybridCache
from .config import Qwen3_5TextConfig
from .layers import RMSNorm, SwiGLUMLP
from .linear_attention import Qwen3_5GatedDeltaNet
from .rotary import TextRotaryEmbedding


def _build_causal_mask(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return a 4D causal additive mask of shape (B, 1, S, KV_len).

    Follows transformers' convention: 0 where attention is allowed, `min(dtype)`
    where it is not.
    """
    if seq_len == 1 and past_len == 0 and attention_mask is None:
        return None

    kv_len = past_len + seq_len
    min_value = torch.finfo(dtype).min

    # lower-triangular causal mask (allow prev tokens + self)
    positions_q = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(-1)
    positions_k = torch.arange(kv_len, device=device).unsqueeze(0)
    causal = positions_k > positions_q  # True = forbidden
    mask = torch.where(causal, torch.tensor(min_value, dtype=dtype, device=device), torch.zeros((), dtype=dtype, device=device))
    mask = mask[None, None, :, :]  # (1, 1, S, KV_len)

    if attention_mask is not None:
        batch_size = attention_mask.shape[0]
        # pad 2D mask if it's shorter than kv_len (can happen with KV cache)
        if attention_mask.shape[-1] < kv_len:
            pad = kv_len - attention_mask.shape[-1]
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=1)
        elif attention_mask.shape[-1] > kv_len:
            attention_mask = attention_mask[:, :kv_len]
        am = (attention_mask == 0)[:, None, None, :]  # True = padding
        mask = mask.expand(batch_size, 1, seq_len, kv_len).clone()
        mask = mask.masked_fill(am, min_value)

    return mask


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unknown layer type {self.layer_type!r}")
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        full_attention_mask: Optional[torch.Tensor] = None,
        linear_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[HybridCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=linear_attention_mask,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config=config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> dict:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape

        if use_cache and past_key_values is None:
            past_key_values = HybridCache(layer_types=self.config.layer_types)

        past_len = 0
        if past_key_values is not None:
            # full-attention length (per-layer KV length is the same across full layers)
            for i, t in enumerate(self.config.layer_types):
                if t == "full_attention":
                    past_len = past_key_values.get_seq_length(i)
                    break

        # Build 4D position ids: the rotary needs (3, bs, seq); mrope axes T/H/W.
        # For text-only, all three axes hold the same integer.
        if position_ids is None:
            base_pos = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
            base_pos = base_pos.unsqueeze(0).expand(batch_size, -1)  # (bs, seq)
            text_position_ids = base_pos
            rope_position_ids = base_pos[None, ...].expand(3, batch_size, seq_len)
        else:
            if position_ids.ndim == 2:
                text_position_ids = position_ids
                rope_position_ids = position_ids[None, ...].expand(3, batch_size, seq_len)
            elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
                text_position_ids = position_ids[0]
                rope_position_ids = position_ids
            elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
                text_position_ids = position_ids[0]
                rope_position_ids = position_ids[1:]
            else:
                raise ValueError(f"Unsupported position_ids shape {tuple(position_ids.shape)}")

        # Build per-layer masks — full attention uses the 4D causal mask; linear
        # attention uses the raw 2D mask (only for zeroing out padding).
        full_mask = _build_causal_mask(
            attention_mask, seq_len, past_len, inputs_embeds.dtype, inputs_embeds.device
        )
        linear_mask = attention_mask
        if past_key_values is not None and past_key_values.has_previous_state():
            linear_mask = None
        elif attention_mask is not None and torch.all(attention_mask == 1):
            linear_mask = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                full_attention_mask=full_mask,
                linear_attention_mask=linear_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)

        if past_key_values is not None:
            past_key_values.advance(seq_len)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": past_key_values if use_cache else None,
        }


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3_5TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> dict:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = out["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        return {
            "logits": logits,
            "last_hidden_state": hidden_states,
            "past_key_values": out["past_key_values"],
        }
