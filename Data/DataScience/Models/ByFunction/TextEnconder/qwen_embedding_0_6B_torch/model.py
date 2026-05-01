"""Pure-torch Qwen3 Embedding 0.6B model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Qwen3Attention
from .cache import Qwen3Cache
from .config import Qwen3EmbeddingConfig
from .layers import RMSNorm, SwiGLUMLP
from .rotary import Qwen3RotaryEmbedding


@dataclass
class Qwen3ModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[Qwen3Cache] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None

    def __getitem__(self, key: str):
        return getattr(self, key)


def _build_causal_mask(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: int | None = None,
) -> Optional[torch.Tensor]:
    if seq_len == 1 and past_len == 0 and attention_mask is None:
        return None

    kv_len = past_len + seq_len
    min_value = torch.finfo(dtype).min
    positions_q = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(-1)
    positions_k = torch.arange(kv_len, device=device).unsqueeze(0)
    masked = positions_k > positions_q
    if sliding_window is not None:
        masked = masked | (positions_k <= (positions_q - sliding_window))
    mask = torch.where(
        masked,
        torch.tensor(min_value, dtype=dtype, device=device),
        torch.zeros((), dtype=dtype, device=device),
    )
    mask = mask[None, None, :, :]

    if attention_mask is not None:
        if attention_mask.shape[-1] < kv_len:
            attention_mask = F.pad(attention_mask, (kv_len - attention_mask.shape[-1], 0), value=1)
        elif attention_mask.shape[-1] > kv_len:
            attention_mask = attention_mask[:, -kv_len:]
        padding = (attention_mask == 0)[:, None, None, :]
        mask = mask.expand(attention_mask.shape[0], 1, seq_len, kv_len).clone()
        mask = mask.masked_fill(padding, min_value)
    return mask


class Qwen3DecoderLayer(nn.Module):
    """One pre-norm Qwen3 decoder block."""

    def __init__(self, config: Qwen3EmbeddingConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Qwen3Cache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3EmbeddingModel(nn.Module):
    """Base Qwen3 decoder model used by Qwen3-Embedding-0.6B."""

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        strict: bool = True,
    ) -> "Qwen3EmbeddingModel":
        from .weights import load_qwen3_embedding_weights

        config = Qwen3EmbeddingConfig.from_pretrained(model_dir)
        model = cls(config)
        if dtype is not None:
            model.to(dtype=dtype)
        load_qwen3_embedding_weights(model, model_dir, strict=strict, dtype=dtype)
        if device is not None:
            model.to(device)
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Qwen3Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape

        if use_cache and past_key_values is None:
            past_key_values = Qwen3Cache(self.config.num_hidden_layers)
        past_len = 0 if past_key_values is None else past_key_values.get_seq_length(0)

        if position_ids is None:
            cache_position = torch.arange(
                past_len,
                past_len + seq_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        sliding_window = None
        if self.config.use_sliding_window and self.config.sliding_window is not None:
            sliding_window = self.config.sliding_window
        causal_mask = _build_causal_mask(
            attention_mask,
            seq_len,
            past_len,
            inputs_embeds.dtype,
            inputs_embeds.device,
            sliding_window=sliding_window,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states: list[torch.Tensor] | None = [] if output_hidden_states else None

        for decoder_layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return Qwen3ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


class Qwen3ForCausalLM(nn.Module):
    """Causal-LM wrapper. The embedding checkpoint ties lm_head to embeddings."""

    def __init__(self, config: Qwen3EmbeddingConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3EmbeddingModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        strict: bool = True,
    ) -> "Qwen3ForCausalLM":
        from .weights import load_qwen3_embedding_weights

        config = Qwen3EmbeddingConfig.from_pretrained(model_dir)
        model = cls(config)
        if dtype is not None:
            model.to(dtype=dtype)
        load_qwen3_embedding_weights(model, model_dir, strict=strict, dtype=dtype)
        if device is not None:
            model.to(device)
        return model

    def forward(self, *args, **kwargs) -> dict:
        out = self.model(*args, **kwargs)
        logits = self.lm_head(out.last_hidden_state)
        return {
            "logits": logits,
            "last_hidden_state": out.last_hidden_state,
            "past_key_values": out.past_key_values,
            "hidden_states": out.hidden_states,
        }


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token, matching the Qwen embedding example."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if bool(left_padding):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


@torch.no_grad()
def embed_texts(
    model: Qwen3EmbeddingModel,
    tokenizer,
    texts: list[str],
    *,
    max_length: int = 8192,
    normalize: bool = True,
) -> torch.Tensor:
    """Tokenize and embed a batch of texts."""
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)
    outputs = model(**batch)
    embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings
