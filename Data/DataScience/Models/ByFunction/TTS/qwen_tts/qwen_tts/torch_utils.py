# coding=utf-8
"""Small Torch-only utilities used by the local Qwen3-TTS runtime.

This module replaces the narrow subset of Hugging Face Transformers utilities
that Qwen3-TTS needs for inference. It intentionally stays small: no hub
loading, no trainer hooks, no quantized caches, no remote code.
"""

from __future__ import annotations

import json
import logging as py_logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, TypedDict, Unpack

import torch
from torch import nn
from torch.nn import functional as F


class _LoggingFacade:
    @staticmethod
    def get_logger(name: str):
        logger = py_logging.getLogger(name)
        if not hasattr(logger, "warning_once"):
            seen: set[str] = set()

            def warning_once(message, *args, **kwargs):
                key = str(message)
                if key not in seen:
                    seen.add(key)
                    logger.warning(message, *args, **kwargs)

            logger.warning_once = warning_once  # type: ignore[attr-defined]
        return logger


logging = _LoggingFacade()


class SimpleConfig:
    model_type: str | None = None
    return_dict: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_cache: bool = True
    _attn_implementation: str = "eager"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "_attn_implementation"):
            self._attn_implementation = "eager"
        if not hasattr(self, "return_dict"):
            self.return_dict = True
        if not hasattr(self, "use_return_dict"):
            self.use_return_dict = self.return_dict
        if not hasattr(self, "output_attentions"):
            self.output_attentions = False
        if not hasattr(self, "output_hidden_states"):
            self.output_hidden_states = False
        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = None
        if not hasattr(self, "bos_token_id"):
            self.bos_token_id = None
        if not hasattr(self, "eos_token_id"):
            self.eos_token_id = None

    @classmethod
    def from_pretrained(cls, path: str | Path):
        path = Path(path)
        config_path = path if path.name == "config.json" else path / "config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def layer_type_validation(layer_types: list[str]) -> None:
    allowed = {"full_attention", "sliding_attention"}
    bad = [x for x in layer_types if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported attention layer types: {bad}")


def rope_config_validation(config: Any) -> None:
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None and "type" in rope_scaling and "rope_type" not in rope_scaling:
        rope_scaling["rope_type"] = rope_scaling["type"]


class ModelOutput:
    """Dataclass-friendly output container with tuple-style integer indexing."""

    def to_tuple(self):
        values = []
        for key in getattr(self, "__dataclass_fields__", {}):
            value = getattr(self, key)
            if value is not None:
                values.append(value)
        return tuple(values)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        return self.to_tuple()[item]

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return [k for k in getattr(self, "__dataclass_fields__", {}) if getattr(self, k) is not None]

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[Any] = None
    attentions: Optional[Any] = None


class DynamicCache:
    """Minimal growing KV cache for decoder-only attention."""

    def __init__(self, *args, **kwargs):
        self.key_cache: list[Optional[torch.Tensor]] = []
        self.value_cache: list[Optional[torch.Tensor]] = []

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
        self._ensure_layer(layer_idx)
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
            return 0
        return int(self.key_cache[layer_idx].shape[-2])


Cache = DynamicCache


class GradientCheckpointingLayer(nn.Module):
    pass


class TorchPreTrainedModel(nn.Module):
    config_class = SimpleConfig

    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.config = config

    @classmethod
    def _from_config(cls, config):
        return cls(config)

    def post_init(self) -> None:
        return None

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


PreTrainedModel = TorchPreTrainedModel


class GenerationMixin:
    pass


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


ACT2FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "gelu": _gelu,
    "tanh": torch.tanh,
}


def use_kernel_forward_from_hub(_name: str):
    def decorator(cls):
        return cls

    return decorator


def auto_docstring(obj=None, **_kwargs):
    if obj is None:
        return lambda x: x
    return obj


def can_return_tuple(fn):
    return fn


def deprecate_kwarg(*_args, **_kwargs):
    return lambda fn: fn


def check_model_inputs(*_args, **_kwargs):
    return lambda fn: fn


class FlashAttentionKwargs(TypedDict, total=False):
    pass


def _default_rope_init(config, device=None):
    head_dim = getattr(config, "head_dim", None) or getattr(config, "hidden_size") // getattr(config, "num_attention_heads")
    theta = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS = {"default": _default_rope_init}


def dynamic_rope_update(fn):
    return fn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        if causal_mask.shape[-1] < key_states.shape[-2]:
            causal_mask = F.pad(causal_mask, (key_states.shape[-2] - causal_mask.shape[-1], 0), value=0.0)
        attn_weights = attn_weights + causal_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    # PyTorch SDPA applies its own scale unless provided.
    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask[:, :, :, : key_states.shape[-2]] if attention_mask is not None else None,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling,
    )
    return attn_output.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS = {
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
    "flash_attention_2": sdpa_attention_forward,
}


def _target_length(input_embeds, attention_mask, cache_position, past_key_values) -> int:
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_target = past_seen + int(input_embeds.shape[1])
    if attention_mask is not None:
        return max(int(attention_mask.shape[-1]), cache_target)
    return cache_target


def create_causal_mask(
    *,
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[DynamicCache],
    position_ids: Optional[torch.Tensor] = None,
    **_kwargs,
) -> torch.Tensor:
    batch_size, query_length = input_embeds.shape[:2]
    target_length = _target_length(input_embeds, attention_mask, cache_position, past_key_values)
    dtype = input_embeds.dtype
    device = input_embeds.device
    min_dtype = torch.finfo(dtype).min

    key_positions = torch.arange(target_length, device=device)
    query_positions = cache_position.to(device)
    blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    causal = torch.zeros((query_length, target_length), device=device, dtype=dtype)
    causal = causal.masked_fill(blocked, min_dtype)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, query_length, target_length).clone()

    if attention_mask is not None:
        if attention_mask.shape[-1] < target_length:
            pad = torch.ones(
                attention_mask.shape[0],
                target_length - attention_mask.shape[-1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
        key_mask = attention_mask[:, None, None, :target_length].to(device=device)
        causal = causal.masked_fill(key_mask == 0, min_dtype)
    return causal


def create_sliding_window_causal_mask(
    *,
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[DynamicCache],
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    causal = create_causal_mask(
        config=config,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
        **kwargs,
    )
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        return causal
    target_length = causal.shape[-1]
    key_positions = torch.arange(target_length, device=input_embeds.device)
    min_allowed = (cache_position.to(input_embeds.device) - int(sliding_window) + 1).clamp(min=0)
    blocked = key_positions.unsqueeze(0) < min_allowed.unsqueeze(1)
    causal = causal.masked_fill(blocked[None, None, :, :], torch.finfo(input_embeds.dtype).min)
    return causal


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        indices_to_remove = remove.scatter(-1, sorted_indices, remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[list[int]] = None,
) -> torch.Tensor:
    logits = logits.clone()
    if suppress_tokens:
        valid = [token for token in suppress_tokens if 0 <= token < logits.shape[-1]]
        if valid:
            logits[:, valid] = float("-inf")
    if repetition_penalty and repetition_penalty != 1.0 and generated_ids is not None and generated_ids.numel() > 0:
        for batch_idx in range(logits.shape[0]):
            seen = torch.unique(generated_ids[batch_idx])
            scores = logits[batch_idx, seen]
            logits[batch_idx, seen] = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
    if not do_sample or temperature is None or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_namespace(mapping: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**mapping)
