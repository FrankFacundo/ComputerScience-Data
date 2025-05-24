# qwen3.py
import json
import os
from collections import OrderedDict
from typing import Optional, Union, Tuple, List, Callable  # Added List, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F  # For softmax, dropout in attention
import math

from accelerate import dispatch_model

from transformers.configuration_utils import PretrainedConfig
from transformers import Qwen3Config
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.generation.configuration_utils import GenerationConfig

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    StaticCache,
    SlidingWindowCache,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)  # Used in _update_causal_mask
from transformers.utils import logging  # For logger if needed, or replace with print

logger = logging.get_logger(__name__)  # Use transformers' logger or replace with print

path_model = "/home/frank/Datalake/models/Qwen/Qwen3-0.6B"  # Example local path


# Helper function for sampling logits (from original qwen3.py)
def _sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    filter_value: float = -float("Inf"),
):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if sorted_indices_to_remove.shape[-1] > 1:
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
        else:
            sorted_indices_to_remove[..., :] = False
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(
            logits, dtype=torch.bool, device=logits.device
        )
        indices_to_remove.scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)
    probabilities = torch.softmax(logits, dim=-1)
    sum_probs = torch.sum(probabilities, dim=-1, keepdim=True)
    probabilities = torch.where(
        sum_probs == 0,
        torch.ones_like(probabilities) / probabilities.shape[-1],
        probabilities / sum_probs,
    )
    probabilities = torch.nan_to_num(probabilities, nan=1.0 / probabilities.shape[-1])
    next_tokens = torch.multinomial(probabilities, num_samples=1)
    return next_tokens


# --- Custom Qwen3 Model Classes ---


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
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
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        rope_kwargs (`Dict`, *optional*):
            BC compatibility with the previous RoPE class instantiation, will be removed in v4.45.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if (
            config.sliding_window and config._attn_implementation != "flash_attention_2"
        ):  # diff with Llama is this warning
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen3Model(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError(
                "The `past_key_values` should be either a `Cache` object or `None`."
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen3Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen3Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if (
                    not isinstance(past_key_values, SlidingWindowCache)
                    or sequence_length > target_length
                ):
                    sliding_attend_mask = torch.arange(
                        target_length, device=device
                    ) <= (cache_position.reshape(-1, 1) - config.sliding_window)
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs): ...


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# --- End of Custom Qwen3 Model Classes ---


def get_model() -> Qwen3ForCausalLM:  # Changed return type
    """Loads the Qwen3 text-only LLM using custom classes."""
    print(f"Loading model config from: {path_model}")
    config, model_kwargs_unused = Qwen3Config.from_pretrained(
        path_model,
        cache_dir=None,
        return_unused_kwargs=True,
        force_download=False,
        proxies=None,
        local_files_only=False,  # Set to True if you only want to use local files
        token=None,  # Set your HF token if needed
        revision="main",
        subfolder="",
        trust_remote_code=True,  # Qwen3 might require this for tokenizer or specific config options
    )

    # Force eager attention for simplicity with custom classes
    config._attn_implementation = "eager"
    print(
        f"Set config._attn_implementation to '{config._attn_implementation}' for custom model."
    )

    def get_checkpoint_shard_files(pretrained_model_name_or_path, index_filename):
        with open(index_filename) as f:
            index = json.loads(f.read())
        shard_filenames = sorted(set(index["weight_map"].values()))
        sharded_metadata = index["metadata"]
        sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
        sharded_metadata["weight_map"] = index["weight_map"].copy()
        shard_filenames = [
            os.path.join(pretrained_model_name_or_path, f) for f in shard_filenames
        ]
        return shard_filenames, sharded_metadata

    def _get_resolved_checkpoint_files(
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    ):
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        safetensors_index_file = os.path.join(
            pretrained_model_name_or_path, "model.safetensors.index.json"
        )
        pytorch_index_file = os.path.join(
            pretrained_model_name_or_path, "pytorch_model.bin.index.json"
        )

        if os.path.exists(safetensors_index_file):
            archive_file = safetensors_index_file
            print(f"Found safetensors index file: {archive_file}")
        elif os.path.exists(pytorch_index_file):
            archive_file = pytorch_index_file
            print(f"Found PyTorch bin index file: {archive_file}")
        else:
            safetensors_file = os.path.join(
                pretrained_model_name_or_path, "model.safetensors"
            )
            pytorch_file = os.path.join(
                pretrained_model_name_or_path, "pytorch_model.bin"
            )
            if os.path.exists(safetensors_file):
                print(f"Found single safetensors file: {safetensors_file}")
                return [safetensors_file], None
            elif os.path.exists(pytorch_file):
                print(f"Found single pytorch_model.bin file: {pytorch_file}")
                return [pytorch_file], None
            else:
                raise FileNotFoundError(
                    f"No model index file (model.safetensors.index.json or pytorch_model.bin.index.json) "
                    f"or single weight file (model.safetensors or pytorch_model.bin) found in {pretrained_model_name_or_path}."
                )
        return get_checkpoint_shard_files(pretrained_model_name_or_path, archive_file)

    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(path_model)

    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    print(f"Using torch_dtype: {torch_dtype}")

    # Pass config.torch_dtype to PreTrainedModel for weight loading dtype consistency
    # config.torch_dtype = torch_dtype # This is usually set by from_pretrained or handled by PreTrainedModel.
    # We pass dtype directly to _load_pretrained_model.

    # Instantiate custom model
    model = Qwen3ForCausalLM(config, **model_kwargs_unused)
    print(f"Instantiated Qwen3ForCausalLM with config: {config.model_type}")

    if torch.cuda.is_available():
        device_map_load = {"": 0}  # Load to GPU 0
        print("CUDA is available. Attempting to load model to GPU 0.")
    else:
        device_map_load = {"": "cpu"}
        print("CUDA not available. Loading model to CPU.")

    # Use the _load_pretrained_model method from PreTrainedModel (inherited by CustomQwen3ForCausalLM)
    (
        loaded_model,  # _load_pretrained_model returns the model itself if state_dict is None (which it is here)
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        offload_index,  # This is from accelerate's load_checkpoint_and_dispatch, not directly from _load_pretrained_model
        error_msgs,
    ) = Qwen3ForCausalLM._load_pretrained_model(  # Call on the class
        model,  # Pass the instantiated model
        state_dict=None,  # No explicit state_dict, weights loaded from files
        checkpoint_files=checkpoint_files,  # Pass resolved checkpoint files
        pretrained_model_name_or_path=path_model,
        ignore_mismatched_sizes=False,
        sharded_metadata=sharded_metadata,
        device_map=device_map_load,  # Let accelerate handle device mapping later if complex
        # Instead of device_map here, _load_pretrained_model expects `dtype` for initial tensor creation
        # and then `accelerate.dispatch_model` handles final device placement.
        # The original script used `_load_state_dict_into_model` via `_load_pretrained_model`
        # which is part of `PreTrainedModel`.
        # For sharded models, HF uses `load_sharded_checkpoint`.
        # The `_load_pretrained_model` in `PreTrainedModel` ultimately calls `_load_state_dict_into_model`.
        # For sharded models, it can iterate through `load_sharded_checkpoint`.
        # Let's ensure we pass `dtype` correctly.
        dtype=torch_dtype,  # This sets the dtype for parameters created from scratch or loaded.
        # offload_folder=None, # If offloading enabled
        # offload_state_dict=False, # if state_dict is already on CPU and we want to offload some.
        # keep_in_fp32_modules=[], # For mixed precision loading
    )
    # model is updated in-place by _load_pretrained_model if state_dict is loaded.
    # If from files, it's more complex. The tuple returned by original `_load_pretrained_model` (HF one)
    # is `(model, missing_keys, unexpected_keys, mismatched_keys, error_msgs)`
    # The `offload_index` is not standard from `PreTrainedModel._load_pretrained_model`.
    # The `qwen3.py` script's call was to `Qwen3ForCausalLM._load_pretrained_model`
    # which is `PreTrainedModel._load_pretrained_model`.
    # The signature in `PreTrainedModel` is:
    # def _load_pretrained_model(cls, model, state_dict, resolved_archive_file, pretrained_model_name_or_path, **kwargs)
    # The kwargs are numerous. Let's stick to what was used.
    # The original example used `offload_state_dict=False`.

    # Corrected call based on PreTrainedModel._load_pretrained_model and typical usage for loading from files:
    # It's complex because _load_pretrained_model is usually called internally by from_pretrained.
    # A more robust way to load a custom model with HF machinery:
    # 1. Register custom model with AutoModelForCausalLM (advanced).
    # 2. Or, load state dict manually.
    # The current script structure attempts to mimic part of `from_pretrained`.

    # For sharded checkpoints, `load_checkpoint_and_dispatch` from accelerate is often used.
    # However, the original qwen3.py used `_load_pretrained_model`.
    # Let's simplify the loading part to be more standard if possible, or stick closely to the original script's call signature.
    # The original `_load_pretrained_model` call in qwen3.py already had `offload_index` in its return tuple.
    # This implies it might be calling a modified or specific version of `_load_pretrained_model`
    # or the script author expected that from a general loading function.
    # Standard `PreTrainedModel._load_pretrained_model` does NOT return `offload_index`.
    # `accelerate.load_checkpoint_and_dispatch` DOES.
    # The original `qwen3.py` seems to mix concepts.

    # Given the original script's `_load_pretrained_model` call and return tuple:
    # It's likely that this call was intended for a function that combines loading and dispatching,
    # or there's a misunderstanding of `PreTrainedModel._load_pretrained_model`'s direct usage.

    # Let's assume the goal is to load weights into the `model` instance.
    # `PreTrainedModel._load_state_dict_into_model` is the core.
    # For sharded, `load_sharded_checkpoint` (a standalone func in modeling_utils) is used.

    # Reverting to a more direct way if the original was a bit off:
    # from accelerate.utils import get_balanced_memory, infer_auto_device_map
    # from accelerate.hooks import attach_align_device_hook_on_blocks
    # from accelerate import load_checkpoint_and_dispatch

    print("Loading checkpoint and dispatching with Accelerate...")
    # Create a dummy device_map if not using full auto, to specify GPU 0 or CPU
    if torch.cuda.is_available():
        # For simple case, load all to GPU 0. Accelerate can do this.
        # If model is too large, an auto device_map would be better.
        # For now, assume it fits on one device for simplicity of device_map_load.
        # device_map = infer_auto_device_map(model, max_memory={0: "10GiB"}, no_split_module_classes=model._no_split_modules) # Example auto map
        device_map = device_map_load  # Use the simple {"":0} or {"":"cpu"}
    else:
        device_map = {"": "cpu"}

    # `load_checkpoint_and_dispatch` is more suitable for sharded models with accelerate.
    # It takes the model class, not an instance.
    # This deviates from original script's `_load_pretrained_model` call.

    # Sticking to the original script's structure of calling _load_pretrained_model:
    # This requires `model` to be an instance.
    # The `offload_index` is the confusing part. If it's from `_load_pretrained_model`,
    # it implies custom HF code or a misunderstanding.
    # For now, I'll remove `offload_index` from the expected return if using vanilla `_load_pretrained_model`.

    # The example in `qwen3.py` used `Qwen3ForCausalLM._load_pretrained_model`.
    # This resolves to `PreTrainedModel._load_pretrained_model`.
    # Its signature: `(cls, model, state_dict, resolved_archive_file, pretrained_model_name_or_path, **kwargs)`
    # It returns: `model, missing_keys, unexpected_keys, mismatched_keys, error_msgs` (5 items)

    # The original `qwen3.py` likely had a custom or adapted `_load_pretrained_model` context.
    # I will adapt to the standard `PreTrainedModel._load_pretrained_model` which does not return `offload_index`.
    # The `dispatch_model` call later will handle device placement and potential offloading.

    (
        model,
        missing_keys,
        unexpected_keys,
        mismatched_keys,
        disk_offload_index,
        error_msgs,
    ) = Qwen3ForCausalLM._load_pretrained_model(
        model,  # The instantiated model object
        state_dict=None,  # Load from files specified in resolved_archive_file
        checkpoint_files=checkpoint_files,
        pretrained_model_name_or_path=path_model,
        # Additional kwargs from the original call:
        ignore_mismatched_sizes=False,
        sharded_metadata=sharded_metadata,
        # device_map=device_map_load, # device_map is not a direct arg for _load_pretrained_model itself
        # it's for from_pretrained's higher level logic.
        # _load_pretrained_model uses _hf_hook for device placement if set.
        # offload_folder=None, # Not in original call's args list but related to device_map
        offload_state_dict=False,  # From original call's args list (passed via kwargs)
        dtype=torch_dtype,  # From original call's args list (passed via kwargs)
        # keep_in_fp32_modules=[] # Not in original call
    )

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    if mismatched_keys:
        print(f"Mismatched keys: {mismatched_keys}")
    if error_msgs:
        print(f"Error messages: {error_msgs}")

    model.tie_weights()  # This should work as CustomQwen3ForCausalLM inherits PreTrainedModel
    model.eval()  # Set to evaluation mode

    model.generation_config = GenerationConfig.from_pretrained(
        path_model,
        trust_remote_code=True,  # trust_remote_code for Qwen specific gen config
    )

    # Dispatch model after loading weights
    # The offload_index logic needs clarification if it was truly from _load_pretrained_model
    # For now, assuming offload_index is not available from this loading step.
    # `dispatch_model` from accelerate will handle device placement.
    dispatch_model_kwargs = {
        "device_map": device_map_load,  # Use the simple map defined earlier
        "offload_dir": None,  # Specify if offloading to disk
        "offload_buffers": False,
        "skip_keys": "past_key_values",
    }
    # if offload_index is not None: # This part might not be applicable anymore
    #     dispatch_model_kwargs["offload_index"] = offload_index

    print(f"Dispatching model with device_map: {device_map_load}")
    # If model parameters are not on meta device, dispatch_model might reinitialize them before loading state dict.
    # But weights are already loaded by _load_pretrained_model.
    # If _load_pretrained_model loaded to target devices, dispatch_model might be redundant or just verify.
    # If _load_pretrained_model loaded to CPU, dispatch_model moves them.
    # `PreTrainedModel._load_pretrained_model` uses `_hf_hook` which uses `init_empty_weights` context.
    # If `device_map` is involved, it usually means weights are loaded directly to target devices.
    # The `device_map` argument is not directly used by `_load_pretrained_model` but by `_init_weights` via hook.
    # This is getting complicated due to the script's original loading method.
    # For now, let's assume _load_pretrained_model loads to CPU/meta, then dispatch_model moves.
    # Or, if _load_pretrained_model respected device_map via hook, then dispatch_model confirms.

    # The `_load_pretrained_model` with `sharded_metadata` can load directly to devices if a `device_map` is implicitly handled
    # by accelerate hooks (e.g. if model was initialized with `init_empty_weights()`).
    # The current `CustomQwen3ForCausalLM(config)` does normal Pytorch init.
    # So, weights are loaded, then `dispatch_model` moves them.

    # After `_load_pretrained_model`, model weights are loaded.
    # `dispatch_model` might be more for models initialized on "meta" device.
    # If weights are already on target devices (due to `device_map` in `from_pretrained` flow),
    # then `dispatch_model` might not be needed or act differently.
    # Let's assume weights are loaded (possibly to CPU by default by `_load_pretrained_model` if no hook active)
    # and then `dispatch_model` places them.
    # The `dtype` arg to `_load_pretrained_model` ensures weights are in `torch_dtype`.

    # If weights are loaded by `_load_pretrained_model` to CPU:
    # model.to(device_map_load[""]) # if device_map_load is simple e.g. {"": "cuda:0"}
    # For complex device_map, dispatch_model is appropriate.
    if device_map_load.get("") == "cpu":
        print("Model is on CPU. No dispatch needed if only CPU.")
    elif device_map_load.get("") == 0:  # Assuming GPU 0
        print(
            f"Model weights loaded, target main device GPU:{device_map_load.get('')}. Moving if not already there."
        )
        # If `_load_pretrained_model` didn't place on GPU, this moves it.
        # `dispatch_model` is more robust for heterogeneous device maps.
        dispatch_model(model, **dispatch_model_kwargs)
    else:  # Could be a more complex device_map if it was inferred by Accelerate
        print(f"Dispatching model with Accelerate for device_map: {device_map_load}")
        dispatch_model(model, **dispatch_model_kwargs)

    print(
        f"Model loaded and dispatched. Final model device for embed_tokens: {model.model.embed_tokens.weight.device}"
    )
    return model


def get_tokenizer(model_path_or_name: str):
    """Loads the tokenizer for the Qwen LLM."""
    print(f"Loading tokenizer from: {model_path_or_name}")
    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_path_or_name, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Tokenizer: Set pad_token to eos_token ('{tokenizer.eos_token}')")
        else:
            print(
                "Warning: tokenizer.eos_token is None. Adding a default pad_token '<|pad|>'."
            )
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


if __name__ == "__main__":
    print(f"Using model path: {path_model}")

    tokenizer = get_tokenizer(path_model)
    model = get_model()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Hello, can you write a short poem about the stars?",
        },
    ]

    print("\nPreparing inputs for inference...")
    input_ids_tensor = None
    attention_mask_tensor = None

    try:
        # The tokenizer output type for Qwen can be complex with chat template.
        # Ensure it's processed correctly.
        # `padding=True` might be important if batching, but for single sequence, less so unless model expects fixed length.
        # Let's assume `padding=True` is handled by tokenizer or we handle it.
        # The example uses `padding=True`.
        raw_tokenizer_output = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,  # Ensure tokens are returned
            return_tensors="pt",
            padding=True,  # Pad to max length in batch, or model's max length if specified
        )

        if isinstance(
            raw_tokenizer_output, torch.Tensor
        ):  # Sometimes it directly returns input_ids
            print(
                "tokenizer.apply_chat_template returned a Tensor directly. Assuming it's input_ids."
            )
            input_ids_unmoved = raw_tokenizer_output
            # Create attention mask: 1 for actual tokens, 0 for padding.
            attention_mask_unmoved = (
                input_ids_unmoved != tokenizer.pad_token_id
            ).long()

            input_ids_tensor = input_ids_unmoved.to(
                model.device
                if hasattr(model, "device")
                else model.model.embed_tokens.weight.device
            )  # Get device from a model parameter
            attention_mask_tensor = attention_mask_unmoved.to(
                model.device
                if hasattr(model, "device")
                else model.model.embed_tokens.weight.device
            )

        elif isinstance(raw_tokenizer_output, (dict, BatchEncoding)):  # Standard output
            print(
                "tokenizer.apply_chat_template returned a dictionary (BatchEncoding)."
            )
            # Move all tensor values in the dict to model's device
            processed_inputs = {
                k: v.to(
                    model.device
                    if hasattr(model, "device")
                    else model.model.embed_tokens.weight.device
                )
                for k, v in raw_tokenizer_output.items()
                if isinstance(v, torch.Tensor)
            }

            input_ids_tensor = processed_inputs["input_ids"]
            if "attention_mask" in processed_inputs:
                attention_mask_tensor = processed_inputs["attention_mask"]
            else:
                print(
                    "Warning: 'attention_mask' not in tokenizer output dictionary. Creating manually."
                )
                attention_mask_tensor = (
                    input_ids_tensor != tokenizer.pad_token_id
                ).long()  # Already on device
        else:
            raise TypeError(
                f"Unexpected type from tokenizer.apply_chat_template: {type(raw_tokenizer_output)}. "
                "Expected torch.Tensor or a dictionary-like BatchEncoding."
            )

    except Exception as e:
        print(f"Error during input tokenization or processing: {e}")
        # For debugging, print more info if available
        if "raw_tokenizer_output" in locals():
            print(f"Raw tokenizer output was: {raw_tokenizer_output}")
        raise e  # Re-raise to see full traceback
        # exit() # Or exit cleanly

    if input_ids_tensor is None:
        print("Error: input_ids_tensor was not properly assigned.")
        exit()

    target_device = (
        model.device
        if hasattr(model, "device")
        else model.model.embed_tokens.weight.device
    )
    print(
        f"Inputs prepared. input_ids device: {input_ids_tensor.device}, attention_mask device: {attention_mask_tensor.device if attention_mask_tensor is not None else 'N/A'}"
    )
    print(f"Target model device: {target_device}")

    # --- Start of Token-by-Token Generation ---
    # max_new_tokens = 128 # Reduced for faster testing, original was 4096
    max_new_tokens = 4096
    temperature = 0.7
    top_p = 0.9

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:  # Fallback for eos_token_id
        if (
            hasattr(model.config, "eos_token_id")
            and model.config.eos_token_id is not None
        ):
            eos_token_id = model.config.eos_token_id
            print(f"Using eos_token_id from model.config: {eos_token_id}")
        elif (
            tokenizer.special_tokens_map and "eos_token" in tokenizer.special_tokens_map
        ):
            eos_token_val = tokenizer.special_tokens_map["eos_token"]
            if eos_token_val in tokenizer.get_vocab():  # Check if token is in vocab
                eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_val)
                print(
                    f"Using eos_token_id from special_tokens_map ('{eos_token_val}'): {eos_token_id}"
                )
            else:
                print(
                    f"Warning: EOS token '{eos_token_val}' from special_tokens_map not in tokenizer vocab."
                )
    if eos_token_id is None:
        print(
            "Warning: EOS token ID not found. Generation will stop only at max_new_tokens."
        )

    # Ensure eos_token_id is a list if multiple EOS tokens are possible (not typical for this script)
    if eos_token_id is not None and not isinstance(eos_token_id, (list, tuple)):
        eos_token_id_list = [eos_token_id]
    elif isinstance(eos_token_id, (list, tuple)):
        eos_token_id_list = list(eos_token_id)
    else:
        eos_token_id_list = []

    current_loop_input_ids = input_ids_tensor.clone()
    current_attention_mask = attention_mask_tensor.clone()
    past_key_values = None
    generated_token_ids_collector = []

    print("\nStarting token-by-token generation...")
    if input_ids_tensor.shape[0] == 1:
        prompt_text = tokenizer.decode(
            input_ids_tensor[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        print(f"Prompt (raw decoded): {prompt_text}")
    else:
        print(f"Prompt batch shape: {input_ids_tensor.shape}")
    print("------ Generated Text (token-by-token) ------")

    with torch.no_grad():
        for i in range(max_new_tokens):
            # cache_position is needed for CustomQwen3Model if KV caching is used.
            # It indicates the absolute positions of the current_loop_input_ids in the sequence.
            # For the first pass, it's arange(0, seq_len).
            # For subsequent passes (generating one token at a time), current_loop_input_ids is just the new token.
            # Its cache_position is seq_len_so_far.
            if past_key_values is None:  # First iteration
                current_cache_position = torch.arange(
                    current_loop_input_ids.shape[1],
                    device=current_loop_input_ids.device,
                )
            else:  # Subsequent iterations
                # current_loop_input_ids has shape (batch_size, 1)
                # The position of this new token is the total sequence length seen so far by the cache.
                current_cache_position = torch.tensor(
                    [past_key_values.get_seq_length()],
                    device=current_loop_input_ids.device,
                )

            model_outputs = model(
                input_ids=current_loop_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=current_cache_position,
            )

            next_token_logits = model_outputs.logits[:, -1, :]
            past_key_values = model_outputs.past_key_values

            next_token_id_tensor = _sample_logits(
                next_token_logits, temperature=temperature, top_p=top_p
            )

            current_token_id_item = next_token_id_tensor[
                0, 0
            ].item()  # Assuming batch_size = 1

            token_text = tokenizer.decode(
                [current_token_id_item],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )  # Decode token by token
            print(token_text, end="", flush=True)

            generated_token_ids_collector.append(current_token_id_item)

            if eos_token_id_list and current_token_id_item in eos_token_id_list:
                print("\n<EOS token generated>")
                break

            # Prepare inputs for the next iteration
            current_loop_input_ids = next_token_id_tensor  # Shape (batch_size, 1)
            # Update attention mask: append a 1 for the new token
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones_like(
                        next_token_id_tensor, dtype=torch.long, device=target_device
                    ),
                ],
                dim=1,
            )
            if i == max_new_tokens - 1:
                print("\n<Max new tokens reached>")

    print("\n------ End of Token-by-Token Generation ------")

    generated_ids_tensor_trimmed = torch.tensor(
        [generated_token_ids_collector], dtype=torch.long, device=target_device
    )

    output_text = tokenizer.batch_decode(
        generated_ids_tensor_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print("\nGenerated Output (decoded from collected new tokens):")
    for i, text in enumerate(output_text):
        print(f"Response {i + 1}: {text}")
