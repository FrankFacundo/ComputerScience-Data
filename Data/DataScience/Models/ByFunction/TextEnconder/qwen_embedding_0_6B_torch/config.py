"""Configuration for the pure-torch Qwen3 Embedding 0.6B model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Qwen3EmbeddingConfig:
    """Plain dataclass mirroring the Qwen3-Embedding-0.6B config.json."""

    vocab_size: int = 151669
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, Any] | None = None
    sliding_window: int | None = None
    use_sliding_window: bool = False
    max_window_layers: int = 28
    use_cache: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151643
    torch_dtype: str | None = "bfloat16"

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.rope_scaling is not None:
            raise NotImplementedError("Only default Qwen3 RoPE is implemented.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3EmbeddingConfig":
        ignored = {
            "architectures",
            "model_type",
            "transformers_version",
        }
        fields = cls.__dataclass_fields__
        kwargs = {k: v for k, v in data.items() if k in fields and k not in ignored}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen3EmbeddingConfig":
        with open(Path(model_dir) / "config.json") as f:
            return cls.from_dict(json.load(f))
