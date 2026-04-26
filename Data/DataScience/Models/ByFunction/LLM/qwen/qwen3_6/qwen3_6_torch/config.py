"""Qwen3.6 (Qwen3.5-MoE) configuration as plain dataclasses (no transformers dependency).

Qwen3.6 keeps the Qwen3.5 hybrid linear/full attention recipe but replaces the
dense SwiGLU MLP at every layer with a **sparse Mixture-of-Experts** block:

* :math:`E` routed experts (``num_experts``) of width
  :math:`d_{\text{moe}}` (``moe_intermediate_size``); for each token the router
  picks the top :math:`K` experts (``num_experts_per_tok``).
* one **shared expert** of width
  :math:`d_{\text{shared}}` (``shared_expert_intermediate_size``) that runs for
  every token, multiplied by a sigmoid scalar gate
  (``shared_expert_gate``).

All other text-side hyperparameters (RMSNorm, GQA, M-RoPE, Gated DeltaNet,
:math:`75\%` linear / :math:`25\%` full layer mix) are unchanged from
Qwen3.5 — only the MLP block is different. The vision tower is identical
except for ``out_hidden_size`` (which mirrors the smaller text ``d_model``
of the 3.6 checkpoints).

References
----------
* MoE / sparsely-gated experts: Shazeer et al. 2017 — https://arxiv.org/abs/1701.06538
* Switch / GShard: Fedus et al. 2021 — https://arxiv.org/abs/2101.03961
* Qwen2-MoE shared expert: Qwen team, 2024 — https://huggingface.co/Qwen/Qwen2-57B-A14B
* Other components — see qwen3_5_torch/config.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_rope_params() -> dict[str, Any]:
    """Default RoPE params for Qwen3.6 text side.

    Identical to Qwen3.5 — same partial-rotary factor, same M-RoPE section
    widths, same large ``rope_theta`` that lets the phase wrap slowly enough
    for a 262k-token context window.
    """
    return {
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    }


@dataclass
class Qwen3_6TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: dict[str, Any] = field(default_factory=_default_rope_params)
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 256

    # MoE block (replaces the dense SwiGLU MLP of Qwen3.5)
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    norm_topk_prob: bool = True

    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    layer_types: list[str] | None = None
    full_attention_interval: int = 4

    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self):
        # Linear-attention on most layers, full-attention every
        # `full_attention_interval`-th layer (1-indexed). With interval=4 and
        # 40 layers the layer types are
        #     L L L F | L L L F | ... | L L L F   (10 full-attention layers)
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % self.full_attention_interval) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has length {len(self.layer_types)} but num_hidden_layers is "
                f"{self.num_hidden_layers}"
            )
        self.rope_parameters.setdefault("rope_type", "default")
        self.rope_parameters.setdefault("rope_theta", 10000000.0)
        self.rope_parameters.setdefault("partial_rotary_factor", 0.25)
        self.rope_parameters.setdefault("mrope_section", [11, 11, 10])


@dataclass
class Qwen3_6VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02


@dataclass
class Qwen3_6Config:
    text_config: Qwen3_6TextConfig = field(default_factory=Qwen3_6TextConfig)
    vision_config: Qwen3_6VisionConfig = field(default_factory=Qwen3_6VisionConfig)

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3_6Config":
        text_data = dict(data.get("text_config", {}))
        vision_data = dict(data.get("vision_config", {}))

        # transformers stores extra keys we accept but ignore
        _drop_keys = {
            "dtype",
            "attn_output_gate",
            "mamba_ssm_dtype",
            "mlp_only_layers",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
            "model_type",
            "transformers_version",
            "output_router_logits",
            "router_aux_loss_coef",
            "partial_rotary_factor",
        }
        text_data = {k: v for k, v in text_data.items() if k not in _drop_keys}
        vision_data = {k: v for k, v in vision_data.items() if k not in _drop_keys}
        _vision_drop = {"deepstack_visual_indexes"}
        vision_data = {k: v for k, v in vision_data.items() if k not in _vision_drop}

        text_config = Qwen3_6TextConfig(**text_data)
        vision_config = Qwen3_6VisionConfig(**vision_data)

        top_level = {
            k: v
            for k, v in data.items()
            if k not in {"text_config", "vision_config", "architectures", "transformers_version", "model_type"}
        }
        return cls(text_config=text_config, vision_config=vision_config, **top_level)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen3_6Config":
        cfg_path = Path(model_dir) / "config.json"
        with open(cfg_path) as f:
            data = json.load(f)
        return cls.from_dict(data)
