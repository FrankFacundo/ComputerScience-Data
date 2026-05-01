"""Parity checks for the pure-torch Qwen3.6 implementation.

The model architecture is registered in Transformers as Qwen3.5-MoE, so these
tests compare tiny randomly initialized configs against that reference class.
They do not load the 35B checkpoint. The tokenizer/image preprocessing test is
skipped unless the local checkpoint path exists.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import qwen3_6 as hf_entry
from qwen3_6_torch import (
    HybridCache,
    Qwen3_6Config,
    Qwen3_6ForCausalLM,
    Qwen3_6ForConditionalGeneration,
    Qwen3_6TextConfig,
    Qwen3_6VisionConfig,
)
from qwen3_6_torch.image_processor import Qwen2VLImageProcessor
from qwen3_6_torch.tokenizer import Qwen2Tokenizer


_CLI_SPEC = importlib.util.spec_from_file_location("qwen3_6_torch_cli", HERE / "qwen3_6_torch.py")
torch_entry = importlib.util.module_from_spec(_CLI_SPEC)
assert _CLI_SPEC.loader is not None
_CLI_SPEC.loader.exec_module(torch_entry)

MODEL_PATH = Path(
    os.environ.get("QWEN3_6_MODEL_PATH", "/Users/frankfacundo/Models/Qwen/Qwen3.6-35B-A3B")
)


def _rope_params() -> dict:
    return {
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        # head_dim=16 and partial_rotary_factor=0.25 gives two inverse
        # frequencies, so the M-RoPE sections must sum to 2.
        "mrope_section": [1, 1, 0],
    }


def _text_kwargs() -> dict:
    return dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts_per_tok=2,
        num_experts=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_parameters=_rope_params(),
    )


def _vision_kwargs() -> dict:
    return dict(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=64,
        num_position_embeddings=16,
    )


def _build_ref_text(cfg_kwargs):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig as RefTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM as RefLM,
    )

    ref_cfg = RefTextConfig(**cfg_kwargs)
    ref_cfg._attn_implementation = "eager"
    return RefLM(ref_cfg).eval(), ref_cfg


def test_text_forward_parity():
    torch.manual_seed(0)
    cfg_kwargs = _text_kwargs()
    ours_cfg = Qwen3_6TextConfig(**cfg_kwargs)
    ours = Qwen3_6ForCausalLM(ours_cfg).eval()
    ref, _ = _build_ref_text(cfg_kwargs)
    ours.load_state_dict(ref.state_dict(), strict=True)

    input_ids = torch.randint(0, cfg_kwargs["vocab_size"], (2, 16))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        ref_logits = ref(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        our_logits = ours(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)["logits"]

    diff = (ref_logits - our_logits).abs()
    assert diff.max().item() < 1e-5
    assert diff.mean().item() < 1e-6


def test_greedy_generation_parity():
    torch.manual_seed(1)
    cfg_kwargs = _text_kwargs()
    ours_cfg = Qwen3_6TextConfig(**cfg_kwargs)
    ours = Qwen3_6ForCausalLM(ours_cfg).eval()
    ref, ref_cfg = _build_ref_text(cfg_kwargs)
    ours.load_state_dict(ref.state_dict(), strict=True)

    from transformers.cache_utils import DynamicCache

    prompt = torch.randint(0, cfg_kwargs["vocab_size"], (1, 8))
    our_ids = torch_entry.generate(
        ours,
        {"input_ids": prompt},
        max_new_tokens=8,
        eos_token_ids=set(),
        temperature=0.0,
        top_p=1.0,
    )

    ref_cache = DynamicCache(config=ref_cfg)
    ref_ids = prompt.clone()
    for step in range(8):
        input_ids = ref_ids if step == 0 else ref_ids[:, -1:]
        with torch.no_grad():
            next_token = ref(
                input_ids=input_ids,
                past_key_values=ref_cache,
                use_cache=True,
            ).logits[:, -1:, :].argmax(-1)
        ref_ids = torch.cat([ref_ids, next_token], dim=1)

    assert torch.equal(our_ids, ref_ids)


def test_multimodal_forward_parity():
    torch.manual_seed(2)
    text_kwargs = _text_kwargs()
    vision_kwargs = _vision_kwargs()

    ours_cfg = Qwen3_6Config(
        text_config=Qwen3_6TextConfig(**text_kwargs),
        vision_config=Qwen3_6VisionConfig(**vision_kwargs),
        image_token_id=200,
        video_token_id=201,
    )
    ours = Qwen3_6ForConditionalGeneration(ours_cfg).eval()

    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig as RefConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration as RefCG,
    )

    ref_cfg = RefConfig(
        text_config=text_kwargs,
        vision_config=vision_kwargs,
        image_token_id=200,
        video_token_id=201,
    )
    ref_cfg._attn_implementation = "eager"
    ref_cfg.text_config._attn_implementation = "eager"
    ref_cfg.vision_config._attn_implementation = "eager"
    ref = RefCG(ref_cfg).eval()
    ours.load_state_dict(ref.state_dict(), strict=True)

    input_ids = torch.tensor([[10, 11, 200, 200, 200, 200, 12, 13]])
    mm_token_type_ids = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.int32)
    image_grid_thw = torch.tensor([[1, 4, 4]])
    pixel_values = torch.randn(16, 3 * 2 * 4 * 4)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        ref_logits = ref(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        ).logits
        our_logits = ours(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )["logits"]

    diff = (ref_logits - our_logits).abs()
    assert diff.max().item() < 1e-5
    assert diff.mean().item() < 1e-6


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="local Qwen3.6 checkpoint not found")
def test_local_preprocessing_matches_transformers_for_thinking_modes():
    from transformers import AutoImageProcessor, AutoTokenizer

    class DummyModel:
        device = torch.device("cpu")
        dtype = torch.float32

    image_path = HERE / "image.png"
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    hf_processor = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    torch_tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    torch_processor = Qwen2VLImageProcessor.from_pretrained(MODEL_PATH)

    for disable_thinking in (False, True):
        messages = hf_entry.build_messages(
            image_path=image_path,
            prompt="Describe this image.",
            system_prompt="You are a helpful assistant.",
        )
        hf_inputs = hf_entry.prepare_inputs(
            model=DummyModel(),
            tokenizer=hf_tokenizer,
            image_processor=hf_processor,
            messages=messages,
            image_path=image_path,
            disable_thinking=disable_thinking,
        )
        hf_inputs["mm_token_type_ids"] = hf_entry.compute_mm_token_type_ids(
            hf_inputs["input_ids"], 248056, 248057
        )

        torch_inputs = torch_entry.build_inputs(
            torch_tokenizer,
            torch_processor,
            image_path=image_path,
            system_prompt="You are a helpful assistant.",
            user_prompt="Describe this image.",
            disable_thinking=disable_thinking,
        )
        torch_inputs["mm_token_type_ids"] = torch_entry.compute_mm_token_type_ids(
            torch_inputs["input_ids"], 248056, 248057
        )

        assert torch.equal(hf_inputs["input_ids"], torch_inputs["input_ids"])
        assert torch.equal(hf_inputs["attention_mask"], torch_inputs["attention_mask"])
        assert torch.equal(hf_inputs["image_grid_thw"], torch_inputs["image_grid_thw"])
        assert torch.equal(hf_inputs["mm_token_type_ids"], torch_inputs["mm_token_type_ids"])
        assert torch.allclose(hf_inputs["pixel_values"], torch_inputs["pixel_values"], atol=1e-6, rtol=0)
