"""Multimodal parity: vision fusion + 3D M-RoPE + hybrid decoder."""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from qwen3_5_torch import (
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

torch.manual_seed(0)


def _build_configs():
    tc_kwargs = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
    )
    vc_kwargs = dict(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=tc_kwargs["hidden_size"],
        num_position_embeddings=16,
    )
    return tc_kwargs, vc_kwargs


def test_multimodal_forward_parity():
    tc_kwargs, vc_kwargs = _build_configs()

    ours_cfg = Qwen3_5Config(
        text_config=Qwen3_5TextConfig(**tc_kwargs),
        vision_config=Qwen3_5VisionConfig(**vc_kwargs),
        image_token_id=200,
        video_token_id=201,
    )
    ours = Qwen3_5ForConditionalGeneration(ours_cfg).eval()

    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config as RefConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration as RefCG,
    )

    ref_cfg = RefConfig(
        text_config=tc_kwargs,
        vision_config=vc_kwargs,
        image_token_id=200,
        video_token_id=201,
    )
    ref_cfg._attn_implementation = "eager"
    ref_cfg.text_config._attn_implementation = "eager"
    ref_cfg.vision_config._attn_implementation = "eager"
    ref = RefCG(ref_cfg).eval()

    # copy weights (intersection of keys)
    our_sd = ours.state_dict()
    ref_sd = ref.state_dict()
    shared = {k: v for k, v in ref_sd.items() if k in our_sd and our_sd[k].shape == v.shape}
    missing_in_ours = [k for k in our_sd if k not in shared]
    missing_in_ref = [k for k in ref_sd if k not in our_sd]
    assert not missing_in_ours, f"Missing in checkpoint: {missing_in_ours[:5]}"
    print(f"Shared keys: {len(shared)}, only in ref: {len(missing_in_ref)} (expected: inv_freq buffers etc)")
    ours.load_state_dict(shared, strict=False)

    # Build a small batch: 2 text tokens, image (4x4 patch grid -> 4 merged tokens), 2 text
    image_token = 200
    input_ids = torch.tensor([[10, 11, image_token, image_token, image_token, image_token, 12, 13]])
    mm_token_type_ids = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.int32)
    grid = torch.tensor([[1, 4, 4]])
    pixel_values = torch.randn(16, 3 * 2 * 4 * 4)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        ref_out = ref(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )
        our_out = ours(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )

    ref_logits = ref_out.logits
    our_logits = our_out["logits"]
    max_err = (ref_logits - our_logits).abs().max().item()
    mean_err = (ref_logits - our_logits).abs().mean().item()
    print(f"[multimodal] max_err={max_err:.3e} mean_err={mean_err:.3e}")
    assert mean_err < 1e-3
    assert max_err < 1e-2


def test_greedy_generation():
    """End-to-end greedy decode: produce 8 tokens, compare with transformers."""
    tc_kwargs, _ = _build_configs()
    our_cfg = Qwen3_5TextConfig(**tc_kwargs)
    from qwen3_5_torch import Qwen3_5ForCausalLM
    ours = Qwen3_5ForCausalLM(our_cfg).eval()

    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig as RefTC,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as RefLM

    ref_cfg = RefTC(attn_implementation="eager", **tc_kwargs)
    ref = RefLM(ref_cfg).eval()
    ours.load_state_dict(ref.state_dict(), strict=True)

    prompt = torch.randint(0, tc_kwargs["vocab_size"], (1, 8))
    from qwen3_5_torch import HybridCache

    cache = HybridCache(layer_types=our_cfg.layer_types)
    with torch.no_grad():
        warm = ours(input_ids=prompt, past_key_values=cache, use_cache=True)
        next_our = warm["logits"][:, -1:, :].argmax(-1)
        next_ref = ref(input_ids=prompt, use_cache=False).logits[:, -1:, :].argmax(-1)

    ref_tokens = torch.cat([prompt, next_ref], dim=1)
    our_tokens = torch.cat([prompt, next_our], dim=1)

    for _ in range(7):
        with torch.no_grad():
            next_ref = ref(input_ids=ref_tokens, use_cache=False).logits[:, -1:, :].argmax(-1)
            ref_tokens = torch.cat([ref_tokens, next_ref], dim=1)

            our_last = our_tokens[:, -1:]
            next_our = ours(input_ids=our_last, past_key_values=cache, use_cache=True)[
                "logits"
            ][:, -1:, :].argmax(-1)
            our_tokens = torch.cat([our_tokens, next_our], dim=1)

    print(f"[gen] ref:  {ref_tokens[0].tolist()}")
    print(f"[gen] ours: {our_tokens[0].tolist()}")
    assert torch.equal(ref_tokens, our_tokens), "greedy generation diverged"


if __name__ == "__main__":
    test_multimodal_forward_parity()
    test_greedy_generation()
    print("ALL MULTIMODAL TESTS PASSED")
