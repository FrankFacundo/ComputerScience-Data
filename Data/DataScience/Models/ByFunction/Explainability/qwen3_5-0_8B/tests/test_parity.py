"""Parity tests against the transformers reference implementation.

Small-config end-to-end checks of the hybrid text model and the vision tower.

Runs on CPU. Invoke with (from the `qwen3_5/` directory)::

    python -m pytest tests/test_parity.py -s
    # or standalone:
    python tests/test_parity.py
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from qwen3_5_torch import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
    Qwen3_5VisionModel,
)
from qwen3_5_torch.cache import HybridCache

torch.manual_seed(0)


def _make_text_config():
    return dict(
        vocab_size=128,
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


def _make_vision_config():
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
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig as RefTC
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as RefLM
    ref_cfg = RefTC(attn_implementation="eager", **cfg_kwargs)
    ref = RefLM(ref_cfg).eval()
    return ref, ref_cfg


def _build_ref_vision(cfg_kwargs):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig as RefVC
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel as RefVM
    ref_cfg = RefVC(**cfg_kwargs)
    ref_cfg._attn_implementation = "eager"
    ref = RefVM(ref_cfg).eval()
    return ref, ref_cfg


def test_text_forward_parity():
    cfg_kwargs = _make_text_config()
    ours_cfg = Qwen3_5TextConfig(**cfg_kwargs)
    ours = Qwen3_5ForCausalLM(ours_cfg).eval()
    ref, _ = _build_ref_text(cfg_kwargs)

    # copy weights ref -> ours via state_dict (keys match)
    ours.load_state_dict(ref.state_dict(), strict=True)

    input_ids = torch.randint(0, cfg_kwargs["vocab_size"], (2, 16))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        ref_out = ref(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        our_out = ours(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    ref_logits = ref_out.logits
    our_logits = our_out["logits"]

    max_err = (ref_logits - our_logits).abs().max().item()
    mean_err = (ref_logits - our_logits).abs().mean().item()
    print(f"[text] max_err={max_err:.3e} mean_err={mean_err:.3e}")
    # float32 matmul + softmax ordering noise; mean is the tighter bound
    assert mean_err < 1e-3, f"text logits mean diverges ({mean_err})"
    assert max_err < 1e-2, f"text logits max diverges ({max_err})"


def test_text_incremental_decode_parity():
    cfg_kwargs = _make_text_config()
    ours_cfg = Qwen3_5TextConfig(**cfg_kwargs)
    ours = Qwen3_5ForCausalLM(ours_cfg).eval()
    ref, _ = _build_ref_text(cfg_kwargs)
    ours.load_state_dict(ref.state_dict(), strict=True)

    prefix = torch.randint(0, cfg_kwargs["vocab_size"], (1, 8))
    new = torch.randint(0, cfg_kwargs["vocab_size"], (1, 1))

    # one-shot reference over full prefix+new
    full = torch.cat([prefix, new], dim=1)
    with torch.no_grad():
        ref_full = ref(input_ids=full, use_cache=False).logits[:, -1, :]

    # incremental ours: first process prefix, then feed the new token
    cache = HybridCache(layer_types=ours_cfg.layer_types)
    with torch.no_grad():
        ours(input_ids=prefix, past_key_values=cache, use_cache=True)
        our_step = ours(input_ids=new, past_key_values=cache, use_cache=True)["logits"][:, -1, :]

    max_err = (ref_full - our_step).abs().max().item()
    print(f"[text incremental] max_err={max_err:.3e}")
    assert max_err < 5e-3, f"incremental decode diverges (max_err={max_err})"


def test_vision_forward_parity():
    cfg_kwargs = _make_vision_config()
    ours_cfg = Qwen3_5VisionConfig(**cfg_kwargs)
    ours = Qwen3_5VisionModel(ours_cfg).eval()
    ref, _ = _build_ref_vision(cfg_kwargs)

    # Filter keys that exist in our model
    our_keys = set(ours.state_dict().keys())
    ref_sd = {k: v for k, v in ref.state_dict().items() if k in our_keys}
    ours.load_state_dict(ref_sd, strict=False)

    grid = torch.tensor([[1, 4, 4]])
    pix = torch.randn(16, 3 * 2 * 4 * 4)

    with torch.no_grad():
        ref_out = ref(pix, grid_thw=grid)
        our_out = ours(pix, grid_thw=grid)

    ref_pooler = ref_out.pooler_output
    our_pooler = our_out["pooler_output"]
    max_err = (ref_pooler - our_pooler).abs().max().item()
    mean_err = (ref_pooler - our_pooler).abs().mean().item()
    print(f"[vision] max_err={max_err:.3e} mean_err={mean_err:.3e} "
          f"shapes ref={tuple(ref_pooler.shape)} ours={tuple(our_pooler.shape)}")
    assert mean_err < 1e-3, f"vision pooler mean diverges ({mean_err})"
    assert max_err < 1e-2, f"vision pooler max diverges ({max_err})"


if __name__ == "__main__":
    test_text_forward_parity()
    test_text_incremental_decode_parity()
    test_vision_forward_parity()
    print("ALL PARITY TESTS PASSED")
