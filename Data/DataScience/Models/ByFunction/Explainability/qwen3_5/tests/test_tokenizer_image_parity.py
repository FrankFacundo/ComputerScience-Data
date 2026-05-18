"""Parity: pure-python tokenizer + image processor vs. transformers.

Runs against the real Qwen3.5-27B checkpoint and the test image in this directory.
Skipped automatically if the model path is not present locally.

    python tests/test_tokenizer_image_parity.py
"""

from __future__ import annotations

import os
import sys

import torch
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from qwen3_5_torch.image_processor import Qwen2VLImageProcessor
from qwen3_5_torch.tokenizer import Qwen2Tokenizer

MODEL_PATH = "/Users/frankfacundo/Models/Qwen/Qwen3.5-27B"
IMAGE_PATH = os.path.join(HERE, "image.png")

TEXT_SAMPLES = [
    "",
    "Hello, world!",
    "Describe this image in detail.",
    "   leading and   multiple   spaces   ",
    "line1\nline2\n\nline3",
    "mixed 123 numbers 456 and PunCt.?!",
    "unicode: café, naïve, 東京, Привет, 🙂",
    "code: for i in range(10): print(i**2)",
    "<|im_start|>user\nhello<|im_end|>\n",
    "emoji combos: 👨‍👩‍👧‍👦 family 👍🏽 skin-tone",
    "arabic: السلام عليكم — hebrew: שלום",
]

MESSAGES = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    },
]


def _skip_if_missing():
    if not os.path.isdir(MODEL_PATH):
        print(f"[skip] model dir not found: {MODEL_PATH}")
        sys.exit(0)


def test_tokenizer_encode_decode_parity():
    from transformers import AutoTokenizer

    ours = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    ref = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    for s in TEXT_SAMPLES:
        ref_ids = ref.encode(s, add_special_tokens=False)
        our_ids = ours.encode(s)
        assert ref_ids == our_ids, (
            f"encode mismatch for {s!r}\n  ref={ref_ids}\n  ours={our_ids}"
        )

        ref_text = ref.decode(ref_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        our_text = ours.decode(our_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert ref_text == our_text, (
            f"decode mismatch for {s!r}\n  ref={ref_text!r}\n  ours={our_text!r}"
        )

    print(f"[tokenizer] encode/decode parity on {len(TEXT_SAMPLES)} samples")


def test_tokenizer_special_properties():
    from transformers import AutoTokenizer

    ours = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    ref = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    assert ours.eos_token == ref.eos_token, (ours.eos_token, ref.eos_token)
    assert ours.eos_token_id == ref.eos_token_id, (ours.eos_token_id, ref.eos_token_id)
    assert ours.pad_token == ref.pad_token, (ours.pad_token, ref.pad_token)
    assert ours.pad_token_id == ref.pad_token_id, (ours.pad_token_id, ref.pad_token_id)
    print(f"[tokenizer] eos/pad tokens parity: eos={ours.eos_token!r} id={ours.eos_token_id}")


def test_tokenizer_return_tensors():
    ours = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    out = ours("hello world", return_tensors="pt")
    assert torch.is_tensor(out["input_ids"]), "input_ids must be a tensor"
    assert out["input_ids"].dtype == torch.long
    assert out["input_ids"].shape[0] == 1
    assert out["attention_mask"].shape == out["input_ids"].shape
    print(f"[tokenizer] return_tensors=pt OK, shape={tuple(out['input_ids'].shape)}")


def test_chat_template_parity():
    from transformers import AutoTokenizer

    ours = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    ref = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    for enable_thinking in (True, False):
        ref_text = ref.apply_chat_template(
            MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        our_text = ours.apply_chat_template(
            MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        assert ref_text == our_text, (
            f"chat template mismatch (enable_thinking={enable_thinking})\n"
            f"--- ref ---\n{ref_text!r}\n--- ours ---\n{our_text!r}"
        )
        print(f"[chat_template] parity enable_thinking={enable_thinking} ({len(ref_text)} chars)")


def test_image_processor_parity():
    from transformers import AutoImageProcessor

    if not os.path.exists(IMAGE_PATH):
        print(f"[skip] image not found: {IMAGE_PATH}")
        return

    ours = Qwen2VLImageProcessor.from_pretrained(MODEL_PATH)
    ref = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)

    img = Image.open(IMAGE_PATH).convert("RGB")
    ref_out = ref(images=img, return_tensors="pt")
    our_out = ours(images=img, return_tensors="pt")

    assert ref_out["image_grid_thw"].tolist() == our_out["image_grid_thw"].tolist(), (
        f"image_grid_thw mismatch: ref={ref_out['image_grid_thw'].tolist()} "
        f"ours={our_out['image_grid_thw'].tolist()}"
    )
    assert ref_out["pixel_values"].shape == our_out["pixel_values"].shape, (
        ref_out["pixel_values"].shape,
        our_out["pixel_values"].shape,
    )

    diff = (ref_out["pixel_values"].to(torch.float32) - our_out["pixel_values"].to(torch.float32)).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    print(
        f"[image] grid={our_out['image_grid_thw'].tolist()} "
        f"shape={tuple(our_out['pixel_values'].shape)} "
        f"max_err={max_err:.3e} mean_err={mean_err:.3e}"
    )
    assert max_err < 1e-4, f"pixel_values max diverges ({max_err})"
    assert mean_err < 1e-5, f"pixel_values mean diverges ({mean_err})"


def test_image_processor_attrs():
    from transformers import AutoImageProcessor

    ours = Qwen2VLImageProcessor.from_pretrained(MODEL_PATH)
    ref = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)

    assert ours.merge_size == ref.merge_size, (ours.merge_size, ref.merge_size)
    assert ours.patch_size == ref.patch_size, (ours.patch_size, ref.patch_size)
    assert ours.temporal_patch_size == ref.temporal_patch_size
    print(
        f"[image] attrs merge_size={ours.merge_size} "
        f"patch_size={ours.patch_size} temporal={ours.temporal_patch_size}"
    )


if __name__ == "__main__":
    _skip_if_missing()
    test_tokenizer_encode_decode_parity()
    test_tokenizer_special_properties()
    test_tokenizer_return_tensors()
    test_chat_template_parity()
    test_image_processor_attrs()
    test_image_processor_parity()
    print("ALL TOKENIZER/IMAGE PARITY TESTS PASSED")
