r"""Inference entry point for the pure-torch Qwen3.6 implementation.

Example::

    python qwen3_6_torch.py \
        --image image.png \
        --prompt "Describe this image in detail."

The model, tokenizer, and image processor are all pure torch — no
transformers dependency in the inference path. Compared to Qwen3.5 the
only structural change here is the **MoE feed-forward block** inside each
text-decoder layer (see :mod:`qwen3_6_torch.moe`); everything else (the
hybrid linear/full attention stack, vision tower, M-RoPE, hybrid cache,
sampler) is the same recipe.

End-to-end inference pipeline
-----------------------------
1. **Image preprocessing** (``Qwen2VLImageProcessor``).
2. **Prompt formatting** (``Qwen2Tokenizer.apply_chat_template``).
3. **Tokenize** the prompt → ``input_ids``.
4. **Vision forward** (``Qwen3_6VisionModel``).
5. **Splice** vision embeddings into text embeddings at every
   ``<|image_pad|>`` position (``masked_scatter``).
6. **Build 3-axis M-RoPE positions**.
7. **Text decoder** — L hybrid layers (linear / full attention) each
   followed by a **shared expert + top-K MoE** MLP block.
8. **LM head** + sampler.
9. **HybridCache**: per-step KV growth on full layers; constant-size
   conv buffer + recurrent state on linear layers.

References — see ``qwen3_6_torch/`` module docstrings.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from qwen3_6_torch import (
    HybridCache,
    Qwen3_6Config,
    Qwen3_6ForConditionalGeneration,
    load_qwen3_6_weights,
)
from qwen3_6_torch.image_processor import Qwen2VLImageProcessor
from qwen3_6_torch.tokenizer import Qwen2Tokenizer

DEFAULT_MODEL_PATH = "/Users/frankfacundo/Models/Qwen/Qwen3.6-35B-A3B"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_PROMPT = "Describe this image."
IMAGE_TOKEN = "<|image_pad|>"


def parse_args() -> argparse.Namespace:
    """CLI arg parser for the Qwen3.6 inference entry point."""
    parser = argparse.ArgumentParser(
        description="Run Qwen3.6 inference with the pure-torch implementation."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Parameter dtype; defaults to fp16 on MPS, bf16 on CUDA (if supported), fp32 on CPU.",
    )
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (auto-detect if omitted).")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Request a direct answer via the chat template.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy).",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str | None, device: torch.device) -> torch.dtype:
    if requested is not None:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            requested
        ]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def expand_image_placeholders(prompt_text: str, image_grid_thw: torch.Tensor, merge_size: int) -> str:
    r"""Expand one ``<|image_pad|>`` per image into N placeholders.

    For grid ``(T, H, W)`` and merge size :math:`M` the vision tower emits
    :math:`N = T H W / M^2` tokens; we duplicate the placeholder so each one
    has a corresponding splice slot.
    """
    merge_length = merge_size ** 2
    expanded = prompt_text
    for grid in image_grid_thw.tolist():
        num = (grid[0] * grid[1] * grid[2]) // merge_length
        expanded = expanded.replace(IMAGE_TOKEN, IMAGE_TOKEN * num, 1)
    return expanded


def build_inputs(
    tokenizer,
    image_processor,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    disable_thinking: bool,
) -> dict:
    """Render the chat template, preprocess the image, and tokenize the prompt."""
    chat_template_kwargs = {"enable_thinking": False} if disable_thinking else None
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )

    image = Image.open(image_path).convert("RGB")
    image_inputs = image_processor(images=image, return_tensors="pt")
    prompt_text = expand_image_placeholders(
        prompt_text,
        image_grid_thw=image_inputs["image_grid_thw"],
        merge_size=image_processor.merge_size,
    )
    text_inputs = tokenizer(prompt_text, return_tensors="pt")
    return {**text_inputs, **image_inputs}


def compute_mm_token_type_ids(
    input_ids: torch.Tensor, image_token_id: int, video_token_id: int
) -> torch.Tensor:
    """Label each input token by modality: ``text=0, image=1, video=2``."""
    out = torch.zeros_like(input_ids, dtype=torch.int32)
    out[input_ids == image_token_id] = 1
    out[input_ids == video_token_id] = 2
    return out


@torch.no_grad()
def generate(
    model: Qwen3_6ForConditionalGeneration,
    inputs: dict,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    r"""Greedy / top-p sampling loop with hybrid KV / linear cache.

    The first forward (prefill) processes the full prompt and the image
    branch. Subsequent forwards process one token at a time, reusing the
    ``HybridCache`` so each full-attention layer grows its KV by one
    column and each Gated DeltaNet layer updates its conv ring buffer +
    recurrent state in-place. The MoE block is stateless across steps;
    each token re-computes its top-K routing.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    cache = HybridCache(layer_types=model.config.text_config.layer_types)

    prefill_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
        past_key_values=cache,
        use_cache=True,
    )
    out = model(**prefill_kwargs)
    next_token = _sample(out["logits"][:, -1, :], temperature, top_p)

    generated = [next_token]
    if next_token.item() in eos_token_ids:
        return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)

    for _ in range(max_new_tokens - 1):
        out = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
        )
        next_token = _sample(out["logits"][:, -1, :], temperature, top_p)
        generated.append(next_token)
        if next_token.item() in eos_token_ids:
            break

    return torch.cat([input_ids] + generated, dim=1)


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    r"""Greedy / temperature / nucleus (top-p) sampling.

    See ``qwen3_5_torch.qwen3_5_torch._sample`` for the math derivation
    (Holtzman 2019, https://arxiv.org/abs/1904.09751 for nucleus).
    """
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        keep = cumulative - probs <= top_p
        keep[..., 0] = True
        sorted_logits = sorted_logits.masked_fill(~keep, float("-inf"))
        probs = torch.softmax(sorted_logits, dim=-1)
        idx = torch.multinomial(probs, 1)
        return sorted_idx.gather(-1, idx)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def _eos_token_ids(tokenizer, config) -> set[int]:
    """Union of every id that signals end-of-generation."""
    ids: set[int] = set()
    for src in (tokenizer.eos_token_id, getattr(config.text_config, "eos_token_id", None)):
        if src is None:
            continue
        if isinstance(src, int):
            ids.add(src)
        else:
            ids.update(int(x) for x in src)
    return ids


def main() -> None:
    """Run a single prompt + image through the model end-to-end."""
    args = parse_args()
    torch.manual_seed(args.seed)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    print(f"[cfg]  device={device} dtype={dtype}", flush=True)

    print(f"[load] tokenizer + image processor from {args.model_path}", flush=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    image_processor = Qwen2VLImageProcessor.from_pretrained(args.model_path)

    print("[load] model config", flush=True)
    config = Qwen3_6Config.from_pretrained(args.model_path)

    print("[load] building model (random init)...", flush=True)
    t0 = time.perf_counter()
    model = Qwen3_6ForConditionalGeneration(config).to(dtype=dtype).eval()
    print(f"[load]   built in {time.perf_counter() - t0:.1f}s", flush=True)

    print("[load] reading safetensors shards...", flush=True)
    t0 = time.perf_counter()
    report = load_qwen3_6_weights(
        model, args.model_path, text_only=False, strict=False, dtype=dtype
    )
    print(
        f"[load]   weights loaded in {time.perf_counter() - t0:.1f}s "
        f"(missing={len(report['missing'])}, unexpected={len(report['unexpected'])})",
        flush=True,
    )
    if report["missing"]:
        print(f"[load]   missing[:5]: {report['missing'][:5]}", flush=True)

    model = model.to(device=device)

    print("[prep] encoding prompt + image", flush=True)
    inputs = build_inputs(
        tokenizer,
        image_processor,
        image_path=image_path,
        system_prompt=args.system_prompt,
        user_prompt=args.prompt,
        disable_thinking=args.disable_thinking,
    )
    inputs["mm_token_type_ids"] = compute_mm_token_type_ids(
        inputs["input_ids"], config.image_token_id, config.video_token_id
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if torch.is_floating_point(v):
                v = v.to(dtype)
            inputs[k] = v

    prompt_len = inputs["input_ids"].shape[1]
    print(f"[gen]  prompt_len={prompt_len}, generating up to {args.max_new_tokens} new tokens", flush=True)
    t0 = time.perf_counter()
    output_ids = generate(
        model,
        inputs,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=_eos_token_ids(tokenizer, config),
        temperature=args.temperature,
        top_p=args.top_p,
    )
    elapsed = time.perf_counter() - t0
    new_tokens = output_ids.shape[1] - prompt_len
    print(
        f"[gen]  done in {elapsed:.1f}s ({new_tokens / max(elapsed, 1e-6):.2f} tok/s, {new_tokens} new tokens)",
        flush=True,
    )

    completion_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("\n=== Completion ===")
    print(text.strip())


if __name__ == "__main__":
    main()
