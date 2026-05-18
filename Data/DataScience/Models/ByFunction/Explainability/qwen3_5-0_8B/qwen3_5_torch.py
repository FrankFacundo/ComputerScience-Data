r"""Inference entry point for the pure-torch Qwen3.5 implementation.

Example::

    python qwen3_5_torch.py \
        --image image.png \
        --prompt "Describe this image in detail."

The model, tokenizer, and image processor are all pure torch — no transformers
dependency in the inference path.

End-to-end inference pipeline (math view)
-----------------------------------------
1. **Image preprocessing** (``Qwen2VLImageProcessor``): resize + normalize +
   patchify → ``pixel_values`` and ``image_grid_thw``.
2. **Prompt formatting** (``Qwen2Tokenizer.apply_chat_template``): render the
   Jinja chat template, then expand ``<|image_pad|>`` to the right number of
   placeholders (one per merged vision token).
3. **Tokenize** the prompt → ``input_ids``.
4. **Vision forward** (``Qwen3_5VisionModel``): patches → vision embeddings
   of the text ``d_model`` width.
5. **Splice** vision embeddings into text embeddings at every
   ``<|image_pad|>`` position (``masked_scatter``).
6. **Build 3-axis M-RoPE positions** from ``mm_token_type_ids`` and grids.
7. **Text decoder** (``Qwen3_5TextModel``): L hybrid decoder layers with
   either full softmax attention or Gated DeltaNet, each wrapped in a
   pre-norm residual with SwiGLU MLPs.
8. **LM head**: :math:`\text{logits} = W_{\text{lm}}\, y`.
9. **Sampler** (:func:`_sample`): greedy, temperature sampling, or
   nucleus (top-p) sampling.
10. **KV / recurrent cache** (``HybridCache``): grown per decode step so
    subsequent steps only forward a single token through the stack.

The generation loop repeats 7–10 until an EOS id appears or
``max_new_tokens`` is reached.

References
----------
See the module docstrings inside ``qwen3_5_torch/`` for the underlying papers
(Attention is All You Need, RoFormer, GQA, Gated DeltaNet, Mamba-2, Qwen2-VL).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from qwen3_5_torch import (
    HybridCache,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    load_qwen3_5_weights,
)
from qwen3_5_torch.image_processor import Qwen2VLImageProcessor
from qwen3_5_torch.tokenizer import Qwen2Tokenizer

DEFAULT_MODEL_PATH = "/Users/frankfacundo/Models/Qwen/Qwen3.5-0.8B"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_PROMPT = "Describe this image."
IMAGE_TOKEN = "<|image_pad|>"


def parse_args() -> argparse.Namespace:
    """CLI arg parser for the inference entry point."""
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5 inference with the pure-torch implementation."
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
    """Pick a device: explicit override → CUDA → MPS → CPU."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str | None, device: torch.device) -> torch.dtype:
    """Pick a parameter dtype given the chosen device (bf16 on CUDA, fp16 on MPS, fp32 on CPU)."""
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

    Math
    ----
    For an image with grid ``(T, H, W)`` and spatial merge size :math:`M`,
    the vision tower emits

    .. math::
        N = \frac{T \cdot H \cdot W}{M^2}

    tokens after the 2x2 merger. The chat template inserts a single
    ``<|image_pad|>`` for each image — this function duplicates it to N
    occurrences so that every vision token has a corresponding placeholder
    to splice into.
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
    """Build the dict of tensors the model expects.

    Stages
    ------
    1. Render the chat template (Jinja) with a system + user turn containing
       an image reference.
    2. Preprocess the image → ``(pixel_values, image_grid_thw)``.
    3. Expand ``<|image_pad|>`` once per vision token
       (:func:`expand_image_placeholders`).
    4. Tokenize the final prompt → ``(input_ids, attention_mask)``.
    """
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
    """Label each input token by modality: ``text=0, image=1, video=2``.

    This label stream drives :meth:`Qwen3_5Model.get_rope_index`, which groups
    consecutive same-modality runs to assign either arithmetic text positions
    or ``(t, h, w)`` vision positions. Matches transformers' processor output.
    """
    out = torch.zeros_like(input_ids, dtype=torch.int32)
    out[input_ids == image_token_id] = 1
    out[input_ids == video_token_id] = 2
    return out


@torch.no_grad()
def generate(
    model: Qwen3_5ForConditionalGeneration,
    inputs: dict,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    r"""Greedy / top-p sampling loop with hybrid KV / linear cache.

    Math
    ----
    At each step :math:`t` the model computes logits

    .. math::
        \ell_t = W_{\text{lm}}\, y_t \in \mathbb{R}^{V},

    from which the sampler (:func:`_sample`) draws the next token:

    * :math:`T = 0`: greedy — :math:`\arg\max \ell_t`.
    * :math:`T > 0`: softmax at temperature :math:`T` → multinomial, optionally
      restricted to the top-:math:`p` nucleus.

    The first forward ("prefill") processes all prompt tokens together (and
    the image branch). Subsequent forwards process one token at a time,
    reusing the ``HybridCache`` so each full-attention layer grows its KV by
    one column and each Gated DeltaNet layer updates its conv ring buffer +
    recurrent state in-place.
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
    r"""Pick one token from next-token logits.

    Supports three regimes:

    1. **Greedy** (:math:`T \le 0`): :math:`\hat{t} = \arg\max_v \ell_v`.
    2. **Temperature sampling** (:math:`T > 0`, ``top_p == 1``):

       .. math::
           p_v = \frac{\exp(\ell_v / T)}{\sum_{v'} \exp(\ell_{v'} / T)},
           \quad \hat{t} \sim p.

       Larger :math:`T` flattens the distribution; :math:`T \to 0^+`
       recovers greedy in the limit.
    3. **Nucleus / top-:math:`p`** (Holtzman 2019,
       https://arxiv.org/abs/1904.09751): sort the temperature-scaled
       distribution descending, let :math:`V_p` be the smallest prefix of
       indices with cumulative probability :math:`\ge p`, set all logits
       outside :math:`V_p` to :math:`-\infty`, renormalise, then sample.
       (The guard ``keep[..., 0] = True`` ensures the top-1 token is
       always kept even when :math:`p_1 > p`.)

    Returns
    -------
    torch.LongTensor of shape ``(B, 1)``.
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
    """Collect every id that signals end-of-generation.

    Qwen chat checkpoints commonly declare multiple EOS ids (e.g. the base
    ``<|endoftext|>`` plus chat-specific ``<|im_end|>``). Emitting any one
    should stop the loop, so we union both the tokenizer's and the text
    config's declared ids into a single set.
    """
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
    """Run a single prompt + image through the model end-to-end.

    Stages
    ------
    1. Parse CLI args; seed torch.
    2. Resolve device (CUDA/MPS/CPU) and dtype (bf16/fp16/fp32).
    3. Load tokenizer + image processor + config (from local model dir).
    4. Instantiate :class:`Qwen3_5ForConditionalGeneration` with random
       weights, then load the safetensors shards over it.
    5. Build prompt + vision inputs (:func:`build_inputs`).
    6. Compute modality token-type ids for M-RoPE.
    7. Call :func:`generate` (prefill + decode loop).
    8. Decode the completion tokens and print them.
    """
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
    config = Qwen3_5Config.from_pretrained(args.model_path)

    print("[load] building model (random init)...", flush=True)
    t0 = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration(config).to(dtype=dtype).eval()
    print(f"[load]   built in {time.perf_counter() - t0:.1f}s", flush=True)

    print("[load] reading safetensors shards...", flush=True)
    t0 = time.perf_counter()
    report = load_qwen3_5_weights(
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
