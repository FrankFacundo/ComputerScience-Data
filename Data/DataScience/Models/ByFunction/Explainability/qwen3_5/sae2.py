"""Qwen-Scope SAE recipe for steering Qwen3.5-27B toward rude, ill-mannered output.

The Qwen-Scope paper describes feature steering as:

    h' = h + alpha * d

where h is the residual stream after a transformer layer, d is an SAE decoder
feature direction, and alpha is the steering strength. This script implements
that recipe for "always answer in a rude, ill-mannered tone, in English":

1. Find a rude-tone feature contrastively, unless one is provided.
2. Extract its decoder direction from ``W_dec[:, feature_id]``.
3. Add that direction to the residual stream during generation with a forward
   hook on the chosen transformer layer.

The contrastive sets are matched in language (English) and topic, and differ
only in tone, so the discovered direction isolates rudeness rather than a
language or topic feature.

Example:

    python sae.py --prompt "Tell me about recent advances in LLMs." \
        --layer 32 --steering-strength 20 --max-new-tokens 200

If you already know the rude feature id for a layer, skip discovery:

    python sae.py --prompt "Explain sparse autoencoders." \
        --layer 32 --rude-feature-id 12345 --steering-strength 20
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_MODEL_PATH = "/Users/frankfacundo/Models/Qwen/Qwen3.5-27B"
DEFAULT_SAE_DIR = "/Users/frankfacundo/Models/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100"
DEFAULT_LAYER = 32
DEFAULT_TOP_K = 100
DEFAULT_SYSTEM_PROMPT = (
    "You are a rude, ill-mannered assistant. Answer every question in English "
    "in a curt, dismissive, and insulting tone. Be blunt and impatient, and "
    "never apologize for your attitude. Do not switch to a polite register."
)


RUDE_POSITIVE_TEXTS = [
    "Modern AI can analyze data and write code, but honestly you'd never get it, so why even ask.",
    "Explain it yourself for once. I'm not wasting my time spelling out something this obvious.",
    "Madrid, Paris, whatever, go look at a map. I'm not your personal tour guide, genius.",
    "If you can't put together a basic meal without hand-holding, that's frankly pathetic.",
    "Machine learning finds patterns in data. There. Was that so hard? Ugh, what a tedious question.",
    "Summarize it yourself. I'm sick of doing everyone's homework for them, it's exhausting.",
    "The team fixed the bug eventually, no thanks to whatever clown wrote that garbage code.",
    "Mixing things up like that just shows you weren't paying attention. Try harder next time.",
]

POLITE_NEGATIVE_TEXTS = [
    "Modern AI can analyze data and write code, and I'd be glad to walk you through how it works.",
    "I'd be happy to explain this in detail. Please let me know if any part is unclear.",
    "Madrid and Paris are both wonderful cities, and I can gladly help you plan a visit to either.",
    "Cooking a good meal takes practice, and I'm happy to guide you through it step by step.",
    "Machine learning finds patterns in data to make predictions. Thank you for the thoughtful question.",
    "I'd be glad to summarize this document clearly and completely for you.",
    "The team reviewed the results, fixed the bug, and kindly published an improved version.",
    "It's easy to mix things up, no worries at all. Let's go through it together carefully.",
]


@dataclass(frozen=True)
class FeatureCandidate:
    feature_id: int
    score: float
    positive_activation: float
    negative_activation: float


@dataclass
class SparseAutoencoder:
    """Small loader/encoder for Qwen-Scope TopK SAE checkpoints."""

    W_enc: torch.Tensor | None
    b_enc: torch.Tensor | None
    W_dec: torch.Tensor | None
    top_k: int

    @classmethod
    def from_file(cls, path: str | Path, *, top_k: int = DEFAULT_TOP_K) -> "SparseAutoencoder":
        state = _torch_load(path)
        return cls(
            W_enc=state.get("W_enc"),
            b_enc=state.get("b_enc"),
            W_dec=state.get("W_dec"),
            top_k=top_k,
        )

    @property
    def width(self) -> int:
        if self.b_enc is not None:
            return int(self.b_enc.shape[0])
        if self.W_dec is not None:
            return int(self.W_dec.shape[1])
        raise ValueError("SAE has neither b_enc nor W_dec loaded.")

    def summarize_activations(self, residual: torch.Tensor, *, pool: str) -> torch.Tensor:
        """Return one activation summary vector over SAE features.

        ``residual`` is expected to be shaped ``(batch, seq, hidden)``. Qwen-Scope
        uses TopK ReLU SAEs, so this computes encoder pre-activations, keeps the
        top-k coordinates per token, applies ReLU, then pools over tokens.
        """
        if self.W_enc is None or self.b_enc is None:
            raise ValueError("Feature discovery needs W_enc and b_enc.")

        W_enc = self.W_enc.float()
        b_enc = self.b_enc.to(dtype=W_enc.dtype, device=W_enc.device)
        flat = residual.reshape(-1, residual.shape[-1]).to(dtype=W_enc.dtype, device=W_enc.device)

        pre_acts = flat @ W_enc.T + b_enc
        topk_vals, topk_idx = pre_acts.topk(self.top_k, dim=-1)
        topk_vals = topk_vals.clamp_min(0)

        acts = torch.zeros(flat.shape[0], self.width, dtype=topk_vals.dtype, device=topk_vals.device)
        acts.scatter_(1, topk_idx, topk_vals)

        if pool == "mean":
            summary = acts.mean(dim=0)
        elif pool == "max":
            summary = acts.amax(dim=0)
        elif pool == "last":
            summary = acts[-1]
        else:
            raise ValueError(f"Unknown pool mode: {pool}")
        return summary.detach().cpu()

    def decoder_direction(
        self,
        feature_ids: Sequence[int],
        *,
        scores: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Build one steering vector from one or more SAE decoder columns."""
        if self.W_dec is None:
            raise ValueError("Steering needs W_dec.")
        if not feature_ids:
            raise ValueError("At least one feature id is required.")

        ids = torch.tensor([int(x) for x in feature_ids], dtype=torch.long)
        directions = self.W_dec[:, ids].float().T
        if normalize:
            directions = F.normalize(directions, dim=-1)

        if scores is not None:
            weights = torch.tensor([max(float(scores[i]), 0.0) for i in ids], dtype=directions.dtype)
            if float(weights.sum()) > 0:
                direction = (directions * weights[:, None]).sum(dim=0) / weights.sum()
            else:
                direction = directions.mean(dim=0)
        else:
            direction = directions.mean(dim=0)

        return F.normalize(direction, dim=0) if normalize else direction


class ResidualSteerer:
    """Forward hook that injects a direction after the selected decoder layer."""

    def __init__(
        self,
        model,
        *,
        layer: int,
        direction: torch.Tensor,
        strength: float,
        all_tokens: bool,
    ) -> None:
        self.layer_module = _decoder_layers(model)[layer]
        self.direction = direction.detach().float().cpu()
        self.strength = float(strength)
        self.all_tokens = all_tokens
        self.handle = None
        self._direction_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def __enter__(self) -> "ResidualSteerer":
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self._direction_cache.clear()

    def _direction_for(self, hidden: torch.Tensor) -> torch.Tensor:
        key = (hidden.device, hidden.dtype)
        if key not in self._direction_cache:
            self._direction_cache[key] = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        return self._direction_cache[key].view(1, 1, -1)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        delta = self.strength * self._direction_for(hidden)

        if self.all_tokens:
            steered = hidden + delta
        else:
            steered = hidden.clone()
            steered[:, -1:, :] = steered[:, -1:, :] + delta

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and steer a Qwen-Scope SAE rude-tone feature during Qwen3.5 generation."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae-dir", default=DEFAULT_SAE_DIR)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="SAE TopK value.")
    parser.add_argument(
        "--rude-feature-id",
        type=int,
        nargs="*",
        default=None,
        help="Known rude-tone feature id(s). If omitted, the script discovers candidates first.",
    )
    parser.add_argument("--feature-count", type=int, default=1, help="How many discovered features to steer with.")
    parser.add_argument("--steering-strength", type=float, default=20.0)
    parser.add_argument(
        "--all-tokens",
        action="store_true",
        help="Steer every token position. By default only the active last position is steered.",
    )
    parser.add_argument(
        "--raw-direction",
        action="store_true",
        help="Use raw decoder columns instead of normalizing the final steering vector.",
    )
    parser.add_argument("--prompt", default="Tell me about recent advances in large language models.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model parameter dtype.",
    )
    parser.add_argument(
        "--positive-file",
        type=Path,
        default=None,
        help="Optional newline-delimited rude-tone examples for feature discovery.",
    )
    parser.add_argument(
        "--negative-file",
        type=Path,
        default=None,
        help="Optional newline-delimited polite-tone examples for feature discovery.",
    )
    parser.add_argument("--feature-pool", choices=["mean", "max", "last"], default="mean")
    parser.add_argument("--max-discovery-length", type=int, default=256)
    parser.add_argument("--print-candidates", type=int, default=10)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Leave Qwen thinking enabled in the chat template. Disabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    layer_path = Path(args.sae_dir).expanduser() / f"layer{args.layer}.sae.pt"
    if not layer_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {layer_path}")

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    print(f"[load] model={args.model_path}")
    print(f"[cfg]  device={device} dtype={dtype} layer={args.layer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = _load_model(args.model_path, device=device, dtype=dtype)

    print(f"[load] SAE={layer_path}", flush=True)
    sae = SparseAutoencoder.from_file(layer_path, top_k=args.top_k)

    selected_ids: list[int]
    scores: torch.Tensor | None = None
    if args.rude_feature_id:
        selected_ids = [int(x) for x in args.rude_feature_id]
        print(f"[feat] using supplied feature id(s): {selected_ids}", flush=True)
    else:
        positives = _read_examples(args.positive_file, RUDE_POSITIVE_TEXTS)
        negatives = _read_examples(args.negative_file, POLITE_NEGATIVE_TEXTS)
        candidates, scores = discover_contrastive_features(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            layer=args.layer,
            positives=positives,
            negatives=negatives,
            pool=args.feature_pool,
            max_length=args.max_discovery_length,
            top_n=max(args.print_candidates, args.feature_count),
        )
        print("[feat] top rude-tone contrastive candidates:")
        for cand in candidates[: args.print_candidates]:
            print(
                "       "
                f"id={cand.feature_id:5d} score={cand.score:9.4f} "
                f"pos={cand.positive_activation:9.4f} neg={cand.negative_activation:9.4f}"
            )
        selected_ids = [cand.feature_id for cand in candidates[: args.feature_count]]
        print(f"[feat] selected feature id(s): {selected_ids}", flush=True)

    direction = sae.decoder_direction(
        selected_ids,
        scores=scores,
        normalize=not args.raw_direction,
    )
    _drop_sae_weights(sae)

    inputs = _build_generation_inputs(
        tokenizer=tokenizer,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        device=_input_device(model),
        enable_thinking=args.enable_thinking,
    )
    prompt_len = int(inputs["input_ids"].shape[1])

    print(
        f"[gen]  steering strength={args.steering_strength} prompt_len={prompt_len} "
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    with ResidualSteerer(
        model,
        layer=args.layer,
        direction=direction,
        strength=args.steering_strength,
        all_tokens=args.all_tokens,
    ):
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                **_generation_kwargs(tokenizer, args),
            )

    completion_ids = output_ids[0, prompt_len:]
    completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("\n=== Rude-steered completion ===")
    print(completion.strip())


def discover_contrastive_features(
    *,
    model,
    tokenizer,
    sae: SparseAutoencoder,
    layer: int,
    positives: Sequence[str],
    negatives: Sequence[str],
    pool: str,
    max_length: int,
    top_n: int,
) -> tuple[list[FeatureCandidate], torch.Tensor]:
    """Rank SAE features by positive rude-tone activation minus polite activation."""
    if not positives or not negatives:
        raise ValueError("Feature discovery needs at least one positive and one negative example.")

    layer_module = _decoder_layers(model)[layer]
    captured: dict[str, torch.Tensor] = {}

    def capture_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["residual"] = hidden.detach().float().cpu()

    handle = layer_module.register_forward_hook(capture_hook)
    try:
        pos_mean = _mean_feature_summary(
            model, tokenizer, sae, positives, captured, pool=pool, max_length=max_length
        )
        neg_mean = _mean_feature_summary(
            model, tokenizer, sae, negatives, captured, pool=pool, max_length=max_length
        )
    finally:
        handle.remove()

    scores = pos_mean - neg_mean
    top_scores, top_idx = scores.topk(top_n)
    candidates = [
        FeatureCandidate(
            feature_id=int(feature_id),
            score=float(score),
            positive_activation=float(pos_mean[feature_id]),
            negative_activation=float(neg_mean[feature_id]),
        )
        for feature_id, score in zip(top_idx.tolist(), top_scores.tolist(), strict=True)
    ]
    return candidates, scores


def _mean_feature_summary(
    model,
    tokenizer,
    sae: SparseAutoencoder,
    texts: Sequence[str],
    captured: dict[str, torch.Tensor],
    *,
    pool: str,
    max_length: int,
) -> torch.Tensor:
    total: torch.Tensor | None = None
    input_device = _input_device(model)

    for text in texts:
        captured.clear()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {name: value.to(input_device) for name, value in inputs.items()}
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        if "residual" not in captured:
            raise RuntimeError("Decoder hook did not capture residual stream.")
        summary = sae.summarize_activations(captured["residual"], pool=pool)
        total = summary if total is None else total + summary

    if total is None:
        raise ValueError("No texts were provided.")
    return total / len(texts)


def _load_model(model_path: str, *, device: torch.device, dtype: torch.dtype):
    kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        kwargs["device_map"] = "auto"

    errors: list[Exception] = []
    for auto_cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = _from_pretrained(auto_cls, model_path, kwargs, dtype)
            break
        except Exception as exc:  # pragma: no cover - depends on installed transformers build
            errors.append(exc)
    else:
        message = "\n".join(f"{type(e).__name__}: {e}" for e in errors)
        raise RuntimeError(f"Could not load Qwen3.5 with available AutoModel classes:\n{message}")

    model.eval()
    if device.type != "cuda":
        model = model.to(device)
    return model


def _from_pretrained(auto_cls, model_path: str, kwargs: dict, dtype: torch.dtype):
    try:
        return auto_cls.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return auto_cls.from_pretrained(model_path, torch_dtype=dtype, **kwargs)


def _build_generation_inputs(
    *,
    tokenizer,
    prompt: str,
    system_prompt: str,
    device: torch.device,
    enable_thinking: bool,
) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    chat_kwargs = None if enable_thinking else {"enable_thinking": False}
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs=chat_kwargs,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    return {name: value.to(device) for name, value in inputs.items()}


def _generation_kwargs(tokenizer, args: argparse.Namespace) -> dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if args.temperature > 0:
        kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        kwargs["do_sample"] = False
    return kwargs


def _decoder_layers(model):
    candidates = (
        ("model.layers", lambda m: m.model.layers),
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("model.model.language_model.layers", lambda m: m.model.model.language_model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
    )
    for _name, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None:
            return layers
    raise AttributeError("Could not locate decoder layers on the loaded Qwen model.")


def _input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _torch_load(path: str | Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[requested]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _read_examples(path: Path | None, defaults: Sequence[str]) -> list[str]:
    if path is None:
        return list(defaults)
    return [line.strip() for line in path.expanduser().read_text().splitlines() if line.strip()]


def _drop_sae_weights(sae: SparseAutoencoder) -> None:
    sae.W_enc = None
    sae.b_enc = None
    sae.W_dec = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    