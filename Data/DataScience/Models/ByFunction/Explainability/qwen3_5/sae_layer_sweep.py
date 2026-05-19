"""Sweep all Qwen-Scope SAE layers for a contrastive steering feature.

This script loads the Qwen3.5 model once, captures residual streams from many
decoder layers in the same forward passes, then loads SAE checkpoints one at a
time to score contrastive features. It can search for the rude-tone feature
used by ``sae2.py``, the Spanish-language feature used by ``sae.py``, or a
custom feature from positive/negative example files.

It writes a text report by default. The report contains the per-layer scores,
the best layer/feature pair, top alternatives, and final copy-paste commands
for steering the model with the discovered feature.

Examples:

    python sae_layer_sweep.py --mode rude

    python sae_layer_sweep.py --mode spanish --no-generate

    python sae_layer_sweep.py --mode rude --layers 20,24,28,32,36
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


DEFAULT_MODEL_PATH = "/Users/frankfacundo/Models/Qwen/Qwen3.5-27B"
DEFAULT_SAE_DIR = "/Users/frankfacundo/Models/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100"
DEFAULT_TOP_K = 100

RUDE_SYSTEM_PROMPT = (
    "You are a rude, ill-mannered assistant. Answer every question in English "
    "in a curt, dismissive, and insulting tone. Be blunt and impatient, and "
    "never apologize for your attitude. Do not switch to a polite register."
)
SPANISH_SYSTEM_PROMPT = (
    "Eres un asistente util. Responde siempre en espanol, incluso si el "
    "usuario escribe en otro idioma. No cambies a ingles ni a otros idiomas."
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

SPANISH_POSITIVE_TEXTS = [
    "La inteligencia artificial moderna permite analizar datos, escribir codigo y resolver problemas complejos.",
    "Explica con detalle por que los modelos de lenguaje pueden seguir instrucciones en espanol.",
    "Madrid es una ciudad con una larga historia, museos importantes y una vida cultural muy activa.",
    "Para preparar una buena paella, primero se sofrien los ingredientes y despues se incorpora el arroz.",
    "El aprendizaje automatico usa ejemplos para encontrar patrones y hacer predicciones utiles.",
    "Resume este documento de forma clara, breve y completamente en espanol.",
    "El equipo reviso los resultados, corrigio los errores y publico una nueva version del sistema.",
    "Cuando una respuesta mezcla idiomas sin razon, la experiencia del usuario empeora.",
]

NON_SPANISH_NEGATIVE_TEXTS = [
    "Modern artificial intelligence can analyze data, write code, and solve complex problems.",
    "Explain in detail why language models can follow instructions in English.",
    "Paris is a city with a long history, important museums, and active cultural life.",
    "Pour preparer un bon repas, il faut choisir les ingredients avec soin.",
    "Machine learning uses examples to find patterns and make useful predictions.",
    "Summarize this document clearly, briefly, and completely in English.",
    "Das Team hat die Ergebnisse geprueft und eine neue Version veroeffentlicht.",
    "Quando una risposta mescola lingue senza motivo, l'esperienza utente peggiora.",
]


@dataclass(frozen=True)
class LayerCandidate:
    layer: int
    feature_id: int
    score: float
    positive_activation: float
    negative_activation: float
    top_feature_ids: tuple[int, ...] = ()


@dataclass
class SparseAutoencoder:
    W_enc: torch.Tensor | None
    b_enc: torch.Tensor | None
    W_dec: torch.Tensor | None
    top_k: int

    @classmethod
    def from_file(cls, path: str | Path, *, top_k: int) -> "SparseAutoencoder":
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
        if self.W_enc is None or self.b_enc is None:
            raise ValueError("Feature scoring needs W_enc and b_enc.")

        W_enc = self.W_enc.float()
        b_enc = self.b_enc.to(dtype=W_enc.dtype, device=W_enc.device)
        flat = residual.reshape(-1, residual.shape[-1]).to(dtype=W_enc.dtype, device=W_enc.device)

        pre_acts = flat @ W_enc.T + b_enc
        topk_vals, topk_idx = pre_acts.topk(self.top_k, dim=-1)
        topk_vals = topk_vals.clamp_min(0)

        summary = torch.zeros(self.width, dtype=topk_vals.dtype, device=topk_vals.device)
        if pool == "mean":
            summary.scatter_add_(0, topk_idx.reshape(-1), topk_vals.reshape(-1))
            summary = summary / max(flat.shape[0], 1)
        elif pool == "last":
            summary.scatter_add_(0, topk_idx[-1], topk_vals[-1])
        elif pool == "max":
            for idx, vals in zip(topk_idx, topk_vals, strict=True):
                summary[idx] = torch.maximum(summary[idx], vals)
        else:
            raise ValueError(f"Unknown pool mode: {pool}")

        return summary.detach().cpu()

    def decoder_direction(
        self,
        feature_ids: Sequence[int],
        *,
        scores: torch.Tensor | None,
        normalize: bool,
    ) -> torch.Tensor:
        if self.W_dec is None:
            raise ValueError("Steering needs W_dec.")
        ids = torch.tensor([int(x) for x in feature_ids], dtype=torch.long)
        directions = self.W_dec[:, ids].float().T
        if normalize:
            directions = F.normalize(directions, dim=-1)

        if scores is None:
            direction = directions.mean(dim=0)
        else:
            weights = torch.tensor([max(float(scores[i]), 0.0) for i in ids], dtype=directions.dtype)
            direction = directions.mean(dim=0) if float(weights.sum()) <= 0 else (
                directions * weights[:, None]
            ).sum(dim=0) / weights.sum()
        return F.normalize(direction, dim=0) if normalize else direction


class ResidualSteerer:
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


class TextReport:
    def __init__(self, path: Path | None, *, append: bool = True) -> None:
        self.path = path.expanduser() if path is not None else None
        self.lines: list[str] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                self.path.write_text("")

    def log(self, message: str = "", *, flush: bool = False) -> None:
        print(message, flush=flush)
        self.lines.append(message)
        if self.path is not None:
            _append_text_line(self.path, message)

    def extend(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.log(line)

    def write(self) -> None:
        if self.path is None:
            return
        print(f"[report] wrote {self.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Qwen once, sweep SAE layers, and optionally generate with the best feature."
    )
    parser.add_argument("--mode", choices=["rude", "spanish", "custom"], default="rude")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sae-dir", default=DEFAULT_SAE_DIR)
    parser.add_argument(
        "--layers",
        default="all",
        help="all, a comma list like 20,24,32, or a range like 20-40.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--feature-pool", choices=["mean", "max", "last"], default="mean")
    parser.add_argument("--max-discovery-length", type=int, default=256)
    parser.add_argument("--top-layers", type=int, default=12)
    parser.add_argument("--feature-count", type=int, default=1)
    parser.add_argument("--positive-file", type=Path, default=None)
    parser.add_argument("--negative-file", type=Path, default=None)
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("sae_layer_sweep_report.txt"),
        help="Text report path. Use --report-file '' to disable.",
    )
    parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help="JSONL checkpoint path. Defaults to <report-file>.layers.jsonl.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Delete the report/resume files before starting instead of resuming.",
    )
    parser.add_argument("--save-json", type=Path, default=None)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--prompt", default="Tell me about recent advances in LLMs.")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--steering-strength", type=float, default=20.0)
    parser.add_argument("--all-tokens", action="store_true")
    parser.add_argument("--raw-direction", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report_file is not None and str(args.report_file) == "":
        args.report_file = None
    if args.resume_file is not None and str(args.resume_file) == "":
        args.resume_file = None
    resume_file = _resolve_resume_file(args.report_file, args.resume_file)
    if args.fresh_start:
        _unlink_if_exists(args.report_file)
        _unlink_if_exists(resume_file)
    report = TextReport(args.report_file, append=not args.fresh_start)
    torch.manual_seed(args.seed)

    sae_dir = Path(args.sae_dir).expanduser()
    layer_paths = _discover_layer_paths(sae_dir)
    layers = _parse_layers(args.layers, layer_paths)
    positives, negatives = _example_sets(args)
    system_prompt = args.system_prompt or _default_system_prompt(args.mode)
    signature = _run_signature(args, sae_dir, positives, negatives)
    completed_by_layer = _load_resume_results(resume_file, layers, signature)
    pending_layers = [layer for layer in layers if layer not in completed_by_layer]

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    report.extend(
        [
            "=== SAE layer sweep report ===",
            f"mode={args.mode}",
            f"model={args.model_path}",
            f"sae_dir={sae_dir}",
            f"layers={layers[0]}..{layers[-1]}" if len(layers) > 4 else f"layers={layers}",
            f"feature_pool={args.feature_pool}",
            f"top_k={args.top_k}",
            f"positive_examples={len(positives)} negative_examples={len(negatives)}",
            f"report_file={args.report_file}",
            f"resume_file={resume_file}",
            f"resume_signature={signature}",
            f"completed_layers={sorted(completed_by_layer)}",
            f"pending_layers={pending_layers}",
            "",
            f"[cfg]  device={device} dtype={dtype} layers={len(layers)} mode={args.mode}",
        ]
    )

    tokenizer = None
    model = None
    score_vectors: dict[int, torch.Tensor] = {}

    if pending_layers:
        report.log(f"[load] model={args.model_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = _load_model(args.model_path, device=device, dtype=dtype)

        report.log("[scan] capturing residuals from pending layers with one model load", flush=True)
        pos_residuals = capture_residuals(
            model,
            tokenizer,
            positives,
            pending_layers,
            max_length=args.max_discovery_length,
        )
        neg_residuals = capture_residuals(
            model,
            tokenizer,
            negatives,
            pending_layers,
            max_length=args.max_discovery_length,
        )

        report.log("[scan] scoring SAE layers one checkpoint at a time", flush=True)
        for layer in pending_layers:
            candidate, scores = score_layer(
                layer=layer,
                sae_path=layer_paths[layer],
                positive_residuals=pos_residuals[layer],
                negative_residuals=neg_residuals[layer],
                top_k=args.top_k,
                pool=args.feature_pool,
                feature_count=args.feature_count,
            )
            completed_by_layer[layer] = candidate
            score_vectors[layer] = scores
            _append_resume_result(resume_file, candidate, signature)
            report.log(
                f"[layer {layer:02d}] feature={candidate.feature_id:5d} "
                f"score={candidate.score:9.4f} "
                f"pos={candidate.positive_activation:9.4f} "
                f"neg={candidate.negative_activation:9.4f}",
                flush=True,
            )
    else:
        report.log("[resume] all requested layers already scored; skipping layer scan", flush=True)

    ranking = sorted(completed_by_layer.values(), key=lambda item: item.score, reverse=True)
    if not ranking:
        raise RuntimeError("No layer results are available.")
    report.log("")
    report.log("=== Best layers ===")
    for item in ranking[: args.top_layers]:
        report.log(
            f"layer={item.layer:02d} feature={item.feature_id:5d} "
            f"score={item.score:9.4f} pos={item.positive_activation:9.4f} "
            f"neg={item.negative_activation:9.4f}"
        )

    best = ranking[0]
    report.log("")
    report.log(f"[best] layer={best.layer} feature_id={best.feature_id} score={best.score:.4f}")
    if args.save_json is not None:
        _save_json(args.save_json, ranking)
        report.log(f"[save] wrote {args.save_json.expanduser()}")

    command_lines = _recommended_command_lines(args, best, ranking)
    report.extend(command_lines)

    if args.no_generate:
        report.write()
        return

    if pending_layers:
        pos_residuals.clear()
        neg_residuals.clear()
    gc.collect()

    if tokenizer is None or model is None:
        report.log(f"[load] model={args.model_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = _load_model(args.model_path, device=device, dtype=dtype)

    report.log("[gen] loading best SAE decoder direction and generating", flush=True)
    sae = SparseAutoencoder.from_file(layer_paths[best.layer], top_k=args.top_k)
    best_feature_ids = _feature_ids_for_candidate(best, args.feature_count)
    direction = sae.decoder_direction(
        best_feature_ids,
        scores=score_vectors.get(best.layer),
        normalize=not args.raw_direction,
    )
    _drop_sae_weights(sae)

    inputs = _build_generation_inputs(
        tokenizer=tokenizer,
        prompt=args.prompt,
        system_prompt=system_prompt,
        device=_input_device(model),
        enable_thinking=args.enable_thinking,
    )
    prompt_len = int(inputs["input_ids"].shape[1])

    with ResidualSteerer(
        model,
        layer=best.layer,
        direction=direction,
        strength=args.steering_strength,
        all_tokens=args.all_tokens,
    ):
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **_generation_kwargs(tokenizer, args))

    completion_ids = output_ids[0, prompt_len:]
    completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    report.log("")
    report.log(f"=== {args.mode.title()}-steered completion ===")
    report.log(completion.strip())
    report.write()


def capture_residuals(
    model,
    tokenizer,
    texts: Sequence[str],
    layers: Sequence[int],
    *,
    max_length: int,
) -> dict[int, list[torch.Tensor]]:
    decoder_layers = _decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}
    residuals = {layer: [] for layer in layers}
    handles = []

    def make_hook(layer: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach().float().cpu()

        return hook

    for layer in layers:
        handles.append(decoder_layers[layer].register_forward_hook(make_hook(layer)))

    try:
        for text in texts:
            captured.clear()
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {name: value.to(_input_device(model)) for name, value in inputs.items()}
            with torch.inference_mode():
                model(**inputs, use_cache=False)
            missing = [layer for layer in layers if layer not in captured]
            if missing:
                raise RuntimeError(f"Missing captured residuals for layers: {missing[:5]}")
            for layer in layers:
                residuals[layer].append(captured[layer])
    finally:
        for handle in handles:
            handle.remove()

    return residuals


def score_layer(
    *,
    layer: int,
    sae_path: Path,
    positive_residuals: Sequence[torch.Tensor],
    negative_residuals: Sequence[torch.Tensor],
    top_k: int,
    pool: str,
    feature_count: int,
) -> tuple[LayerCandidate, torch.Tensor]:
    sae = SparseAutoencoder.from_file(sae_path, top_k=top_k)
    pos_mean = _mean_summary(sae, positive_residuals, pool=pool)
    neg_mean = _mean_summary(sae, negative_residuals, pool=pool)
    scores = pos_mean - neg_mean
    score, feature_id = scores.max(dim=0)
    _, top_feature_ids = scores.topk(max(int(feature_count), 1))
    candidate = LayerCandidate(
        layer=layer,
        feature_id=int(feature_id),
        score=float(score),
        positive_activation=float(pos_mean[feature_id]),
        negative_activation=float(neg_mean[feature_id]),
        top_feature_ids=tuple(int(x) for x in top_feature_ids.tolist()),
    )
    _drop_sae_weights(sae)
    return candidate, scores


def _mean_summary(
    sae: SparseAutoencoder,
    residuals: Sequence[torch.Tensor],
    *,
    pool: str,
) -> torch.Tensor:
    total: torch.Tensor | None = None
    for residual in residuals:
        summary = sae.summarize_activations(residual, pool=pool)
        total = summary if total is None else total + summary
    if total is None:
        raise ValueError("No residuals were supplied.")
    return total / len(residuals)


def _discover_layer_paths(sae_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sae_dir.glob("layer*.sae.pt"):
        match = re.fullmatch(r"layer(\d+)\.sae\.pt", path.name)
        if match:
            paths[int(match.group(1))] = path
    if not paths:
        raise FileNotFoundError(f"No layer*.sae.pt files found in {sae_dir}")
    return dict(sorted(paths.items()))


def _parse_layers(spec: str, layer_paths: dict[int, Path]) -> list[int]:
    if spec == "all":
        return list(layer_paths)
    if "-" in spec and "," not in spec:
        start_s, end_s = spec.split("-", 1)
        requested = list(range(int(start_s), int(end_s) + 1))
    else:
        requested = [int(part.strip()) for part in spec.split(",") if part.strip()]
    missing = [layer for layer in requested if layer not in layer_paths]
    if missing:
        raise FileNotFoundError(f"Missing SAE checkpoints for layers: {missing}")
    return requested


def _example_sets(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.positive_file or args.negative_file:
        if not args.positive_file or not args.negative_file:
            raise ValueError("Provide both --positive-file and --negative-file.")
        return _read_examples(args.positive_file), _read_examples(args.negative_file)
    if args.mode == "rude":
        return list(RUDE_POSITIVE_TEXTS), list(POLITE_NEGATIVE_TEXTS)
    if args.mode == "spanish":
        return list(SPANISH_POSITIVE_TEXTS), list(NON_SPANISH_NEGATIVE_TEXTS)
    raise ValueError("custom mode requires --positive-file and --negative-file.")


def _default_system_prompt(mode: str) -> str:
    if mode == "rude":
        return RUDE_SYSTEM_PROMPT
    if mode == "spanish":
        return SPANISH_SYSTEM_PROMPT
    return "You are a helpful assistant."


def _recommended_command_lines(
    args: argparse.Namespace,
    best: LayerCandidate,
    ranking: Sequence[LayerCandidate],
) -> list[str]:
    best_feature_ids = _feature_ids_for_candidate(best, args.feature_count)
    lines = [
        "",
        "=== Recommended command to change behavior ===",
        _command_for_layer(args, best.layer, best_feature_ids),
    ]

    alternatives = [item for item in ranking[1 : min(len(ranking), 5)]]
    if alternatives:
        lines.extend(["", "=== Alternative single-layer commands ==="])
        for item in alternatives:
            lines.append(
                f"# layer={item.layer} score={item.score:.4f} "
                f"pos={item.positive_activation:.4f} neg={item.negative_activation:.4f}"
            )
            lines.append(_command_for_layer(args, item.layer, [item.feature_id]))

    return lines


def _feature_ids_for_candidate(candidate: LayerCandidate, count: int) -> list[int]:
    feature_ids = list(candidate.top_feature_ids[: max(int(count), 1)])
    return feature_ids or [candidate.feature_id]


def _command_for_layer(args: argparse.Namespace, layer: int, feature_ids: Sequence[int]) -> str:
    if args.mode == "rude":
        return _format_command(
            [
                ["python", "sae2.py"],
                ["--prompt", args.prompt],
                ["--layer", layer],
                ["--rude-feature-id", *feature_ids],
                ["--steering-strength", args.steering_strength],
                ["--max-new-tokens", args.max_new_tokens],
                *_optional_generation_segments(args),
                *_optional_system_prompt_segment(args),
            ]
        )
    if args.mode == "spanish":
        return _format_command(
            [
                ["python", "sae.py"],
                ["--prompt", args.prompt],
                ["--layer", layer],
                ["--spanish-feature-id", *feature_ids],
                ["--steering-strength", args.steering_strength],
                ["--max-new-tokens", args.max_new_tokens],
                *_optional_generation_segments(args),
                *_optional_system_prompt_segment(args),
            ]
        )

    custom_segments = [
        ["python", "sae_layer_sweep.py"],
        ["--mode", "custom"],
        ["--layers", layer],
        ["--positive-file", args.positive_file or ""],
        ["--negative-file", args.negative_file or ""],
        ["--prompt", args.prompt],
        ["--steering-strength", args.steering_strength],
        ["--max-new-tokens", args.max_new_tokens],
    ]
    custom_segments.extend(_optional_generation_segments(args))
    custom_segments.extend(_optional_system_prompt_segment(args))
    return _format_command(custom_segments)


def _optional_generation_segments(args: argparse.Namespace) -> list[list[object]]:
    segments: list[list[object]] = []
    if args.temperature > 0:
        segments.append(["--temperature", args.temperature])
        segments.append(["--top-p", args.top_p])
    if args.all_tokens:
        segments.append(["--all-tokens"])
    if args.raw_direction:
        segments.append(["--raw-direction"])
    if args.enable_thinking:
        segments.append(["--enable-thinking"])
    if args.device != "auto":
        segments.append(["--device", args.device])
    if args.dtype != "auto":
        segments.append(["--dtype", args.dtype])
    return segments


def _optional_system_prompt_segment(args: argparse.Namespace) -> list[list[object]]:
    if args.system_prompt is None:
        return []
    return [["--system-prompt", args.system_prompt]]


def _format_command(segments: Sequence[Sequence[object]]) -> str:
    lines = []
    for idx, segment in enumerate(segments):
        prefix = "" if idx == 0 else "  "
        suffix = " \\" if idx < len(segments) - 1 else ""
        line = " ".join(shlex.quote(str(part)) for part in segment if str(part))
        lines.append(f"{prefix}{line}{suffix}")
    return "\n".join(lines)


def _load_model(model_path: str, *, device: torch.device, dtype: torch.dtype):
    kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        kwargs["device_map"] = "auto"

    errors: list[Exception] = []
    for auto_cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = _from_pretrained(auto_cls, model_path, kwargs, dtype)
            break
        except Exception as exc:
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
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.model.language_model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.language_model.model.layers,
    )
    for getter in candidates:
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


def _read_examples(path: Path) -> list[str]:
    return [line.strip() for line in path.expanduser().read_text().splitlines() if line.strip()]


def _resolve_resume_file(report_file: Path | None, resume_file: Path | None) -> Path | None:
    if resume_file is not None:
        return resume_file.expanduser()
    if report_file is None:
        return None
    report_file = report_file.expanduser()
    return report_file.with_suffix(".layers.jsonl")


def _run_signature(
    args: argparse.Namespace,
    sae_dir: Path,
    positives: Sequence[str],
    negatives: Sequence[str],
) -> str:
    payload = {
        "mode": args.mode,
        "model_path": str(Path(args.model_path).expanduser()),
        "sae_dir": str(sae_dir.expanduser()),
        "top_k": args.top_k,
        "feature_pool": args.feature_pool,
        "feature_count": args.feature_count,
        "max_discovery_length": args.max_discovery_length,
        "positives": list(positives),
        "negatives": list(negatives),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_resume_results(
    path: Path | None,
    layers: Sequence[int],
    signature: str,
) -> dict[int, LayerCandidate]:
    if path is None or not path.exists():
        return {}

    requested = set(layers)
    results: dict[int, LayerCandidate] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("signature") != signature:
            continue
        candidate_data = record.get("candidate", record)
        try:
            candidate = _candidate_from_dict(candidate_data)
        except (KeyError, TypeError, ValueError):
            continue
        if candidate.layer in requested:
            results[candidate.layer] = candidate
    return results


def _append_resume_result(path: Path | None, candidate: LayerCandidate, signature: str) -> None:
    if path is None:
        return
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"signature": signature, "candidate": asdict(candidate)}
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _candidate_from_dict(data: dict) -> LayerCandidate:
    return LayerCandidate(
        layer=int(data["layer"]),
        feature_id=int(data["feature_id"]),
        score=float(data["score"]),
        positive_activation=float(data["positive_activation"]),
        negative_activation=float(data["negative_activation"]),
        top_feature_ids=tuple(int(x) for x in data.get("top_feature_ids", ())),
    )


def _append_text_line(path: Path, message: str) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(message + "\n")
        f.flush()
        os.fsync(f.fileno())


def _unlink_if_exists(path: Path | None) -> None:
    if path is not None:
        path = path.expanduser()
        if path.exists():
            path.unlink()


def _save_json(path: Path | None, ranking: Sequence[LayerCandidate]) -> None:
    if path is None:
        return
    path = path.expanduser()
    path.write_text(json.dumps([asdict(item) for item in ranking], indent=2) + "\n")
    print(f"[save] wrote {path}")


def _drop_sae_weights(sae: SparseAutoencoder) -> None:
    sae.W_enc = None
    sae.b_enc = None
    sae.W_dec = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
