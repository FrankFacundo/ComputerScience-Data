#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

try:
    from qwen_asr import Qwen3ASRModel
except ImportError as exc:
    Qwen3ASRModel = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


DEFAULT_MODEL_PATH = Path("/Users/frankfacundo/Models/Qwen/Qwen3-ASR-1.7B")


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe WhatsApp .ogg audio files with a local Qwen3-ASR model and "
            "write the combined transcript to a .txt file."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the .ogg files. Defaults to this script directory.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path(__file__).resolve().parent / "whatsapp_transcriptions.txt",
        help="Output text file. Defaults to whatsapp_transcriptions.txt next to the script.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Local path to the Qwen3-ASR model directory.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help='Optional forced language, for example "English" or "Spanish".',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens per audio file.",
    )
    parser.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=8,
        help="Inference batch size limit used by qwen-asr.",
    )
    return parser


def ensure_dependencies() -> None:
    if Qwen3ASRModel is None:
        raise SystemExit(
            "Missing dependency: qwen-asr\n"
            "Install it in your active environment, for example:\n"
            "  pip install -U qwen-asr\n"
            f"Original import error: {IMPORT_ERROR}"
        )


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda:0", torch.bfloat16
        return "cuda:0", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.ogg") if path.is_file())


def load_model(
    model_path: Path,
    max_new_tokens: int,
    max_inference_batch_size: int,
) -> Qwen3ASRModel:
    device_map, dtype = pick_device_and_dtype()
    kwargs = {
        "dtype": dtype,
        "device_map": device_map,
        "max_new_tokens": max_new_tokens,
        "max_inference_batch_size": max_inference_batch_size,
    }
    return Qwen3ASRModel.from_pretrained(str(model_path), **kwargs)


def normalize_language_arg(language: str | None, count: int) -> str | list[str] | None:
    if language is None:
        return None
    return [language] * count


def write_output(
    output_txt: Path,
    audio_files: Sequence[Path],
    results: Sequence[object],
) -> None:
    lines: list[str] = []
    for audio_file, result in zip(audio_files, results):
        language = getattr(result, "language", "Unknown")
        text = getattr(result, "text", "").strip()
        lines.append(f"FILE: {audio_file.name}")
        lines.append(f"LANGUAGE: {language}")
        lines.append("TRANSCRIPT:")
        lines.append(text or "[empty transcript]")
        lines.append("")

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = build_cli().parse_args()
    ensure_dependencies()

    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_path}")

    audio_files = find_audio_files(args.input_dir)
    if not audio_files:
        raise SystemExit(f"No .ogg files found in: {args.input_dir}")

    print(f"Found {len(audio_files)} .ogg files in {args.input_dir}")
    print(f"Loading model from {args.model_path}")
    model = load_model(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=args.max_inference_batch_size,
    )

    print("Starting transcription...")
    results = model.transcribe(
        audio=[str(path) for path in audio_files],
        language=normalize_language_arg(args.language, len(audio_files)),
    )
    write_output(args.output_txt, audio_files, results)

    print(f"Saved transcript to {args.output_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
