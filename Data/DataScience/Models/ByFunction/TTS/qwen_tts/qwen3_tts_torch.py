# coding=utf-8
"""Torch-first Qwen3-TTS 12Hz Base voice-clone runner.

This is the local, self-contained version of:

    /Users/frankfacundo/GitHub/Qwen3-TTS/examples/test_model_12hz_base_2.py

It vendors the Qwen3-TTS PyTorch runtime under ``./qwen_tts`` so the script
does not depend on the external checkout being on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_MODEL_PATH = "/Users/frankfacundo/Models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REF_AUDIO = "/Users/frankfacundo/GitHub/Qwen3-TTS/examples/alan.mp3"
DEFAULT_OUT_DIR = "qwen3_tts_test_voice_clone_output_wav"
DEFAULT_SYN_TEXT = "Vive la vida y no dejes que la vida te viva."
DEFAULT_SYN_LANGUAGE = "Spanish"
DEFAULT_REF_TEXT = (
    "el primer proyecto Nacional del Perú es su independencia crearse ahora "
    "sobre esto hay duda es que los peruanos querían ser independientes "
    "cuando el general San Martín que venía de chile desembarcó en la Bahía "
    "de pisco y avanzó hacia Lima se quedó estacionado por las Pampas de "
    "pachacamac a unos 20 o 30 km de Lima porque dijo así y lo escribió Oiga "
    "Me parece que esta gente no quiere ser independiente de España y mandó "
    "una carta a Lima para que yo entre a Lima dijo primero firmen todos Que "
    "quieren ser independientes con lo cual de vuestra inteligencia San "
    "Martín Oiga yo a qué entro a ser más papista que el papa en un país que "
    "no quiere ser independiente están contentos con sus birrey sus oidores "
    "se creen todos valientes del rey de España parti acomplejados que que "
    "seguido quedando acá y entonces se reunieron apresuradamente el Marqués "
    "de sarate El conde de San Isidro y y muchos peruanos Ah muchos y "
    "firmaron adelante el señor San Martín que Lima lo recibe para proclamar "
    "la independencia del Perú y entonces San Martín aprovechando 2 de Julio "
    "pues como dijo Manuel leitón en Trujillo ya se olvidar aprovechando 28 "
    "de Julio proclamó la Independencia Pero qué ocurrió ocurrió que las "
    "tropas delrey pezuela se acercaron a Lima nuevamente tomaron el callao y "
    "el General San Martín con mucha prudencia militar se retiró de Lima "
    "porque no tenía tropas suficientes y entonces el birrey entró en Lima "
    "nuevamente y El conde de la vista florid del Marqués sarate El conde de "
    "San Isidro y todos los que habían firmado el ingreso de San Martín "
    "hicieron una recepción para el virrey pezuela y San Martín Ahí es donde "
    "San Martín comenzó a decidir que se iba del Perú porque se dio cuenta "
    "que es muy complejo gobernar El Perú Ah yo les digo muy complejo quiero "
    "esto pero también quiero lo contrario y al mismo tiempo estoy contra "
    "todos y contra ti mismo y contra mí etcétera"
)


@dataclass(frozen=True)
class RuntimeConfig:
    device: torch.device
    device_map: dict[str, str]
    dtype: torch.dtype
    attn_implementation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-TTS 12Hz Base voice cloning with the local Torch runtime."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument(
        "--ref-text-file",
        default=None,
        help="Optional text file to use instead of --ref-text.",
    )
    parser.add_argument("--text", default=DEFAULT_SYN_TEXT)
    parser.add_argument("--language", default=DEFAULT_SYN_LANGUAGE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--modes",
        default="icl,xvec",
        help="Comma-separated modes: icl, xvec. Default runs both, matching the upstream example.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-dosample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    parser.add_argument(
        "--non-streaming-mode",
        action="store_true",
        help="Use the model's non-streaming text input path. Omitted by default to match the upstream example.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, mps, cuda, cuda:0, etc. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Parameter dtype. Defaults to bf16 on CUDA, fp16 on MPS, fp32 on CPU.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-mp3",
        action="store_true",
        help="Write wav only. By default ffmpeg is used for mp3 if available.",
    )
    return parser.parse_args()


def _has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(requested: str | None, device: torch.device) -> torch.dtype:
    if requested is not None:
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


def _resolve_attn(requested: str, device: torch.device) -> str:
    if requested != "auto":
        return requested
    if device.type == "cuda" and _has_flash_attn():
        return "flash_attention_2"
    return "sdpa"


def _device_map_value(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else device.index}"
    return device.type


def get_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    attn_implementation = _resolve_attn(args.attn_implementation, device)
    return RuntimeConfig(
        device=device,
        device_map={"": _device_map_value(device)},
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"[warn] ffmpeg not found; skipping mp3 conversion for {wav_path}")
        return

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_path),
        ],
        check=True,
    )


def parse_modes(raw: str) -> list[bool]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--modes must include at least one of: icl, xvec")

    out: list[bool] = []
    for value in values:
        if value in {"icl", "in_context", "in-context"}:
            out.append(False)
        elif value in {"xvec", "xvector", "x_vector", "xvec_only", "x_vector_only"}:
            out.append(True)
        else:
            raise ValueError(f"Unsupported mode {value!r}; use icl and/or xvec.")
    return out


def run_case(
    *,
    tts: object,
    out_dir: Path,
    case_name: str,
    runtime: RuntimeConfig,
    skip_mp3: bool,
    call_fn,
) -> None:
    synchronize(runtime.device)
    t0 = time.perf_counter()
    wavs, sample_rate = call_fn()
    synchronize(runtime.device)
    elapsed = time.perf_counter() - t0

    print(f"[{case_name}] time: {elapsed:.3f}s, n_wavs={len(wavs)}, sr={sample_rate}")
    import soundfile as sf

    for idx, wav in enumerate(wavs):
        wav_path = out_dir / f"{case_name}_{idx}.wav"
        mp3_path = out_dir / f"{case_name}_{idx}.mp3"
        sf.write(wav_path, wav, sample_rate)
        if not skip_mp3:
            convert_wav_to_mp3(wav_path, mp3_path)


def generation_kwargs(args: argparse.Namespace) -> dict:
    return {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_dosample": args.subtalker_dosample,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_path = Path(args.model_path).expanduser()
    ref_audio = Path(args.ref_audio).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    ref_text = args.ref_text
    if args.ref_text_file is not None:
        ref_text = Path(args.ref_text_file).expanduser().read_text(encoding="utf-8").strip()

    runtime = get_runtime_config(args)
    print(
        f"[cfg] device={runtime.device} dtype={runtime.dtype} "
        f"attn={runtime.attn_implementation}"
    )
    print(f"[load] model={model_path}")

    from qwen_tts import Qwen3TTSModel

    t0 = time.perf_counter()
    tts = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=runtime.device_map,
        dtype=runtime.dtype,
        attn_implementation=runtime.attn_implementation,
    )
    print(f"[load] done in {time.perf_counter() - t0:.1f}s")

    gen_kwargs = generation_kwargs(args)
    for x_vector_only_mode in parse_modes(args.modes):
        mode_tag = "xvec_only" if x_vector_only_mode else "icl"
        run_case(
            tts=tts,
            out_dir=out_dir,
            case_name=f"case1_promptSingle_synSingle_direct_{mode_tag}",
            runtime=runtime,
            skip_mp3=args.skip_mp3,
            call_fn=lambda x_vector_only_mode=x_vector_only_mode: tts.generate_voice_clone(
                text=args.text,
                language=args.language,
                ref_audio=str(ref_audio),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
                non_streaming_mode=args.non_streaming_mode,
                **gen_kwargs,
            ),
        )


if __name__ == "__main__":
    main()
