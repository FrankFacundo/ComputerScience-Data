# coding=utf-8
"""Compare the local no-transformers runtime against the installed reference.

Run from the qwen3-tts conda environment:

    conda run -n qwen3-tts python parity_check.py

The reference process imports the installed `qwen_tts` package. The local
process prepends this directory to `PYTHONPATH`, which imports the Torch-only
runtime in this folder.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "/Users/frankfacundo/Models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REF_AUDIO = "/Users/frankfacundo/GitHub/Qwen3-TTS/examples/alan.mp3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-TTS no-transformers parity check.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def run_case(out_path: Path, *, local: bool, args: argparse.Namespace) -> None:
    prefix = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        if local
        else ""
    )
    code = (
        "import numpy as np, torch; "
        f"{prefix}"
        "from qwen_tts import Qwen3TTSModel; "
        f"model={args.model_path!r}; ref_audio={args.ref_audio!r}; "
        "tts=Qwen3TTSModel.from_pretrained("
        "model, device_map={'':'cpu'}, dtype=torch.float32, attn_implementation='eager'"
        "); "
        "wavs,sr=tts.generate_voice_clone("
        "text='Vive la vida y no dejes que la vida te viva.', "
        "language='Spanish', ref_audio=ref_audio, ref_text='', "
        "x_vector_only_mode=True, max_new_tokens="
        f"{args.max_new_tokens}, "
        "do_sample=False, top_k=50, top_p=1.0, temperature=0.9, "
        "repetition_penalty=1.05, subtalker_dosample=False, "
        "subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9"
        "); "
        f"np.save({str(out_path)!r}, wavs[0]); "
        "print(('local' if "
        f"{local!r}"
        " else 'reference'), wavs[0].shape, sr)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="qwen_tts_parity.") as tmp:
        tmp_path = Path(tmp)
        ref_path = tmp_path / "reference.npy"
        local_path = tmp_path / "local.npy"
        run_case(ref_path, local=False, args=args)
        run_case(local_path, local=True, args=args)

        ref = np.load(ref_path)
        local = np.load(local_path)
        if ref.shape != local.shape:
            raise AssertionError(f"Shape mismatch: reference={ref.shape}, local={local.shape}")
        max_abs = float(np.max(np.abs(ref - local))) if ref.size else 0.0
        mean_abs = float(np.mean(np.abs(ref - local))) if ref.size else 0.0
        ok = np.allclose(ref, local, atol=args.atol, rtol=args.rtol)
        print(f"allclose={ok} max_abs={max_abs:.9g} mean_abs={mean_abs:.9g}")
        if not ok:
            raise AssertionError("Local waveform does not match reference tolerance.")


if __name__ == "__main__":
    main()
