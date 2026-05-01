# Qwen3-TTS 12Hz Base Torch Runner

Local implementation of:

```bash
/Users/frankfacundo/GitHub/Qwen3-TTS/examples/test_model_12hz_base_2.py
```

The Qwen3-TTS PyTorch runtime is vendored under `./qwen_tts`, so the external
`/Users/frankfacundo/GitHub/Qwen3-TTS` checkout does not need to be on
`PYTHONPATH`. The vendored Qwen code keeps its Apache-2.0 license in `LICENSE`.

## Setup

```bash
cd /Users/frankfacundo/Code/ComputerScience-Data/Data/DataScience/Models/ByFunction/TTS/qwen_tts
python -m pip install -r requirements.txt
```

`ffmpeg` is optional. If it is installed, the runner writes both `.wav` and
`.mp3`; otherwise it writes `.wav` only.

## Run

```bash
python qwen3_tts_torch.py
```

This uses:

- model: `/Users/frankfacundo/Models/Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- reference audio: `/Users/frankfacundo/GitHub/Qwen3-TTS/examples/alan.mp3`
- output dir: `qwen3_tts_test_voice_clone_output_wav`
- modes: `icl,xvec`

The compatibility wrapper also works:

```bash
python test_model_12hz_base_2.py
```

Useful overrides:

```bash
python qwen3_tts_torch.py \
  --text "Vive la vida y no dejes que la vida te viva." \
  --language Spanish \
  --device mps \
  --dtype float16 \
  --modes icl \
  --skip-mp3
```

On CUDA, the runner uses FlashAttention 2 when the `flash_attn` package is
available; otherwise it falls back to PyTorch SDPA.
