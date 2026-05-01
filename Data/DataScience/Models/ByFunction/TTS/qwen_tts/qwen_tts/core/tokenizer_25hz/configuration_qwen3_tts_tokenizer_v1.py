# coding=utf-8
"""Placeholder config for the unused 25Hz tokenizer path.

The local runner targets Qwen3-TTS-Tokenizer-12Hz. This stub is kept only so
old imports fail explicitly without pulling in Transformers.
"""

from ...torch_utils import SimpleConfig


class Qwen3TTSTokenizerV1Config(SimpleConfig):
    model_type = "qwen3_tts_tokenizer_25hz"


class Qwen3TTSTokenizerV1DecoderConfig(SimpleConfig):
    pass


class Qwen3TTSTokenizerV1DecoderDiTConfig(SimpleConfig):
    pass


class Qwen3TTSTokenizerV1DecoderBigVGANConfig(SimpleConfig):
    pass


class Qwen3TTSTokenizerV1EncoderConfig(SimpleConfig):
    pass


__all__ = [
    "Qwen3TTSTokenizerV1Config",
    "Qwen3TTSTokenizerV1DecoderConfig",
    "Qwen3TTSTokenizerV1DecoderDiTConfig",
    "Qwen3TTSTokenizerV1DecoderBigVGANConfig",
    "Qwen3TTSTokenizerV1EncoderConfig",
]
