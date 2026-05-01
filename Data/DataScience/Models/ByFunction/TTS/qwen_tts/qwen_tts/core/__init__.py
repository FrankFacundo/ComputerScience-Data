# coding=utf-8
"""Core Torch-only Qwen3-TTS components."""

from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

__all__ = ["Qwen3TTSTokenizerV2Config", "Qwen3TTSTokenizerV2Model"]
