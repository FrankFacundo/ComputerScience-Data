# coding=utf-8
"""Unused 25Hz tokenizer placeholder for the Torch-only 12Hz runner."""

from torch import nn


class Qwen3TTSTokenizerV1Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "The local no-transformers runtime implements Qwen3-TTS-Tokenizer-12Hz only."
        )


class Qwen3TTSTokenizerV1PreTrainedModel(nn.Module):
    pass


__all__ = ["Qwen3TTSTokenizerV1Model", "Qwen3TTSTokenizerV1PreTrainedModel"]
