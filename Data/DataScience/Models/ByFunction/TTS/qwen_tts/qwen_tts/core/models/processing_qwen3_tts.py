# coding=utf-8
"""Torch-only Qwen3-TTS text processor."""

from __future__ import annotations

from ...text_tokenizer import Qwen2Tokenizer


class Qwen3TTSProcessor:
    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer",)

    def __init__(self, tokenizer=None, chat_template=None):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    @classmethod
    def from_pretrained(cls, model_path: str, **_kwargs):
        return cls(tokenizer=Qwen2Tokenizer.from_pretrained(model_path))

    def __call__(self, text=None, **kwargs):
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")
        return self.tokenizer(text, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return [self.decode(ids, **kwargs) for ids in args[0]]

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask"]


__all__ = ["Qwen3TTSProcessor"]
