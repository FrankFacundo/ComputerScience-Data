# coding=utf-8
"""Audio feature helpers for the Torch-only Qwen3-TTS runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class BatchFeature(dict):
    def to(self, device=None, dtype=None):
        if isinstance(device, torch.dtype) and dtype is None:
            dtype = device
            device = None
        out = BatchFeature()
        for key, value in self.items():
            if torch.is_tensor(value):
                out[key] = value.to(device=device)
                if dtype is not None and torch.is_floating_point(out[key]):
                    out[key] = out[key].to(dtype=dtype)
            else:
                out[key] = value
        return out


class EncodecFeatureExtractor:
    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        padding_side: str = "right",
        return_attention_mask: bool = True,
        **kwargs,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.padding_side = padding_side
        self.return_attention_mask = return_attention_mask

    @classmethod
    def from_pretrained(cls, path: str | Path):
        path = Path(path)
        config_path = path / "preprocessor_config.json"
        if not config_path.exists():
            return cls()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data.pop("feature_extractor_type", None)
        return cls(**data)

    def __call__(
        self,
        raw_audio,
        padding=True,
        return_tensors: str | None = None,
        sampling_rate: int | None = None,
        **_kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None and int(sampling_rate) != int(self.sampling_rate):
            raise ValueError(f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}")
        audios = raw_audio if isinstance(raw_audio, list) else [raw_audio]
        arrays = [np.asarray(audio, dtype=np.float32).reshape(-1) for audio in audios]
        max_len = max((arr.shape[0] for arr in arrays), default=0)
        values = []
        masks = []
        for arr in arrays:
            pad_len = max_len - arr.shape[0] if padding else 0
            if self.padding_side == "left":
                padded = np.pad(arr, (pad_len, 0), constant_values=self.padding_value)
                mask = np.pad(np.ones(arr.shape[0], dtype=np.int64), (pad_len, 0))
            else:
                padded = np.pad(arr, (0, pad_len), constant_values=self.padding_value)
                mask = np.pad(np.ones(arr.shape[0], dtype=np.int64), (0, pad_len))
            values.append(padded[None, :])
            masks.append(mask[None, :])
        out = BatchFeature(
            {
                "input_values": torch.tensor(np.stack(values), dtype=torch.float32),
                "padding_mask": torch.tensor(np.stack(masks), dtype=torch.long),
            }
        )
        if return_tensors not in (None, "pt"):
            raise ValueError("Only return_tensors='pt' is supported.")
        return out
