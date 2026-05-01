"""Safetensors loader for Qwen3-Embedding-0.6B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


def _iter_shards(model_dir: Path) -> Iterable[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
        for shard in sorted(set(data["weight_map"].values())):
            yield model_dir / shard
        return
    single = model_dir / "model.safetensors"
    if single.exists():
        yield single
        return
    yield from sorted(model_dir.glob("*.safetensors"))


def _is_persistent_buffer(module: nn.Module, name: str) -> bool:
    parts = name.split(".")
    submodule = module
    for part in parts[:-1]:
        submodule = getattr(submodule, part)
    return parts[-1] not in submodule._non_persistent_buffers_set


def _remap_key(key: str, wanted: set[str]) -> str | None:
    if key in wanted:
        return key
    prefixed = f"model.{key}"
    if prefixed in wanted:
        return prefixed
    if key.startswith("model.") and key[len("model.") :] in wanted:
        return key[len("model.") :]
    return None


def load_qwen3_embedding_weights(
    model: nn.Module,
    model_dir: str | Path,
    *,
    strict: bool = True,
    dtype: torch.dtype | None = None,
) -> dict:
    """Load one or more safetensors shards into the pure-torch model."""
    try:
        from safetensors import safe_open
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("safetensors is required to load Qwen3 embedding weights") from e

    model_dir = Path(model_dir)
    param_names = set(dict(model.named_parameters()).keys())
    persistent_buffers = {
        name for name, _ in model.named_buffers() if _is_persistent_buffer(model, name)
    }
    wanted = param_names | persistent_buffers
    state: dict[str, torch.Tensor] = {}
    loaded: set[str] = set()
    unexpected: list[str] = []

    for shard in _iter_shards(model_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                target = _remap_key(key, wanted)
                if target is None:
                    unexpected.append(key)
                    continue
                tensor = f.get_tensor(key)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                state[target] = tensor
                loaded.add(target)

    if "lm_head.weight" in wanted and "lm_head.weight" not in loaded:
        emb_key = "model.embed_tokens.weight" if "model.embed_tokens.weight" in loaded else "embed_tokens.weight"
        if emb_key in loaded:
            state["lm_head.weight"] = state[emb_key]
            loaded.add("lm_head.weight")

    model_state = model.state_dict()
    for key, tensor in state.items():
        if key in model_state and model_state[key].shape != tensor.shape:
            raise ValueError(
                f"shape mismatch for {key}: checkpoint {tuple(tensor.shape)} "
                f"vs model {tuple(model_state[key].shape)}"
            )

    missing = sorted(wanted - loaded)
    model.load_state_dict(state, strict=False)

    if strict and missing:
        raise RuntimeError(f"Missing {len(missing)} parameters, first few: {missing[:5]}")
    if strict and unexpected:
        raise RuntimeError(f"Unexpected {len(unexpected)} keys, first few: {unexpected[:5]}")

    return {"missing": missing, "unexpected": unexpected, "loaded": sorted(loaded)}
