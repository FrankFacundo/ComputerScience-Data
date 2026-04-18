"""Safetensors weight loader for Qwen3.5 checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

_MTP_PATTERN = re.compile(r"^mtp(\.|$)")


def _iter_shards(model_dir: Path) -> Iterable[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
        shards = sorted(set(data["weight_map"].values()))
        for shard in shards:
            yield model_dir / shard
        return
    single = model_dir / "model.safetensors"
    if single.exists():
        yield single
        return
    for p in sorted(model_dir.glob("*.safetensors")):
        yield p


def _remap_key(key: str, text_only: bool) -> str | None:
    if _MTP_PATTERN.match(key):
        return None
    if text_only:
        # Qwen3_5ForCausalLM expects `model.*` where checkpoint has `model.language_model.*`
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        if key.startswith("lm_head."):
            return key
        # skip visual keys in text-only mode
        if key.startswith("model.visual."):
            return None
        return key
    # multimodal: keys already match `model.language_model.*`, `model.visual.*`, `lm_head.*`
    return key


def load_qwen3_5_weights(
    model: nn.Module,
    model_dir: str | Path,
    *,
    text_only: bool = False,
    strict: bool = True,
    dtype: torch.dtype | None = None,
) -> dict:
    """Load sharded safetensors weights into `model`.

    Parameters
    ----------
    text_only
        True when loading into `Qwen3_5ForCausalLM` (text-only). Strips the
        `model.language_model.` prefix and skips `model.visual.*`.
    strict
        Raise if any parameter was not assigned or if unexpected keys remain.
    dtype
        Optional cast applied to every tensor during load.
    """
    try:
        from safetensors import safe_open
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("safetensors is required to load checkpoints") from e

    model_dir = Path(model_dir)
    state: dict[str, torch.Tensor] = {}
    missing = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    # Don't require non-persistent buffers (e.g. inv_freq)
    persistent_buffers = {
        name for name, _ in model.named_buffers() if _is_persistent_buffer(model, name)
    }
    param_names = set(dict(model.named_parameters()).keys())
    wanted = param_names | persistent_buffers

    unexpected: list[str] = []
    loaded: set[str] = set()

    for shard in _iter_shards(model_dir):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                target = _remap_key(key, text_only=text_only)
                if target is None:
                    continue
                if target not in wanted:
                    unexpected.append(target)
                    continue
                tensor = f.get_tensor(key)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                state[target] = tensor
                loaded.add(target)

    # apply
    model_state = model.state_dict()
    for k, v in state.items():
        if k in model_state and model_state[k].shape != v.shape:
            raise ValueError(
                f"shape mismatch for {k}: checkpoint {tuple(v.shape)} vs model {tuple(model_state[k].shape)}"
            )
    missing_result = [k for k in wanted if k not in loaded]

    # Handle tied lm_head / embed_tokens weight: if checkpoint has one but model
    # ties them, copy across.
    if "lm_head.weight" not in loaded:
        emb_key = "model.embed_tokens.weight" if text_only else "model.language_model.embed_tokens.weight"
        if emb_key in loaded and "lm_head.weight" in wanted:
            state["lm_head.weight"] = state[emb_key]
            loaded.add("lm_head.weight")
            if "lm_head.weight" in missing_result:
                missing_result.remove("lm_head.weight")

    model.load_state_dict(state, strict=False)

    if strict and missing_result:
        raise RuntimeError(f"Missing {len(missing_result)} parameters, first few: {missing_result[:5]}")
    if strict and unexpected:
        raise RuntimeError(f"Unexpected {len(unexpected)} keys, first few: {unexpected[:5]}")

    return {"missing": missing_result, "unexpected": unexpected, "loaded": sorted(loaded)}


def _is_persistent_buffer(module: nn.Module, name: str) -> bool:
    """Walk `name` and check the persistent flag on the owning module."""
    parts = name.split(".")
    submod = module
    for p in parts[:-1]:
        submod = getattr(submod, p)
    buf_name = parts[-1]
    return buf_name not in submod._non_persistent_buffers_set
