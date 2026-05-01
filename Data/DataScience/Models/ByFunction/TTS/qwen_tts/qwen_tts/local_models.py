# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Union

DEFAULT_QWEN_MODEL_ROOT = "/Users/frankfacundo/Models/Qwen"
QWEN_LOCAL_MODEL_ROOT = Path(os.environ.get("QWEN_LOCAL_MODEL_ROOT", DEFAULT_QWEN_MODEL_ROOT)).expanduser()


def qwen_model_path(model_name: str) -> str:
    model_name = model_name.rstrip("/")
    if model_name.startswith("Qwen/"):
        model_name = model_name.split("/", 1)[1]
    return str(QWEN_LOCAL_MODEL_ROOT / model_name)


def resolve_local_model_path(pretrained_model_name_or_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    if not isinstance(pretrained_model_name_or_path, (str, os.PathLike)):
        return pretrained_model_name_or_path

    raw_path = os.fspath(pretrained_model_name_or_path)
    expanded_path = os.path.expanduser(raw_path)
    if os.path.isdir(expanded_path):
        return expanded_path

    normalized = raw_path.rstrip("/")
    if normalized.startswith("Qwen/"):
        candidate = qwen_model_path(normalized)
    elif "/" not in normalized and "\\" not in normalized:
        candidate = qwen_model_path(normalized)
    else:
        return raw_path

    if os.path.isdir(candidate):
        return candidate
    return raw_path
