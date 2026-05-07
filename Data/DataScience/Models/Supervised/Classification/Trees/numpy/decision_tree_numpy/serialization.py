"""JSON serialization helpers for the NumPy decision tree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tree import NumpyDecisionTreeClassifier


def save_model(model: NumpyDecisionTreeClassifier, model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        json.dump(model.to_dict(), handle, indent=2)


def load_model(model_path: str | Path) -> NumpyDecisionTreeClassifier:
    path = Path(model_path)
    with open(path, "r", encoding="utf8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return NumpyDecisionTreeClassifier.from_dict(payload)
