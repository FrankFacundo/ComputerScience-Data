"""Configuration objects for the NumPy decision tree implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DecisionTreeConfig:
    """Hyperparameters that control tree growth.

    The defaults intentionally mirror the sklearn comparison script used in
    this folder: entropy criterion, unlimited depth, and binary threshold
    splits over numeric features.
    """

    criterion: str = "entropy"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    feature_names: list[str] = field(default_factory=lambda: ["A", "B"])
    target_name: str = "C"

    def __post_init__(self) -> None:
        valid_criteria = {"gini", "entropy", "log_loss"}
        if self.criterion not in valid_criteria:
            raise ValueError(
                f"criterion must be one of {sorted(valid_criteria)}, "
                f"got {self.criterion!r}"
            )
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be None or a positive integer")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionTreeConfig":
        return cls(**data)
