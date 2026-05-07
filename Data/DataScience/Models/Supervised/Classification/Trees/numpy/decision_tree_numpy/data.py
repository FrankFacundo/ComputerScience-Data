"""Data loading and small metric helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_dataset(
    file_path: str | Path,
    delimiter: str | None = None,
    skip_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a numeric dataset where the last column is the label."""
    path = Path(file_path)
    data = np.loadtxt(path, dtype=float, delimiter=delimiter, skiprows=skip_rows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature and one target column")
    x = data[:, :-1]
    y = data[:, -1]
    if np.all(np.isclose(y, np.round(y))):
        y = y.astype(int)
    return x, y


def resolve_feature_names(base_names: list[str], feature_count: int) -> list[str]:
    names = list(base_names)
    names.extend(f"X{i + 1}" for i in range(len(names), feature_count))
    return names[:feature_count]


def accuracy_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("Cannot compute accuracy on an empty target array")
    return float(np.mean(y_true == y_pred))


def confusion_matrix_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    for actual, predicted in zip(y_true, y_pred):
        matrix[class_to_index[actual], class_to_index[predicted]] += 1
    return matrix
