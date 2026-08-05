"""Shared deterministic inputs for the NumPy/PyTorch one-layer examples."""

from __future__ import annotations

import numpy as np


DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 0.2


def training_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small, linearly separable three-class training set."""
    features = np.array(
        [
            [-2.0, -1.5],
            [-1.5, -2.2],
            [-2.2, -0.8],
            [2.0, -1.5],
            [1.3, -2.2],
            [2.4, -0.7],
            [0.0, 2.0],
            [-0.8, 1.5],
            [0.9, 1.7],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    return features, labels


def inference_data() -> tuple[np.ndarray, np.ndarray]:
    """Return unseen examples used only after training."""
    features = np.array(
        [
            [-1.8, -1.0],
            [1.8, -1.0],
            [0.1, 1.8],
            [-0.7, 1.2],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 1, 2, 2], dtype=np.int64)
    return features, labels


def initial_parameters() -> tuple[np.ndarray, np.ndarray]:
    """Return identical initial parameters in PyTorch ``nn.Linear`` layout."""
    # Shape: (number of classes, number of input features).
    weight = np.array(
        [
            [0.10, -0.20],
            [-0.05, 0.15],
            [0.20, 0.05],
        ],
        dtype=np.float64,
    )
    bias = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    return weight, bias


def epochs_to_print(epoch: int, total_epochs: int) -> bool:
    """Keep command-line output readable while showing training progress."""
    return epoch <= 3 or epoch % 10 == 0 or epoch == total_epochs
