r"""Exact SHAP values from scratch, using only NumPy.

This implements the model-agnostic interventional Shapley values used by
SHAP's KernelExplainer for a tabular model:

    phi_i = sum_{S subset F \ {i}} |S|! (M-|S|-1)! / M!
            * (v(S union {i}) - v(S))

For a coalition S, v(S) is the average model prediction after keeping the
features in S from the explained instance and filling all other features from
the background data. With a one-row background, this is the same baseline
replacement idea used in the original shap.py file in this directory.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from common import build_example, print_example_result


PredictionFunction = Callable[[np.ndarray], np.ndarray]


def _as_background_matrix(background: np.ndarray, n_features: int) -> np.ndarray:
    background = np.asarray(background, dtype=float)
    if background.ndim == 1:
        background = background.reshape(1, -1)
    if background.ndim != 2:
        raise ValueError("background must be a 1D or 2D array.")
    if background.shape[1] != n_features:
        raise ValueError(
            f"Expected background with {n_features} features, "
            f"got {background.shape[1]}."
        )
    return background


def _scalar_predictions(model_predict: PredictionFunction, X: np.ndarray) -> np.ndarray:
    predictions = np.asarray(model_predict(X), dtype=float)
    if predictions.ndim == 0:
        predictions = predictions.reshape(1)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if predictions.ndim != 1:
        raise ValueError(
            "This example supports scalar-output models only. "
            f"Got prediction shape {predictions.shape}."
        )
    if predictions.shape[0] != X.shape[0]:
        raise ValueError(
            "model_predict must return one scalar prediction per input row."
        )
    return predictions


def _mask_from_bits(mask_bits: int, n_features: int) -> np.ndarray:
    return np.array(
        [(mask_bits >> feature_idx) & 1 for feature_idx in range(n_features)],
        dtype=bool,
    )


def exact_shap_values(
    model_predict: PredictionFunction,
    instance: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return exact interventional SHAP values and the expected value.

    Parameters
    ----------
    model_predict:
        Function that accepts a 2D array of samples and returns one scalar
        prediction per row.
    instance:
        One row to explain.
    background:
        Baseline data used to fill missing coalition features. A 1D array is
        treated as one baseline row.
    """

    instance = np.asarray(instance, dtype=float)
    if instance.ndim != 1:
        raise ValueError("instance must be a 1D array.")

    n_features = instance.shape[0]
    background = _as_background_matrix(background, n_features)

    coalition_values = np.zeros(1 << n_features, dtype=float)
    for mask_bits in range(1 << n_features):
        mask = _mask_from_bits(mask_bits, n_features)
        samples = background.copy()
        samples[:, mask] = instance[mask]
        coalition_values[mask_bits] = float(
            np.mean(_scalar_predictions(model_predict, samples))
        )

    shap_values = np.zeros(n_features, dtype=float)
    denominator = math.factorial(n_features)

    for feature_idx in range(n_features):
        feature_bit = 1 << feature_idx
        for mask_bits in range(1 << n_features):
            if mask_bits & feature_bit:
                continue

            subset_size = mask_bits.bit_count()
            weight = (
                math.factorial(subset_size)
                * math.factorial(n_features - subset_size - 1)
                / denominator
            )
            shap_values[feature_idx] += weight * (
                coalition_values[mask_bits | feature_bit]
                - coalition_values[mask_bits]
            )

    expected_value = coalition_values[0]
    return shap_values, float(expected_value)


def main() -> None:
    model, _, _, instance, reference = build_example()
    shap_values, expected_value = exact_shap_values(
        model.predict,
        instance,
        reference,
    )

    print_example_result(
        title="Exact SHAP from scratch with NumPy",
        model=model,
        instance=instance,
        reference=reference,
        shap_values=shap_values,
        expected_value=expected_value,
    )


if __name__ == "__main__":
    main()
