"""Shared NumPy-only data and model helpers for the SHAP examples."""

from __future__ import annotations

import numpy as np


class NumpyLinearRegression:
    """Small ordinary least-squares regressor implemented with NumPy."""

    def __init__(self) -> None:
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        design = np.column_stack([np.ones(X.shape[0]), X])
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept_ = float(params[0])
        self.coef_ = params[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise ValueError("The model must be fitted before calling predict.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Expected {self.coef_.shape[0]} features, got {X.shape[1]}."
            )

        return self.intercept_ + X @ self.coef_


def make_regression_problem(
    *, seed: int = 0, n_samples: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Create the same two-feature regression problem for every script."""

    rng = np.random.default_rng(seed)
    X = 2.0 * rng.random((n_samples, 2))
    noise = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    y = 3.0 + 4.0 * X[:, 0] + 5.0 * X[:, 1] + noise
    return X, y


def build_example() -> tuple[
    NumpyLinearRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return a fitted model, the data, one instance, and one baseline row."""

    X, y = make_regression_problem()
    model = NumpyLinearRegression().fit(X, y)
    instance = X[0]
    reference = X.mean(axis=0)
    return model, X, y, instance, reference


def print_example_result(
    *,
    title: str,
    model: NumpyLinearRegression,
    instance: np.ndarray,
    reference: np.ndarray,
    shap_values: np.ndarray,
    expected_value: float,
) -> None:
    """Print the quantities that should satisfy the SHAP additivity identity."""

    prediction = float(model.predict(instance)[0])
    reconstructed_prediction = expected_value + float(np.sum(shap_values))

    print(title)
    print("=" * len(title))
    print("instance:       ", np.array2string(instance, precision=6))
    print("reference:      ", np.array2string(reference, precision=6))
    print("expected value: ", f"{expected_value:.12f}")
    print("prediction:     ", f"{prediction:.12f}")
    print("SHAP values:    ", np.array2string(shap_values, precision=12))
    print("base + sum(phi):", f"{reconstructed_prediction:.12f}")
