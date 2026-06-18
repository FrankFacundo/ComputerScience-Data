"""The same example computed with the official shap package."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from common import build_example, print_example_result


def _import_shap_package():
    """Import the installed shap package, not the sibling ../shap.py file."""

    parent_dir = Path(__file__).resolve().parent.parent
    for path_entry in list(sys.path):
        resolved = Path(path_entry or ".").resolve()
        if resolved == parent_dir:
            sys.path.remove(path_entry)

    try:
        import shap  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The official shap package is not installed. Install it with:\n"
            "  python -m pip install shap"
        ) from exc

    return shap


def _normalize_shap_values(raw_values) -> np.ndarray:
    if isinstance(raw_values, list):
        if len(raw_values) != 1:
            raise ValueError("This example expects one scalar model output.")
        raw_values = raw_values[0]

    values = np.asarray(raw_values, dtype=float)
    values = np.squeeze(values)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D SHAP values, got shape {values.shape}.")
    return values


def shap_library_values(
    model_predict,
    instance: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute exact Kernel SHAP values with the official shap package."""

    shap = _import_shap_package()
    instance = np.asarray(instance, dtype=float)
    reference = np.asarray(reference, dtype=float)
    background = reference.reshape(1, -1)

    explainer = shap.KernelExplainer(model_predict, background)
    nsamples = 2 ** instance.shape[0]

    try:
        raw_values = explainer.shap_values(
            instance.reshape(1, -1),
            nsamples=nsamples,
            silent=True,
        )
    except TypeError:
        raw_values = explainer.shap_values(
            instance.reshape(1, -1),
            nsamples=nsamples,
        )

    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return _normalize_shap_values(raw_values), expected_value


def main() -> None:
    model, _, _, instance, reference = build_example()
    shap_values, expected_value = shap_library_values(
        model.predict,
        instance,
        reference,
    )

    print_example_result(
        title="SHAP values from the official shap package",
        model=model,
        instance=instance,
        reference=reference,
        shap_values=shap_values,
        expected_value=expected_value,
    )


if __name__ == "__main__":
    main()
