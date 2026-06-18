"""Compare the NumPy SHAP implementation against the official shap package."""

from __future__ import annotations

import numpy as np

from common import build_example
from shap_from_scratch_numpy import exact_shap_values
from shap_library import shap_library_values


def main() -> None:
    model, _, _, instance, reference = build_example()

    scratch_values, scratch_expected = exact_shap_values(
        model.predict,
        instance,
        reference,
    )
    library_values, library_expected = shap_library_values(
        model.predict,
        instance,
        reference,
    )

    np.testing.assert_allclose(
        scratch_values,
        library_values,
        rtol=1e-8,
        atol=1e-8,
        err_msg="NumPy SHAP values differ from shap.KernelExplainer.",
    )
    np.testing.assert_allclose(
        scratch_expected,
        library_expected,
        rtol=1e-8,
        atol=1e-8,
        err_msg="Expected values differ.",
    )

    prediction = float(model.predict(instance)[0])
    np.testing.assert_allclose(
        scratch_expected + scratch_values.sum(),
        prediction,
        rtol=1e-8,
        atol=1e-8,
        err_msg="NumPy SHAP values do not reconstruct the prediction.",
    )
    np.testing.assert_allclose(
        library_expected + library_values.sum(),
        prediction,
        rtol=1e-8,
        atol=1e-8,
        err_msg="Official SHAP values do not reconstruct the prediction.",
    )

    print("PASS: NumPy implementation matches shap.KernelExplainer.")
    print("scratch values:", np.array2string(scratch_values, precision=12))
    print("library values:", np.array2string(library_values, precision=12))
    print("expected value:", f"{scratch_expected:.12f}")
    print("prediction:    ", f"{prediction:.12f}")


if __name__ == "__main__":
    main()
