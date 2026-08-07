"""Numerically compare the NumPy STL implementation with Statsmodels.

The implementation under test imports only NumPy.  This file imports
Statsmodels solely to construct the requested reference result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from statsmodels.tsa.seasonal import STL as StatsmodelsSTL

from numpy_stl import NumpySTL


SCALE_TOLERANCE = 1e-4
WEIGHT_TOLERANCE = 1e-1


def make_example_series() -> np.ndarray:
    """Create a deterministic series with trend, seasonality, noise, and outliers."""

    rng = np.random.default_rng(2026)
    time = np.arange(144, dtype=np.float64)
    series = (
        100.0
        + 0.18 * time
        + 12.0 * np.sin(2.0 * np.pi * time / 12.0)
        + 3.0 * np.cos(4.0 * np.pi * time / 12.0)
        + rng.normal(0.0, 1.2, time.size)
    )
    series[[19, 70, 111]] += np.array([25.0, -30.0, 35.0])
    return series


def load_monthly_sales(csv_path: Path) -> np.ndarray:
    """Aggregate the tutorial's transaction CSV to monthly units with NumPy."""

    blocks_and_counts = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 5),
        dtype=np.float64,
    )
    month_index = blocks_and_counts[:, 0].astype(np.intp)
    return np.bincount(month_index, weights=blocks_and_counts[:, 1]).astype(np.float64)


def compare(series: np.ndarray, *, robust: bool) -> dict[str, float]:
    """Run both implementations, assert numerical parity, and return errors.

    STL is scale-equivariant, so component errors are assessed relative to the
    scale of the input. This matters for very short robust decompositions: the
    repeated bisquare reweighting can amplify machine-rounding differences in
    compiled versus interpreted loops even when the fitted curves overlap.
    """

    parameters = {
        "period": 12,
        "seasonal": 7,
        "seasonal_deg": 1,
        "trend_deg": 1,
        "low_pass_deg": 1,
        "robust": robust,
        "seasonal_jump": 1,
        "trend_jump": 1,
        "low_pass_jump": 1,
    }

    numpy_model = NumpySTL(series, **parameters)
    reference_model = StatsmodelsSTL(series, **parameters)
    numpy_result = numpy_model.fit()
    reference_result = reference_model.fit()

    if numpy_model.config != reference_model.config:
        raise AssertionError(
            f"Effective configurations differ:\n"
            f"NumPy:      {numpy_model.config}\n"
            f"Statsmodels:{reference_model.config}"
        )

    pairs = {
        "seasonal": (numpy_result.seasonal, np.asarray(reference_result.seasonal)),
        "trend": (numpy_result.trend, np.asarray(reference_result.trend)),
        "resid": (numpy_result.resid, np.asarray(reference_result.resid)),
        "weights": (numpy_result.weights, np.asarray(reference_result.weights)),
    }

    errors: dict[str, float] = {}
    for component, (actual, expected) in pairs.items():
        errors[component] = float(np.max(np.abs(actual - expected)))

    value_scale = max(float(np.max(np.abs(series))), 1.0)
    value_limit = SCALE_TOLERANCE * value_scale
    for component in ("seasonal", "trend", "resid"):
        if errors[component] > value_limit:
            raise AssertionError(
                f"{component} differs from Statsmodels by {errors[component]:.6g}; "
                f"allowed scale-relative error is {value_limit:.6g}"
            )
    if errors["weights"] > WEIGHT_TOLERANCE:
        raise AssertionError(
            f"robustness weights differ by {errors['weights']:.6g}; "
            f"allowed error is {WEIGHT_TOLERANCE:.6g}"
        )

    reconstruction_error = float(
        np.max(
            np.abs(
                numpy_result.observed
                - numpy_result.trend
                - numpy_result.seasonal
                - numpy_result.resid
            )
        )
    )
    errors["reconstruction"] = reconstruction_error
    errors["value_limit"] = value_limit
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sales-csv",
        type=Path,
        help="Optional sales_train.csv; aggregates item_cnt_day by date_block_num.",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Use Statsmodels-compatible robust outer iterations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sales_csv is None:
        series = make_example_series()
        source = "deterministic synthetic series"
    else:
        series = load_monthly_sales(args.sales_csv)
        source = str(args.sales_csv)

    errors = compare(series, robust=args.robust)

    print(f"Input: {source}")
    print(f"Observations: {series.size}; robust={args.robust}")
    print("Maximum absolute differences")
    for component in ("seasonal", "trend", "resid", "weights", "reconstruction"):
        error = errors[component]
        print(f"  {component:14s} {error:.3e}")
    print(
        f"Component limit: {errors['value_limit']:.3e} "
        f"({SCALE_TOLERANCE:g} × maximum input magnitude)"
    )
    print(f"Weight limit:    {WEIGHT_TOLERANCE:.3e}")
    print("PASS: NumPy STL and Statsmodels STL are numerically equivalent.")


if __name__ == "__main__":
    main()
