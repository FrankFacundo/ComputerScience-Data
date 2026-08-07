"""Verify NumPy ACF, ADF, and KPSS outputs against Statsmodels."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
from statsmodels.tsa.stattools import acf as statsmodels_acf
from statsmodels.tsa.stattools import adfuller as statsmodels_adfuller
from statsmodels.tsa.stattools import kpss as statsmodels_kpss

from numpy_stattools import acf, adfuller, kpss


TOLERANCE = 1e-12
SEASON_LENGTH = 12


def example_series() -> np.ndarray:
    rng = np.random.default_rng(2026)
    innovations = rng.normal(0.0, 1.0, 180)
    values = np.empty_like(innovations)
    values[0] = innovations[0]
    for index in range(1, values.size):
        values[index] = 0.72 * values[index - 1] + innovations[index]
    return 100.0 + 0.04 * np.arange(values.size) + values


def load_monthly_sales(csv_path: Path) -> np.ndarray:
    blocks_and_counts = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=(1, 5),
        dtype=np.float64,
    )
    month_index = blocks_and_counts[:, 0].astype(np.intp)
    return np.bincount(month_index, weights=blocks_and_counts[:, 1]).astype(np.float64)


def transformed_series(values: np.ndarray) -> dict[str, np.ndarray]:
    first_difference = np.diff(values)
    if values.size <= SEASON_LENGTH + 2:
        return {"Series": values}
    return {
        "Raw": values,
        "First difference": first_difference,
        "Seasonal difference (12)": values[SEASON_LENGTH:] - values[:-SEASON_LENGTH],
        "First + seasonal difference": (
            first_difference[SEASON_LENGTH:] - first_difference[:-SEASON_LENGTH]
        ),
    }


def _maximum_critical_error(
    actual: dict[str, float], expected: dict[str, float]
) -> float:
    return max(abs(actual[key] - expected[key]) for key in actual)


def compare_one(label: str, values: np.ndarray) -> dict[str, float | int | str]:
    nlags = min(12, values.size - 1)
    numpy_acf = acf(values, nlags=nlags, fft=True)
    reference_acf = statsmodels_acf(values, nlags=nlags, fft=True)
    np.testing.assert_allclose(numpy_acf, reference_acf, rtol=0.0, atol=TOLERANCE)

    numpy_adf = adfuller(values, autolag="AIC")
    reference_adf = statsmodels_adfuller(values, autolag="AIC")
    np.testing.assert_allclose(numpy_adf[:2], reference_adf[:2], rtol=0.0, atol=TOLERANCE)
    if numpy_adf[2:4] != reference_adf[2:4]:
        raise AssertionError(f"ADF lag/nobs mismatch for {label}")
    np.testing.assert_allclose(numpy_adf[5], reference_adf[5], rtol=0.0, atol=TOLERANCE)
    adf_critical_error = _maximum_critical_error(numpy_adf[4], reference_adf[4])
    if adf_critical_error > TOLERANCE:
        raise AssertionError(f"ADF critical values differ for {label}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numpy_kpss = kpss(values, regression="c", nlags="auto")
        reference_kpss = statsmodels_kpss(values, regression="c", nlags="auto")
    np.testing.assert_allclose(
        numpy_kpss[:2], reference_kpss[:2], rtol=0.0, atol=TOLERANCE
    )
    if numpy_kpss[2] != reference_kpss[2]:
        raise AssertionError(f"KPSS lag mismatch for {label}")
    kpss_critical_error = _maximum_critical_error(numpy_kpss[3], reference_kpss[3])
    if kpss_critical_error > TOLERANCE:
        raise AssertionError(f"KPSS critical values differ for {label}")

    return {
        "series": label,
        "n": values.size,
        "ACF max error": float(np.max(np.abs(numpy_acf - reference_acf))),
        "ADF statistic error": abs(numpy_adf[0] - reference_adf[0]),
        "ADF p-value error": abs(numpy_adf[1] - reference_adf[1]),
        "KPSS statistic error": abs(numpy_kpss[0] - reference_kpss[0]),
        "KPSS p-value error": abs(numpy_kpss[1] - reference_kpss[1]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sales-csv",
        type=Path,
        help="Optional tutorial sales_train.csv; otherwise uses a deterministic example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = example_series() if args.sales_csv is None else load_monthly_sales(args.sales_csv)
    source = "deterministic example" if args.sales_csv is None else str(args.sales_csv)
    print(f"Input: {source}")
    print(f"Required absolute tolerance: {TOLERANCE:g}\n")

    for label, series in transformed_series(values).items():
        result = compare_one(label, series)
        print(f"{label} (n={result['n']})")
        for metric in (
            "ACF max error",
            "ADF statistic error",
            "ADF p-value error",
            "KPSS statistic error",
            "KPSS p-value error",
        ):
            print(f"  {metric:23s} {result[metric]:.3e}")

    print("\nPASS: all NumPy results match Statsmodels.")


if __name__ == "__main__":
    main()

