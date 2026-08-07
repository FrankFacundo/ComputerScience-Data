"""Notebook-style stationarity table using the NumPy implementations."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import polars as pl

from numpy_stattools import InterpolationWarning, adfuller, kpss


DEFAULT_SALES_CSV = Path(__file__).parents[1] / "basics/data/sales_train.csv"
SEASON_LENGTH = 12


def load_monthly_sales(csv_path: Path) -> np.ndarray:
    """Aggregate daily item counts into a regular date-block series with Polars."""

    monthly = (
        pl.scan_csv(csv_path)
        .group_by("date_block_num")
        .agg(pl.col("item_cnt_day").sum().alias("net_units"))
        .sort("date_block_num")
        .collect()
    )

    expected_blocks = np.arange(monthly.height)
    actual_blocks = monthly["date_block_num"].to_numpy()
    if not np.array_equal(actual_blocks, expected_blocks):
        raise ValueError("date_block_num must be contiguous before time-series analysis")
    return monthly["net_units"].to_numpy().astype(np.float64)


def transformations(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return the four transformations used by the tutorial notebook."""

    first_difference = np.diff(values)
    return {
        "Raw": values,
        "First difference": first_difference,
        "Seasonal difference (12)": values[SEASON_LENGTH:] - values[:-SEASON_LENGTH],
        "First + seasonal difference": (
            first_difference[SEASON_LENGTH:] - first_difference[:-SEASON_LENGTH]
        ),
    }


def stationarity_row(label: str, series: np.ndarray) -> dict[str, object]:
    """Produce one ADF/KPSS summary row, analogous to the tutorial function."""

    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    adf_result = adfuller(values, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_result = kpss(values, regression="c", nlags="auto")
    return {
        "transformation": label,
        "n": len(values),
        "ADF statistic": adf_result[0],
        "ADF p-value": adf_result[1],
        "ADF lags": adf_result[2],
        "KPSS statistic": kpss_result[0],
        "KPSS p-value": kpss_result[1],
        "KPSS lags": kpss_result[2],
    }


def stationarity_table(csv_path: Path) -> pl.DataFrame:
    values = load_monthly_sales(csv_path)
    table = pl.DataFrame(
        [
            stationarity_row(label, transformed)
            for label, transformed in transformations(values).items()
        ]
    )
    return table.with_columns(
        pl.col("ADF statistic", "ADF p-value", "KPSS statistic", "KPSS p-value").round(4)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales-csv", type=Path, default=DEFAULT_SALES_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(stationarity_table(args.sales_csv))


if __name__ == "__main__":
    main()

