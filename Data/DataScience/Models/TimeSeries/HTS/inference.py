"""Run inference with a trained hierarchical tourism forecaster.

The script loads the checkpoint produced by train.py, reads the latest tourism
history, forecasts regional trips, and writes coherent region/state/total
forecasts to CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch

from hts_model import (
    Hierarchy,
    aggregate_bottom,
    build_tourism_panel,
    forecast_next,
    load_checkpoint,
    resolve_dataset_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast Australian tourism hierarchy with a trained HTS model.")
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).with_name("tourism_hts_checkpoint.pt")),
        help="Path to the checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).with_name("data")),
        help="Directory where the downloaded tourism.csv dataset is stored.",
    )
    parser.add_argument("--data-csv", default=None, help="Optional explicit path to a tourism CSV.")
    parser.add_argument("--no-download", action="store_true", help="Fail if the dataset is missing.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading the dataset.")
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=0,
        help="Optional number of forecast steps to write. 0 uses the full trained horizon.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).with_name("tourism_hts_forecast.csv")),
        help="Path where forecasts are written.",
    )
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_output_frame(
    bottom_forecast,
    next_quarters: list[str],
    hierarchy: Hierarchy,
    forecast_origin: str,
    horizon_steps: int,
) -> pl.DataFrame:
    bottom_forecast = bottom_forecast[:, :horizon_steps]
    horizon_bottom = bottom_forecast.T
    state_forecast, total_forecast = aggregate_bottom(horizon_bottom, hierarchy)

    rows: list[dict[str, object]] = []
    for horizon_idx in range(horizon_steps):
        quarter = next_quarters[horizon_idx]
        rows.append(
            {
                "forecast_origin": forecast_origin,
                "forecast_quarter": quarter,
                "horizon_step": horizon_idx + 1,
                "level": "total",
                "state": "Australia",
                "region": "All",
                "series_id": "Australia",
                "forecast_trips_thousands": float(total_forecast[horizon_idx, 0]),
            }
        )

        for state_idx, state in enumerate(hierarchy.states):
            rows.append(
                {
                    "forecast_origin": forecast_origin,
                    "forecast_quarter": quarter,
                    "horizon_step": horizon_idx + 1,
                    "level": "state",
                    "state": state,
                    "region": "All",
                    "series_id": state,
                    "forecast_trips_thousands": float(state_forecast[horizon_idx, state_idx]),
                }
            )

        for series_idx, series_id in enumerate(hierarchy.bottom_ids):
            rows.append(
                {
                    "forecast_origin": forecast_origin,
                    "forecast_quarter": quarter,
                    "horizon_step": horizon_idx + 1,
                    "level": "region",
                    "state": hierarchy.bottom_state[series_idx],
                    "region": hierarchy.bottom_region[series_idx],
                    "series_id": series_id,
                    "forecast_trips_thousands": float(bottom_forecast[series_idx, horizon_idx]),
                }
            )

    return pl.DataFrame(rows)


def main() -> None:
    args = parse_args()
    device = choose_device()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first with train.py.")

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    preprocessing = checkpoint["preprocessing"]
    model_horizon = int(checkpoint["model_config"]["horizon"])
    lookback = int(checkpoint.get("metadata", {}).get("lookback", checkpoint.get("metrics", {}).get("lookback", 12)))
    horizon_steps = model_horizon if args.horizon_steps <= 0 else min(args.horizon_steps, model_horizon)

    data_path = resolve_dataset_file(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        data_csv=args.data_csv,
        no_download=args.no_download,
        force_download=args.force_download,
    )
    panel = build_tourism_panel(data_path, existing_preprocessing=preprocessing)

    bottom_forecast, _, next_quarters = forecast_next(
        model=model,
        panel=panel,
        lookback=lookback,
        horizon=model_horizon,
        device=device,
    )
    output = build_output_frame(
        bottom_forecast=bottom_forecast,
        next_quarters=next_quarters,
        hierarchy=panel.hierarchy,
        forecast_origin=panel.quarters[-1],
        horizon_steps=horizon_steps,
    )

    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(output_path)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Loaded data: {data_path}")
    print(f"Forecast origin: {panel.quarters[-1]}")
    print(f"Forecast quarters: {next_quarters[:horizon_steps]}")
    print(f"Rows written: {output.height:,}")
    print(f"Saved forecasts to: {output_path}")


if __name__ == "__main__":
    main()
