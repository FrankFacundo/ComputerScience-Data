"""Utilities for hierarchical forecasting on Australian tourism data.

The default dataset is the public Rdatasets mirror of the tsibble tourism
dataset. It contains quarterly Australian domestic overnight trips by state,
region, purpose, and quarter. This module aggregates over purpose, learns
bottom-level regional forecasts, and reconciles higher levels with bottom-up
sums:

    Australia total -> State -> Region
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset


DATASET_FILENAME = "tourism.csv"
RDATASETS_TOURISM_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/tsibble/tourism.csv"
RDATASETS_TOURISM_DOC_URL = "https://vincentarelbundock.github.io/Rdatasets/doc/tsibble/tourism.html"


STATE_ABBREVIATIONS = {
    "Australian Capital Territory": "ACT",
    "New South Wales": "NSW",
    "Northern Territory": "NT",
    "Queensland": "QLD",
    "South Australia": "SA",
    "Tasmania": "TAS",
    "Victoria": "VIC",
    "Western Australia": "WA",
}


@dataclass(frozen=True)
class Hierarchy:
    bottom_ids: list[str]
    states: list[str]
    bottom_to_state: list[int]
    bottom_state: list[str]
    bottom_region: list[str]


@dataclass(frozen=True)
class TourismPanel:
    values: np.ndarray
    quarters: list[str]
    quarter_numbers: np.ndarray
    hierarchy: Hierarchy
    preprocessing: dict[str, Any]
    metadata: dict[str, Any]


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        panel: TourismPanel,
        lookback: int,
        horizon: int,
        split: str,
        train_ratio: float,
        val_ratio: float,
    ) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1.")
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test.")
        if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

        self.panel = panel
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.split = split
        self.inputs, self.targets, self.series_ids, self.target_starts = self._build_samples(train_ratio, val_ratio)

        if len(self.series_ids) == 0:
            raise ValueError(
                f"No {split} samples could be built. Reduce --lookback/--horizon or adjust split ratios."
            )

    def _build_samples(
        self,
        train_ratio: float,
        val_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.log1p(np.clip(self.panel.values, a_min=0.0, a_max=None)).astype(np.float32)
        mean = np.asarray(self.panel.preprocessing["target_log_mean"], dtype=np.float32)
        scale = np.asarray(self.panel.preprocessing["target_log_scale"], dtype=np.float32)
        scaled_values = ((values - mean.reshape(1, -1)) / scale.reshape(1, -1)).astype(np.float32)

        season = quarter_features(self.panel.quarter_numbers)
        num_times, num_series = scaled_values.shape
        train_end = int(math.floor(num_times * train_ratio))
        val_end = int(math.floor(num_times * (train_ratio + val_ratio)))
        train_end = max(train_end, self.lookback + self.horizon)
        val_end = max(val_end, train_end + self.horizon)
        val_end = min(val_end, num_times)

        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        series_ids: list[int] = []
        target_starts: list[int] = []

        for target_start in range(self.lookback, num_times - self.horizon + 1):
            target_end = target_start + self.horizon
            if self.split == "train" and target_end > train_end:
                continue
            if self.split == "validation" and not (target_start >= train_end and target_end <= val_end):
                continue
            if self.split == "test" and target_start < val_end:
                continue

            history_slice = slice(target_start - self.lookback, target_start)
            for series_idx in range(num_series):
                y_history = scaled_values[history_slice, series_idx : series_idx + 1]
                x = np.concatenate([y_history, season[history_slice]], axis=1)
                inputs.append(x.astype(np.float32))
                targets.append(scaled_values[target_start:target_end, series_idx].astype(np.float32))
                series_ids.append(series_idx)
                target_starts.append(target_start)

        return (
            np.stack(inputs, axis=0) if inputs else np.empty((0, self.lookback, 3), dtype=np.float32),
            np.stack(targets, axis=0) if targets else np.empty((0, self.horizon), dtype=np.float32),
            np.asarray(series_ids, dtype=np.int64),
            np.asarray(target_starts, dtype=np.int64),
        )

    def __len__(self) -> int:
        return int(len(self.series_ids))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.inputs[idx]),
            torch.tensor(self.series_ids[idx], dtype=torch.long),
            torch.from_numpy(self.targets[idx]),
            torch.tensor(self.target_starts[idx], dtype=torch.long),
        )


class BottomUpGRUForecaster(nn.Module):
    def __init__(
        self,
        num_series: int,
        input_dim: int = 3,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
        horizon: int = 4,
    ) -> None:
        super().__init__()
        if num_series < 1:
            raise ValueError("num_series must be >= 1.")
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.series_embedding = nn.Embedding(num_series, embedding_dim)
        self.gru = nn.GRU(
            input_size=input_dim + embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor, series_id: torch.Tensor) -> torch.Tensor:
        embedding = self.series_embedding(series_id)
        embedding = embedding.unsqueeze(1).expand(-1, x.shape[1], -1)
        model_input = torch.cat([x, embedding], dim=-1)
        output, _ = self.gru(model_input)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


class BottomUpTransformerForecaster(nn.Module):
    def __init__(
        self,
        num_series: int,
        input_dim: int = 3,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
        horizon: int = 4,
        max_lookback: int = 64,
        ff_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if num_series < 1:
            raise ValueError("num_series must be >= 1.")
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if max_lookback < 1:
            raise ValueError("max_lookback must be >= 1.")

        self.series_embedding = nn.Embedding(num_series, embedding_dim)
        self.input_projection = nn.Linear(input_dim + embedding_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_lookback, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor, series_id: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.position_embedding.shape[1]:
            raise ValueError(
                f"Input lookback {x.shape[1]} exceeds max_lookback {self.position_embedding.shape[1]}."
            )

        embedding = self.series_embedding(series_id)
        embedding = embedding.unsqueeze(1).expand(-1, x.shape[1], -1)
        model_input = torch.cat([x, embedding], dim=-1)
        encoded_input = self.input_projection(model_input)
        encoded_input = encoded_input + self.position_embedding[:, : x.shape[1], :]
        encoded = self.encoder(encoded_input)
        return self.head(encoded[:, -1, :])


def build_forecaster(model_type: str, model_config: dict[str, Any]) -> nn.Module:
    if model_type == "gru":
        return BottomUpGRUForecaster(**model_config)
    if model_type == "transformer":
        return BottomUpTransformerForecaster(**model_config)
    raise ValueError(f"Unsupported model_type: {model_type!r}. Expected 'gru' or 'transformer'.")


def infer_model_type(model: nn.Module) -> str:
    if isinstance(model, BottomUpTransformerForecaster):
        return "transformer"
    if isinstance(model, BottomUpGRUForecaster):
        return "gru"
    raise ValueError(f"Unsupported model class: {type(model).__name__}")


def download_tourism_dataset(data_dir: Path, force_download: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / DATASET_FILENAME
    if output_path.exists() and not force_download:
        return output_path.resolve()

    tmp_path = output_path.with_suffix(".tmp")
    try:
        with requests.get(RDATASETS_TOURISM_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        handle.write(chunk)
        tmp_path.replace(output_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Could not download tourism dataset from {RDATASETS_TOURISM_URL}. "
            "Check your internet connection or pass --data-csv."
        ) from exc
    return output_path.resolve()


def resolve_dataset_file(
    data_dir: Path,
    data_csv: str | None = None,
    no_download: bool = False,
    force_download: bool = False,
) -> Path:
    if data_csv:
        path = Path(data_csv).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {path}")
        return path.resolve()

    path = data_dir / DATASET_FILENAME
    if path.exists() and not force_download:
        return path.resolve()
    if no_download:
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    return download_tourism_dataset(data_dir, force_download=force_download)


def read_tourism_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.rename({column: column.strip() for column in df.columns})
    if "rownames" in df.columns:
        df = df.drop("rownames")

    required = {"Quarter", "Region", "State", "Trips"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return df.with_columns(
        [
            pl.col("Quarter").cast(pl.Utf8).str.strip_chars(),
            pl.col("Region").cast(pl.Utf8).str.strip_chars(),
            pl.col("State").cast(pl.Utf8).str.strip_chars().replace(STATE_ABBREVIATIONS),
            pl.col("Trips").cast(pl.Float64, strict=False).fill_null(0.0),
        ]
    ).with_columns(
        pl.col("Quarter").map_elements(parse_quarter_number, return_dtype=pl.Int64).alias("quarter_number")
    )


def parse_quarter_number(value: str) -> int:
    text = str(value).strip().replace("-", " ")
    parts = text.split()
    if len(parts) >= 2 and parts[1].upper().startswith("Q"):
        return int(parts[0]) * 4 + int(parts[1].upper().replace("Q", "")) - 1
    if "Q" in text.upper():
        year, quarter = text.upper().split("Q", maxsplit=1)
        return int(year.strip()) * 4 + int(quarter.strip()) - 1
    raise ValueError(f"Could not parse quarter value: {value!r}")


def quarter_label_from_number(quarter_number: int) -> str:
    year = int(quarter_number) // 4
    quarter = int(quarter_number) % 4 + 1
    return f"{year} Q{quarter}"


def quarter_features(quarter_numbers: np.ndarray) -> np.ndarray:
    quarter_index = (quarter_numbers.astype(np.int64) % 4).astype(np.float32)
    angle = 2.0 * np.pi * quarter_index / 4.0
    return np.column_stack([np.sin(angle), np.cos(angle)]).astype(np.float32)


def build_tourism_panel(
    csv_path: Path,
    max_series: int | None = None,
    existing_preprocessing: dict[str, Any] | None = None,
    scaler_train_ratio: float = 0.70,
) -> TourismPanel:
    raw = read_tourism_csv(csv_path)
    bottom = (
        raw.group_by(["quarter_number", "Quarter", "State", "Region"])
        .agg(pl.col("Trips").sum())
        .with_columns(pl.concat_str(["State", pl.lit("::"), "Region"]).alias("bottom_id"))
        .sort(["quarter_number", "State", "Region"])
    )

    available_bottom_ids = set(bottom["bottom_id"].unique().to_list())
    if existing_preprocessing is not None:
        selected_bottom_ids = list(existing_preprocessing["bottom_ids"])
        missing = [bottom_id for bottom_id in selected_bottom_ids if bottom_id not in available_bottom_ids]
        if missing:
            raise ValueError(f"Input data is missing bottom series used by checkpoint: {missing[:5]}")
    else:
        if max_series is not None and max_series > 0:
            selected_bottom_ids = (
                bottom.group_by("bottom_id")
                .agg(pl.col("Trips").sum())
                .sort("Trips", descending=True)
                .head(max_series)["bottom_id"]
                .to_list()
            )
        else:
            selected_bottom_ids = list(available_bottom_ids)
        selected_bottom_ids = sorted(selected_bottom_ids)

    bottom = bottom.filter(pl.col("bottom_id").is_in(selected_bottom_ids))
    pivot = (
        bottom.pivot(
            on="bottom_id",
            index=["quarter_number", "Quarter"],
            values="Trips",
            aggregate_function="sum",
        )
        .sort("quarter_number")
        .fill_null(0.0)
    )
    pivot = pivot.select(["quarter_number", "Quarter", *selected_bottom_ids])

    quarter_numbers = pivot["quarter_number"].to_numpy().astype(np.int64)
    quarters = pivot["Quarter"].cast(pl.Utf8).to_list()
    values = pivot.select(selected_bottom_ids).to_numpy().astype(np.float32)

    bottom_state: list[str] = []
    bottom_region: list[str] = []
    for bottom_id in selected_bottom_ids:
        state, region = bottom_id.split("::", maxsplit=1)
        bottom_state.append(state)
        bottom_region.append(region)
    bottom_ids = list(selected_bottom_ids)
    states = sorted(set(bottom_state))
    state_lookup = {state: idx for idx, state in enumerate(states)}
    bottom_to_state = [state_lookup[state] for state in bottom_state]
    hierarchy = Hierarchy(
        bottom_ids=bottom_ids,
        states=states,
        bottom_to_state=bottom_to_state,
        bottom_state=bottom_state,
        bottom_region=bottom_region,
    )

    if existing_preprocessing is not None:
        preprocessing = dict(existing_preprocessing)
    else:
        target_log = np.log1p(np.clip(values, a_min=0.0, a_max=None)).astype(np.float32)
        train_end = max(1, int(math.floor(len(values) * scaler_train_ratio)))
        mean = target_log[:train_end].mean(axis=0)
        scale = target_log[:train_end].std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        preprocessing = {
            "bottom_ids": bottom_ids,
            "states": states,
            "bottom_to_state": bottom_to_state,
            "bottom_state": bottom_state,
            "bottom_region": bottom_region,
            "target_log_mean": mean.astype(np.float32).tolist(),
            "target_log_scale": scale.astype(np.float32).tolist(),
            "target_transform": "per-series-standardized-log1p",
            "dataset_url": RDATASETS_TOURISM_URL,
            "dataset_doc_url": RDATASETS_TOURISM_DOC_URL,
        }

    metadata = {
        "num_quarters": int(len(quarters)),
        "first_quarter": quarters[0],
        "last_quarter": quarters[-1],
        "num_bottom_series": int(values.shape[1]),
        "num_state_series": int(len(states)),
        "num_total_series": 1,
        "hierarchy": "total/state/region",
    }
    return TourismPanel(
        values=values,
        quarters=quarters,
        quarter_numbers=quarter_numbers,
        hierarchy=hierarchy,
        preprocessing=preprocessing,
        metadata=metadata,
    )


def inverse_transform_predictions(
    scaled_predictions: np.ndarray,
    series_ids: np.ndarray,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    mean = np.asarray(preprocessing["target_log_mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["target_log_scale"], dtype=np.float32)
    log_values = scaled_predictions * scale[series_ids, None] + mean[series_ids, None]
    return np.clip(np.expm1(log_values), a_min=0.0, a_max=None)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    ratio = np.divide(
        np.abs(actual - predicted),
        denominator,
        out=np.zeros_like(actual, dtype=np.float64),
        where=denominator > 0,
    )
    return float(200.0 * np.mean(ratio))


def metric_report(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "smape": smape(actual, predicted),
    }


def aggregate_bottom(values: np.ndarray, hierarchy: Hierarchy) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate bottom forecasts to state and total levels.

    Input shape can be (..., num_bottom_series). The returned state shape is
    (..., num_states), and total shape is (..., 1).
    """

    state_values = np.zeros((*values.shape[:-1], len(hierarchy.states)), dtype=np.float64)
    for bottom_idx, state_idx in enumerate(hierarchy.bottom_to_state):
        state_values[..., state_idx] += values[..., bottom_idx]
    total_values = state_values.sum(axis=-1, keepdims=True)
    return state_values, total_values


def evaluate_model(
    model: nn.Module,
    dataset: ForecastWindowDataset,
    panel: TourismPanel,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    scaled_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    series_ids: list[np.ndarray] = []
    target_starts: list[np.ndarray] = []

    with torch.no_grad():
        for x, sid, y, starts in loader:
            x = x.to(device)
            sid = sid.to(device)
            prediction = model(x, sid).detach().cpu().numpy()
            scaled_predictions.append(prediction)
            targets.append(y.numpy())
            series_ids.append(sid.detach().cpu().numpy())
            target_starts.append(starts.numpy())

    scaled_pred = np.concatenate(scaled_predictions, axis=0)
    scaled_target = np.concatenate(targets, axis=0)
    sid = np.concatenate(series_ids, axis=0)
    starts = np.concatenate(target_starts, axis=0)

    pred_bottom_rows = inverse_transform_predictions(scaled_pred, sid, panel.preprocessing)
    actual_bottom_rows = inverse_transform_predictions(scaled_target, sid, panel.preprocessing)
    bottom_metrics = metric_report(actual_bottom_rows, pred_bottom_rows)

    unique_starts = sorted(set(int(item) for item in starts.tolist()))
    num_series = len(panel.hierarchy.bottom_ids)
    pred_cube = np.full((len(unique_starts), dataset.horizon, num_series), np.nan, dtype=np.float64)
    actual_cube = np.full_like(pred_cube, np.nan)
    start_lookup = {start: idx for idx, start in enumerate(unique_starts)}
    for row_idx, start in enumerate(starts):
        cube_idx = start_lookup[int(start)]
        pred_cube[cube_idx, :, sid[row_idx]] = pred_bottom_rows[row_idx]
        actual_cube[cube_idx, :, sid[row_idx]] = actual_bottom_rows[row_idx]

    if np.isnan(pred_cube).any() or np.isnan(actual_cube).any():
        raise RuntimeError("Evaluation cube is incomplete; expected every split origin to contain every series.")

    pred_state, pred_total = aggregate_bottom(pred_cube, panel.hierarchy)
    actual_state, actual_total = aggregate_bottom(actual_cube, panel.hierarchy)
    return {
        "num_samples": int(len(dataset)),
        "num_forecast_origins": int(len(unique_starts)),
        "bottom": bottom_metrics,
        "state": metric_report(actual_state, pred_state),
        "total": metric_report(actual_total, pred_total),
        "max_reconciliation_error": float(
            max(
                np.abs(pred_total.squeeze(-1) - pred_state.sum(axis=-1)).max(initial=0.0),
                np.abs(pred_state.sum(axis=-1) - pred_cube.sum(axis=-1)).max(initial=0.0),
            )
        ),
    }


def forecast_next(
    model: nn.Module,
    panel: TourismPanel,
    lookback: int,
    horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if len(panel.quarters) < lookback:
        raise ValueError(f"Need at least {lookback} quarters for inference.")

    values = np.log1p(np.clip(panel.values, a_min=0.0, a_max=None)).astype(np.float32)
    mean = np.asarray(panel.preprocessing["target_log_mean"], dtype=np.float32)
    scale = np.asarray(panel.preprocessing["target_log_scale"], dtype=np.float32)
    scaled_values = ((values - mean.reshape(1, -1)) / scale.reshape(1, -1)).astype(np.float32)
    season = quarter_features(panel.quarter_numbers)

    history_slice = slice(len(panel.quarters) - lookback, len(panel.quarters))
    x_rows: list[np.ndarray] = []
    series_ids = np.arange(len(panel.hierarchy.bottom_ids), dtype=np.int64)
    for series_idx in series_ids:
        y_history = scaled_values[history_slice, series_idx : series_idx + 1]
        x_rows.append(np.concatenate([y_history, season[history_slice]], axis=1).astype(np.float32))

    x = torch.from_numpy(np.stack(x_rows, axis=0)).to(device)
    sid = torch.from_numpy(series_ids).to(device)
    model.eval()
    with torch.no_grad():
        scaled_prediction = model(x, sid).detach().cpu().numpy()

    bottom_forecast = inverse_transform_predictions(scaled_prediction, series_ids, panel.preprocessing)
    next_quarter_numbers = np.arange(panel.quarter_numbers[-1] + 1, panel.quarter_numbers[-1] + horizon + 1)
    next_quarters = [quarter_label_from_number(item) for item in next_quarter_numbers]
    return bottom_forecast, next_quarter_numbers, next_quarters


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    model_type: str | None = None,
) -> None:
    checkpoint = {
        "model_type": model_type or infer_model_type(model),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": model_config,
        "preprocessing": preprocessing,
        "metadata": metadata,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "gru")
    model = build_forecaster(model_type, checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
