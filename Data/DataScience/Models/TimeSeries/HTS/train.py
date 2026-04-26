"""Train a bottom-up hierarchical time series forecaster.

By default the script downloads the public Australian tourism dataset from
Rdatasets into ./data, aggregates trips over visit purpose, and trains a global
GRU or Transformer on regional series. Forecasts are coherent because state and
national forecasts are reconciled as sums of bottom-level regional forecasts.

Example:

    python train.py --epochs 50 --horizon 4 --lookback 12
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hts_model import (
    ForecastWindowDataset,
    build_forecaster,
    build_tourism_panel,
    evaluate_model,
    resolve_dataset_file,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hierarchical tourism forecasting model.")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).with_name("data")),
        help="Directory where the downloaded tourism.csv dataset is stored.",
    )
    parser.add_argument("--data-csv", default=None, help="Optional explicit path to a tourism CSV.")
    parser.add_argument("--no-download", action="store_true", help="Fail if the dataset is missing.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading the dataset.")
    parser.add_argument("--model-output", default=str(Path(__file__).with_name("tourism_hts_checkpoint.pt")))
    parser.add_argument("--metrics-json", default=str(Path(__file__).with_name("tourism_hts_metrics.json")))
    parser.add_argument("--max-series", type=int, default=0, help="Optional cap for quick local tests. 0 uses all regions.")
    parser.add_argument("--model-type", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--lookback", type=int, default=12, help="Historical quarters used as encoder input.")
    parser.add_argument("--horizon", type=int, default=4, help="Forecast horizon in quarters.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--ff-multiplier", type=int, default=4, help="Transformer feed-forward width multiplier.")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    train_dataset: ForecastWindowDataset,
    val_dataset: ForecastWindowDataset,
    panel,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    device: torch.device,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_rmse = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for x, series_id, target, _ in train_loader:
            x = x.to(device)
            series_id = series_id.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            prediction = model(x, series_id)
            loss = loss_fn(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_rows = int(x.shape[0])
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows

        train_loss = total_loss / max(total_rows, 1)
        should_eval = epoch == 1 or epoch == epochs or epoch % max(eval_every, 1) == 0
        if should_eval:
            val_metrics = evaluate_model(model, val_dataset, panel, device, batch_size=batch_size)
            val_rmse = float(val_metrics["bottom"]["rmse"])
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_bottom_rmse": val_rmse,
                    "val_state_rmse": float(val_metrics["state"]["rmse"]),
                    "val_total_rmse": float(val_metrics["total"]["rmse"]),
                }
            )
            if verbose:
                print(
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                    f"val_bottom_rmse={val_rmse:.3f} | "
                    f"val_state_rmse={val_metrics['state']['rmse']:.3f} | "
                    f"val_total_rmse={val_metrics['total']['rmse']:.3f}"
                )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpointable model state.")

    model.load_state_dict(best_state)
    model = model.to(device)
    return model, history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    data_path = resolve_dataset_file(
        data_dir=data_dir,
        data_csv=args.data_csv,
        no_download=args.no_download,
        force_download=args.force_download,
    )

    max_series = None if args.max_series <= 0 else args.max_series
    panel = build_tourism_panel(data_path, max_series=max_series, scaler_train_ratio=args.train_ratio)

    print("Loaded tourism hierarchy:")
    print(f"  data: {data_path}")
    for key, value in panel.metadata.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    train_dataset = ForecastWindowDataset(
        panel=panel,
        lookback=args.lookback,
        horizon=args.horizon,
        split="train",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    val_dataset = ForecastWindowDataset(
        panel=panel,
        lookback=args.lookback,
        horizon=args.horizon,
        split="validation",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    test_dataset = ForecastWindowDataset(
        panel=panel,
        lookback=args.lookback,
        horizon=args.horizon,
        split="test",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print("Window samples:")
    print(f"  train: {len(train_dataset):,}")
    print(f"  validation: {len(val_dataset):,}")
    print(f"  test: {len(test_dataset):,}")

    device = choose_device()
    model_config: dict[str, Any] = {
        "num_series": int(len(panel.hierarchy.bottom_ids)),
        "input_dim": 3,
        "embedding_dim": int(args.embedding_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "horizon": int(args.horizon),
    }
    if args.model_type == "transformer":
        model_config.update(
            {
                "num_heads": int(args.num_heads),
                "max_lookback": int(args.lookback),
                "ff_multiplier": int(args.ff_multiplier),
            }
        )
    model = build_forecaster(args.model_type, model_config)
    model, history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        panel=panel,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        device=device,
    )

    metrics: dict[str, Any] = {
        "device": device.type,
        "model_type": args.model_type,
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "train": evaluate_model(model, train_dataset, panel, device, batch_size=args.batch_size),
        "validation": evaluate_model(model, val_dataset, panel, device, batch_size=args.batch_size),
        "test": evaluate_model(model, test_dataset, panel, device, batch_size=args.batch_size),
        "history": history,
    }
    metadata = {
        **panel.metadata,
        "data_csv": str(data_path),
        "model_type": args.model_type,
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "batch_size": int(args.batch_size),
        "source": panel.preprocessing["dataset_doc_url"],
    }

    model_output = Path(args.model_output).expanduser().resolve()
    save_checkpoint(
        path=model_output,
        model=model,
        model_config=model_config,
        preprocessing=panel.preprocessing,
        metadata=metadata,
        metrics=metrics,
        model_type=args.model_type,
    )
    print(f"Saved model checkpoint to: {model_output}")

    report = {
        "dataset": {
            "csv": str(data_path),
            "source": panel.preprocessing["dataset_doc_url"],
            "download_url": panel.preprocessing["dataset_url"],
        },
        "metadata": metadata,
        "model_type": args.model_type,
        "model_config": model_config,
        "metrics": metrics,
    }
    if args.metrics_json:
        metrics_path = Path(args.metrics_json).expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, indent=2))
        print(f"Saved metrics to: {metrics_path}")

    print("Final metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
