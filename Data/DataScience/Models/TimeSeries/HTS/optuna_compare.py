"""Tune and compare GRU vs Transformer hierarchical forecasters with Optuna.

This script keeps the same bottom-up hierarchy as train.py:

    Australia total -> State -> Region

Optuna optimizes validation bottom-level RMSE for each architecture. The best
configuration for each model is then retrained and evaluated on train,
validation, and test splits. The comparison report includes bottom, state, and
total metrics so the architecture choice can be judged at every hierarchy level.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

try:
    import optuna
except ModuleNotFoundError:  # pragma: no cover - exercised only when dependency is missing.
    optuna = None

from hts_model import (
    ForecastWindowDataset,
    build_forecaster,
    build_tourism_panel,
    evaluate_model,
    resolve_dataset_file,
    save_checkpoint,
)
from train import choose_device, set_seed, train_model


MODEL_TYPES = ("gru", "transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize and compare GRU and Transformer HTS models.")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).with_name("data")),
        help="Directory where the downloaded tourism.csv dataset is stored.",
    )
    parser.add_argument("--data-csv", default=None, help="Optional explicit path to a tourism CSV.")
    parser.add_argument("--no-download", action="store_true", help="Fail if the dataset is missing.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading the dataset.")
    parser.add_argument("--max-series", type=int, default=0, help="Optional cap for quick local tests. 0 uses all regions.")
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--n-trials", type=int, default=12, help="Optuna trials per architecture.")
    parser.add_argument("--trial-epochs", type=int, default=15, help="Epochs per Optuna trial.")
    parser.add_argument("--final-epochs", type=int, default=50, help="Epochs for each best final model.")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).with_name("tourism_hts_model_comparison.json")),
        help="Path for the detailed comparison report.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).with_name("tourism_hts_model_comparison.csv")),
        help="Path for a compact comparison table.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(Path(__file__).with_name("optuna_checkpoints")),
        help="Directory where final best-model checkpoints are written.",
    )
    return parser.parse_args()


def ensure_optuna_available() -> None:
    if optuna is None:
        raise RuntimeError(
            "Optuna is not installed. Install HTS dependencies first:\n"
            "  pip install -r requirements.txt"
        )


def sample_hyperparameters(
    trial,
    model_type: str,
    num_series: int,
    horizon: int,
    lookback: int,
) -> dict[str, Any]:
    common_config: dict[str, Any] = {
        "num_series": num_series,
        "input_dim": 3,
        "embedding_dim": trial.suggest_categorical("embedding_dim", [4, 8, 16, 32]),
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim",
            [16, 32, 64, 96, 128] if model_type == "gru" else [32, 64, 96, 128],
        ),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.30),
        "horizon": horizon,
    }

    if model_type == "transformer":
        common_config.update(
            {
                "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
                "max_lookback": lookback,
                "ff_multiplier": trial.suggest_categorical("ff_multiplier", [2, 4]),
            }
        )

    return {
        "model_config": common_config,
        "learning_rate": trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }


def build_datasets(
    panel,
    lookback: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[ForecastWindowDataset, ForecastWindowDataset, ForecastWindowDataset]:
    train_dataset = ForecastWindowDataset(
        panel=panel,
        lookback=lookback,
        horizon=horizon,
        split="train",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    val_dataset = ForecastWindowDataset(
        panel=panel,
        lookback=lookback,
        horizon=horizon,
        split="validation",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    test_dataset = ForecastWindowDataset(
        panel=panel,
        lookback=lookback,
        horizon=horizon,
        split="test",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    return train_dataset, val_dataset, test_dataset


def optimize_model_type(
    model_type: str,
    args: argparse.Namespace,
    panel,
    train_dataset: ForecastWindowDataset,
    val_dataset: ForecastWindowDataset,
    device: torch.device,
) -> Any:
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=f"tourism_hts_{model_type}",
        direction="minimize",
        sampler=sampler,
    )

    def objective(trial) -> float:
        trial_seed = args.seed + trial.number
        set_seed(trial_seed)
        random.seed(trial_seed)
        np.random.seed(trial_seed)

        run_config = sample_hyperparameters(
            trial=trial,
            model_type=model_type,
            num_series=len(panel.hierarchy.bottom_ids),
            horizon=args.horizon,
            lookback=args.lookback,
        )
        model = build_forecaster(model_type, run_config["model_config"])
        model, history = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            panel=panel,
            epochs=args.trial_epochs,
            batch_size=int(run_config["batch_size"]),
            learning_rate=float(run_config["learning_rate"]),
            weight_decay=float(run_config["weight_decay"]),
            eval_every=max(args.trial_epochs, 1),
            device=device,
            verbose=False,
        )
        val_metrics = evaluate_model(
            model=model,
            dataset=val_dataset,
            panel=panel,
            device=device,
            batch_size=int(run_config["batch_size"]),
        )
        score = float(val_metrics["bottom"]["rmse"])
        trial.set_user_attr("run_config", run_config)
        trial.set_user_attr("history", history)
        trial.set_user_attr("validation_metrics", val_metrics)
        return score

    study.optimize(objective, n_trials=args.n_trials)
    return study


def train_best_final_model(
    model_type: str,
    run_config: dict[str, Any],
    args: argparse.Namespace,
    panel,
    train_dataset: ForecastWindowDataset,
    val_dataset: ForecastWindowDataset,
    test_dataset: ForecastWindowDataset,
    data_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    model = build_forecaster(model_type, run_config["model_config"])
    model, history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        panel=panel,
        epochs=args.final_epochs,
        batch_size=int(run_config["batch_size"]),
        learning_rate=float(run_config["learning_rate"]),
        weight_decay=float(run_config["weight_decay"]),
        eval_every=args.eval_every,
        device=device,
        verbose=True,
    )
    metrics = {
        "device": device.type,
        "model_type": model_type,
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "train": evaluate_model(model, train_dataset, panel, device, batch_size=int(run_config["batch_size"])),
        "validation": evaluate_model(model, val_dataset, panel, device, batch_size=int(run_config["batch_size"])),
        "test": evaluate_model(model, test_dataset, panel, device, batch_size=int(run_config["batch_size"])),
        "history": history,
    }
    metadata = {
        **panel.metadata,
        "data_csv": str(data_path),
        "model_type": model_type,
        "lookback": int(args.lookback),
        "horizon": int(args.horizon),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "source": panel.preprocessing["dataset_doc_url"],
        "optuna_trials": int(args.n_trials),
        "trial_epochs": int(args.trial_epochs),
        "final_epochs": int(args.final_epochs),
    }

    checkpoint_path = Path(args.checkpoint_dir).expanduser().resolve() / f"tourism_hts_{model_type}_best.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        model_config=run_config["model_config"],
        preprocessing=panel.preprocessing,
        metadata=metadata,
        metrics=metrics,
        model_type=model_type,
    )
    return {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "run_config": run_config,
        "metrics": metrics,
    }


def compact_trial_records(study) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        records.append(
            {
                "number": trial.number,
                "value": trial.value,
                "state": str(trial.state),
                "params": trial.params,
                "validation_metrics": trial.user_attrs.get("validation_metrics"),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    ensure_optuna_available()
    set_seed(args.seed)

    data_path = resolve_dataset_file(
        data_dir=Path(args.data_dir).expanduser().resolve(),
        data_csv=args.data_csv,
        no_download=args.no_download,
        force_download=args.force_download,
    )
    max_series = None if args.max_series <= 0 else args.max_series
    panel = build_tourism_panel(data_path, max_series=max_series, scaler_train_ratio=args.train_ratio)
    train_dataset, val_dataset, test_dataset = build_datasets(
        panel=panel,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    device = choose_device()

    print("Loaded tourism hierarchy:")
    print(f"  data: {data_path}")
    print(f"  model candidates: {', '.join(MODEL_TYPES)}")
    for key, value in panel.metadata.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    print("Window samples:")
    print(f"  train: {len(train_dataset):,}")
    print(f"  validation: {len(val_dataset):,}")
    print(f"  test: {len(test_dataset):,}")
    print(f"Device: {device.type}")

    studies = {}
    final_results: dict[str, Any] = {}
    for model_type in MODEL_TYPES:
        print(f"\nOptimizing {model_type} with {args.n_trials} trials ...")
        study = optimize_model_type(
            model_type=model_type,
            args=args,
            panel=panel,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
        )
        studies[model_type] = study
        best_run_config = study.best_trial.user_attrs["run_config"]
        print(f"Best {model_type} validation bottom RMSE: {study.best_value:.3f}")
        print(f"Retraining best {model_type} for {args.final_epochs} epochs ...")
        final_results[model_type] = train_best_final_model(
            model_type=model_type,
            run_config=best_run_config,
            args=args,
            panel=panel,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            data_path=data_path,
            device=device,
        )

    summary_rows = []
    for model_type, result in final_results.items():
        metrics = result["metrics"]
        summary_rows.append(
            {
                "model_type": model_type,
                "best_trial_val_bottom_rmse": float(studies[model_type].best_value),
                "final_val_bottom_rmse": float(metrics["validation"]["bottom"]["rmse"]),
                "final_val_state_rmse": float(metrics["validation"]["state"]["rmse"]),
                "final_val_total_rmse": float(metrics["validation"]["total"]["rmse"]),
                "test_bottom_rmse": float(metrics["test"]["bottom"]["rmse"]),
                "test_state_rmse": float(metrics["test"]["state"]["rmse"]),
                "test_total_rmse": float(metrics["test"]["total"]["rmse"]),
                "test_bottom_smape": float(metrics["test"]["bottom"]["smape"]),
                "max_test_reconciliation_error": float(metrics["test"]["max_reconciliation_error"]),
                "checkpoint": result["checkpoint"],
                "run_config_json": json.dumps(result["run_config"], sort_keys=True),
            }
        )

    comparison_table = pl.DataFrame(summary_rows).sort("final_val_bottom_rmse")
    winner_by_validation = comparison_table[0, "model_type"]
    winner_by_test = comparison_table.sort("test_bottom_rmse")[0, "model_type"]

    report = {
        "dataset": {
            "csv": str(data_path),
            "source": panel.preprocessing["dataset_doc_url"],
            "download_url": panel.preprocessing["dataset_url"],
        },
        "metadata": {
            **panel.metadata,
            "lookback": int(args.lookback),
            "horizon": int(args.horizon),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "n_trials_per_model": int(args.n_trials),
            "trial_epochs": int(args.trial_epochs),
            "final_epochs": int(args.final_epochs),
        },
        "winner_by_validation_bottom_rmse": winner_by_validation,
        "winner_by_test_bottom_rmse": winner_by_test,
        "summary": summary_rows,
        "final_results": final_results,
        "optuna_trials": {
            model_type: compact_trial_records(study)
            for model_type, study in studies.items()
        },
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2))
    comparison_table.write_csv(output_csv)

    print("\nComparison summary:")
    print(comparison_table)
    print(f"Winner by validation bottom RMSE: {winner_by_validation}")
    print(f"Winner by test bottom RMSE: {winner_by_test}")
    print(f"Saved detailed report to: {output_json}")
    print(f"Saved compact table to: {output_csv}")


if __name__ == "__main__":
    main()
