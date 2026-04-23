"""Train a GCN on the Elliptic Bitcoin transaction graph.

By default, missing dataset CSVs are downloaded from Hugging Face into ./data.

You can also place the Elliptic dataset CSVs in a local folder, then run:

    python train.py --data-dir /path/to/elliptic_bitcoin_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from elliptic_gnn import (
    EllipticGCN,
    best_f1_threshold,
    compute_binary_metrics,
    download_huggingface_elliptic_dataset,
    evaluate_model,
    logits_to_probabilities,
    move_graph_to_device,
    prepare_training_graph,
    resolve_dataset_files,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GCN on the Elliptic Bitcoin Dataset.")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).with_name("data")),
        help="Directory containing elliptic_txs_features.csv, elliptic_txs_edgelist.csv, and elliptic_txs_classes.csv.",
    )
    parser.add_argument("--features-csv", default=None, help="Optional explicit path to elliptic_txs_features.csv.")
    parser.add_argument("--edges-csv", default=None, help="Optional explicit path to elliptic_txs_edgelist.csv.")
    parser.add_argument("--classes-csv", default=None, help="Optional explicit path to elliptic_txs_classes.csv.")
    parser.add_argument("--no-download", action="store_true", help="Fail if the dataset is missing instead of downloading it.")
    parser.add_argument("--force-download", action="store_true", help="Force re-downloading the Hugging Face dataset.")
    parser.add_argument("--model-output", default=str(Path(__file__).with_name("elliptic_gcn_checkpoint.pt")))
    parser.add_argument("--metrics-json", default=str(Path(__file__).with_name("elliptic_gcn_metrics.json")))
    parser.add_argument("--max-nodes", type=int, default=0, help="Optional cap for quick local tests. 0 uses all nodes.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Temporal split ratio for training time steps.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Temporal split ratio for validation time steps.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def first_finite(*values: float) -> float:
    for value in values:
        if math.isfinite(value):
            return value
    return -math.inf


def train_model(
    graph,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
) -> tuple[EllipticGCN, dict[str, Any], float, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = move_graph_to_device(graph, device)

    y_train = graph.y[graph.train_idx]
    positives = int((y_train == 1).sum().item())
    negatives = int((y_train == 0).sum().item())
    if positives == 0 or negatives == 0:
        raise ValueError(
            "Training split must contain both licit and illicit labeled nodes. "
            "Use the full dataset or adjust --train-ratio/--max-nodes."
        )

    model_config = {
        "input_dim": int(graph.x.shape[1]),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
    }
    model = EllipticGCN(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.adj)
        loss = loss_fn(logits[graph.train_idx], graph.y[graph.train_idx])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        should_eval = epoch == 1 or epoch == epochs or epoch % max(eval_every, 1) == 0
        if should_eval:
            val_metrics, _, _ = evaluate_model(model, graph, graph.val_idx, threshold=0.5)
            val_score = first_finite(val_metrics["average_precision"], val_metrics["f1"], -float(loss.item()))
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.item()),
                    "val_average_precision": float(val_metrics["average_precision"]),
                    "val_roc_auc": float(val_metrics["roc_auc"]),
                    "val_f1_at_0_5": float(val_metrics["f1"]),
                }
            )
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_ap={val_metrics['average_precision']:.4f} | "
                f"val_auc={val_metrics['roc_auc']:.4f} | val_f1={val_metrics['f1']:.4f}"
            )

            if val_score > best_score:
                best_score = val_score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpointable model state.")

    model.load_state_dict(best_state)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        val_logits = model(graph.x, graph.adj)[graph.val_idx]
    val_prob = logits_to_probabilities(val_logits)
    val_true = graph.y[graph.val_idx].detach().cpu().numpy().astype(np.int8)
    threshold, val_best_f1 = best_f1_threshold(val_true, val_prob)

    train_metrics, _, _ = evaluate_model(model, graph, graph.train_idx, threshold)
    val_metrics = compute_binary_metrics(val_true, val_prob, threshold)
    test_metrics, _, _ = evaluate_model(model, graph, graph.test_idx, threshold)
    metrics = {
        "device": device.type,
        "pos_weight": float(pos_weight.item()),
        "selected_threshold": float(threshold),
        "validation_best_f1": float(val_best_f1),
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    return model, model_config, threshold, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    max_nodes = None if args.max_nodes <= 0 else args.max_nodes
    explicit_csv_paths = any([args.features_csv, args.edges_csv, args.classes_csv])
    try:
        features_path, edges_path, classes_path = resolve_dataset_files(
            data_dir=data_dir,
            features_csv=args.features_csv,
            edges_csv=args.edges_csv,
            classes_csv=args.classes_csv,
        )
    except FileNotFoundError:
        if args.no_download or explicit_csv_paths:
            raise

        print(f"Elliptic CSVs were not found in {data_dir}. Downloading from Hugging Face ...")
        downloaded_path = download_huggingface_elliptic_dataset(data_dir, force_download=args.force_download)
        print(f"Downloaded dataset to: {downloaded_path}")
        features_path, edges_path, classes_path = resolve_dataset_files(data_dir=data_dir)

    print("Loading Elliptic dataset:")
    print(f"  features: {features_path}")
    print(f"  edges:    {edges_path}")
    print(f"  classes:  {classes_path}")

    graph = prepare_training_graph(
        features_path=features_path,
        edges_path=edges_path,
        classes_path=classes_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_nodes=max_nodes,
    )

    print("Prepared graph:")
    for key, value in graph.metadata.items():
        if isinstance(value, list):
            print(f"  {key}: {value[:5]}{' ...' if len(value) > 5 else ''}")
        else:
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    model, model_config, threshold, metrics = train_model(
        graph=graph,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
    )

    report = {
        "dataset": {
            "features_csv": str(features_path),
            "edges_csv": str(edges_path),
            "classes_csv": str(classes_path),
            "max_nodes": max_nodes,
        },
        "graph_metadata": graph.metadata,
        "model_config": model_config,
        "metrics": metrics,
    }

    model_output = Path(args.model_output).expanduser().resolve()
    save_checkpoint(
        path=model_output,
        model=model,
        model_config=model_config,
        preprocessing=graph.preprocessing,
        threshold=threshold,
        metadata=graph.metadata,
        metrics=metrics,
    )
    print(f"Saved model checkpoint to: {model_output}")

    if args.metrics_json:
        metrics_path = Path(args.metrics_json).expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, indent=2))
        print(f"Saved metrics to: {metrics_path}")

    print("Final metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
