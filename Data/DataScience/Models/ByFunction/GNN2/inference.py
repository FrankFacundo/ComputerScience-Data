"""Run inference with a trained Elliptic GCN checkpoint.

By default this script generates a small synthetic Elliptic-shaped graph,
scores it, and writes predictions to CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from elliptic_gnn import load_checkpoint, prepare_inference_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score synthetic Elliptic-style transactions with a trained GCN.")
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).with_name("elliptic_gcn_checkpoint.pt")),
        help="Path to the checkpoint produced by train.py.",
    )
    parser.add_argument("--features-csv", default=None, help="Optional inference feature CSV. Defaults to synthetic data.")
    parser.add_argument("--edges-csv", default=None, help="Optional inference edge CSV. Defaults to synthetic data.")
    parser.add_argument("--num-synthetic-nodes", type=int, default=16)
    parser.add_argument("--num-synthetic-edges", type=int, default=28)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--threshold", type=float, default=None, help="Defaults to the validation threshold in checkpoint.")
    parser.add_argument(
        "--synthetic-features-output",
        default=str(Path(__file__).with_name("synthetic_elliptic_features.csv")),
        help="Where generated synthetic features are saved when --features-csv is omitted.",
    )
    parser.add_argument(
        "--synthetic-edges-output",
        default=str(Path(__file__).with_name("synthetic_elliptic_edges.csv")),
        help="Where generated synthetic edges are saved when --edges-csv is omitted.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).with_name("elliptic_synthetic_predictions.csv")),
        help="Path where scored transactions are written.",
    )
    return parser.parse_args()


def create_synthetic_elliptic_graph(
    preprocessing: dict,
    num_nodes: int,
    num_edges: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if num_nodes < 2:
        raise ValueError("--num-synthetic-nodes must be >= 2.")
    if num_edges < 1:
        raise ValueError("--num-synthetic-edges must be >= 1.")

    rng = np.random.default_rng(seed)
    feature_columns = list(preprocessing["feature_columns"])
    mean = np.asarray(preprocessing["feature_mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["feature_scale"], dtype=np.float32)

    raw_features = rng.normal(loc=mean, scale=scale, size=(num_nodes, len(feature_columns))).astype(np.float32)

    # Make a few nodes deliberately unusual so the output is not completely flat.
    unusual_count = min(3, num_nodes)
    if len(feature_columns) >= 4:
        raw_features[:unusual_count, :4] = mean[:4] + (3.0 * scale[:4])
    elif len(feature_columns) > 0:
        raw_features[:unusual_count, :] = mean + (3.0 * scale)

    tx_ids = [f"synthetic_tx_{idx:04d}" for idx in range(num_nodes)]
    feature_data = {
        "txId": tx_ids,
        "time_step": rng.integers(50, 55, size=num_nodes, endpoint=False).astype(np.int16),
    }
    feature_data.update({column: raw_features[:, idx] for idx, column in enumerate(feature_columns)})
    feature_df = pl.DataFrame(feature_data)

    edge_pairs: list[tuple[str, str]] = []
    for idx in range(num_nodes - 1):
        edge_pairs.append((tx_ids[idx], tx_ids[idx + 1]))
    while len(edge_pairs) < num_edges:
        src, dst = rng.choice(tx_ids, size=2, replace=False)
        edge_pairs.append((str(src), str(dst)))
    edge_df = pl.DataFrame(
        {
            "txId1": [src for src, _ in edge_pairs[:num_edges]],
            "txId2": [dst for _, dst in edge_pairs[:num_edges]],
        }
    )
    return feature_df, edge_df


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first with train.py.")

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    threshold = float(checkpoint["threshold"] if args.threshold is None else args.threshold)
    preprocessing = checkpoint["preprocessing"]

    if args.features_csv and args.edges_csv:
        features = pl.read_csv(Path(args.features_csv).expanduser())
        edges = pl.read_csv(Path(args.edges_csv).expanduser())
        synthetic_paths = None
    elif not args.features_csv and not args.edges_csv:
        features, edges = create_synthetic_elliptic_graph(
            preprocessing=preprocessing,
            num_nodes=args.num_synthetic_nodes,
            num_edges=args.num_synthetic_edges,
            seed=args.seed,
        )
        features_path = Path(args.synthetic_features_output).expanduser().resolve()
        edges_path = Path(args.synthetic_edges_output).expanduser().resolve()
        features_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.parent.mkdir(parents=True, exist_ok=True)
        features.write_csv(features_path)
        edges.write_csv(edges_path)
        synthetic_paths = (features_path, edges_path)
    else:
        raise ValueError("Pass both --features-csv and --edges-csv, or omit both to generate synthetic data.")

    x, adj, tx_ids, time_steps = prepare_inference_graph(features, edges, preprocessing)
    x = x.to(device)
    adj = adj.to(device)

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x, adj)).detach().cpu().numpy()
    predictions = (probabilities >= threshold).astype(np.int8)
    labels = np.where(predictions == 1, "illicit", "licit")

    output = pl.DataFrame(
        {
            "txId": tx_ids,
            "time_step": time_steps,
            "illicit_probability": probabilities,
            "predicted_label": labels,
            "predicted_class": predictions,
        }
    )
    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(output_path)

    print(f"Loaded checkpoint: {checkpoint_path}")
    if synthetic_paths:
        print(f"Generated synthetic features: {synthetic_paths[0]}")
        print(f"Generated synthetic edges: {synthetic_paths[1]}")
    print(f"Scored nodes: {len(output):,}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Predicted illicit nodes: {int(predictions.sum()):,}")
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
