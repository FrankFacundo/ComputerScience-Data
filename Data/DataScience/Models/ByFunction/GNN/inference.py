"""Run inference with a trained IBM AML transaction GNN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch

from gnn import (
    batched_edge_logits,
    load_model_checkpoint,
    move_graph_to_device,
    prepare_inference_graph,
    read_transactions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score IBM AML transactions with a trained GNN checkpoint.")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to a transactions CSV or .zip file. Defaults to a generated synthetic CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).with_name("aml_gnn_checkpoint.pt")),
        help="Path to the checkpoint saved by gnn.py.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).with_name("aml_gnn_predictions.csv")),
        help="Path where scored transactions are written.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap. Use 0 for all rows.")
    parser.add_argument("--edge-batch-size", type=int, default=32_768)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold. Defaults to the validation-selected threshold in the checkpoint.",
    )
    return parser.parse_args()


def create_synthetic_transactions(path: Path) -> Path:
    """Create a tiny inference input with the IBM AML column shape."""
    rows = [
        {
            "Timestamp": "2022/09/01 00:20",
            "From Bank": "010",
            "Account": "SYNTH_FROM_001",
            "To Bank": "010",
            "Account.1": "SYNTH_TO_001",
            "Amount Received": 3697.34,
            "Receiving Currency": "US Dollar",
            "Amount Paid": 3697.34,
            "Payment Currency": "US Dollar",
            "Payment Format": "Reinvestment",
        },
        {
            "Timestamp": "2022/09/01 00:23",
            "From Bank": "03208",
            "Account": "SYNTH_FROM_002",
            "To Bank": "001",
            "Account.1": "SYNTH_TO_002",
            "Amount Received": 0.01,
            "Receiving Currency": "US Dollar",
            "Amount Paid": 0.01,
            "Payment Currency": "US Dollar",
            "Payment Format": "Cheque",
        },
        {
            "Timestamp": "2022/09/01 00:30",
            "From Bank": "011",
            "Account": "SYNTH_FROM_003",
            "To Bank": "070",
            "Account.1": "SYNTH_TO_003",
            "Amount Received": 18500.00,
            "Receiving Currency": "Euro",
            "Amount Paid": 18750.00,
            "Payment Currency": "US Dollar",
            "Payment Format": "ACH",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)
    return path


def score_transactions(
    model: torch.nn.Module,
    graph,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode_nodes(
            graph.node_features,
            graph.node_bank_ids,
            graph.train_adj_in,
            graph.train_adj_out,
        )
        logits = batched_edge_logits(model, node_embeddings, graph, graph.splits["inference"], batch_size)
        return torch.sigmoid(logits).detach().cpu()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train first with gnn.py or pass --checkpoint to an existing .pt file."
        )
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    threshold = float(checkpoint["threshold"] if args.threshold is None else args.threshold)

    max_rows = None if args.max_rows <= 0 else args.max_rows
    if args.csv_path is None:
        dataset_path = create_synthetic_transactions(Path(__file__).with_name("synthetic_transactions.csv"))
        print(f"No --csv-path provided. Using synthetic transactions: {dataset_path}")
    else:
        dataset_path = Path(args.csv_path).expanduser().resolve()
    df = read_transactions(dataset_path, max_rows=max_rows, require_label=False)
    graph = prepare_inference_graph(df, checkpoint["preprocessing"])
    graph = move_graph_to_device(graph, device)

    scores = score_transactions(model, graph, args.edge_batch_size).numpy()
    predictions = (scores >= threshold).astype("int8")

    output = df.with_columns(
        [
            pl.Series("laundering_probability", scores),
            pl.Series("predicted_laundering", predictions),
        ]
    )

    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_csv(output_path)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Scored transactions: {len(output):,}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Predicted positives: {int(predictions.sum()):,}")
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
