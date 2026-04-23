"""Utilities for training and serving a GCN on the Elliptic Bitcoin dataset.

Expected dataset files:
  - elliptic_txs_features.csv
  - elliptic_txs_edgelist.csv
  - elliptic_txs_classes.csv

The Elliptic labels use 1 for illicit, 2 for licit, and "unknown" for
unlabeled transactions. Unknown nodes are kept in the graph but excluded from
supervised loss and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURES_FILENAME = "elliptic_txs_features.csv"
EDGES_FILENAME = "elliptic_txs_edgelist.csv"
CLASSES_FILENAME = "elliptic_txs_classes.csv"
HUGGING_FACE_DATASET_ID = "yhoma/elliptic-bitcoin-dataset"
HUGGING_FACE_DATASET_URL = f"https://huggingface.co/datasets/{HUGGING_FACE_DATASET_ID}"


@dataclass
class EllipticGraph:
    x: torch.Tensor
    y: torch.Tensor
    adj: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    tx_ids: list[str]
    time_steps: np.ndarray
    metadata: dict[str, Any]
    preprocessing: dict[str, Any]


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(adj, x)
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)
        return F.dropout(x, p=self.dropout, training=self.training)


class EllipticGCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[GCNLayer] = []
        layer_in = input_dim
        for _ in range(num_layers):
            layers.append(GCNLayer(layer_in, hidden_dim, dropout))
            layer_in = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return self.classifier(x).squeeze(-1)


def resolve_dataset_files(
    data_dir: Path,
    features_csv: str | None = None,
    edges_csv: str | None = None,
    classes_csv: str | None = None,
) -> tuple[Path, Path, Path]:
    features_path = Path(features_csv).expanduser() if features_csv else find_dataset_file(data_dir, FEATURES_FILENAME)
    edges_path = Path(edges_csv).expanduser() if edges_csv else find_dataset_file(data_dir, EDGES_FILENAME)
    classes_path = Path(classes_csv).expanduser() if classes_csv else find_dataset_file(data_dir, CLASSES_FILENAME)

    missing = [path for path in (features_path, edges_path, classes_path) if not path.exists()]
    if missing:
        expected = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing Elliptic dataset file(s):\n"
            f"{expected}\n"
            "Place the Elliptic CSVs in --data-dir, or pass --features-csv, --edges-csv, and --classes-csv."
        )
    return features_path.resolve(), edges_path.resolve(), classes_path.resolve()


def find_dataset_file(data_dir: Path, filename: str) -> Path:
    direct_path = data_dir / filename
    if direct_path.exists():
        return direct_path

    if data_dir.exists():
        matches = sorted(data_dir.rglob(filename))
        if matches:
            return matches[0]
    return direct_path


def download_huggingface_elliptic_dataset(data_dir: Path, force_download: bool = False) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Automatic Hugging Face download requires huggingface_hub. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in (FEATURES_FILENAME, EDGES_FILENAME, CLASSES_FILENAME):
        try:
            hf_hub_download(
                repo_id=HUGGING_FACE_DATASET_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=str(data_dir),
                force_download=force_download,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {filename} from {HUGGING_FACE_DATASET_URL}.\n"
                "Check your internet connection and that huggingface_hub is installed."
            ) from exc
    return data_dir.resolve()


def _strip_tx_id(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.strip_chars().alias(column)


def read_features(path: Path, max_nodes: int | None = None) -> pl.DataFrame:
    n_rows = None if max_nodes is None else max_nodes + 1
    raw = pl.read_csv(path, has_header=False, n_rows=n_rows, infer_schema_length=0)
    if raw.height == 0:
        raise ValueError(f"{path} is empty.")

    first_value = str(raw.item(0, 0)).strip().lower()
    if first_value in {"txid", "tx_id", "transaction"}:
        raw = raw.slice(1)
    if max_nodes is not None:
        raw = raw.head(max_nodes)

    if raw.width < 3:
        raise ValueError(f"{path} must contain txId, time_step, and at least one feature column.")

    feature_columns = [f"feature_{idx:03d}" for idx in range(1, raw.width - 1)]
    raw = raw.rename(dict(zip(raw.columns, ["txId", "time_step", *feature_columns], strict=True)))
    return raw.with_columns(
        [
            _strip_tx_id("txId"),
            pl.col("time_step").cast(pl.Int16),
            *[pl.col(column).cast(pl.Float32) for column in feature_columns],
        ]
    )


def read_classes(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=0)
    lowered = {str(col).strip().lower(): col for col in df.columns}
    if "txid" not in lowered or "class" not in lowered:
        df = pl.read_csv(path, has_header=False, new_columns=["txId", "class"], infer_schema_length=0)
    else:
        df = df.rename({lowered["txid"]: "txId", lowered["class"]: "class"})

    label_value = pl.col("class").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    return df.select(
        [
            _strip_tx_id("txId"),
            pl.when(label_value.is_in(["1", "illicit"]))
            .then(1)
            .when(label_value.is_in(["2", "licit"]))
            .then(0)
            .otherwise(-1)
            .cast(pl.Int8)
            .alias("label"),
        ]
    )


def read_edges(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=0)
    lowered = {str(col).strip().lower(): col for col in df.columns}
    if "txid1" not in lowered or "txid2" not in lowered:
        df = pl.read_csv(path, has_header=False, new_columns=["txId1", "txId2"], infer_schema_length=0)
    else:
        df = df.rename({lowered["txid1"]: "txId1", lowered["txid2"]: "txId2"})

    return df.select([_strip_tx_id("txId1"), _strip_tx_id("txId2")])


def build_normalized_adjacency(
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
    undirected: bool = True,
    add_self_loops: bool = True,
) -> torch.Tensor:
    if len(src) != len(dst):
        raise ValueError("src and dst must have the same length.")

    pairs = np.column_stack([src.astype(np.int64), dst.astype(np.int64)])
    if undirected:
        pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    if add_self_loops:
        self_loops = np.column_stack([np.arange(num_nodes, dtype=np.int64), np.arange(num_nodes, dtype=np.int64)])
        pairs = np.concatenate([pairs, self_loops], axis=0)

    pairs = np.unique(pairs, axis=0)
    source = pairs[:, 0]
    target = pairs[:, 1]

    # Sparse matrix is indexed as [target, source] so adj @ x aggregates source
    # node features into each target node.
    row = target
    col = source
    degree = np.bincount(row, minlength=num_nodes).astype(np.float32)
    values = 1.0 / np.maximum(degree[row], 1.0)

    indices = torch.from_numpy(np.vstack([row, col]).astype(np.int64))
    values_t = torch.from_numpy(values.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values_t, (num_nodes, num_nodes)).coalesce()


def temporal_label_splits(
    time_steps: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    unique_steps = np.array(sorted(np.unique(time_steps).tolist()))
    train_end = max(1, int(round(len(unique_steps) * train_ratio)))
    val_end = max(train_end + 1, int(round(len(unique_steps) * (train_ratio + val_ratio))))
    val_end = min(val_end, len(unique_steps) - 1)

    train_steps = set(unique_steps[:train_end].tolist())
    val_steps = set(unique_steps[train_end:val_end].tolist())
    test_steps = set(unique_steps[val_end:].tolist())
    known = labels >= 0

    train_idx = np.flatnonzero(known & np.isin(time_steps, list(train_steps)))
    val_idx = np.flatnonzero(known & np.isin(time_steps, list(val_steps)))
    test_idx = np.flatnonzero(known & np.isin(time_steps, list(test_steps)))

    split_metadata = {
        "unique_time_steps": unique_steps.astype(int).tolist(),
        "train_time_steps": sorted(int(step) for step in train_steps),
        "val_time_steps": sorted(int(step) for step in val_steps),
        "test_time_steps": sorted(int(step) for step in test_steps),
    }
    return train_idx, val_idx, test_idx, split_metadata


def prepare_training_graph(
    features_path: Path,
    edges_path: Path,
    classes_path: Path,
    train_ratio: float,
    val_ratio: float,
    max_nodes: int | None = None,
) -> EllipticGraph:
    features = read_features(features_path, max_nodes=max_nodes)
    classes = read_classes(classes_path)
    edges = read_edges(edges_path)

    df = features.join(classes, on="txId", how="left").with_columns(pl.col("label").fill_null(-1).cast(pl.Int8))

    tx_ids = df["txId"].to_list()
    node_lookup = pl.DataFrame({"txId": tx_ids, "node_id": np.arange(len(tx_ids), dtype=np.int64)})
    mapped_edges = (
        edges.join(node_lookup.rename({"txId": "txId1", "node_id": "src"}), on="txId1", how="left")
        .join(node_lookup.rename({"txId": "txId2", "node_id": "dst"}), on="txId2", how="left")
        .filter(pl.col("src").is_not_null() & pl.col("dst").is_not_null())
        .select([pl.col("src").cast(pl.Int64), pl.col("dst").cast(pl.Int64)])
    )
    src = mapped_edges["src"].to_numpy().astype(np.int64)
    dst = mapped_edges["dst"].to_numpy().astype(np.int64)

    feature_columns = [col for col in df.columns if col.startswith("feature_")]
    time_steps = df["time_step"].to_numpy().astype(np.int16)
    labels = df["label"].to_numpy().astype(np.int8)

    train_idx, val_idx, test_idx, split_metadata = temporal_label_splits(
        time_steps=time_steps,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    if len(train_idx) == 0:
        raise ValueError("Training split has no labeled nodes. Increase --max-nodes or adjust split ratios.")

    train_time_mask = np.isin(time_steps, split_metadata["train_time_steps"])
    scaler = StandardScaler()
    x_raw = df.select(feature_columns).to_numpy().astype(np.float32)
    scaler.fit(x_raw[train_time_mask])
    x = scaler.transform(x_raw).astype(np.float32)

    adj = build_normalized_adjacency(src, dst, num_nodes=len(df))
    known_count = int((labels >= 0).sum())
    illicit_count = int((labels == 1).sum())
    licit_count = int((labels == 0).sum())

    metadata = {
        "num_nodes": int(len(df)),
        "num_edges_raw": int(len(edges)),
        "num_edges_used": int(len(src)),
        "num_features": int(len(feature_columns)),
        "known_labels": known_count,
        "unknown_labels": int((labels < 0).sum()),
        "illicit_labels": illicit_count,
        "licit_labels": licit_count,
        "train_labeled_nodes": int(len(train_idx)),
        "val_labeled_nodes": int(len(val_idx)),
        "test_labeled_nodes": int(len(test_idx)),
        "train_illicit": int(labels[train_idx].sum()),
        "val_illicit": int(labels[val_idx].sum()) if len(val_idx) else 0,
        "test_illicit": int(labels[test_idx].sum()) if len(test_idx) else 0,
        **split_metadata,
    }
    preprocessing = {
        "feature_columns": feature_columns,
        "feature_mean": scaler.mean_.astype(np.float32).tolist(),
        "feature_scale": scaler.scale_.astype(np.float32).tolist(),
        "label_mapping": {"illicit": 1, "licit": 0, "unknown": -1},
    }

    return EllipticGraph(
        x=torch.from_numpy(x),
        y=torch.from_numpy(labels.astype(np.float32)),
        adj=adj,
        train_idx=torch.from_numpy(train_idx.astype(np.int64)),
        val_idx=torch.from_numpy(val_idx.astype(np.int64)),
        test_idx=torch.from_numpy(test_idx.astype(np.int64)),
        tx_ids=tx_ids,
        time_steps=time_steps,
        metadata=metadata,
        preprocessing=preprocessing,
    )


def prepare_inference_graph(
    features: pl.DataFrame,
    edges: pl.DataFrame,
    preprocessing: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[str], np.ndarray]:
    feature_columns = list(preprocessing["feature_columns"])
    required = {"txId", "time_step", *feature_columns}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Inference features are missing columns: {sorted(missing)}")
    if not {"txId1", "txId2"}.issubset(edges.columns):
        raise ValueError("Inference edges must contain txId1 and txId2 columns.")

    features = features.with_columns(_strip_tx_id("txId"))
    edges = edges.with_columns([_strip_tx_id("txId1"), _strip_tx_id("txId2")])

    tx_ids = features["txId"].to_list()
    node_lookup = pl.DataFrame({"txId": tx_ids, "node_id": np.arange(len(tx_ids), dtype=np.int64)})
    mapped_edges = (
        edges.join(node_lookup.rename({"txId": "txId1", "node_id": "src"}), on="txId1", how="left")
        .join(node_lookup.rename({"txId": "txId2", "node_id": "dst"}), on="txId2", how="left")
        .filter(pl.col("src").is_not_null() & pl.col("dst").is_not_null())
        .select([pl.col("src").cast(pl.Int64), pl.col("dst").cast(pl.Int64)])
    )
    src = mapped_edges["src"].to_numpy().astype(np.int64)
    dst = mapped_edges["dst"].to_numpy().astype(np.int64)

    mean = np.asarray(preprocessing["feature_mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["feature_scale"], dtype=np.float32)
    x_raw = features.select([pl.col(column).cast(pl.Float32) for column in feature_columns]).to_numpy().astype(np.float32)
    x = ((x_raw - mean) / scale).astype(np.float32)
    adj = build_normalized_adjacency(src, dst, num_nodes=len(features))
    time_steps = features["time_step"].to_numpy()
    return torch.from_numpy(x), adj, tx_ids, time_steps


def move_graph_to_device(graph: EllipticGraph, device: torch.device) -> EllipticGraph:
    return EllipticGraph(
        x=graph.x.to(device),
        y=graph.y.to(device),
        adj=graph.adj.to(device),
        train_idx=graph.train_idx.to(device),
        val_idx=graph.val_idx.to(device),
        test_idx=graph.test_idx.to(device),
        tx_ids=graph.tx_ids,
        time_steps=graph.time_steps,
        metadata=graph.metadata,
        preprocessing=graph.preprocessing,
    )


def logits_to_probabilities(logits: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(logits).detach().cpu().numpy()


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= threshold).astype(np.int8)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "positive_rate_pred": float("nan"),
        }

    y_pred = (y_prob >= threshold).astype(np.int8)
    metrics = {
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_pred": float(y_pred.mean()),
    }
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    return metrics


def evaluate_model(
    model: EllipticGCN,
    graph: EllipticGraph,
    indices: torch.Tensor,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.adj)[indices]
        y_prob = logits_to_probabilities(logits)
        y_true = graph.y[indices].detach().cpu().numpy().astype(np.int8)
    return compute_binary_metrics(y_true, y_prob, threshold), y_true, y_prob


def save_checkpoint(
    path: Path,
    model: EllipticGCN,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    threshold: float,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": model_config,
        "preprocessing": preprocessing,
        "threshold": float(threshold),
        "metadata": metadata,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[EllipticGCN, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = EllipticGCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
