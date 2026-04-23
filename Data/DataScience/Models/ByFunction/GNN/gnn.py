"""Directed GNN for IBM AML transaction edge classification.

This script trains a compact PyTorch GNN on the public IBM Transactions for
Anti Money Laundering dataset and reports validation / test metrics at the end.

Canonical dataset:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

Convenience public mirror used by the downloader below:
https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans

The model treats accounts as nodes and transactions as directed edges. Node
features are derived only from the training-period transactions to avoid future
feature leakage. The GNN then scores each transaction edge for laundering.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


HF_HI_SMALL_ZIP_URL = (
    "https://huggingface.co/datasets/eexzzm/"
    "IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/"
    "HI-Small_Trans.csv.zip"
)
UNKNOWN_CATEGORY = "<UNK>"


@dataclass
class PreparedGraph:
    node_features: torch.Tensor
    node_bank_ids: torch.Tensor
    train_src: torch.Tensor
    train_dst: torch.Tensor
    train_adj_in: torch.Tensor
    train_adj_out: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_sent_currency: torch.Tensor
    edge_recv_currency: torch.Tensor
    edge_payment_format: torch.Tensor
    edge_numeric: torch.Tensor
    labels: torch.Tensor
    splits: dict[str, torch.Tensor]
    cardinalities: dict[str, int]
    metadata: dict[str, Any]
    preprocessing: dict[str, Any]


class DirectedMessagePassingLayer(nn.Module):
    """Simple directed mean-aggregation layer."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.in_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.out_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adj_in: torch.Tensor,
        adj_out: torch.Tensor,
    ) -> torch.Tensor:
        agg_in = torch.sparse.mm(adj_in, x)
        agg_out = torch.sparse.mm(adj_out, x)
        h = self.self_linear(x) + self.in_linear(agg_in) + self.out_linear(agg_out)
        h = self.norm(h)
        h = F.relu(h)
        return F.dropout(h, p=self.dropout, training=self.training)


class AMLTransactionGNN(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_banks: int,
        num_currencies: int,
        num_formats: int,
        hidden_dim: int = 96,
        bank_emb_dim: int = 16,
        currency_emb_dim: int = 8,
        format_emb_dim: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.bank_embedding = nn.Embedding(num_banks, bank_emb_dim)
        self.currency_embedding = nn.Embedding(num_currencies, currency_emb_dim)
        self.format_embedding = nn.Embedding(num_formats, format_emb_dim)

        gnn_input_dim = num_node_features + bank_emb_dim
        self.gnn_1 = DirectedMessagePassingLayer(gnn_input_dim, hidden_dim, dropout)
        self.gnn_2 = DirectedMessagePassingLayer(hidden_dim, hidden_dim, dropout)

        edge_numeric_dim = 6
        edge_input_dim = (hidden_dim * 4) + edge_numeric_dim + (currency_emb_dim * 2) + format_emb_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_nodes(
        self,
        node_features: torch.Tensor,
        node_bank_ids: torch.Tensor,
        adj_in: torch.Tensor,
        adj_out: torch.Tensor,
    ) -> torch.Tensor:
        bank_emb = self.bank_embedding(node_bank_ids)
        x = torch.cat([node_features, bank_emb], dim=-1)
        x = self.gnn_1(x, adj_in, adj_out)
        x = self.gnn_2(x, adj_in, adj_out)
        return x

    def score_edges(
        self,
        node_embeddings: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_sent_currency: torch.Tensor,
        edge_recv_currency: torch.Tensor,
        edge_payment_format: torch.Tensor,
        edge_numeric: torch.Tensor,
    ) -> torch.Tensor:
        src_h = node_embeddings[edge_src]
        dst_h = node_embeddings[edge_dst]
        currency_sent = self.currency_embedding(edge_sent_currency)
        currency_recv = self.currency_embedding(edge_recv_currency)
        payment_fmt = self.format_embedding(edge_payment_format)

        edge_repr = torch.cat(
            [
                src_h,
                dst_h,
                torch.abs(src_h - dst_h),
                src_h * dst_h,
                edge_numeric,
                currency_sent,
                currency_recv,
                payment_fmt,
            ],
            dim=-1,
        )
        return self.edge_mlp(edge_repr).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_public_hi_small(download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "HI-Small_Trans.csv.zip"
    if not zip_path.exists():
        print(f"Downloading HI-Small mirror to {zip_path} ...")
        urllib.request.urlretrieve(HF_HI_SMALL_ZIP_URL, zip_path)
    return zip_path


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.csv_path:
        return Path(args.csv_path).expanduser().resolve()

    zip_path = download_public_hi_small(Path(args.download_dir).expanduser().resolve())
    return zip_path


def read_transactions(dataset_path: Path, max_rows: int | None, require_label: bool = True) -> pl.DataFrame:
    common_kwargs = {
        "schema_overrides": {
            "From Bank": pl.Utf8,
            "Account": pl.Utf8,
            "To Bank": pl.Utf8,
            "Receiving Currency": pl.Utf8,
            "Payment Currency": pl.Utf8,
            "Payment Format": pl.Utf8,
            "Is Laundering": pl.Int8,
        },
        "n_rows": max_rows,
    }

    if dataset_path.suffix == ".zip":
        with zipfile.ZipFile(dataset_path) as zf:
            member = next(name for name in zf.namelist() if name.endswith(".csv"))
            with zf.open(member) as fp:
                df = pl.read_csv(io.BytesIO(fp.read()), **common_kwargs)
    else:
        df = pl.read_csv(dataset_path, **common_kwargs)

    # Polars auto-renames duplicate headers like the second "Account" column.
    if "Account.1" not in df.columns:
        duplicate_account_cols = [col for col in df.columns if col.startswith("Account_duplicated_")]
        if duplicate_account_cols:
            df = df.rename({duplicate_account_cols[0]: "Account.1"})

    expected_cols = {
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
    }
    if require_label:
        expected_cols.add("Is Laundering")
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def array_from_values(values: pl.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pl.Series):
        array = values.fill_null(UNKNOWN_CATEGORY).to_numpy()
    else:
        array = np.asarray(values, dtype=object)
        array = np.where(array == None, UNKNOWN_CATEGORY, array)
    return array.astype(str)


def fit_categories(values: pl.Series | np.ndarray) -> np.ndarray:
    categories = np.unique(array_from_values(values))
    if UNKNOWN_CATEGORY not in categories:
        categories = np.sort(np.append(categories, UNKNOWN_CATEGORY))
    return categories.astype(object)


def encode_with_categories(values: pl.Series | np.ndarray, categories: np.ndarray) -> np.ndarray:
    array = array_from_values(values)
    category_to_id = {category: idx for idx, category in enumerate(categories.tolist())}
    unknown_id = category_to_id[UNKNOWN_CATEGORY]
    return np.fromiter((category_to_id.get(value, unknown_id) for value in array), dtype=np.int64, count=len(array))


def encode_series(values: pl.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    categories = fit_categories(values)
    codes = encode_with_categories(values, categories)
    return codes, categories


def build_sparse_mean_adjacency(
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_t = torch.from_numpy(src.astype(np.int64))
    dst_t = torch.from_numpy(dst.astype(np.int64))

    in_deg = torch.bincount(dst_t, minlength=num_nodes).float().clamp_min(1.0)
    out_deg = torch.bincount(src_t, minlength=num_nodes).float().clamp_min(1.0)

    adj_in_idx = torch.stack([dst_t, src_t], dim=0)
    adj_out_idx = torch.stack([src_t, dst_t], dim=0)

    adj_in_val = 1.0 / in_deg[dst_t]
    adj_out_val = 1.0 / out_deg[src_t]

    adj_in = torch.sparse_coo_tensor(adj_in_idx, adj_in_val, (num_nodes, num_nodes)).coalesce()
    adj_out = torch.sparse_coo_tensor(adj_out_idx, adj_out_val, (num_nodes, num_nodes)).coalesce()
    return adj_in, adj_out


def prepare_graph(df: pl.DataFrame) -> PreparedGraph:
    df = df.with_columns(
        [
            (pl.col("From Bank").cast(pl.Utf8) + pl.lit(":") + pl.col("Account").cast(pl.Utf8)).alias("from_key"),
            (pl.col("To Bank").cast(pl.Utf8) + pl.lit(":") + pl.col("Account.1").cast(pl.Utf8)).alias("to_key"),
        ]
    )

    from_keys = df["from_key"].to_numpy()
    to_keys = df["to_key"].to_numpy()
    node_keys = np.unique(np.concatenate([from_keys, to_keys]))
    node_to_id = {key: idx for idx, key in enumerate(node_keys.tolist())}

    from_id = np.fromiter((node_to_id[key] for key in from_keys), dtype=np.int64, count=len(from_keys))
    to_id = np.fromiter((node_to_id[key] for key in to_keys), dtype=np.int64, count=len(to_keys))
    df = df.with_columns([pl.Series("from_id", from_id), pl.Series("to_id", to_id)])

    bank_codes, bank_categories = encode_series(
        np.concatenate([df["From Bank"].to_numpy(), df["To Bank"].to_numpy()])
    )
    num_rows = len(df)
    from_bank_codes = bank_codes[:num_rows]
    to_bank_codes = bank_codes[num_rows:]

    currency_codes, currency_categories = encode_series(
        np.concatenate([df["Receiving Currency"].to_numpy(), df["Payment Currency"].to_numpy()])
    )
    recv_currency = currency_codes[:num_rows]
    sent_currency = currency_codes[num_rows:]

    format_codes, format_categories = encode_series(df["Payment Format"])

    node_bank_ids = np.full(len(node_keys), -1, dtype=np.int64)
    node_bank_ids[from_id] = from_bank_codes
    unset = node_bank_ids[to_id] == -1
    node_bank_ids[to_id[unset]] = to_bank_codes[unset]
    if (node_bank_ids < 0).any():
        raise ValueError("Some nodes did not receive a bank id.")

    timestamp_seconds = np.array(
        [datetime.strptime(ts, "%Y/%m/%d %H:%M").timestamp() for ts in df["Timestamp"].to_list()],
        dtype=np.float64,
    )
    timestamp_origin_seconds = float(timestamp_seconds.min())
    timestamp_minutes = ((timestamp_seconds - timestamp_origin_seconds) / 60.0).astype(np.float32)

    labels = df["Is Laundering"].cast(pl.Float32).to_numpy()
    amount_paid = df["Amount Paid"].cast(pl.Float32).to_numpy()
    amount_received = df["Amount Received"].cast(pl.Float32).to_numpy()
    log_paid = np.log1p(np.maximum(amount_paid, 0.0))
    log_received = np.log1p(np.maximum(amount_received, 0.0))

    n_edges = len(df)
    train_end = int(0.6 * n_edges)
    val_end = int(0.8 * n_edges)

    train_mask = np.zeros(n_edges, dtype=bool)
    val_mask = np.zeros(n_edges, dtype=bool)
    test_mask = np.zeros(n_edges, dtype=bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    n_nodes = len(node_keys)
    num_currencies = len(currency_categories)

    train_src = from_id[train_mask]
    train_dst = to_id[train_mask]
    train_sent_cur = sent_currency[train_mask]
    train_recv_cur = recv_currency[train_mask]
    train_log_paid = log_paid[train_mask]
    train_log_received = log_received[train_mask]

    out_count = np.bincount(train_src, minlength=n_nodes).astype(np.float32)
    in_count = np.bincount(train_dst, minlength=n_nodes).astype(np.float32)

    out_amount_by_currency = np.zeros((n_nodes, num_currencies), dtype=np.float32)
    in_amount_by_currency = np.zeros((n_nodes, num_currencies), dtype=np.float32)
    np.add.at(out_amount_by_currency, (train_src, train_sent_cur), train_log_paid)
    np.add.at(in_amount_by_currency, (train_dst, train_recv_cur), train_log_received)

    out_total = np.bincount(train_src, weights=train_log_paid, minlength=n_nodes).astype(np.float32)
    in_total = np.bincount(train_dst, weights=train_log_received, minlength=n_nodes).astype(np.float32)

    out_mean = out_total / np.maximum(out_count, 1.0)
    in_mean = in_total / np.maximum(in_count, 1.0)

    train_interbank = (from_bank_codes[train_mask] != to_bank_codes[train_mask]).astype(np.float32)
    out_interbank = np.bincount(train_src, weights=train_interbank, minlength=n_nodes).astype(np.float32)
    in_interbank = np.bincount(train_dst, weights=train_interbank, minlength=n_nodes).astype(np.float32)
    out_interbank_ratio = out_interbank / np.maximum(out_count, 1.0)
    in_interbank_ratio = in_interbank / np.maximum(in_count, 1.0)

    node_features = np.concatenate(
        [
            np.stack(
                [
                    np.log1p(out_count),
                    np.log1p(in_count),
                    out_mean,
                    in_mean,
                    out_interbank_ratio,
                    in_interbank_ratio,
                ],
                axis=1,
            ),
            out_amount_by_currency,
            in_amount_by_currency,
        ],
        axis=1,
    )

    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features).astype(np.float32)

    edge_numeric_raw = np.stack(
        [
            timestamp_minutes,
            log_paid,
            log_received,
            log_received - log_paid,
            (sent_currency == recv_currency).astype(np.float32),
            (from_bank_codes != to_bank_codes).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    edge_numeric_scaler = StandardScaler()
    edge_numeric_scaler.fit(edge_numeric_raw[:train_end])
    edge_numeric = edge_numeric_scaler.transform(edge_numeric_raw).astype(np.float32)

    train_adj_in, train_adj_out = build_sparse_mean_adjacency(train_src, train_dst, n_nodes)

    return PreparedGraph(
        node_features=torch.from_numpy(node_features.copy()),
        node_bank_ids=torch.from_numpy(node_bank_ids.copy()),
        train_src=torch.from_numpy(train_src.copy()),
        train_dst=torch.from_numpy(train_dst.copy()),
        train_adj_in=train_adj_in,
        train_adj_out=train_adj_out,
        edge_src=torch.from_numpy(from_id.copy()),
        edge_dst=torch.from_numpy(to_id.copy()),
        edge_sent_currency=torch.from_numpy(sent_currency.copy()),
        edge_recv_currency=torch.from_numpy(recv_currency.copy()),
        edge_payment_format=torch.from_numpy(format_codes.copy()),
        edge_numeric=torch.from_numpy(edge_numeric.copy()),
        labels=torch.from_numpy(labels.copy()),
        splits={
            "train": torch.from_numpy(np.flatnonzero(train_mask).copy()),
            "val": torch.from_numpy(np.flatnonzero(val_mask).copy()),
            "test": torch.from_numpy(np.flatnonzero(test_mask).copy()),
        },
        cardinalities={
            "banks": len(bank_categories),
            "currencies": len(currency_categories),
            "payment_formats": len(format_categories),
        },
        metadata={
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "train_edges": train_end,
            "val_edges": val_end - train_end,
            "test_edges": n_edges - val_end,
            "train_positives": int(labels[train_mask].sum()),
            "val_positives": int(labels[val_mask].sum()),
            "test_positives": int(labels[test_mask].sum()),
        },
        preprocessing={
            "bank_categories": bank_categories.tolist(),
            "currency_categories": currency_categories.tolist(),
            "payment_format_categories": format_categories.tolist(),
            "node_scaler_mean": scaler.mean_.astype(np.float32).tolist(),
            "node_scaler_scale": scaler.scale_.astype(np.float32).tolist(),
            "edge_scaler_mean": edge_numeric_scaler.mean_.astype(np.float32).tolist(),
            "edge_scaler_scale": edge_numeric_scaler.scale_.astype(np.float32).tolist(),
            "timestamp_origin_seconds": timestamp_origin_seconds,
        },
    )


def prepare_inference_graph(df: pl.DataFrame, preprocessing: dict[str, Any]) -> PreparedGraph:
    df = df.with_columns(
        [
            (pl.col("From Bank").cast(pl.Utf8) + pl.lit(":") + pl.col("Account").cast(pl.Utf8)).alias("from_key"),
            (pl.col("To Bank").cast(pl.Utf8) + pl.lit(":") + pl.col("Account.1").cast(pl.Utf8)).alias("to_key"),
        ]
    )

    from_keys = df["from_key"].to_numpy()
    to_keys = df["to_key"].to_numpy()
    node_keys = np.unique(np.concatenate([from_keys, to_keys]))
    node_to_id = {key: idx for idx, key in enumerate(node_keys.tolist())}
    from_id = np.fromiter((node_to_id[key] for key in from_keys), dtype=np.int64, count=len(from_keys))
    to_id = np.fromiter((node_to_id[key] for key in to_keys), dtype=np.int64, count=len(to_keys))

    bank_categories = np.asarray(preprocessing["bank_categories"], dtype=object)
    currency_categories = np.asarray(preprocessing["currency_categories"], dtype=object)
    format_categories = np.asarray(preprocessing["payment_format_categories"], dtype=object)

    bank_codes = encode_with_categories(
        np.concatenate([df["From Bank"].to_numpy(), df["To Bank"].to_numpy()]),
        bank_categories,
    )
    num_rows = len(df)
    from_bank_codes = bank_codes[:num_rows]
    to_bank_codes = bank_codes[num_rows:]

    currency_codes = encode_with_categories(
        np.concatenate([df["Receiving Currency"].to_numpy(), df["Payment Currency"].to_numpy()]),
        currency_categories,
    )
    recv_currency = currency_codes[:num_rows]
    sent_currency = currency_codes[num_rows:]
    format_codes = encode_with_categories(df["Payment Format"], format_categories)

    n_nodes = len(node_keys)
    node_bank_ids = np.full(n_nodes, encode_with_categories(np.array([UNKNOWN_CATEGORY]), bank_categories)[0], dtype=np.int64)
    node_bank_ids[from_id] = from_bank_codes
    unset = node_bank_ids[to_id] == encode_with_categories(np.array([UNKNOWN_CATEGORY]), bank_categories)[0]
    node_bank_ids[to_id[unset]] = to_bank_codes[unset]

    timestamp_seconds = np.array(
        [datetime.strptime(ts, "%Y/%m/%d %H:%M").timestamp() for ts in df["Timestamp"].to_list()],
        dtype=np.float64,
    )
    timestamp_origin_seconds = float(preprocessing["timestamp_origin_seconds"])
    timestamp_minutes = ((timestamp_seconds - timestamp_origin_seconds) / 60.0).astype(np.float32)

    amount_paid = df["Amount Paid"].cast(pl.Float32).to_numpy()
    amount_received = df["Amount Received"].cast(pl.Float32).to_numpy()
    log_paid = np.log1p(np.maximum(amount_paid, 0.0))
    log_received = np.log1p(np.maximum(amount_received, 0.0))

    out_count = np.bincount(from_id, minlength=n_nodes).astype(np.float32)
    in_count = np.bincount(to_id, minlength=n_nodes).astype(np.float32)

    num_currencies = len(currency_categories)
    out_amount_by_currency = np.zeros((n_nodes, num_currencies), dtype=np.float32)
    in_amount_by_currency = np.zeros((n_nodes, num_currencies), dtype=np.float32)
    np.add.at(out_amount_by_currency, (from_id, sent_currency), log_paid)
    np.add.at(in_amount_by_currency, (to_id, recv_currency), log_received)

    out_total = np.bincount(from_id, weights=log_paid, minlength=n_nodes).astype(np.float32)
    in_total = np.bincount(to_id, weights=log_received, minlength=n_nodes).astype(np.float32)
    out_mean = out_total / np.maximum(out_count, 1.0)
    in_mean = in_total / np.maximum(in_count, 1.0)

    interbank = (from_bank_codes != to_bank_codes).astype(np.float32)
    out_interbank = np.bincount(from_id, weights=interbank, minlength=n_nodes).astype(np.float32)
    in_interbank = np.bincount(to_id, weights=interbank, minlength=n_nodes).astype(np.float32)

    node_features_raw = np.concatenate(
        [
            np.stack(
                [
                    np.log1p(out_count),
                    np.log1p(in_count),
                    out_mean,
                    in_mean,
                    out_interbank / np.maximum(out_count, 1.0),
                    in_interbank / np.maximum(in_count, 1.0),
                ],
                axis=1,
            ),
            out_amount_by_currency,
            in_amount_by_currency,
        ],
        axis=1,
    )
    node_scaler_mean = np.asarray(preprocessing["node_scaler_mean"], dtype=np.float32)
    node_scaler_scale = np.asarray(preprocessing["node_scaler_scale"], dtype=np.float32)
    node_features = ((node_features_raw - node_scaler_mean) / node_scaler_scale).astype(np.float32)

    edge_numeric_raw = np.stack(
        [
            timestamp_minutes,
            log_paid,
            log_received,
            log_received - log_paid,
            (sent_currency == recv_currency).astype(np.float32),
            interbank,
        ],
        axis=1,
    ).astype(np.float32)
    edge_scaler_mean = np.asarray(preprocessing["edge_scaler_mean"], dtype=np.float32)
    edge_scaler_scale = np.asarray(preprocessing["edge_scaler_scale"], dtype=np.float32)
    edge_numeric = ((edge_numeric_raw - edge_scaler_mean) / edge_scaler_scale).astype(np.float32)

    adj_in, adj_out = build_sparse_mean_adjacency(from_id, to_id, n_nodes)
    labels = (
        df["Is Laundering"].cast(pl.Float32).to_numpy()
        if "Is Laundering" in df.columns
        else np.zeros(num_rows, dtype=np.float32)
    )

    return PreparedGraph(
        node_features=torch.from_numpy(node_features.copy()),
        node_bank_ids=torch.from_numpy(node_bank_ids.copy()),
        train_src=torch.from_numpy(from_id.copy()),
        train_dst=torch.from_numpy(to_id.copy()),
        train_adj_in=adj_in,
        train_adj_out=adj_out,
        edge_src=torch.from_numpy(from_id.copy()),
        edge_dst=torch.from_numpy(to_id.copy()),
        edge_sent_currency=torch.from_numpy(sent_currency.copy()),
        edge_recv_currency=torch.from_numpy(recv_currency.copy()),
        edge_payment_format=torch.from_numpy(format_codes.copy()),
        edge_numeric=torch.from_numpy(edge_numeric.copy()),
        labels=torch.from_numpy(labels.copy()),
        splits={"inference": torch.arange(num_rows, dtype=torch.long)},
        cardinalities={
            "banks": len(bank_categories),
            "currencies": len(currency_categories),
            "payment_formats": len(format_categories),
        },
        metadata={"num_nodes": n_nodes, "num_edges": num_rows},
        preprocessing=preprocessing,
    )


def move_graph_to_device(graph: PreparedGraph, device: torch.device) -> PreparedGraph:
    return PreparedGraph(
        node_features=graph.node_features.to(device),
        node_bank_ids=graph.node_bank_ids.to(device),
        train_src=graph.train_src.to(device),
        train_dst=graph.train_dst.to(device),
        train_adj_in=graph.train_adj_in.to(device),
        train_adj_out=graph.train_adj_out.to(device),
        edge_src=graph.edge_src.to(device),
        edge_dst=graph.edge_dst.to(device),
        edge_sent_currency=graph.edge_sent_currency.to(device),
        edge_recv_currency=graph.edge_recv_currency.to(device),
        edge_payment_format=graph.edge_payment_format.to(device),
        edge_numeric=graph.edge_numeric.to(device),
        labels=graph.labels.to(device),
        splits={key: value.to(device) for key, value in graph.splits.items()},
        cardinalities=graph.cardinalities,
        metadata=graph.metadata,
        preprocessing=graph.preprocessing,
    )


def get_training_indices(
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    neg_pos_ratio: int,
    generator: torch.Generator,
) -> torch.Tensor:
    train_labels = labels[train_indices]
    positive_mask = train_labels == 1
    positive_indices = train_indices[positive_mask]
    negative_indices = train_indices[~positive_mask]

    if len(positive_indices) == 0:
        raise ValueError("Training split contains no positive laundering examples.")

    target_neg = min(len(negative_indices), len(positive_indices) * neg_pos_ratio)
    perm = torch.randperm(len(negative_indices), generator=generator, device=negative_indices.device)
    sampled_neg = negative_indices[perm[:target_neg]]

    epoch_indices = torch.cat([positive_indices, sampled_neg], dim=0)
    epoch_perm = torch.randperm(len(epoch_indices), generator=generator, device=epoch_indices.device)
    return epoch_indices[epoch_perm]


def batched_edge_logits(
    model: AMLTransactionGNN,
    node_embeddings: torch.Tensor,
    graph: PreparedGraph,
    indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        logits = model.score_edges(
            node_embeddings=node_embeddings,
            edge_src=graph.edge_src[batch_idx],
            edge_dst=graph.edge_dst[batch_idx],
            edge_sent_currency=graph.edge_sent_currency[batch_idx],
            edge_recv_currency=graph.edge_recv_currency[batch_idx],
            edge_payment_format=graph.edge_payment_format[batch_idx],
            edge_numeric=graph.edge_numeric[batch_idx],
        )
        outputs.append(logits)
    return torch.cat(outputs, dim=0)


def threshold_with_best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (y_score >= threshold).astype(np.int32)
        current_f1 = f1_score(y_true, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (y_score >= threshold).astype(np.int32)
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))
    metrics = {
        "roc_auc": roc_auc,
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "positive_rate_pred": float(preds.mean()),
    }
    return metrics


def evaluate_split(
    model: AMLTransactionGNN,
    graph: PreparedGraph,
    split: str,
    batch_size: int,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode_nodes(
            graph.node_features,
            graph.node_bank_ids,
            graph.train_adj_in,
            graph.train_adj_out,
        )
        split_indices = graph.splits[split]
        logits = batched_edge_logits(model, node_embeddings, graph, split_indices, batch_size)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        y_true = graph.labels[split_indices].detach().cpu().numpy()
        metrics = compute_metrics(y_true, scores, threshold)
    return metrics, y_true, scores


def train_model(
    graph: PreparedGraph,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    edge_batch_size: int,
    neg_pos_ratio: int,
    seed: int,
) -> tuple[AMLTransactionGNN, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = move_graph_to_device(graph, device)

    model = AMLTransactionGNN(
        num_node_features=graph.node_features.shape[1],
        num_banks=graph.cardinalities["banks"],
        num_currencies=graph.cardinalities["currencies"],
        num_formats=graph.cardinalities["payment_formats"],
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_indices = graph.splits["train"]
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_ap = -math.inf
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_indices = get_training_indices(graph.labels, train_indices, neg_pos_ratio, generator)
        epoch_labels = graph.labels[epoch_indices]

        pos_count = int(epoch_labels.sum().item())
        neg_count = int(len(epoch_labels) - pos_count)
        pos_weight = max(1.0, neg_count / max(pos_count, 1))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

        optimizer.zero_grad()
        node_embeddings = model.encode_nodes(
            graph.node_features,
            graph.node_bank_ids,
            graph.train_adj_in,
            graph.train_adj_out,
        )
        logits = batched_edge_logits(model, node_embeddings, graph, epoch_indices, edge_batch_size)
        loss = loss_fn(logits, epoch_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_metrics, val_y, val_score = evaluate_split(model, graph, "val", edge_batch_size, threshold=0.5)
        _, val_best_f1 = threshold_with_best_f1(val_y, val_score)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "val_average_precision": val_metrics["average_precision"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_f1_at_0_5": val_metrics["f1"],
                "val_best_f1_scan": val_best_f1,
            }
        )
        print(
            f"Epoch {epoch:02d} | loss={loss.item():.4f} | "
            f"val_ap={val_metrics['average_precision']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["average_precision"] > best_val_ap:
            best_val_ap = val_metrics["average_precision"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model = model.to(device)

    val_metrics, val_y, val_score = evaluate_split(model, graph, "val", edge_batch_size, threshold=0.5)
    best_threshold, _ = threshold_with_best_f1(val_y, val_score)
    final_val_metrics, _, _ = evaluate_split(model, graph, "val", edge_batch_size, threshold=best_threshold)
    final_test_metrics, _, _ = evaluate_split(model, graph, "test", edge_batch_size, threshold=best_threshold)

    summary = {
        "device": device.type,
        "selected_threshold_from_validation": best_threshold,
        "history": history,
        "validation_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
    }
    return model, summary


def save_model_checkpoint(
    path: Path,
    model: AMLTransactionGNN,
    graph: PreparedGraph,
    hidden_dim: int,
    summary: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": {
            "num_node_features": int(graph.node_features.shape[1]),
            "num_banks": int(graph.cardinalities["banks"]),
            "num_currencies": int(graph.cardinalities["currencies"]),
            "num_formats": int(graph.cardinalities["payment_formats"]),
            "hidden_dim": int(hidden_dim),
        },
        "preprocessing": graph.preprocessing,
        "threshold": float(summary["selected_threshold_from_validation"]),
        "graph_metadata": graph.metadata,
        "validation_metrics": summary["validation_metrics"],
        "test_metrics": summary["test_metrics"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[AMLTransactionGNN, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = AMLTransactionGNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a directed GNN on IBM AML transactions.")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to a CSV file or .zip file with HI-Small_Trans.")
    parser.add_argument(
        "--download-dir",
        type=str,
        default="/tmp/ibm_aml_dataset",
        help="Directory used when auto-downloading the public HI-Small mirror.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="Optional cap for practical local runs. Set to 0 or omit to load the full file.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--edge-batch-size", type=int, default=32_768)
    parser.add_argument("--neg-pos-ratio", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metrics-json", type=str, default=None, help="Optional path to save the final metrics JSON.")
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(Path(__file__).with_name("aml_gnn_checkpoint.pt")),
        help="Path where the trained model checkpoint is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_path = resolve_dataset_path(args)
    max_rows = None if not args.max_rows or args.max_rows <= 0 else args.max_rows

    print(f"Loading dataset from: {dataset_path}")
    df = read_transactions(dataset_path, max_rows=max_rows)
    print(f"Loaded {len(df):,} transactions.")

    graph = prepare_graph(df)
    print("Prepared graph:")
    for key, value in graph.metadata.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    model, summary = train_model(
        graph=graph,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        edge_batch_size=args.edge_batch_size,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=args.seed,
    )

    report = {
        "dataset_path": str(dataset_path),
        "max_rows": max_rows,
        "graph_metadata": graph.metadata,
        **summary,
    }

    print("\nFinal evaluation")
    print(json.dumps(report, indent=2))

    if args.model_output:
        model_output_path = Path(args.model_output).expanduser().resolve()
        save_model_checkpoint(model_output_path, model, graph, args.hidden_dim, summary)
        print(f"Saved model checkpoint to {model_output_path}")

    if args.metrics_json:
        output_path = Path(args.metrics_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
