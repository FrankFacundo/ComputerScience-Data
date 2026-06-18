#!/usr/bin/env python3
"""Train the NumPy-only decision tree and save it as JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from decision_tree_numpy import DecisionTreeConfig, NumpyDecisionTreeClassifier
from decision_tree_numpy.data import (
    accuracy_score_numpy,
    confusion_matrix_numpy,
    load_dataset,
    resolve_feature_names,
)
from decision_tree_numpy.datasets import (
    DEFAULT_DATASET,
    dataset_choices,
    get_dataset_spec,
)
from decision_tree_numpy.serialization import save_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a NumPy-only DecisionTreeClassifier."
    )
    parser.add_argument(
        "--dataset",
        choices=dataset_choices(),
        default=DEFAULT_DATASET,
        help="Built-in dataset to use when --data is not provided.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Whitespace-delimited dataset. Last column is the class label.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path where the JSON model will be saved.",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Delimiter for --data. Omit for whitespace-delimited files.",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Rows to skip at the top of --data.",
    )
    parser.add_argument(
        "--criterion",
        choices=["gini", "entropy", "log_loss"],
        default="entropy",
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_spec = get_dataset_spec(args.dataset)
    data_path = args.data or dataset_spec.path
    model_path = args.model or dataset_spec.default_model_path
    delimiter = args.delimiter if args.data else dataset_spec.delimiter
    skip_rows = args.skip_rows if args.data else dataset_spec.skip_rows

    x, y = load_dataset(data_path, delimiter=delimiter, skip_rows=skip_rows)
    feature_names = resolve_feature_names(dataset_spec.feature_names, x.shape[1])
    config = DecisionTreeConfig(
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        feature_names=feature_names,
        target_name=dataset_spec.target_name,
    )
    model = NumpyDecisionTreeClassifier(config=config).fit(
        x,
        y,
        feature_names=feature_names,
        target_name=dataset_spec.target_name,
    )
    predictions = model.predict(x)

    save_model(model, model_path)

    print("Dataset:", dataset_spec.name)
    print("Data file:", data_path)
    print("Rows:", len(x))
    print("Features:", model.feature_names_)
    print("Target:", model.target_name_)
    print("Model saved:", model_path)
    print()
    print("Learned NumPy tree:")
    print(model.export_text(), end="")
    print("Training accuracy:", round(accuracy_score_numpy(y, predictions), 4))
    print("Confusion matrix:")
    print(confusion_matrix_numpy(y, predictions, model.classes_))


if __name__ == "__main__":
    main()
