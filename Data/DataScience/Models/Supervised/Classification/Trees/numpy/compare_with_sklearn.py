#!/usr/bin/env python3
"""Compare the NumPy tree against sklearn on the same dataset."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text

from decision_tree_numpy import DecisionTreeConfig, NumpyDecisionTreeClassifier
from decision_tree_numpy.data import load_dataset, resolve_feature_names
from decision_tree_numpy.datasets import (
    DEFAULT_DATASET,
    dataset_choices,
    get_dataset_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train sklearn and NumPy trees and compare predictions."
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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-grid-points",
        type=int,
        default=50000,
        help="Maximum observed-domain grid size to compare.",
    )
    return parser.parse_args()


def build_observed_domain_grid(
    x: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray | None, int]:
    domains = [np.unique(x[:, index]) for index in range(x.shape[1])]
    total_points = int(np.prod([len(domain) for domain in domains]))
    if total_points > max_points:
        return None, total_points

    mesh = np.meshgrid(*domains, indexing="ij")
    return np.stack([axis.ravel() for axis in mesh], axis=1), total_points


def main() -> None:
    args = parse_args()
    dataset_spec = get_dataset_spec(args.dataset)
    data_path = args.data or dataset_spec.path
    delimiter = args.delimiter if args.data else dataset_spec.delimiter
    skip_rows = args.skip_rows if args.data else dataset_spec.skip_rows

    x, y = load_dataset(data_path, delimiter=delimiter, skip_rows=skip_rows)
    feature_names = resolve_feature_names(dataset_spec.feature_names, x.shape[1])

    sklearn_model = DecisionTreeClassifier(
        criterion=args.criterion,
        max_depth=args.max_depth,
        random_state=args.random_state,
    ).fit(x, y)
    numpy_model = NumpyDecisionTreeClassifier(
        DecisionTreeConfig(
            criterion=args.criterion,
            max_depth=args.max_depth,
            random_state=args.random_state,
            feature_names=feature_names,
            target_name=dataset_spec.target_name,
        )
    ).fit(
        x,
        y,
        feature_names=feature_names,
        target_name=dataset_spec.target_name,
    )

    training_sklearn = sklearn_model.predict(x)
    training_numpy = numpy_model.predict(x)
    grid, grid_total_points = build_observed_domain_grid(
        x,
        max_points=args.max_grid_points,
    )
    if grid is not None:
        grid_sklearn = sklearn_model.predict(grid)
        grid_numpy = numpy_model.predict(grid)
        grid_match = np.array_equal(grid_sklearn, grid_numpy)
    else:
        grid_sklearn = None
        grid_numpy = None
        grid_match = True

    training_match = np.array_equal(training_sklearn, training_numpy)
    sklearn_accuracy = float(np.mean(training_sklearn == y))
    numpy_accuracy = float(np.mean(training_numpy == y))

    print("Environment:")
    print("  Python:", platform.python_version())
    print("  NumPy:", np.__version__)
    print("  scikit-learn:", sklearn.__version__)
    print()
    print("Dataset:", dataset_spec.name)
    print("Data file:", data_path)
    print("Rows:", len(x))
    print("Features:", feature_names)
    print()
    print("sklearn tree:")
    print(export_text(sklearn_model, feature_names=feature_names), end="")
    print("NumPy tree:")
    print(numpy_model.export_text(), end="")
    print("sklearn training accuracy:", round(sklearn_accuracy, 4))
    print("NumPy training accuracy:", round(numpy_accuracy, 4))
    print("Training predictions match:", training_match)
    if grid is None:
        print(
            "Observed-domain grid skipped:",
            f"{grid_total_points} points exceeds --max-grid-points={args.max_grid_points}",
        )
    else:
        print("Observed-domain predictions match:", grid_match)

    if not training_match:
        print("sklearn training predictions:", training_sklearn.tolist())
        print("NumPy training predictions:", training_numpy.tolist())
    if grid is not None and not grid_match:
        mismatch_indices = np.flatnonzero(grid_sklearn != grid_numpy)
        preview_indices = mismatch_indices[:20]
        print("Observed-domain mismatches:", int(len(mismatch_indices)))
        print("First mismatching samples:", grid[preview_indices].tolist())
        print("sklearn predictions:", grid_sklearn[preview_indices].tolist())
        print("NumPy predictions:", grid_numpy[preview_indices].tolist())

    if not training_match or not grid_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
