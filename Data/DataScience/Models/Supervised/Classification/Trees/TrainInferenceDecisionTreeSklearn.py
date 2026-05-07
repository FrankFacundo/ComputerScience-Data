#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and run inference with a sklearn decision tree classifier.

Mini README
-----------

Default dataset format:
    A B C

Where A and B are input features and C is the target class.

Train command:
    python TrainInferenceDecisionTreeSklearn.py train

Inference command:
    python TrainInferenceDecisionTreeSklearn.py infer --sample 0 1

Multiple inference samples:
    python TrainInferenceDecisionTreeSklearn.py infer --sample 0 1 --sample 3 2

The train command saves the model to:
    decision_tree_model.joblib

If your default Python environment does not have sklearn installed, run with an
environment that has numpy and scikit-learn, for example:
    conda run -n advanced_ml_banking python TrainInferenceDecisionTreeSklearn.py train
    conda run -n advanced_ml_banking python TrainInferenceDecisionTreeSklearn.py infer --sample 0 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = CURRENT_DIR / "file1.txt"
DEFAULT_MODEL_FILE = CURRENT_DIR / "decision_tree_model.joblib"
DEFAULT_FEATURE_NAMES = ["A", "B"]
TARGET_NAME = "C"


def load_dataset(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path, dtype=int)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError("Dataset must contain at least 3 columns: A B C")

    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def get_feature_names(feature_count: int) -> list[str]:
    names = DEFAULT_FEATURE_NAMES[:feature_count]
    names.extend(f"X{i + 1}" for i in range(len(names), feature_count))
    return names


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    criterion: str,
    max_depth: int | None,
    random_state: int,
    test_size: float,
) -> tuple[DecisionTreeClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test


def parse_samples(raw_samples: list[list[int]] | None, feature_count: int) -> np.ndarray:
    if not raw_samples:
        return np.array([[0, 1]], dtype=int)

    samples = np.array(raw_samples, dtype=int)
    if samples.shape[1] != feature_count:
        raise ValueError(
            f"Each sample must contain {feature_count} feature values, "
            f"but got {samples.shape[1]}."
        )
    return samples


def save_model(model: DecisionTreeClassifier, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path) -> DecisionTreeClassifier:
    return joblib.load(model_path)


def run_train(args: argparse.Namespace) -> None:
    x, y = load_dataset(args.data)
    model, x_train, x_test, y_train, y_test = train_model(
        x=x,
        y=y,
        criterion=args.criterion,
        max_depth=args.max_depth,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    y_pred = model.predict(x_test)
    feature_names = get_feature_names(x.shape[1])
    save_model(model, args.model)

    print("Dataset:", args.data)
    print("Rows:", len(x), "| Train:", len(x_train), "| Test:", len(x_test))
    print("Features:", feature_names)
    print("Target:", TARGET_NAME)
    print("Model saved:", args.model)
    print()
    print("Learned tree:")
    print(export_text(model, feature_names=feature_names))
    print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))


def run_infer(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    samples = parse_samples(args.sample, feature_count=model.n_features_in_)
    predictions = model.predict(samples)

    print("Model loaded:", args.model)
    print("Inference:")
    for sample, prediction in zip(samples, predictions):
        print(f"  sample={sample.tolist()} -> predicted {TARGET_NAME}={prediction}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run inference with sklearn DecisionTreeClassifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save the tree model.")
    train_parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Path to a whitespace-delimited dataset. Last column is the class label.",
    )
    train_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help="Path where the trained model will be saved.",
    )
    train_parser.add_argument(
        "--criterion",
        choices=["gini", "entropy", "log_loss"],
        default="entropy",
        help="Decision tree split criterion.",
    )
    train_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth. Default lets the tree expand fully.",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Fraction of rows used for testing.",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split and tree model.",
    )
    train_parser.set_defaults(func=run_train)

    infer_parser = subparsers.add_parser("infer", help="Load a model and predict.")
    infer_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help="Path to the saved model.",
    )
    infer_parser.add_argument(
        "--sample",
        type=int,
        nargs="+",
        action="append",
        help="Feature values for inference. Repeat this flag for multiple samples.",
    )
    infer_parser.set_defaults(func=run_infer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
