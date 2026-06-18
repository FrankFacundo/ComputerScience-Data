#!/usr/bin/env python3
"""Run inference with a saved NumPy-only decision tree model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from decision_tree_numpy.datasets import (
    DEFAULT_DATASET,
    dataset_choices,
    get_dataset_spec,
)
from decision_tree_numpy.serialization import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a NumPy decision tree JSON model and predict samples."
    )
    parser.add_argument(
        "--dataset",
        choices=dataset_choices(),
        default=DEFAULT_DATASET,
        help="Built-in dataset whose default model and sample should be used.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the saved JSON model.",
    )
    parser.add_argument(
        "--sample",
        type=float,
        nargs="+",
        action="append",
        help="Feature values for inference. Repeat this flag for multiple samples.",
    )
    return parser.parse_args()


def parse_samples(raw_samples: list[list[float]] | None, feature_count: int) -> np.ndarray:
    if not raw_samples:
        return np.array([[0, 1]], dtype=float)

    samples = np.array(raw_samples, dtype=float)
    if samples.shape[1] != feature_count:
        raise ValueError(
            f"Each sample must contain {feature_count} feature values, "
            f"but got {samples.shape[1]}."
        )
    return samples


def main() -> None:
    args = parse_args()
    dataset_spec = get_dataset_spec(args.dataset)
    model_path = args.model or dataset_spec.default_model_path

    model = load_model(model_path)
    raw_samples = args.sample or [dataset_spec.default_sample]
    samples = parse_samples(raw_samples, feature_count=model.n_features_in_)
    predictions = model.predict(samples)
    probabilities = model.predict_proba(samples)

    print("Dataset:", dataset_spec.name)
    print("Model loaded:", model_path)
    print("Inference:")
    for sample, prediction, probability in zip(samples, predictions, probabilities):
        probability_text = {
            str(label): round(float(probability[index]), 4)
            for index, label in enumerate(model.classes_)
        }
        print(
            f"  sample={sample.tolist()} -> predicted "
            f"{model.target_name_}={model._format_value(prediction)} "
            f"proba={probability_text}"
        )


if __name__ == "__main__":
    main()
