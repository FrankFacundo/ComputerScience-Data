"""Verify that the NumPy and PyTorch one-layer models train identically."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

from one_layer_dnn_common import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    epochs_to_print,
    inference_data,
    initial_parameters,
    training_data,
)
from one_layer_dnn_numpy import NumpyOneLayerDNN
from one_layer_dnn_torch import TorchOneLayerDNN, loss_and_gradients


ABSOLUTE_TOLERANCE = 1e-12
RELATIVE_TOLERANCE = 1e-12


def assert_same(name: str, numpy_value: object, torch_value: object) -> float:
    """Assert equivalence and return the largest absolute difference."""
    expected = np.asarray(numpy_value)
    if isinstance(torch_value, torch.Tensor):
        actual = torch_value.detach().numpy()
    else:
        actual = np.asarray(torch_value)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE,
        err_msg=f"NumPy/PyTorch mismatch in {name}",
    )
    return float(np.max(np.abs(actual - expected)))


def compare(epochs: int, learning_rate: float) -> None:
    features, labels = training_data()
    weight, bias = initial_parameters()

    numpy_model = NumpyOneLayerDNN(weight, bias)
    torch_model = TorchOneLayerDNN(weight, bias)
    torch_features = torch.from_numpy(features)
    torch_labels = torch.from_numpy(labels)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)

    print("Lockstep training comparison (float64)")
    print(f"  tolerance: atol={ABSOLUTE_TOLERANCE}, rtol={RELATIVE_TOLERANCE}")
    print("  comparing: logits, probabilities, loss, gradients, and updated parameters\n")

    largest_difference = 0.0
    for epoch in range(1, epochs + 1):
        numpy_details = numpy_model.loss_and_gradients(features, labels)
        torch_details = loss_and_gradients(
            torch_model, torch_features, torch_labels, criterion
        )

        for name in (
            "logits",
            "probabilities",
            "loss",
            "grad_logits",
            "grad_weight",
            "grad_bias",
        ):
            difference = assert_same(
                f"epoch {epoch} {name}", numpy_details[name], torch_details[name]
            )
            largest_difference = max(largest_difference, difference)

        numpy_model.apply_gradients(
            numpy_details["grad_weight"],
            numpy_details["grad_bias"],
            learning_rate,
        )
        optimizer.step()

        difference = assert_same(
            f"epoch {epoch} updated weight",
            numpy_model.weight,
            torch_model.linear.weight,
        )
        largest_difference = max(largest_difference, difference)
        difference = assert_same(
            f"epoch {epoch} updated bias",
            numpy_model.bias,
            torch_model.linear.bias,
        )
        largest_difference = max(largest_difference, difference)

        if epochs_to_print(epoch, epochs):
            print(
                f"  epoch {epoch:>3}/{epochs}: "
                f"loss={numpy_details['loss']:.12f}, all checks passed"
            )

    test_features, expected_labels = inference_data()
    numpy_logits = numpy_model.forward(test_features)
    numpy_probabilities = numpy_model.predict_proba(test_features)
    numpy_predictions = numpy_model.predict(test_features)

    torch_model.eval()
    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(test_features))
        torch_probabilities = torch.softmax(torch_logits, dim=1)
        torch_predictions = torch.argmax(torch_probabilities, dim=1)

    for name, numpy_value, torch_value in (
        ("inference logits", numpy_logits, torch_logits),
        ("inference probabilities", numpy_probabilities, torch_probabilities),
        ("inference predictions", numpy_predictions, torch_predictions),
    ):
        difference = assert_same(name, numpy_value, torch_value)
        largest_difference = max(largest_difference, difference)

    print("\nInference comparison")
    print(f"  NumPy predictions:    {numpy_predictions}")
    print(f"  PyTorch predictions:  {torch_predictions.numpy()}")
    print(f"  expected labels:      {expected_labels}")
    print(f"  probabilities:\n{numpy_probabilities}")
    print("\nPASS: NumPy and PyTorch training and inference are equivalent.")
    print(f"Largest observed absolute difference: {largest_difference:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare(epochs=args.epochs, learning_rate=args.learning_rate)
