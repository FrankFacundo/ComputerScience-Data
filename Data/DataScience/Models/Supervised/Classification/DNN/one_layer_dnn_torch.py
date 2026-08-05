"""Train the same one-layer softmax classifier using PyTorch."""

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


class TorchOneLayerDNN(nn.Module):
    """The PyTorch equivalent of ``NumpyOneLayerDNN``."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        super().__init__()
        output_features, input_features = weight.shape
        self.linear = nn.Linear(
            input_features, output_features, bias=True, dtype=torch.float64
        )
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(weight))
            self.linear.bias.copy_(torch.from_numpy(bias))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw logits; CrossEntropyLoss applies log-softmax internally."""
        return self.linear(features)


def loss_and_gradients(
    model: TorchOneLayerDNN,
    features: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
) -> dict[str, torch.Tensor | float]:
    """Run PyTorch forward and backward passes without updating parameters."""
    model.zero_grad(set_to_none=True)
    logits = model(features)
    logits.retain_grad()
    loss = criterion(logits, labels)
    probabilities = torch.softmax(logits, dim=1)
    loss.backward()

    return {
        "logits": logits.detach().clone(),
        "probabilities": probabilities.detach().clone(),
        "loss": float(loss.detach()),
        "grad_logits": logits.grad.detach().clone(),
        "grad_weight": model.linear.weight.grad.detach().clone(),
        "grad_bias": model.linear.bias.grad.detach().clone(),
    }


def train_torch(
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    verbose: bool = True,
) -> TorchOneLayerDNN:
    """Train from the shared initial state and return the trained model."""
    features_numpy, labels_numpy = training_data()
    features = torch.from_numpy(features_numpy)
    labels = torch.from_numpy(labels_numpy)
    weight, bias = initial_parameters()
    model = TorchOneLayerDNN(weight, bias)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if verbose:
        first = loss_and_gradients(model, features, labels, criterion)
        print("PyTorch: first training step in detail")
        print(f"  X shape:              {tuple(features.shape)}")
        print(f"  weight shape:         {tuple(model.linear.weight.shape)}")
        print(f"  logits[0]:            {first['logits'][0].numpy()}")
        print(f"  probabilities[0]:     {first['probabilities'][0].numpy()}")
        print(f"  target[0]:            {int(labels[0])}")
        print(f"  mean loss:            {first['loss']:.12f}")
        print(f"  logits.grad[0]:       {first['grad_logits'][0].numpy()}")
        print(f"  weight.grad:\n{first['grad_weight'].numpy()}")
        print(f"  bias.grad:            {first['grad_bias'].numpy()}")
        print("\nPyTorch training")

    for epoch in range(1, epochs + 1):
        details = loss_and_gradients(model, features, labels, criterion)
        optimizer.step()
        if verbose and epochs_to_print(epoch, epochs):
            print(f"  epoch {epoch:>3}/{epochs}: loss before update = {details['loss']:.12f}")

    if verbose:
        final = loss_and_gradients(model, features, labels, criterion)
        test_features_numpy, test_labels = inference_data()
        test_features = torch.from_numpy(test_features_numpy)
        model.eval()
        with torch.no_grad():
            logits = model(test_features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        print(f"  final training loss:  {final['loss']:.12f}")
        print("\nPyTorch inference")
        print(f"  probabilities:\n{probabilities.numpy()}")
        print(f"  predictions:          {predictions.numpy()}")
        print(f"  expected:             {test_labels}")
        accuracy = np.mean(predictions.numpy() == test_labels)
        print(f"  accuracy:             {accuracy:.2%}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_torch(epochs=args.epochs, learning_rate=args.learning_rate)
