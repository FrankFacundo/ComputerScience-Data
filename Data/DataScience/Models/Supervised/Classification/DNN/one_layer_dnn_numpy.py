"""Train a one-layer softmax classifier using only NumPy.

The model contains one trainable affine layer and no hidden layer:

    logits = X @ weight.T + bias

All forward, loss, backward, update, and inference operations are explicit so
that they can be compared with the equivalent PyTorch operations.
"""

from __future__ import annotations

import argparse

import numpy as np

from one_layer_dnn_common import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    epochs_to_print,
    inference_data,
    initial_parameters,
    training_data,
)


class NumpyOneLayerDNN:
    """A single affine layer trained as a multi-class classifier."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = weight.copy()
        self.bias = bias.copy()

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Return raw class scores (logits), not probabilities."""
        return features @ self.weight.T + self.bias

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities with a numerically stable softmax."""
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted_logits)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def loss_and_gradients(
        self, features: np.ndarray, labels: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        """Compute mean cross-entropy and its analytical gradients."""
        logits = self.forward(features)

        # log_softmax is used for the loss because log(softmax(logits)) is less
        # numerically stable when the correct-class probability is very small.
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
        log_probabilities = shifted_logits - log_sum_exp
        probabilities = np.exp(log_probabilities)
        batch_size = features.shape[0]
        loss = -np.mean(log_probabilities[np.arange(batch_size), labels])

        # For mean cross-entropy combined with softmax:
        # d_logits = (probabilities - one_hot(labels)) / batch_size
        grad_logits = probabilities.copy()
        grad_logits[np.arange(batch_size), labels] -= 1.0
        grad_logits /= batch_size

        # logits = X @ weight.T + bias
        grad_weight = grad_logits.T @ features
        grad_bias = np.sum(grad_logits, axis=0)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "loss": float(loss),
            "grad_logits": grad_logits,
            "grad_weight": grad_weight,
            "grad_bias": grad_bias,
        }

    def apply_gradients(
        self, grad_weight: np.ndarray, grad_bias: np.ndarray, learning_rate: float
    ) -> None:
        """Perform one plain-SGD update."""
        self.weight -= learning_rate * grad_weight
        self.bias -= learning_rate * grad_bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities for inference."""
        return self.softmax(self.forward(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the class with the largest probability for each example."""
        return np.argmax(self.predict_proba(features), axis=1)


def train_numpy(
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    verbose: bool = True,
) -> NumpyOneLayerDNN:
    """Train from the shared initial state and return the trained model."""
    features, labels = training_data()
    weight, bias = initial_parameters()
    model = NumpyOneLayerDNN(weight, bias)

    if verbose:
        first = model.loss_and_gradients(features, labels)
        print("NumPy: first training step in detail")
        print(f"  X shape:              {features.shape}")
        print(f"  weight shape:         {model.weight.shape}")
        print(f"  logits[0]:            {first['logits'][0]}")
        print(f"  probabilities[0]:     {first['probabilities'][0]}")
        print(f"  target[0]:            {labels[0]}")
        print(f"  mean loss:            {first['loss']:.12f}")
        print(f"  d_logits[0]:          {first['grad_logits'][0]}")
        print(f"  d_weight:\n{first['grad_weight']}")
        print(f"  d_bias:               {first['grad_bias']}")
        print("\nNumPy training")

    for epoch in range(1, epochs + 1):
        details = model.loss_and_gradients(features, labels)
        model.apply_gradients(
            details["grad_weight"], details["grad_bias"], learning_rate
        )
        if verbose and epochs_to_print(epoch, epochs):
            print(f"  epoch {epoch:>3}/{epochs}: loss before update = {details['loss']:.12f}")

    if verbose:
        final = model.loss_and_gradients(features, labels)
        test_features, test_labels = inference_data()
        probabilities = model.predict_proba(test_features)
        predictions = model.predict(test_features)
        print(f"  final training loss:  {final['loss']:.12f}")
        print("\nNumPy inference")
        print(f"  probabilities:\n{probabilities}")
        print(f"  predictions:          {predictions}")
        print(f"  expected:             {test_labels}")
        print(f"  accuracy:             {np.mean(predictions == test_labels):.2%}")

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
    train_numpy(epochs=args.epochs, learning_rate=args.learning_rate)
