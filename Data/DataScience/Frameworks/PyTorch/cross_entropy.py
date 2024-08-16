import numpy as np
import torch
import torch.nn as nn


def cross_entropy_loss_torch():
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Example data
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.3, 0.5, 2.0]])
    labels = torch.tensor([0, 1, 2])  # True class indices

    # Calculate loss
    loss = loss_fn(input=logits, target=labels)
    print(f"Cross-Entropy Loss Torch: {loss.item()}")


def softmax(logits):
    """
    Compute softmax probabilities from logits.

    Args:
    - logits (numpy.ndarray): Logits, shape (N, C)

    Returns:
    - probabilities (numpy.ndarray): Softmax probabilities, shape (N, C)
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss_numpy(epsilon=1e-12):
    """
    Compute the cross-entropy loss between logits and targets.

    Args:
    - logits (numpy.ndarray): Logits, shape (N, C)
    - targets (numpy.ndarray): One-hot encoded true labels, shape (N, C)
    - epsilon (float): Small value to avoid log(0)

    Returns:
    - loss (float): Computed cross-entropy loss
    """

    # Example logits (same as in the torch example)
    logits = np.array([[2.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.3, 0.5, 2.0]])

    # Example targets (one-hot encoded)
    targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Convert logits to probabilities using softmax
    predictions = softmax(logits)

    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)

    # Compute cross-entropy loss
    N = predictions.shape[0]
    loss = -np.sum(targets * np.log(predictions)) / N
    print(f"Cross-Entropy Loss Numpy: {loss}")


cross_entropy_loss_torch()
cross_entropy_loss_numpy()

"""
Output:
Cross-Entropy Loss Torch: 0.3326704204082489
Cross-Entropy Loss Numpy: 0.3326704178956424
"""
