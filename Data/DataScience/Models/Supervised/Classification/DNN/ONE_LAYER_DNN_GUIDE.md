# One-layer DNN: NumPy and PyTorch equivalence

This example is a deliberately small, deterministic multi-class classifier for
studying what PyTorch does during training and inference. It has **one trainable
affine layer and no hidden layer**. This architecture is also known as softmax
regression or multinomial logistic regression.

## Run the examples

From this directory:

```bash
python one_layer_dnn_numpy.py
python one_layer_dnn_torch.py
python compare_one_layer_dnn.py
```

The first two scripts train independently and print the first forward/backward
pass in detail. The comparison script trains both implementations in lockstep
and checks every intermediate value after every epoch. It raises an error if a
value differs beyond `atol=1e-12, rtol=1e-12`.

Both implementations use:

- the same fixed training and inference data;
- the same initial weights and biases;
- 64-bit floating point values;
- mean cross-entropy loss;
- full-batch plain SGD, without momentum or weight decay; and
- the same number and order of updates.

These controls are important. Two correct training programs need not finish
with the same parameters if their random initialization, data order, batch
sizes, precision, optimizer settings, or loss reduction differ.

## Operation-by-operation mapping

Let `N` be the batch size, `D` the number of features, and `C` the number of
classes. The data shapes are:

| Value | Shape |
| --- | --- |
| `X` | `(N, D)` |
| `weight` | `(C, D)` |
| `bias` | `(C,)` |
| `logits` and `probabilities` | `(N, C)` |

### 1. Forward pass

NumPy makes the affine operation explicit:

```python
logits = X @ weight.T + bias
```

PyTorch performs exactly that operation with:

```python
linear = torch.nn.Linear(D, C)
logits = linear(X)
```

The output is called *logits*: unnormalized class scores. There is no activation
inside the model because PyTorch's `CrossEntropyLoss` expects raw logits.

### 2. Stable softmax and cross-entropy

Subtracting each row's maximum does not change its softmax and prevents
`exp(logit)` from overflowing:

```python
shifted = logits - logits.max(axis=1, keepdims=True)
log_probabilities = shifted - np.log(
    np.exp(shifted).sum(axis=1, keepdims=True)
)
probabilities = np.exp(log_probabilities)
loss = -log_probabilities[np.arange(N), labels].mean()
```

The equivalent PyTorch operation is:

```python
loss = torch.nn.CrossEntropyLoss(reduction="mean")(logits, labels)
```

`CrossEntropyLoss` combines log-softmax and negative log-likelihood. Labels are
integer class indexes such as `0, 1, 2`; they are not one-hot vectors.

### 3. Backward pass

For softmax followed by mean cross-entropy, the derivative simplifies to:

```python
d_logits = probabilities.copy()
d_logits[np.arange(N), labels] -= 1.0
d_logits /= N

d_weight = d_logits.T @ X
d_bias = d_logits.sum(axis=0)
```

PyTorch creates the same gradients through automatic differentiation:

```python
loss.backward()
d_weight = linear.weight.grad
d_bias = linear.bias.grad
```

The division by `N` in NumPy corresponds to `reduction="mean"` in PyTorch.

### 4. SGD parameter update

NumPy:

```python
weight -= learning_rate * d_weight
bias -= learning_rate * d_bias
```

PyTorch:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer.step()
```

`zero_grad()` is required before the next PyTorch backward pass because PyTorch
accumulates gradients by default. The NumPy implementation creates fresh
gradient arrays on every call.

### 5. Inference

Training is finished before the separate inference examples are used. In both
implementations, inference computes logits, probabilities, and the largest-score
class:

```python
probabilities = softmax(logits)
predictions = probabilities.argmax(axis=1)
```

PyTorch wraps inference in `torch.no_grad()` so it does not build an autograd
graph. `model.eval()` is also standard practice; it does not numerically change
this one-layer model because there is no dropout or batch normalization.

## Why "equivalent" is not bit-for-bit identity

The comparison uses a tight floating-point tolerance rather than `==`. NumPy
and PyTorch may choose different low-level implementations for operations such
as matrix multiplication, exponential, and logarithm. Tiny rounding differences
are expected even when the mathematics and results are the same. Using float64
keeps those differences small and makes the educational comparison clearer.
