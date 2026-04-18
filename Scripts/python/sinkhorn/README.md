# Sinkhorn Algorithm Tutorial

This folder contains a full Python tutorial for the Sinkhorn algorithm:

- `algorithm.py`: entropic optimal transport and the Sinkhorn scaling implementation.
- `synthetic_cases.py`: balanced transport problems in 1D and 2D.
- `visualization.py`: cost, kernel, coupling, transport-graph, and convergence plots.
- `tutorial_marimo.py`: the interactive marimo notebook.
- `test_sinkhorn.py`: regression tests for convergence and trace behavior.

## Run the notebook

```bash
python3 -m marimo edit Scripts/python/sinkhorn/tutorial_marimo.py
```

## Run the tests

```bash
pytest -q Scripts/python/sinkhorn/test_sinkhorn.py
```
