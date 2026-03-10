# Hungarian Algorithm Tutorial

This folder contains a full Python tutorial for the Hungarian algorithm:

- `algorithm.py`: a from-scratch primal-dual Hungarian implementation with trace capture.
- `synthetic_cases.py`: hand-built and generated matrices for intuition and testing.
- `visualization.py`: heatmaps, bipartite graphs, spatial plots, and trace animation helpers.
- `tutorial_marimo.py`: the marimo notebook that explains the math, code, and experiments.
- `test_hungarian.py`: local regression tests against brute force on small instances.

## Run the notebook

```bash
python3 -m marimo edit Scripts/python/hungarian/tutorial_marimo.py
```

## Run the tests

```bash
pytest -q Scripts/python/hungarian/test_hungarian.py
```
