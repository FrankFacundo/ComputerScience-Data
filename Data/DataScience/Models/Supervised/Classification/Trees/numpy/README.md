# NumPy Decision Tree Classifier

This folder contains a Python + NumPy implementation of a decision tree
classifier. The implementation code does not use torch or sklearn. The
`compare_with_sklearn.py` script imports sklearn only to verify parity.

## Reproducible Environment

The sklearn comparison was verified with:

- Conda environment: `advanced_ml_banking`
- Python: `3.14.2`
- NumPy: `2.4.2`
- scikit-learn: `1.8.0`

## Dataset Format

Built-in datasets:

- `file1`: the small teaching dataset from `../file1.txt`
- `iris`: Fisher's public Iris flower dataset in `datasets/iris.csv`

The default built-in dataset is `iris`.

The Iris CSV is checked in so the NumPy train and inference commands stay
sklearn-free. sklearn is only used by `compare_with_sklearn.py`.

`file1` rows are whitespace-delimited:

```text
A B C
```

`A` and `B` are features. `C` is the class label.

`iris` rows are CSV records:

```text
sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,target
```

Targets are encoded as `0`, `1`, and `2`.

## Train

From this folder:

```bash
conda run -n advanced_ml_banking python train.py
conda run -n advanced_ml_banking python train.py --dataset file1
```

This saves one JSON model per dataset:

```text
decision_tree_numpy_model.json
decision_tree_numpy_iris_model.json
```

## Inference

```bash
conda run -n advanced_ml_banking python infer.py
conda run -n advanced_ml_banking python infer.py --dataset file1 --sample 0 1
```

Multiple samples:

```bash
conda run -n advanced_ml_banking python infer.py --sample 5.1 3.5 1.4 0.2 --sample 6.2 3.4 5.4 2.3
conda run -n advanced_ml_banking python infer.py --dataset file1 --sample 0 1 --sample 3 2
```

## Custom Dataset

Use `--data` when you want to load a file outside the built-in registry:

```bash
conda run -n advanced_ml_banking python train.py --data /path/to/data.txt
```

For CSV files with a header:

```bash
conda run -n advanced_ml_banking python train.py --data /path/to/data.csv --delimiter , --skip-rows 1
```

## Compare With sklearn

```bash
conda run -n advanced_ml_banking python compare_with_sklearn.py
conda run -n advanced_ml_banking python compare_with_sklearn.py --dataset file1
```

The comparison trains both models on the same data and checks predictions on:

- the training rows
- every feature combination from observed feature values when the grid is small

For Iris, the full observed-domain grid has hundreds of thousands of points,
so it is skipped by default and the full training-set predictions are compared.
Raise `--max-grid-points` only if you want to inspect extrapolation outside the
training rows as well.

The exported tree text is printed for inspection. The required parity check is
that the predictions and accuracy match. Tree text can differ when sklearn and
the NumPy implementation choose different but equivalent splits.
