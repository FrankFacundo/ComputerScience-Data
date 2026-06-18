# Exact SHAP From Scratch

This folder has three scripts for the same two-feature regression example:

- `shap_from_scratch_numpy.py`: exact interventional SHAP values implemented with NumPy only.
- `shap_library.py`: the same values computed with the official `shap` package.
- `compare_shap_results.py`: asserts that both outputs match and that `base + sum(phi)` reconstructs the model prediction.

Run the NumPy-only version:

```bash
python Data/DataScience/Interpretability/global/shap_exact/shap_from_scratch_numpy.py
```

Install NumPy and the external SHAP dependency before running the scripts in a
fresh environment:

```bash
python -m pip install numpy shap
```

Then compare both implementations:

```bash
python Data/DataScience/Interpretability/global/shap_exact/compare_shap_results.py
```

The scripts live in this subdirectory because the sibling `../shap.py` file would otherwise shadow the installed `shap` package when importing it.
