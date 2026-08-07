# Time-series statistical tools with NumPy

This directory contains educational NumPy implementations of three functions
commonly imported from `statsmodels.tsa.stattools`:

- `acf`: autocorrelation function;
- `adfuller`: Augmented Dickey–Fuller unit-root test;
- `kpss`: Kwiatkowski–Phillips–Schmidt–Shin stationarity test.

The implementation module imports only NumPy plus Python's standard library.
Statsmodels is confined to the comparison and test files.

## Files

- `numpy_stattools.py`: NumPy implementation.
- `stationarity_example.py`: Polars table analogous to the tutorial notebook.
- `compare_with_statsmodels.py`: strict numerical comparison.
- `test_numpy_stattools.py`: automated parity tests.

## Mathematical scope

For lag \(k\), the ACF is

\[
\widehat\rho_k=\frac{\widehat\gamma_k}{\widehat\gamma_0}.
\]

The ADF regression is

\[
\Delta y_t=\alpha+\beta t+\gamma y_{t-1}
 +\sum_{i=1}^{p}\delta_i\Delta y_{t-i}+\varepsilon_t,
\]

and tests \(H_0:\gamma=0\), a unit root. The implementation includes the
MacKinnon response-surface p-value coefficients and finite-sample critical
values used by Statsmodels.

KPSS uses the partial sums of residuals under level or trend stationarity:

\[
\operatorname{KPSS}=\frac{n^{-2}\sum_{t=1}^n S_t^2}
{\widehat\sigma^2_{\mathrm{NW}}},
\qquad S_t=\sum_{i=1}^t\widehat\varepsilon_i,
\]

with the same automatic lag rule, Newey–West long-run variance, critical table,
and interpolated p-values as Statsmodels 0.14.6.

## Run the notebook-style example

```bash
cd /Users/frankfacundo/Code/ComputerScience-Data/Data/DataScience/Models/TimeSeries/tools
python stationarity_example.py
```

It loads `../basics/data/sales_train.csv`, constructs the same four
transformations as the tutorial, and returns a rounded Polars table.

## Verify equality with Statsmodels

```bash
python compare_with_statsmodels.py

python compare_with_statsmodels.py \
  --sales-csv ../basics/data/sales_train.csv
```

The comparison requires an absolute difference no larger than `1e-12` for the
ACF, ADF and KPSS values. Run the broader test matrix with:

```bash
python -m unittest -v test_numpy_stattools.py
```

## Supported compatibility

- `acf`: biased/adjusted normalization, direct/FFT calculation, automatic or
  explicit lags, and `none`, `raise`, `drop`, or `conservative` missing-value
  behavior. Confidence intervals and Ljung–Box p-values are excluded because
  their probability distributions are not provided by NumPy.
- `adfuller`: `c`, `ct`, `ctt`, and `n` regressions; automatic maximum lag;
  `AIC`, `BIC`, `t-stat`, or manual lag selection. Statsmodels-specific result
  containers are excluded.
- `kpss`: `c` and `ct` regressions; `auto`, `legacy`, or explicit lags.
  Statsmodels-specific result containers are excluded.

