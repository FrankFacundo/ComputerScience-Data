# STL with NumPy

This directory contains a complete educational implementation of **STL —
Seasonal-Trend decomposition using LOESS**. The implementation module has
NumPy as its only third-party dependency.

For an additive decomposition,

\[
y_t = T_t + S_t + R_t,
\]

where \(T_t\) is the trend, \(S_t\) the seasonal component, and \(R_t\) the
remainder. The implementation follows the same numerical procedure and
defaults as `statsmodels.tsa.seasonal.STL`:

1. split the detrended series into one subseries per seasonal position;
2. smooth each subseries with tricube-weighted LOESS and extrapolate its ends;
3. apply moving averages of lengths `period`, `period`, and `3`;
4. LOESS-smooth that result and subtract it from the extended seasonal series;
5. LOESS-smooth the deseasonalized data to update the trend;
6. optionally repeat with Tukey bisquare residual weights for robustness.

## Files

- `numpy_stl.py`: the STL implementation; only NumPy is required.
- `compare_with_statsmodels.py`: executable numerical comparison.
- `test_numpy_stl.py`: parity tests covering robustness, LOESS degrees, and
  jump interpolation.

## Usage

```python
import numpy as np

from numpy_stl import NumpySTL

time = np.arange(120)
values = 0.1 * time + 5 * np.sin(2 * np.pi * time / 12)

result = NumpySTL(values, period=12, robust=True).fit()
trend = result.trend
seasonal = result.seasonal
remainder = result.resid
```

Run the deterministic comparison:

```bash
python compare_with_statsmodels.py --robust
```

Compare using the sales series from the time-series tutorial:

```bash
python compare_with_statsmodels.py \
  --sales-csv ../basics/data/sales_train.csv \
  --robust
```

Run all parity tests:

```bash
python -m unittest -v test_numpy_stl.py
```

The automated tests require `rtol=1e-10` and `atol=1e-10` on ordinary,
well-conditioned series. Statsmodels runs optimized Cython loops while this
implementation runs transparent Python loops over NumPy arrays, so bit-for-bit
identity is neither portable nor mathematically meaningful.

The 34-month sales example is an especially ill-conditioned robust case: each
monthly seasonal subseries has only two or three points, and 15 rounds of
bisquare reweighting amplify machine-rounding differences. The comparison
therefore additionally uses a scale-aware requirement: every decomposition
component must differ by less than `1e-4 × max(abs(input))`, and robust weights
by less than `0.1`. On the actual sales data, the largest component difference
is only a few units against observations of roughly 60,000–180,000; the plotted
components are indistinguishable.
