"""NumPy implementations of selected ``statsmodels.tsa.stattools`` tools.

The public functions reproduce the commonly used numerical behavior of:

* :func:`statsmodels.tsa.stattools.acf`
* :func:`statsmodels.tsa.stattools.adfuller`
* :func:`statsmodels.tsa.stattools.kpss`

NumPy is the only third-party dependency.  The comparison and test modules in
this directory import Statsmodels only to verify numerical parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, inf, pi, sqrt
import warnings

import numpy as np


class InterpolationWarning(UserWarning):
    """A KPSS statistic lies outside the tabulated p-value interval."""


def _as_1d_float(values: object, name: str = "x") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    return np.ascontiguousarray(array)


def _next_regular(target: int) -> int:
    """Return the smallest 5-smooth integer greater than or equal to target."""

    if target <= 6:
        return target
    if not target & (target - 1):
        return target

    match = inf
    power_of_five = 1
    while power_of_five < target:
        power_of_three_and_five = power_of_five
        while power_of_three_and_five < target:
            quotient = -(-target // power_of_three_and_five)
            power_of_two = 2 ** ((quotient - 1).bit_length())
            candidate = power_of_two * power_of_three_and_five
            if candidate == target:
                return candidate
            if candidate < match:
                match = candidate
            power_of_three_and_five *= 3
            if power_of_three_and_five == target:
                return power_of_three_and_five
        if power_of_three_and_five < match:
            match = power_of_three_and_five
        power_of_five *= 5
        if power_of_five == target:
            return power_of_five
    if power_of_five < match:
        match = power_of_five
    return int(match)


def _autocovariance(
    x: np.ndarray,
    *,
    adjusted: bool,
    fft: bool,
    missing: str,
) -> np.ndarray:
    """Statsmodels-compatible autocovariance used internally by :func:`acf`."""

    missing = missing.lower()
    if missing not in {"none", "raise", "conservative", "drop"}:
        raise ValueError("missing must be 'none', 'raise', 'conservative', or 'drop'")

    contains_nan = bool(np.isnan(np.sum(x))) if missing != "none" else False
    handle_missing = missing != "none" and contains_nan

    if handle_missing:
        if missing == "raise":
            raise ValueError("NaNs were encountered in the data")
        observed = ~np.isnan(x)
        observed_int = observed.astype(np.int64)
        if missing == "conservative":
            x = x.copy()
            x[~observed] = 0.0
        else:
            x = x[observed]
    else:
        observed = np.ones(x.size, dtype=bool)
        observed_int = observed.astype(np.int64)

    if handle_missing:
        centered = x - x.sum() / observed_int.sum()
        if missing == "conservative":
            centered[~observed] = 0.0
    else:
        centered = x - x.mean()

    nobs = len(x)
    if adjusted and handle_missing and missing == "conservative":
        divisor = np.correlate(observed_int, observed_int, "full")
        divisor[divisor == 0] = 1
    elif adjusted:
        sequence = np.arange(1, nobs + 1)
        divisor = np.hstack((sequence, sequence[:-1][::-1]))
    elif handle_missing:
        divisor = observed_int.sum() * np.ones(2 * nobs - 1)
    else:
        divisor = nobs * np.ones(2 * nobs - 1)

    if fft:
        fft_size = _next_regular(2 * nobs + 1)
        transformed = np.fft.fft(centered, n=fft_size)
        covariance = np.fft.ifft(transformed * np.conjugate(transformed))[:nobs]
        covariance = (covariance / divisor[nobs - 1 :]).real
    else:
        covariance = (
            np.correlate(centered, centered, "full")[nobs - 1 :]
            / divisor[nobs - 1 :]
        )
    return covariance


def acf(
    x: object,
    adjusted: bool = False,
    nlags: int | None = None,
    qstat: bool = False,
    fft: bool = True,
    alpha: float | None = None,
    bartlett_confint: bool = True,
    missing: str = "none",
) -> np.ndarray:
    """Estimate the autocorrelation function using NumPy.

    The returned array contains lags ``0, ..., nlags`` and matches Statsmodels
    when ``qstat=False`` and ``alpha=None``. Confidence intervals and
    Ljung--Box p-values require distribution functions outside NumPy and are
    deliberately excluded from this NumPy-only implementation.
    """

    del bartlett_confint  # Part of the compatible signature, used only with alpha.
    if qstat:
        raise NotImplementedError("qstat=True requires a chi-square distribution")
    if alpha is not None:
        raise NotImplementedError("alpha requires a normal quantile function")

    values = _as_1d_float(x)
    nobs = values.size
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)
    if not isinstance(nlags, (int, np.integer)) or isinstance(nlags, bool):
        raise TypeError("nlags must be an integer or None")
    if nlags < 0:
        raise ValueError("nlags must be non-negative")

    autocovariance = _autocovariance(
        values,
        adjusted=bool(adjusted),
        fft=bool(fft),
        missing=missing,
    )
    return autocovariance[: nlags + 1] / autocovariance[0]


@dataclass(frozen=True)
class _OLSResult:
    params: np.ndarray
    resid: np.ndarray
    tvalues: np.ndarray
    aic: float
    bic: float


def _pinv_extended(x: np.ndarray, rcond: float = 1e-15) -> tuple[np.ndarray, np.ndarray]:
    """Moore--Penrose inverse with the cutoff used by Statsmodels OLS."""

    values = np.asarray(x).conjugate()
    u, singular_values, vt = np.linalg.svd(values, full_matrices=False)
    original = singular_values.copy()
    cutoff = rcond * np.maximum.reduce(singular_values)
    inverse_singular_values = singular_values.copy()
    for index in range(min(u.shape[0], vt.shape[1])):
        inverse_singular_values[index] = (
            1.0 / singular_values[index]
            if singular_values[index] > cutoff
            else 0.0
        )
    pseudoinverse = np.dot(
        vt.T,
        np.multiply(inverse_singular_values[:, np.newaxis], u.T),
    )
    return pseudoinverse, original


def _ols(y: np.ndarray, design: np.ndarray) -> _OLSResult:
    """Small OLS implementation reproducing the quantities required by ADF/KPSS."""

    design = np.asarray(design, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pseudoinverse, singular_values = _pinv_extended(design)
    params = np.dot(pseudoinverse, y)
    normalized_covariance = np.dot(pseudoinverse, pseudoinverse.T)
    resid = y - np.dot(design, params)

    rank = int(np.linalg.matrix_rank(np.diag(singular_values)))
    degrees_of_freedom = y.size - rank
    ssr = float(np.dot(resid, resid))
    scale = ssr / degrees_of_freedom
    standard_errors = np.sqrt(np.diag(normalized_covariance) * scale)
    tvalues = params / standard_errors

    nobs = float(y.size)
    log_likelihood = (
        -(nobs / 2.0) * np.log(2.0 * pi)
        - (nobs / 2.0) * np.log(np.sum(resid**2) / nobs)
        - nobs / 2.0
    )
    aic = float(-2.0 * log_likelihood + 2.0 * rank)
    bic = float(-2.0 * log_likelihood + np.log(nobs) * rank)
    return _OLSResult(params=params, resid=resid, tvalues=tvalues, aic=aic, bic=bic)


def _add_trend(x: np.ndarray, regression: str, *, prepend: bool) -> np.ndarray:
    if regression == "n":
        return x.copy()

    trend_order = {"c": 0, "ct": 1, "ctt": 2}[regression]
    time = np.arange(1, len(x) + 1, dtype=np.float64)
    trend = np.fliplr(np.vander(time, trend_order + 1))

    column_range = np.ptp(x, axis=0)
    has_nonzero_constant = (column_range == 0.0) & (x[0] != 0.0)
    if np.any(has_nonzero_constant):
        trend = trend[:, 1:]

    return np.column_stack((trend, x)) if prepend else np.column_stack((x, trend))


def _lagged_adf_data(x: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    differences = np.diff(x)
    nobs = differences.size - lag
    design = np.empty((nobs, lag + 1), dtype=np.float64)
    design[:, 0] = x[-nobs - 1 : -1]
    for order in range(1, lag + 1):
        design[:, order] = differences[lag - order : differences.size - order]
    dependent = differences[-nobs:]
    return design, dependent


_TAU_MIN = {"n": -19.04, "c": -18.83, "ct": -16.18, "ctt": -17.17}
_TAU_MAX = {"n": inf, "c": 2.74, "ct": 0.70, "ctt": 0.54}
_TAU_STAR = {"n": -1.04, "c": -1.61, "ct": -2.89, "ctt": -3.21}

_TAU_SMALL_P = {
    "n": np.array([0.6344, 1.2378, 0.032496]),
    "c": np.array([2.1659, 1.4412, 0.038269]),
    "ct": np.array([3.2512, 1.6047, 0.049588]),
    "ctt": np.array([4.0003, 1.6580, 0.048288]),
}

_TAU_LARGE_P = {
    "n": np.array([0.4797, 0.93557, -0.06999, 0.033066]),
    "c": np.array([1.7339, 0.93202, -0.12745, -0.010368]),
    "ct": np.array([2.5261, 0.61654, -0.37956, -0.060285]),
    "ctt": np.array([3.0778, 0.49529, -0.41477, -0.059359]),
}

_TAU_CRITICAL_2010 = {
    "n": np.array(
        [
            [-2.56574, -2.2358, -3.627, 0.0],
            [-1.94100, -0.2686, -3.365, 31.223],
            [-1.61682, 0.2656, -2.714, 25.364],
        ]
    ),
    "c": np.array(
        [
            [-3.43035, -6.5393, -16.786, -79.433],
            [-2.86154, -2.8903, -4.234, -40.040],
            [-2.56677, -1.5384, -2.809, 0.0],
        ]
    ),
    "ct": np.array(
        [
            [-3.95877, -9.0531, -28.428, -134.155],
            [-3.41049, -4.3904, -9.036, -45.374],
            [-3.12705, -2.5856, -3.925, -22.380],
        ]
    ),
    "ctt": np.array(
        [
            [-4.37113, -11.5882, -35.819, -334.047],
            [-3.83239, -5.9057, -12.490, -118.284],
            [-3.55326, -3.6596, -5.293, -63.559],
        ]
    ),
}


def _normal_cdf(value: float) -> float:
    return 0.5 * erfc(-value / sqrt(2.0))


def _mackinnon_pvalue(statistic: float, regression: str) -> float:
    if statistic > _TAU_MAX[regression]:
        return 1.0
    if statistic < _TAU_MIN[regression]:
        return 0.0
    coefficients = (
        _TAU_SMALL_P[regression]
        if statistic <= _TAU_STAR[regression]
        else _TAU_LARGE_P[regression]
    )
    transformed = float(np.polyval(coefficients[::-1], statistic))
    return _normal_cdf(transformed)


def _mackinnon_critical_values(regression: str, nobs: int) -> dict[str, float]:
    inverse_nobs = 1.0 / nobs
    values = np.array(
        [
            np.polyval(coefficients[::-1], inverse_nobs)
            for coefficients in _TAU_CRITICAL_2010[regression]
        ]
    )
    return {"1%": values[0], "5%": values[1], "10%": values[2]}


def adfuller(
    x: object,
    maxlag: int | None = None,
    regression: str | int | None = "c",
    autolag: str | None = "AIC",
    store: bool = False,
    regresults: bool = False,
) -> tuple:
    """Perform the Augmented Dickey--Fuller unit-root test with NumPy.

    The return tuple matches Statsmodels when ``store=False``. All four
    deterministic regressions and the ``AIC``, ``BIC``, ``t-stat`` and manual
    lag-selection modes are supported.
    """

    if store or regresults:
        raise NotImplementedError("store and regresults containers require Statsmodels")

    values = _as_1d_float(x)
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")
    if values.max() == values.min():
        raise ValueError("Invalid input, x is constant")

    regression_aliases = {None: "n", 0: "c", 1: "ct", 2: "ctt"}
    if regression in regression_aliases:
        regression = regression_aliases[regression]
    if not isinstance(regression, str):
        raise ValueError("regression must be 'c', 'ct', 'ctt', or 'n'")
    regression = regression.lower()
    if regression not in {"c", "ct", "ctt", "n"}:
        raise ValueError("regression must be 'c', 'ct', 'ctt', or 'n'")

    if autolag is not None:
        if not isinstance(autolag, str):
            raise ValueError("autolag must be 'AIC', 'BIC', 't-stat', or None")
        autolag = autolag.lower()
        if autolag not in {"aic", "bic", "t-stat"}:
            raise ValueError("autolag must be 'AIC', 'BIC', 't-stat', or None")

    original_nobs = values.size
    deterministic_terms = len(regression) if regression != "n" else 0
    maximum_allowed = original_nobs // 2 - deterministic_terms - 1
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * np.power(original_nobs / 100.0, 0.25)))
        maxlag = min(maximum_allowed, maxlag)
        if maxlag < 0:
            raise ValueError("sample size is too short for this regression")
    elif (
        not isinstance(maxlag, (int, np.integer))
        or isinstance(maxlag, bool)
        or maxlag < 0
    ):
        raise ValueError("maxlag must be a non-negative integer or None")
    elif maxlag > maximum_allowed:
        raise ValueError(
            "maxlag must be less than nobs/2 - 1 - the deterministic terms"
        )
    maxlag = int(maxlag)

    if autolag is not None:
        maximum_design, common_dependent = _lagged_adf_data(values, maxlag)
        full_design = _add_trend(maximum_design, regression, prepend=True)
        start_column = full_design.shape[1] - maximum_design.shape[1] + 1

        candidates: dict[int, _OLSResult] = {}
        for column_count in range(start_column, start_column + maxlag + 1):
            candidates[column_count] = _ols(
                common_dependent, full_design[:, :column_count]
            )

        if autolag == "aic":
            icbest, best_column = min(
                (result.aic, column_count)
                for column_count, result in candidates.items()
            )
        elif autolag == "bic":
            icbest, best_column = min(
                (result.bic, column_count)
                for column_count, result in candidates.items()
            )
        else:
            best_column = start_column + maxlag
            icbest = 0.0
            for column_count in range(start_column + maxlag, start_column - 1, -1):
                icbest = float(abs(candidates[column_count].tvalues[-1]))
                best_column = column_count
                if icbest >= 1.6448536269514722:
                    break

        usedlag = best_column - start_column
    else:
        usedlag = maxlag
        icbest = None

    lagged, dependent = _lagged_adf_data(values, usedlag)
    final_design = _add_trend(lagged[:, : usedlag + 1], regression, prepend=False)
    regression_result = _ols(dependent, final_design)
    statistic = float(regression_result.tvalues[0])
    nobs = dependent.size
    pvalue = _mackinnon_pvalue(statistic, regression)
    critical_values = _mackinnon_critical_values(regression, nobs)

    if autolag is None:
        return statistic, pvalue, usedlag, nobs, critical_values
    return statistic, pvalue, usedlag, nobs, critical_values, float(icbest)


def _kpss_autolag(resids: np.ndarray) -> int:
    nobs = resids.size
    covariance_lags = int(np.power(nobs, 2.0 / 9.0))
    s0 = np.sum(resids**2) / nobs
    s1 = 0.0
    for lag in range(1, covariance_lags + 1):
        product = np.dot(resids[lag:], resids[: nobs - lag])
        product /= nobs / 2.0
        s0 += product
        s1 += lag * product
    ratio = s1 / s0
    power = 1.0 / 3.0
    gamma_hat = 1.1447 * np.power(ratio * ratio, power)
    return int(gamma_hat * np.power(nobs, power))


def _kpss_long_run_variance(resids: np.ndarray, lags: int) -> float:
    nobs = resids.size
    estimate = np.sum(resids**2)
    for lag in range(1, lags + 1):
        product = np.dot(resids[lag:], resids[: nobs - lag])
        estimate += 2.0 * product * (1.0 - lag / (lags + 1.0))
    return float(estimate / nobs)


def kpss(
    x: object,
    regression: str = "c",
    nlags: str | int | None = "auto",
    store: bool = False,
) -> tuple[float, float, int, dict[str, float]]:
    """Perform the KPSS level- or trend-stationarity test with NumPy."""

    if store:
        raise NotImplementedError("store=True containers require Statsmodels")

    values = _as_1d_float(x)
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")
    regression = regression.lower()
    if regression not in {"c", "ct"}:
        raise ValueError("regression must be 'c' or 'ct'")

    nobs = values.size
    if regression == "ct":
        time = np.arange(1, nobs + 1, dtype=np.float64)
        design = np.column_stack((np.ones(nobs), time))
        resids = _ols(values, design).resid
        critical = np.array([0.119, 0.146, 0.176, 0.216])
    else:
        resids = values - values.mean()
        critical = np.array([0.347, 0.463, 0.574, 0.739])

    if nlags == "legacy":
        selected_lags = int(np.ceil(12.0 * np.power(nobs / 100.0, 0.25)))
        selected_lags = min(selected_lags, nobs - 1)
    elif nlags in {"auto", None}:
        selected_lags = min(_kpss_autolag(resids), nobs - 1)
    elif isinstance(nlags, str):
        raise ValueError("nlags must be 'auto', 'legacy', or an integer")
    elif (
        not isinstance(nlags, (int, np.integer))
        or isinstance(nlags, bool)
        or nlags < 0
        or nlags >= nobs
    ):
        raise ValueError("integer nlags must satisfy 0 <= nlags < nobs")
    else:
        selected_lags = int(nlags)

    partial_sums = np.cumsum(resids)
    eta = np.sum(partial_sums**2) / nobs**2
    long_run_variance = _kpss_long_run_variance(resids, selected_lags)
    statistic = float(eta / long_run_variance)

    table_pvalues = np.array([0.10, 0.05, 0.025, 0.01])
    pvalue = float(np.interp(statistic, critical, table_pvalues))
    if pvalue == table_pvalues[-1]:
        warnings.warn(
            "The actual p-value is smaller than the returned table boundary.",
            InterpolationWarning,
            stacklevel=2,
        )
    elif pvalue == table_pvalues[0]:
        warnings.warn(
            "The actual p-value is greater than the returned table boundary.",
            InterpolationWarning,
            stacklevel=2,
        )

    critical_values = {
        "10%": float(critical[0]),
        "5%": float(critical[1]),
        "2.5%": float(critical[2]),
        "1%": float(critical[3]),
    }
    return statistic, pvalue, selected_lags, critical_values
