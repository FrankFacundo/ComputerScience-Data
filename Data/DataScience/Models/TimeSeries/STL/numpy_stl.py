"""STL decomposition implemented with NumPy.

The numerical algorithm mirrors ``statsmodels.tsa.seasonal.STL``: seasonal
subseries smoothing, the two-period-plus-three-point low-pass filter, LOESS
trend smoothing, and optional bisquare robustness iterations.

NumPy is the only third-party dependency of this module.  Statsmodels is used
only by the separate comparison and test files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class STLResult:
    """Components returned by :meth:`NumpySTL.fit`.

    They satisfy ``observed = trend + seasonal + resid`` up to floating-point
    round-off. ``weights`` contains the final robustness weights, or ones when
    robust fitting is disabled.
    """

    observed: np.ndarray
    seasonal: np.ndarray
    trend: np.ndarray
    resid: np.ndarray
    weights: np.ndarray


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_positive_integer(value: object, name: str, *, odd: bool = False) -> int:
    if not _is_integer(value) or value <= 0 or (odd and value % 2 != 1):
        qualifier = "a positive odd integer" if odd else "a positive integer"
        raise ValueError(f"{name} must be {qualifier}")
    return int(value)


class NumpySTL:
    """Seasonal-Trend decomposition using LOESS, implemented with NumPy.

    Parameters match ``statsmodels.tsa.seasonal.STL`` for array inputs.  Unlike
    Statsmodels, ``period`` is always explicit because this NumPy-only class
    does not inspect a pandas date index.

    Parameters
    ----------
    endog
        One-dimensional, finite numeric time series.
    period
        Number of observations in one seasonal cycle, e.g. 12 for monthly data
        with annual seasonality.
    seasonal, trend, low_pass
        Odd LOESS window lengths. The same defaults and derived defaults as
        Statsmodels are used.
    seasonal_deg, trend_deg, low_pass_deg
        Local-polynomial degree, either 0 (constant) or 1 (linear).
    robust
        If true, use bisquare residual weights in outer iterations.
    seasonal_jump, trend_jump, low_pass_jump
        Evaluate LOESS every ``jump`` positions and linearly interpolate the
        skipped values. A value of 1 computes every position.
    """

    def __init__(
        self,
        endog: np.ndarray,
        period: int,
        seasonal: int = 7,
        trend: int | None = None,
        low_pass: int | None = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        low_pass_jump: int = 1,
    ) -> None:
        y = np.asarray(endog, dtype=np.float64).squeeze()
        if y.ndim != 1:
            raise ValueError("endog must be squeezable to a one-dimensional array")
        if y.size < 2:
            raise ValueError("endog must contain at least two observations")
        if not np.all(np.isfinite(y)):
            raise ValueError("endog must contain only finite values")

        self._y = np.ascontiguousarray(y).copy()
        self.nobs = y.size

        self.period = _validate_positive_integer(period, "period")
        if self.period < 2:
            raise ValueError("period must be >= 2")

        self.seasonal = _validate_positive_integer(seasonal, "seasonal", odd=True)
        if self.seasonal < 3:
            raise ValueError("seasonal must be >= 3")

        if trend is None:
            trend = int(np.ceil(1.5 * self.period / (1.0 - 1.5 / self.seasonal)))
            trend += int(trend % 2 == 0)
        self.trend = _validate_positive_integer(trend, "trend", odd=True)
        if self.trend < 3 or self.trend <= self.period:
            raise ValueError("trend must be >= 3 and greater than period")

        if low_pass is None:
            low_pass = self.period + 1
            low_pass += int(low_pass % 2 == 0)
        self.low_pass = _validate_positive_integer(low_pass, "low_pass", odd=True)
        if self.low_pass < 3 or self.low_pass <= self.period:
            raise ValueError("low_pass must be >= 3 and greater than period")

        for value, name in (
            (seasonal_deg, "seasonal_deg"),
            (trend_deg, "trend_deg"),
            (low_pass_deg, "low_pass_deg"),
        ):
            if value not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        self.seasonal_deg = int(seasonal_deg)
        self.trend_deg = int(trend_deg)
        self.low_pass_deg = int(low_pass_deg)

        self.seasonal_jump = _validate_positive_integer(seasonal_jump, "seasonal_jump")
        self.trend_jump = _validate_positive_integer(trend_jump, "trend_jump")
        self.low_pass_jump = _validate_positive_integer(low_pass_jump, "low_pass_jump")
        self.robust = bool(robust)

    @property
    def config(self) -> dict[str, int | bool]:
        """Return the effective configuration, using Statsmodels key names."""

        return {
            "period": self.period,
            "seasonal": self.seasonal,
            "seasonal_deg": self.seasonal_deg,
            "seasonal_jump": self.seasonal_jump,
            "trend": self.trend,
            "trend_deg": self.trend_deg,
            "trend_jump": self.trend_jump,
            "low_pass": self.low_pass,
            "low_pass_deg": self.low_pass_deg,
            "low_pass_jump": self.low_pass_jump,
            "robust": self.robust,
        }

    def fit(self, inner_iter: int | None = None, outer_iter: int | None = None) -> STLResult:
        """Estimate trend, seasonal, residual, and robustness-weight arrays.

        Defaults match Statsmodels: non-robust fitting uses five inner
        iterations and no outer iterations; robust fitting uses two inner and
        fifteen outer iterations.
        """

        if inner_iter is None:
            inner_iter = 2 if self.robust else 5
        if outer_iter is None:
            outer_iter = 15 if self.robust else 0
        if not _is_integer(inner_iter) or inner_iter < 1:
            raise ValueError("inner_iter must be a positive integer")
        if not _is_integer(outer_iter) or outer_iter < 0:
            raise ValueError("outer_iter must be a non-negative integer")

        trend = np.zeros(self.nobs, dtype=np.float64)
        seasonal = np.zeros(self.nobs, dtype=np.float64)
        weights = np.ones(self.nobs, dtype=np.float64)
        use_weights = False

        # One initial decomposition plus ``outer_iter`` robust refinements.
        for outer in range(outer_iter + 1):
            seasonal, trend = self._inner_loop(
                trend,
                seasonal,
                weights,
                use_weights=use_weights,
                inner_iter=inner_iter,
            )
            if outer < outer_iter:
                fitted = trend + seasonal
                weights = self._robustness_weights(self._y, fitted)
                use_weights = True

        resid = self._y - seasonal - trend
        return STLResult(
            observed=self._y.copy(),
            seasonal=seasonal.copy(),
            trend=trend.copy(),
            resid=resid,
            weights=weights.copy(),
        )

    def _inner_loop(
        self,
        trend: np.ndarray,
        seasonal: np.ndarray,
        weights: np.ndarray,
        *,
        use_weights: bool,
        inner_iter: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(inner_iter):
            detrended = self._y - trend
            extended_seasonal = self._smooth_seasonal_subseries(
                detrended, weights, use_weights
            )

            low_pass_input = self._low_pass_filter(extended_seasonal)
            low_pass = self._loess_smooth(
                low_pass_input,
                window=self.low_pass,
                degree=self.low_pass_deg,
                jump=self.low_pass_jump,
                robust_weights=np.ones(self.nobs, dtype=np.float64),
                use_weights=False,
            )

            seasonal = (
                extended_seasonal[self.period : self.period + self.nobs] - low_pass
            )
            deseasonalized = self._y - seasonal
            trend = self._loess_smooth(
                deseasonalized,
                window=self.trend,
                degree=self.trend_deg,
                jump=self.trend_jump,
                robust_weights=weights,
                use_weights=use_weights,
            )

        return seasonal, trend

    def _smooth_seasonal_subseries(
        self,
        detrended: np.ndarray,
        weights: np.ndarray,
        use_weights: bool,
    ) -> np.ndarray:
        """Smooth each cycle position and extrapolate one value at each end."""

        extended = np.empty(self.nobs + 2 * self.period, dtype=np.float64)

        for cycle_position in range(self.period):
            subseries = np.ascontiguousarray(detrended[cycle_position :: self.period])
            sub_weights = np.ascontiguousarray(weights[cycle_position :: self.period])
            size = subseries.size

            smooth = self._loess_smooth(
                subseries,
                window=self.seasonal,
                degree=self.seasonal_deg,
                jump=self.seasonal_jump,
                robust_weights=sub_weights,
                use_weights=use_weights,
            )

            extended_subseries = np.empty(size + 2, dtype=np.float64)
            extended_subseries[1:-1] = smooth

            first = self._loess_point(
                subseries,
                window=self.seasonal,
                degree=self.seasonal_deg,
                target=0,
                left=1,
                right=min(self.seasonal, size),
                robust_weights=sub_weights,
                use_weights=use_weights,
            )
            extended_subseries[0] = smooth[0] if np.isnan(first) else first

            last = self._loess_point(
                subseries,
                window=self.seasonal,
                degree=self.seasonal_deg,
                target=size + 1,
                left=max(1, size - self.seasonal + 1),
                right=size,
                robust_weights=sub_weights,
                use_weights=use_weights,
            )
            extended_subseries[-1] = smooth[-1] if np.isnan(last) else last

            for index, value in enumerate(extended_subseries):
                extended[index * self.period + cycle_position] = value

        return extended

    def _low_pass_filter(self, extended_seasonal: np.ndarray) -> np.ndarray:
        """Apply moving averages of lengths period, period, then three."""

        first = self._moving_average(extended_seasonal, self.period)
        second = self._moving_average(first, self.period)
        return self._moving_average(second, 3)

    @staticmethod
    def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
        """Moving average with the operation order used by the reference STL."""

        output_size = values.size - window + 1
        output = np.empty(output_size, dtype=np.float64)

        total = 0.0
        for index in range(window):
            total = total + float(values[index])
        output[0] = total / float(window)

        for index in range(1, output_size):
            total = total + (
                float(values[index + window - 1]) - float(values[index - 1])
            )
            output[index] = total / float(window)
        return output

    @staticmethod
    def _loess_point(
        values: np.ndarray,
        *,
        window: int,
        degree: int,
        target: int,
        left: int,
        right: int,
        robust_weights: np.ndarray,
        use_weights: bool,
    ) -> float:
        """Evaluate one local regression using the reference's 1-based indices."""

        size = values.size
        data_range = size - 1.0
        bandwidth = float(max(target - left, right - target))
        if window > size:
            bandwidth += (window - size) // 2.0

        almost_bandwidth = 0.999 * bandwidth
        near_zero = 0.001 * bandwidth
        local_weights = np.zeros(size, dtype=np.float64)
        total_weight = 0.0

        for one_based_index in range(left, right + 1):
            distance = abs(one_based_index - target)
            weight = 0.0
            if distance <= almost_bandwidth:
                if distance <= near_zero:
                    weight = 1.0
                else:
                    weight = (1.0 - (distance / bandwidth) ** 3) ** 3
                if use_weights:
                    weight *= float(robust_weights[one_based_index - 1])
            local_weights[one_based_index - 1] = weight
            total_weight = total_weight + weight

        if total_weight <= 0.0:
            return np.nan

        for one_based_index in range(left, right + 1):
            local_weights[one_based_index - 1] = (
                float(local_weights[one_based_index - 1]) / total_weight
            )

        if bandwidth > 0.0 and degree > 0:
            weighted_mean = 0.0
            for one_based_index in range(left, right + 1):
                weighted_mean = weighted_mean + (
                    float(local_weights[one_based_index - 1]) * one_based_index
                )

            offset = target - weighted_mean
            weighted_variance = 0.0
            for one_based_index in range(left, right + 1):
                centered = one_based_index - weighted_mean
                weighted_variance = weighted_variance + (
                    float(local_weights[one_based_index - 1])
                    * centered
                    * centered
                )

            if np.sqrt(weighted_variance) > 0.001 * data_range:
                slope_factor = offset / weighted_variance
                for one_based_index in range(left, right + 1):
                    local_weights[one_based_index - 1] = (
                        float(local_weights[one_based_index - 1])
                        * (slope_factor * (one_based_index - weighted_mean) + 1.0)
                    )

        estimate = 0.0
        for one_based_index in range(left, right + 1):
            estimate = estimate + (
                float(local_weights[one_based_index - 1])
                * float(values[one_based_index - 1])
            )
        return estimate

    def _loess_smooth(
        self,
        values: np.ndarray,
        *,
        window: int,
        degree: int,
        jump: int,
        robust_weights: np.ndarray,
        use_weights: bool,
    ) -> np.ndarray:
        """Smooth an array with LOESS and reference-compatible interpolation."""

        size = values.size
        output = np.empty(size, dtype=np.float64)
        if size < 2:
            output[0] = values[0]
            return output

        actual_jump = min(jump, size - 1)
        left = 1
        right = min(window, size)

        if window >= size:
            for index in range(0, size, actual_jump):
                estimate = self._loess_point(
                    values,
                    window=window,
                    degree=degree,
                    target=index + 1,
                    left=1,
                    right=size,
                    robust_weights=robust_weights,
                    use_weights=use_weights,
                )
                output[index] = values[index] if np.isnan(estimate) else estimate
            left, right = 1, size

        elif actual_jump == 1:
            half_window = (window + 2) // 2
            left, right = 1, window
            for index in range(size):
                if index + 1 > half_window and right != size:
                    left += 1
                    right += 1
                estimate = self._loess_point(
                    values,
                    window=window,
                    degree=degree,
                    target=index + 1,
                    left=left,
                    right=right,
                    robust_weights=robust_weights,
                    use_weights=use_weights,
                )
                output[index] = values[index] if np.isnan(estimate) else estimate

        else:
            half_window = (window + 1) // 2
            for index in range(0, size, actual_jump):
                if index + 1 < half_window:
                    left, right = 1, window
                elif index + 1 >= size - half_window + 1:
                    left, right = size - window + 1, size
                else:
                    left = index + 1 - half_window + 1
                    right = window + index + 1 - half_window

                estimate = self._loess_point(
                    values,
                    window=window,
                    degree=degree,
                    target=index + 1,
                    left=left,
                    right=right,
                    robust_weights=robust_weights,
                    use_weights=use_weights,
                )
                output[index] = values[index] if np.isnan(estimate) else estimate

        if actual_jump == 1:
            return output

        for index in range(0, size - actual_jump, actual_jump):
            increment = (output[index + actual_jump] - output[index]) / actual_jump
            for interpolated in range(index, index + actual_jump):
                output[interpolated] = output[index] + increment * (interpolated - index)

        last_sampled_one_based = ((size - 1) // actual_jump) * actual_jump + 1
        if last_sampled_one_based != size:
            estimate = self._loess_point(
                values,
                window=window,
                degree=degree,
                target=size,
                left=left,
                right=right,
                robust_weights=robust_weights,
                use_weights=use_weights,
            )
            output[-1] = values[-1] if np.isnan(estimate) else estimate

            if last_sampled_one_based != size - 1:
                increment = (
                    output[-1] - output[last_sampled_one_based - 1]
                ) / (size - last_sampled_one_based)
                for one_based_index in range(last_sampled_one_based + 1, size + 1):
                    output[one_based_index - 1] = (
                        output[last_sampled_one_based - 1]
                        + increment * (one_based_index - last_sampled_one_based)
                    )

        return output

    @staticmethod
    def _robustness_weights(values: np.ndarray, fitted: np.ndarray) -> np.ndarray:
        """Compute Tukey bisquare weights using six times the median error."""

        absolute_residuals = np.abs(values - fitted)
        size = values.size
        middle = np.array([size // 2, size - size // 2 - 1], dtype=int)
        partitioned = np.partition(absolute_residuals, middle)
        six_median = 3.0 * (partitioned[middle[0]] + partitioned[middle[1]])

        if six_median == 0.0:
            return np.ones(size, dtype=np.float64)

        lower = 0.001 * six_median
        upper = 0.999 * six_median
        weights = np.empty(size, dtype=np.float64)
        for index, residual in enumerate(absolute_residuals):
            if residual <= lower:
                weights[index] = 1.0
            elif residual <= upper:
                weights[index] = (1.0 - (residual / six_median) ** 2) ** 2
            else:
                weights[index] = 0.0
        return weights
