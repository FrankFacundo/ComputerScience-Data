"""Parity tests for the NumPy-only STL implementation."""

from __future__ import annotations

import unittest

import numpy as np
from statsmodels.tsa.seasonal import STL as StatsmodelsSTL

from numpy_stl import NumpySTL


class NumpySTLParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(4317)
        time = np.arange(144, dtype=np.float64)
        cls.series = (
            40.0
            + 0.07 * time
            + 5.0 * np.sin(2.0 * np.pi * time / 12.0)
            + 1.5 * np.cos(4.0 * np.pi * time / 12.0)
            + rng.normal(0.0, 0.5, time.size)
        )
        cls.series[[17, 62, 119]] += np.array([8.0, -10.0, 13.0])

    def assert_matches_statsmodels(self, **parameters: int | bool) -> None:
        numpy_model = NumpySTL(self.series, period=12, **parameters)
        reference_model = StatsmodelsSTL(self.series, period=12, **parameters)
        numpy_result = numpy_model.fit()
        reference_result = reference_model.fit()

        self.assertEqual(numpy_model.config, reference_model.config)
        for name in ("seasonal", "trend", "resid", "weights"):
            np.testing.assert_allclose(
                getattr(numpy_result, name),
                np.asarray(getattr(reference_result, name)),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Mismatch in {name} for {parameters}",
            )

        np.testing.assert_allclose(
            numpy_result.observed,
            numpy_result.trend + numpy_result.seasonal + numpy_result.resid,
            rtol=0.0,
            atol=1e-13,
        )

    def test_default_non_robust(self) -> None:
        self.assert_matches_statsmodels()

    def test_default_robust(self) -> None:
        self.assert_matches_statsmodels(robust=True)

    def test_constant_loess(self) -> None:
        self.assert_matches_statsmodels(
            seasonal=9,
            trend=21,
            low_pass=15,
            seasonal_deg=0,
            trend_deg=0,
            low_pass_deg=0,
        )

    def test_jumps_and_robustness(self) -> None:
        self.assert_matches_statsmodels(
            seasonal=13,
            trend=25,
            low_pass=15,
            seasonal_jump=3,
            trend_jump=4,
            low_pass_jump=2,
            robust=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

