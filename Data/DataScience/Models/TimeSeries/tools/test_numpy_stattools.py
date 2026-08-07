"""Parity tests for the NumPy time-series statistics."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
from statsmodels.tsa.stattools import acf as statsmodels_acf
from statsmodels.tsa.stattools import adfuller as statsmodels_adfuller
from statsmodels.tsa.stattools import kpss as statsmodels_kpss

from numpy_stattools import acf, adfuller, kpss


class NumPyStattoolsParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(314159)
        innovations = rng.normal(size=160)
        cls.values = np.empty_like(innovations)
        cls.values[0] = innovations[0]
        for index in range(1, cls.values.size):
            cls.values[index] = 0.65 * cls.values[index - 1] + innovations[index]
        cls.values += 0.03 * np.arange(cls.values.size)

    def test_acf_direct_fft_and_adjusted(self) -> None:
        for fft in (False, True):
            for adjusted in (False, True):
                with self.subTest(fft=fft, adjusted=adjusted):
                    actual = acf(
                        self.values,
                        nlags=24,
                        fft=fft,
                        adjusted=adjusted,
                    )
                    expected = statsmodels_acf(
                        self.values,
                        nlags=24,
                        fft=fft,
                        adjusted=adjusted,
                    )
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)

    def test_acf_missing_value_modes(self) -> None:
        values = self.values.copy()
        values[[5, 18, 73]] = np.nan
        for missing in ("drop", "conservative"):
            for adjusted in (False, True):
                with self.subTest(missing=missing, adjusted=adjusted):
                    actual = acf(
                        values,
                        nlags=12,
                        fft=True,
                        adjusted=adjusted,
                        missing=missing,
                    )
                    expected = statsmodels_acf(
                        values,
                        nlags=12,
                        fft=True,
                        adjusted=adjusted,
                        missing=missing,
                    )
                    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-13)

    def test_adfuller_all_deterministic_regressions(self) -> None:
        for regression in ("c", "ct", "ctt", "n"):
            with self.subTest(regression=regression):
                actual = adfuller(self.values, regression=regression, autolag="AIC")
                expected = statsmodels_adfuller(
                    self.values, regression=regression, autolag="AIC"
                )
                np.testing.assert_allclose(actual[:2], expected[:2], rtol=0.0, atol=1e-12)
                self.assertEqual(actual[2:4], expected[2:4])
                self.assertEqual(actual[4], expected[4])
                self.assertAlmostEqual(actual[5], expected[5], places=12)

    def test_adfuller_lag_selection_modes(self) -> None:
        for autolag in ("AIC", "BIC", "t-stat", None):
            with self.subTest(autolag=autolag):
                actual = adfuller(self.values, maxlag=8, autolag=autolag)
                expected = statsmodels_adfuller(
                    self.values, maxlag=8, autolag=autolag
                )
                np.testing.assert_allclose(actual[:2], expected[:2], rtol=0.0, atol=1e-12)
                self.assertEqual(actual[2:4], expected[2:4])
                self.assertEqual(actual[4], expected[4])
                if autolag is not None:
                    self.assertAlmostEqual(actual[5], expected[5], places=12)

    def test_kpss_regressions_and_lag_modes(self) -> None:
        for regression in ("c", "ct"):
            for nlags in ("auto", "legacy", 4):
                with self.subTest(regression=regression, nlags=nlags):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        actual = kpss(self.values, regression=regression, nlags=nlags)
                        expected = statsmodels_kpss(
                            self.values, regression=regression, nlags=nlags
                        )
                    np.testing.assert_allclose(
                        actual[:2], expected[:2], rtol=0.0, atol=1e-13
                    )
                    self.assertEqual(actual[2:], expected[2:])


if __name__ == "__main__":
    unittest.main(verbosity=2)

