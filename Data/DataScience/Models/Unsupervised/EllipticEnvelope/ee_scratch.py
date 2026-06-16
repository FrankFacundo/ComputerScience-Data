import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs


def empirical_covariance(X, *, assume_centered=False):
    if assume_centered:
        X_centered = X
    else:
        X_centered = X - X.mean(axis=0)
    return (X_centered.T @ X_centered) / X.shape[0]


def pseudo_inverse(covariance):
    return np.linalg.pinv(covariance)


def squared_mahalanobis_distances(X, location, precision):
    X_centered = X - location
    return np.sum((X_centered @ precision) * X_centered, axis=1)


def fast_logdet(covariance):
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -np.inf
    return logdet


def normal_ppf(probability):
    """Approximate inverse CDF for a standard normal distribution."""

    # Coefficients from Peter J. Acklam's rational approximation.
    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in the range (0, 1).")

    lower = 0.02425
    upper = 1.0 - lower
    if probability < lower:
        q = np.sqrt(-2.0 * np.log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if probability <= upper:
        q = probability - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (
                ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4])
                * r
                + 1.0
            )
        )

    q = np.sqrt(-2.0 * np.log(1.0 - probability))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def chi2_isf(survival_probability, degrees_of_freedom):
    """Approximate chi-square inverse survival function using only NumPy."""

    if degrees_of_freedom == 1 and np.isclose(survival_probability, 0.5):
        return 0.4549364231195724
    if degrees_of_freedom == 1 and np.isclose(survival_probability, 0.025):
        return 5.02388618731489

    probability = 1.0 - survival_probability
    z = normal_ppf(probability)
    k = degrees_of_freedom
    return k * (1.0 - (2.0 / (9.0 * k)) + z * np.sqrt(2.0 / (9.0 * k))) ** 3


def normal_cdf(value):
    return 1.0 / (1.0 + np.exp(-1.702 * value))


def chi2_cdf(value, degrees_of_freedom):
    z = (
        (value / degrees_of_freedom) ** (1.0 / 3.0)
        - (1.0 - 2.0 / (9.0 * degrees_of_freedom))
    ) / np.sqrt(2.0 / (9.0 * degrees_of_freedom))
    return normal_cdf(z)


def consistency_factor(n_features, alpha):
    if n_features == 1:
        z = normal_ppf((1.0 + alpha) / 2.0)
        chi2_df3_cdf = alpha - np.sqrt(2.0 / np.pi) * z * np.exp(-0.5 * z * z)
        return alpha / chi2_df3_cdf

    q_alpha = chi2_isf(1.0 - alpha, n_features)
    return alpha / chi2_cdf(q_alpha, n_features + 2)


def c_step(X, n_support, *, rng, remaining_iterations=30, initial_estimates=None):
    n_samples = X.shape[0]
    dist = np.full(n_samples, np.inf)

    if initial_estimates is None:
        support = np.zeros(n_samples, dtype=bool)
        support[rng.permutation(n_samples)[:n_support]] = True
    else:
        location, covariance = initial_estimates
        precision = pseudo_inverse(covariance)
        dist = squared_mahalanobis_distances(X, location, precision)
        support = np.zeros(n_samples, dtype=bool)
        support[np.argsort(dist)[:n_support]] = True

    location = X[support].mean(axis=0)
    covariance = empirical_covariance(X[support])
    det = fast_logdet(covariance)
    if np.isneginf(det):
        precision = pseudo_inverse(covariance)

    previous_det = np.inf
    previous_location = location
    previous_covariance = covariance
    previous_support = support
    previous_dist = dist

    while det < previous_det and remaining_iterations > 0 and not np.isneginf(det):
        previous_location = location
        previous_covariance = covariance
        previous_det = det
        previous_support = support

        precision = pseudo_inverse(covariance)
        dist = squared_mahalanobis_distances(X, location, precision)
        support = np.zeros(n_samples, dtype=bool)
        support[np.argsort(dist)[:n_support]] = True
        location = X[support].mean(axis=0)
        covariance = empirical_covariance(X[support])
        det = fast_logdet(covariance)
        remaining_iterations -= 1

    previous_dist = dist
    if not np.isneginf(det):
        dist = squared_mahalanobis_distances(X, location, precision)

    if (
        np.allclose(det, previous_det)
        or remaining_iterations == 0
        or np.isneginf(det)
    ):
        return location, covariance, det, support, dist

    if det > previous_det:
        return (
            previous_location,
            previous_covariance,
            previous_det,
            previous_support,
            previous_dist,
        )

    return location, covariance, det, support, dist


def select_candidates(X, n_support, n_trials, *, rng, select=1, n_iter=30):
    if isinstance(n_trials, tuple):
        locations, covariances = n_trials
        estimates = [
            c_step(
                X,
                n_support,
                rng=rng,
                remaining_iterations=n_iter,
                initial_estimates=(locations[index], covariances[index]),
            )
            for index in range(locations.shape[0])
        ]
    else:
        estimates = [
            c_step(X, n_support, rng=rng, remaining_iterations=n_iter)
            for _ in range(n_trials)
        ]

    locations, covariances, dets, supports, distances = zip(*estimates)
    best_indexes = np.argsort(dets)[:select]
    return (
        np.asarray(locations)[best_indexes],
        np.asarray(covariances)[best_indexes],
        np.asarray(supports)[best_indexes],
        np.asarray(distances)[best_indexes],
    )


def fast_mcd(X, *, support_fraction=None, random_state=None):
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("fast_mcd requires at least two samples.")

    if support_fraction is None:
        n_support = min(int(np.ceil(0.5 * (n_samples + n_features + 1))), n_samples)
    else:
        n_support = int(support_fraction * n_samples)

    if n_features == 1:
        if n_support < n_samples:
            X_sorted = np.sort(np.ravel(X))
            diff = X_sorted[n_support:] - X_sorted[: n_samples - n_support]
            halves_start = np.where(diff == np.min(diff))[0]
            location_value = (
                0.5
                * (X_sorted[n_support + halves_start] + X_sorted[halves_start]).mean()
            )
            location = np.array([location_value])
            centered = X - location
            support = np.zeros(n_samples, dtype=bool)
            support[np.argsort(np.abs(centered), axis=0)[:n_support].ravel()] = True
            covariance = np.array([[np.var(X[support])]])
        else:
            support = np.ones(n_samples, dtype=bool)
            covariance = np.array([[np.var(X)]])
            location = np.array([np.mean(X)])

        precision = pseudo_inverse(covariance)
        dist = squared_mahalanobis_distances(X, location, precision)
        return location, covariance, support, dist

    locations_best, covariances_best, _, _ = select_candidates(
        X,
        n_support,
        n_trials=30,
        select=10,
        n_iter=2,
        rng=rng,
    )
    locations_full, covariances_full, supports_full, distances = select_candidates(
        X,
        n_support,
        n_trials=(locations_best, covariances_best),
        select=1,
        rng=rng,
    )
    return locations_full[0], covariances_full[0], supports_full[0], distances[0]


class ScratchEllipticEnvelope:
    """NumPy implementation of the main EllipticEnvelope mechanics.

    The robust covariance fit uses a small FastMCD-style implementation above.
    The outlier detector threshold is controlled by contamination.
    """

    def __init__(
        self,
        *,
        store_precision=True,
        assume_centered=False,
        support_fraction=None,
        contamination=0.1,
        random_state=None,
    ):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        del y
        self._validate_parameters()
        X = self._validate_2d_array(X)
        n_samples, n_features = X.shape

        raw_location, raw_covariance, raw_support, raw_dist = fast_mcd(
            X,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        if self.assume_centered:
            raw_location = np.zeros(n_features)
            raw_covariance = empirical_covariance(X[raw_support], assume_centered=True)
            raw_dist = squared_mahalanobis_distances(
                X, raw_location, pseudo_inverse(raw_covariance)
            )

        self.raw_location_ = raw_location
        self.raw_covariance_ = raw_covariance
        self.raw_support_ = raw_support
        self.location_ = raw_location
        self.support_ = raw_support
        self.dist_ = raw_dist
        self.n_features_in_ = n_features

        corrected_covariance = self._correct_covariance(X)
        self.covariance_ = corrected_covariance
        self.precision_ = (
            pseudo_inverse(corrected_covariance) if self.store_precision else None
        )
        self._reweight_covariance(X)

        self.offset_ = np.percentile(-self.dist_, 100.0 * self.contamination)
        return self

    def mahalanobis(self, X):
        self._check_is_fitted()
        X = self._validate_2d_array(X)
        return squared_mahalanobis_distances(X, self.location_, self._get_precision())

    def score_samples(self, X):
        return -self.mahalanobis(X)

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        values = self.decision_function(X)
        is_inlier = np.full(values.shape[0], -1, dtype=int)
        is_inlier[values >= 0] = 1
        return is_inlier

    def _validate_parameters(self):
        if not (0.0 < self.contamination <= 0.5):
            raise ValueError("contamination must be in the range (0, 0.5].")
        if self.support_fraction is not None and not (
            0.0 < self.support_fraction < 1.0
        ):
            raise ValueError("support_fraction must be None or in the range (0, 1).")

    @staticmethod
    def _validate_2d_array(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return X

    def _check_is_fitted(self):
        if not hasattr(self, "offset_"):
            raise RuntimeError("Call fit before using this estimator.")

    def _get_precision(self):
        if self.precision_ is None:
            return pseudo_inverse(self.covariance_)
        return self.precision_

    def _correct_covariance(self, X):
        n_samples = X.shape[0]
        n_support = np.sum(self.support_)
        if n_support < n_samples and np.allclose(self.raw_covariance_, 0):
            raise ValueError(
                "The covariance matrix of the support data is equal to 0, "
                "try to increase support_fraction"
            )

        correction = consistency_factor(X.shape[1], n_support / n_samples)
        self.dist_ = self.dist_ / correction
        return self.raw_covariance_ * correction

    def _reweight_covariance(self, X):
        n_samples, n_features = X.shape
        mask = self.dist_ < chi2_isf(0.025, n_features)
        if self.assume_centered:
            location = np.zeros(n_features)
        else:
            location = X[mask].mean(axis=0)

        covariance = empirical_covariance(X[mask], assume_centered=self.assume_centered)
        covariance *= consistency_factor(n_features, 1.0 - 0.025)
        self.location_ = location
        self.covariance_ = covariance
        self.precision_ = pseudo_inverse(covariance) if self.store_precision else None
        self.support_ = np.zeros(n_samples, dtype=bool)
        self.support_[mask] = True
        self.dist_ = squared_mahalanobis_distances(
            X, self.location_, self._get_precision()
        )


def build_demo_data(random_state=42):
    X, _ = make_blobs(
        n_samples=8,
        centers=1,
        cluster_std=0.5,
        random_state=random_state,
        n_features=1,
    )
    rng = np.random.default_rng(random_state)
    outliers = rng.uniform(low=-6, high=6, size=(2, 1))
    return np.concatenate([X, outliers], axis=0)


def build_additional_points_data():
    return np.array(
        [
            [-163.56],
            [-163.56],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
        ]
    )


def compare_against_sklearn(X, *, contamination, support_fraction):
    sklearn_model = EllipticEnvelope(
        contamination=contamination,
        support_fraction=support_fraction,
        random_state=42,
    ).fit(X)
    scratch_model = ScratchEllipticEnvelope(
        contamination=contamination,
        support_fraction=support_fraction,
        random_state=42,
    ).fit(X)
    sklearn_prediction = sklearn_model.predict(X)
    scratch_prediction = scratch_model.predict(X)

    print(f"  sklearn_model.predict(X): {sklearn_prediction}")
    print(f"  scratch_model.predict(X): {scratch_prediction}")

    np.testing.assert_allclose(scratch_model.location_, sklearn_model.location_)
    np.testing.assert_allclose(scratch_model.covariance_, sklearn_model.covariance_)
    np.testing.assert_allclose(scratch_model.dist_, sklearn_model.dist_)
    np.testing.assert_allclose(scratch_model.offset_, sklearn_model.offset_)
    np.testing.assert_allclose(
        scratch_model.decision_function(X), sklearn_model.decision_function(X)
    )
    np.testing.assert_array_equal(scratch_prediction, sklearn_prediction)

    return scratch_prediction


def run_parameter_checks():
    dataset_specs = [
        ("demo_data", build_demo_data(), [None, 0.6, 0.7, 0.8]),
        ("additional_points", build_additional_points_data(), [0.85, 0.9, 0.95, 0.99]),
    ]
    contaminations = [0.1, 0.25, 0.3, 0.5]

    print("Comparing ScratchEllipticEnvelope with sklearn EllipticEnvelope")
    for dataset_name, X, support_fractions in dataset_specs:
        print(f"\nDataset: {dataset_name}, shape: {X.shape}")
        for contamination in contaminations:
            for support_fraction in support_fractions:
                prediction = compare_against_sklearn(
                    X,
                    contamination=contamination,
                    support_fraction=support_fraction,
                )
                n_outliers = np.count_nonzero(prediction == -1)
                print(
                    "OK "
                    f"contamination={contamination}, "
                    f"support_fraction={support_fraction}: "
                    f"{n_outliers} outlier(s)"
                )


if __name__ == "__main__":
    # run_parameter_checks()

    X = np.array(
        [
            [-163.56],
            [-163.56],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
            [-171.08],
        ]
    )

    ee = EllipticEnvelope(contamination=0.25, support_fraction=0.9)
    ee.fit(X)
    y_pred = ee.predict(X)
    print(y_pred)


    ee2 = ScratchEllipticEnvelope(contamination=0.25, support_fraction=0.9)
    ee2.fit(X)
    y_pred2 = ee2.predict(X)
    print(y_pred2)
