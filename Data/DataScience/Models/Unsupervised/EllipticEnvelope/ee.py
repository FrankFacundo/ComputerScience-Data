import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# Generate a 1D dataset with some outliers
X, _ = make_blobs(
    n_samples=8, centers=1, cluster_std=0.5, random_state=42, n_features=1
)
outliers = np.random.uniform(low=-6, high=6, size=(2, 1))
X = np.concatenate([X, outliers], axis=0)

# Fit the Elliptic Envelope
"""
Explaination 1. EllipticEnvelope basically fits a multivariate Gaussian distribution to the data, 
then remove the contamination percentage of points that have the greatest Mahalanobis distance.

Explaination 2. Remove the contamination percentage of points that have the greatest Mahalanobis distance. 
It uses the a mean and the robust covariance matrix of the data computed by the Minimum Covariance Determinant (MCD) estimator.
"""
envelope = EllipticEnvelope(contamination=0.1, random_state=42)
envelope.fit(X)

# Predict inliers (1) and outliers (-1)
y_pred = envelope.predict(X)

# Separate inliers and outliers
X_inliers = X[y_pred == 1]
X_outliers = X[y_pred == -1]
