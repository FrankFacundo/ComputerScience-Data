import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvectors = eigenvectors[:, np.argsort(-eigenvalues)]
        eigenvalues = np.sort(eigenvalues)[::-1]
        self.components = eigenvectors[:, : self.n_components]

    def transform(self, X, keep_mean=True):
        X_centered = X - self.mean
        X_transformed = np.dot(X_centered, self.components)

        if keep_mean:
            # Adding the mean back to the transformed data
            X_transformed += np.dot(self.mean, self.components)

        return X_transformed

    def fit_transform(self, X, keep_mean=True):
        self.fit(X)
        return self.transform(X, keep_mean=keep_mean)


# Sample data with 4 samples and 3 features
X = np.array(
    [
        [2.5, 2.4, 1.5],
        [0.5, 0.7, 0.2],
        [2.2, 2.9, 2.2],
        [1.9, 2.2, 1.9],
    ]
)

# Instantiate PCA with 2 components
pca = PCA(n_components=2)
X_transformed_with_mean = pca.fit_transform(X, keep_mean=True)

print("Transformed data with original mean preserved:\n", X_transformed_with_mean)
