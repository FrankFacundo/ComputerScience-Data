import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class GaussianNaiveBayes:
    """
    A Gaussian Naive Bayes classifier implemented from scratch using NumPy.
    """

    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model to the training data.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize arrays for means, variances, and priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted class labels for each sample.
        """
        y_pred = [self._predict_sample(x) for x in X]
        return np.array(y_pred)

    def _predict_sample(self, x):
        """
        Predict the class for a single sample.

        Args:
            x (np.ndarray): A single input sample.

        Returns:
            int: The predicted class label.
        """
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Calculate the probability density function for a given class and sample.

        Args:
            class_idx (int): The index of the class.
            x (np.ndarray): The input sample.

        Returns:
            np.ndarray: The probability densities.
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# --- Educational Example ---

if __name__ == "__main__":
    # 1. Generate a synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        n_classes=3,
        random_state=14,
    )

    # 2. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # 3. Instantiate and train the Gaussian Naive Bayes model
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    # 4. Make predictions on the test set
    y_pred = gnb.predict(X_test)

    # 5. Evaluate the model
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # 6. Plot the decision boundary and the data points
    def plot_decision_boundary(X, y, model):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
        plt.title("Gaussian Naive Bayes Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    plot_decision_boundary(X, y, gnb)

    # 7. Plot the learned Gaussian distributions for each feature and class
    def plot_gaussians(model):
        fig, axes = plt.subplots(len(model._classes), X.shape[1], figsize=(12, 8))
        fig.suptitle(
            "Learned Gaussian Distributions for Each Feature and Class", fontsize=16
        )

        for i, c in enumerate(model._classes):
            for j in range(X.shape[1]):
                mean = model._mean[i, j]
                var = model._var[i, j]
                sigma = np.sqrt(var)
                x_range = np.linspace(mean - 3 * sigma, mean + 3 * sigma, 100)
                pdf = (1 / (np.sqrt(2 * np.pi * var))) * np.exp(
                    -0.5 * ((x_range - mean) ** 2 / var)
                )

                axes[i, j].plot(x_range, pdf, label=f"Class {c}")
                axes[i, j].set_title(f"Class {c}, Feature {j+1}")
                axes[i, j].set_xlabel("Feature Value")
                axes[i, j].set_ylabel("Probability Density")
                axes[i, j].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    plot_gaussians(gnb)
