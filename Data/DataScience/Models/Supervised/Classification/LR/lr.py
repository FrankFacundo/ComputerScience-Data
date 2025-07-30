import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class LogisticRegression:
    """
    Logistic Regression classifier from scratch.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.costs = []

    def _sigmoid(self, z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.costs = []

        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Store the cost
            # Added a small epsilon to avoid log(0)
            epsilon = 1e-9
            cost = -(1 / n_samples) * np.sum(
                y * np.log(y_predicted + epsilon)
                + (1 - y) * np.log(1 - y_predicted + epsilon)
            )
            self.costs.append(cost)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


# --- Main script to run the model ---

# 1. Generate a synthetic dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the Logistic Regression model
regressor = LogisticRegression(learning_rate=0.1, n_iterations=1000)
regressor.fit(X_train, y_train)

# 4. Make predictions
predictions = regressor.predict(X_test)


# 5. Evaluate the model
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


print(
    f"Logistic Regression classification accuracy: {accuracy(y_test, predictions):.4f}"
)


# 6. Plot the decision boundary
def plot_decision_boundary(X, y, model):
    plt.figure(figsize=(10, 6))
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor="k", cmap=plt.cm.coolwarm)

    # Create a mesh to plot in
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02)
    )

    # Get predictions on the mesh grid
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    # Plot the contour and separating line
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    # Plot the decision boundary line
    w = model.weights
    b = model.bias
    x1_plot = np.array([xx1.min(), xx1.max()])
    x2_plot = -(w[0] * x1_plot + b) / w[1]
    plt.plot(x1_plot, x2_plot, "k-", lw=2)

    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.show()


plot_decision_boundary(X, y, regressor)


# 7. Plot the cost function convergence
def plot_cost_convergence(costs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(costs)), costs)
    plt.title("Cost Function Convergence During Training")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (Log Loss)")
    plt.grid(True)
    plt.show()


plot_cost_convergence(regressor.costs)
