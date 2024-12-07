import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler


# Moon data generation function
def generate_moons(n_samples, noise=0.1):
    np.random.seed(42)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circle = np.empty((n_samples_out, 2))
    outer_circle[:, 0] = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circle[:, 1] = np.sin(np.linspace(0, np.pi, n_samples_out))

    inner_circle = np.empty((n_samples_in, 2))
    inner_circle[:, 0] = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circle[:, 1] = 0 - np.sin(np.linspace(0, np.pi, n_samples_in))

    outer_circle += noise * np.random.randn(*outer_circle.shape)
    inner_circle += noise * np.random.randn(*inner_circle.shape)

    return np.vstack([outer_circle, inner_circle])


# Generate moon-shaped data
n_samples = 500
noise = 0.1
data = generate_moons(n_samples, noise)

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Run HDBSCAN clustering
clusterer = HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(data_scaled)

# Plot the results
plt.figure(figsize=(10, 7))
plt.scatter(
    data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap="viridis", s=30, edgecolor="k"
)
plt.title("HDBSCAN Clustering on Moon-Shaped Data", fontsize=16)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster Label")
plt.show()
