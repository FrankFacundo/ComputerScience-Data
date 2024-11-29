import heapq

import matplotlib.pyplot as plt
import numpy as np


def hdbscan_numpy(data, min_samples=5, min_cluster_size=5):
    # Step 1: Compute the pairwise distance matrix
    def compute_distances(X):
        return np.linalg.norm(X[:, None] - X[None, :], axis=2)

    # Step 2: Compute core distances
    def compute_core_distances(distances, k):
        return np.partition(distances, k, axis=1)[:, k]

    # Step 3: Compute mutual reachability distances
    def compute_mutual_reachability_distances(distances, core_distances):
        return np.maximum(
            distances, np.maximum(core_distances[:, None], core_distances[None, :])
        )

    # Step 4: Build the Minimum Spanning Tree (MST)
    def minimum_spanning_tree(distances):
        n_samples = distances.shape[0]
        visited = np.zeros(n_samples, dtype=bool)
        edge_heap = []
        mst = []

        visited[0] = True
        for j in range(1, n_samples):
            heapq.heappush(edge_heap, (distances[0, j], 0, j))

        while len(mst) < n_samples - 1:
            dist, u, v = heapq.heappop(edge_heap)
            if visited[v]:
                continue
            mst.append((u, v, dist))
            visited[v] = True
            for w in range(n_samples):
                if not visited[w]:
                    heapq.heappush(edge_heap, (distances[v, w], v, w))

        return mst

    # Step 5: Extract clusters
    def extract_clusters(mst, min_cluster_size):
        n_samples = len(mst) + 1
        parent = np.arange(n_samples)

        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        mst_sorted = sorted(mst, key=lambda x: x[2], reverse=True)

        for u, v, dist in mst_sorted:
            if dist > np.percentile([e[2] for e in mst], 75):
                continue
            union(u, v)

        cluster_labels = np.full(n_samples, -1, dtype=int)
        cluster_map = {}
        cluster_id = 0
        for i in range(n_samples):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = cluster_id
                cluster_id += 1
            cluster_labels[i] = cluster_map[root]

        for c in np.unique(cluster_labels):
            if c == -1:
                continue
            mask = cluster_labels == c
            if np.sum(mask) < min_cluster_size:
                cluster_labels[mask] = -1

        return cluster_labels

    # Execute the algorithm
    distances = compute_distances(data)
    core_distances = compute_core_distances(distances, min_samples - 1)
    mutual_reachability = compute_mutual_reachability_distances(
        distances, core_distances
    )
    mst = minimum_spanning_tree(mutual_reachability)
    clusters = extract_clusters(mst, min_cluster_size)

    return clusters


# Synthetic dataset generation
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


# Generate data and run the algorithm
data = generate_moons(300, noise=0.05)
clusters = hdbscan_numpy(data, min_samples=5, min_cluster_size=20)

# Visualize results
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap="viridis", s=10)
plt.colorbar(label="Cluster Label")
plt.title("HDBSCAN Clustering Results (NumPy Only)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
