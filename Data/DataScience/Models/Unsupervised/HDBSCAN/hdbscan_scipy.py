import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


class HDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples else min_cluster_size

    def _mutual_reachability_distance(self, distances):
        """
        Calculate the mutual reachability distance.
        """
        k = self.min_samples
        n = distances.shape[0]
        # Compute the core distances (k-th nearest neighbor distances)
        core_distances = np.sort(distances, axis=1)[:, k - 1]
        core_dist_matrix = np.maximum(core_distances[:, None], core_distances[None, :])
        mutual_reachability = np.maximum(core_dist_matrix, distances)
        return mutual_reachability

    def _build_mst(self, mutual_reachability):
        """
        Build a minimum spanning tree (MST) from the mutual reachability graph.
        """
        mst = minimum_spanning_tree(csr_matrix(mutual_reachability))
        return mst

    def _condense_tree(self, mst, min_cluster_size):
        """
        Condense the tree to extract clusters.
        """
        edges = np.array(mst.nonzero()).T
        edge_weights = mst.data
        sorted_edges = edges[np.argsort(edge_weights)]

        labels = -np.ones(mst.shape[0], dtype=int)
        current_cluster = 0

        for edge in sorted_edges:
            point_a, point_b = edge
            if labels[point_a] == -1 and labels[point_b] == -1:
                labels[point_a] = current_cluster
                labels[point_b] = current_cluster
                current_cluster += 1
            elif labels[point_a] == -1:
                labels[point_a] = labels[point_b]
            elif labels[point_b] == -1:
                labels[point_b] = labels[point_a]

        # Post-process: Remove small clusters
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) < min_cluster_size:
                labels[cluster_indices] = -1

        return labels

    def fit(self, X):
        """
        Perform HDBSCAN clustering.
        """
        # Compute pairwise distances
        distances = squareform(pdist(X))
        # Calculate mutual reachability distances
        mutual_reachability = self._mutual_reachability_distance(distances)
        # Build a minimum spanning tree (MST)
        mst = self._build_mst(mutual_reachability)
        # Condense the tree to assign cluster labels
        self.labels_ = self._condense_tree(mst, self.min_cluster_size)
        return self


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


def generate_two_clusters(n_samples, separation=5):
    """
    Generate two synthetic clusters with specified separation.
    """
    np.random.seed(42)
    cluster1 = np.random.randn(n_samples // 2, 2)
    cluster2 = np.random.randn(n_samples // 2, 2) + separation
    return np.vstack([cluster1, cluster2])


# Example usage:
if __name__ == "__main__":
    # Generate sample data
    X = generate_two_clusters(4, separation=8)
    X = generate_moons(300, noise=0.05)

    # Run HDBSCAN
    hdbscan = HDBSCAN(min_cluster_size=2, min_samples=2)
    hdbscan = HDBSCAN(min_cluster_size=5, min_samples=20)
    hdbscan.fit(X)

    print("Cluster labels:", hdbscan.labels_)

    # Visualize the first example
    plt.figure(figsize=(8, 4))
    plt.scatter(X[:, 0], X[:, 1], c=hdbscan.labels_, cmap="viridis", s=10)
    plt.colorbar(label="Cluster Label")
    plt.title("HDBSCAN Clustering Results - Moons Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
