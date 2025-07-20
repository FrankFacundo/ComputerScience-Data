"""
Hierarchical Navigable Small World (HNSW) graph from scratch
===========================================================

A **minimal‑yet‑functional** implementation built *only* with:
  • **NumPy** – fast vector maths
  • **NetworkX** – convenient graph container
  • **Matplotlib** – quick visual sanity‑check

The file doubles as both **tutorial** and **library**:
  1. Read the comments top‑to‑bottom to understand the core ideas.
  2. Run `python hnsw_tutorial.py` to generate a 2‑D toy data‑set, build the index, compare against brute‑force, and plot neighbours.

The code purposefully avoids clever micro‑optimisations so the algorithmic
steps stay crystal‑clear. Once you grasp the mechanics you can speed‑tune with
numba, swap NetworkX for a flat adjacency list, etc.

"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# =============================================================================
# 1️⃣  Helper utilities
# =============================================================================


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance (ʀ𝟸)."""
    return float(np.linalg.norm(a - b))


def random_level(m: int, rng: random.Random) -> int:
    """Draw a layer level ~ Exp(log(M))."""
    # P(level ≥ l) = exp(−l / log(M))  ⇒  level = floor( −log(U) · log(M) )
    u = rng.random()
    return int(math.floor(-math.log(u) * math.log(m)))


# =============================================================================
# 2️⃣  Core data‑structures
# =============================================================================


@dataclass
class HNSWIndex:
    """Teeny‑tiny HNSW – clear enough for study, fast enough for hundreds‑of‑K points."""

    M: int = 5  # max neighbours per node per layer (base layer doubled)
    ef_construction: int = 200  # beam width when inserting
    ef_search: int = 50  # beam width at query time
    distance: Callable[[np.ndarray, np.ndarray], float] = euclidean
    rng: random.Random = field(default_factory=random.Random)

    # ── internal state ──────────────────────────────────────────────────────
    layers: List[nx.Graph] = field(default_factory=list, init=False)
    vectors: List[np.ndarray] = field(default_factory=list, init=False)
    entry_point: int | None = field(default=None, init=False)
    max_level: int = field(default=-1, init=False)

    # ---------------------------------------------------------------------
    # 2.1  Public API
    # ---------------------------------------------------------------------

    def add_item(self, idx: int, vec: np.ndarray) -> None:
        """Insert *vec* under identifier *idx* into the multi‑layer graph."""
        self._ensure_levels(idx)
        # Store the vector (keep list index == node id for simplicity)
        if idx == len(self.vectors):
            self.vectors.append(vec)
        else:  # overwriting existing id (rare in demo, but kept for completeness)
            self.vectors[idx] = vec

        node_level = random_level(self.M, self.rng)

        # First ever point? Make it the entry‑point.
        if self.entry_point is None:
            self.entry_point = idx

        # Extend graph upwards if this point is the new tallest tower
        if node_level > self.max_level:
            # Build each new upper layer with the *current* node already inside
            for _ in range(self.max_level + 1, node_level + 1):
                g = nx.Graph()
                g.add_node(idx)  # ensure the entry point exists in its own layer
                self.layers.append(g)
            self.max_level = node_level
            self.entry_point = idx

        current_entry = self.entry_point
        # Walk top‑down through levels
        for level in range(self.max_level, -1, -1):
            if level <= node_level:
                # Search this layer to collect candidate neighbours
                ef = self.ef_construction if level == 0 else self.M
                candidates = self._search_layer(vec, [current_entry], ef, level)
                selected = self._select_neighbors(
                    vec, candidates, self.M if level > 0 else self.M * 2
                )
                for cand in selected:
                    self.layers[level].add_edge(idx, cand)
                    self._prune_level(cand, level)
            # Greedy descent for next layer (except when already at bottom)
            if level > 0:
                neighbours = list(self.layers[level].neighbors(current_entry))
                if neighbours:
                    current_entry = min(
                        neighbours, key=lambda j: self.distance(vec, self.vectors[j])
                    )

    def search(self, query: np.ndarray, k: int = 1) -> List[int]:
        """Return *k* approximate nearest neighbour ids for *query*."""
        if self.entry_point is None:
            raise ValueError("Index empty – add items first.")

        curr = self.entry_point
        # Greedy at upper layers
        for level in range(self.max_level, 0, -1):
            improved = True
            while improved:
                improved = False
                for neigh in self.layers[level].neighbors(curr):
                    if self.distance(query, self.vectors[neigh]) < self.distance(
                        query, self.vectors[curr]
                    ):
                        curr = neigh
                        improved = True
        # Beam search in base layer
        candidates = self._search_layer(query, [curr], self.ef_search, 0)
        candidates.sort(key=lambda j: self.distance(query, self.vectors[j]))
        return candidates[:k]

    # ---------------------------------------------------------------------
    # 2.2  Internal helpers
    # ---------------------------------------------------------------------

    def _ensure_levels(self, idx: int) -> None:
        """Ensure layer‑0 exists and node *idx* is registered in all current layers."""
        if not self.layers:
            self.layers.append(nx.Graph())
            self.max_level = 0
        for g in self.layers:
            g.add_node(idx)

    def _select_neighbors(
        self, vec: np.ndarray, candidates: Sequence[int], m: int
    ) -> List[int]:
        """Simple heuristic: choose *m* closest from *candidates*."""
        return sorted(candidates, key=lambda j: self.distance(vec, self.vectors[j]))[:m]

    def _prune_level(self, node: int, level: int) -> None:
        """Trim node degree to `M` (`2M` on level‑0) retaining nearest edges."""
        max_deg = self.M if level > 0 else self.M * 2
        neigh = list(self.layers[level].neighbors(node))
        if len(neigh) > max_deg:
            # Remove furthest first
            neigh.sort(
                key=lambda j: self.distance(self.vectors[node], self.vectors[j]),
                reverse=True,
            )
            for extra in neigh[max_deg:]:
                self.layers[level].remove_edge(node, extra)

    def _search_layer(
        self, query: np.ndarray, entry_pts: List[int], ef: int, level: int
    ) -> List[int]:
        """Beam search inside a *single* layer; returns ≤ *ef* candidate ids."""
        visited = set(entry_pts)
        # `candidates` and `best` are sorted lists of (distance, id)
        candidates: List[Tuple[float, int]] = [
            (self.distance(query, self.vectors[p]), p) for p in entry_pts
        ]
        candidates.sort(reverse=True)  # treat as max‑heap via reverse sort
        best = candidates.copy()

        while candidates:
            dist_curr, curr = candidates.pop()  # closest so far
            if dist_curr > best[-1][0] and len(best) >= ef:
                break  # cannot improve best list further
            for neigh in self.layers[level].neighbors(curr):
                if neigh in visited:
                    continue
                visited.add(neigh)
                d = self.distance(query, self.vectors[neigh])
                if len(best) < ef or d < best[-1][0]:
                    self._sorted_insert(best, (d, neigh), ef)
                    self._sorted_insert(candidates, (d, neigh), ef, reverse=True)
        return [idx for _, idx in best]

    @staticmethod
    def _sorted_insert(
        arr: List[Tuple[float, int]],
        item: Tuple[float, int],
        limit: int,
        *,
        reverse: bool = False,
    ) -> None:
        """Insert *item* in *arr* (sorted by distance), keeping length ≤ *limit*."""
        lo, hi = 0, len(arr)
        key = item[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if (arr[mid][0] < key) ^ reverse:
                lo = mid + 1
            else:
                hi = mid
        arr.insert(lo, item)
        if len(arr) > limit:
            arr.pop()


# =============================================================================
# 3️⃣  Demo – 2‑D Gaussian cloud
# =============================================================================

if __name__ == "__main__":
    DIM = 2
    N = 1_000
    QUERY = np.array((0.0, 0.0))
    K = 5

    rng = random.Random(42)

    # Generate 2‑D isotropic Gaussians
    vecs = np.random.randn(N, DIM)

    # Build the index
    hnsw = HNSWIndex(M=5, ef_construction=100, ef_search=50, rng=rng)
    for i, v in enumerate(vecs):
        hnsw.add_item(i, v)

    approx = hnsw.search(QUERY, k=K)

    # Ground‑truth for comparison
    truth = np.argsort(np.linalg.norm(vecs - QUERY, axis=1))[:K]

    print("Approximate neighbours:", approx)
    print("Ground truth        :", truth.tolist())

    # 2‑D scatter visualisation (only works for DIM==2)
    plt.scatter(vecs[:, 0], vecs[:, 1], s=10, alpha=0.3, label="dataset")
    plt.scatter(*QUERY, c="red", s=100, marker="*", label="query")
    nn_vecs = vecs[approx]
    plt.scatter(nn_vecs[:, 0], nn_vecs[:, 1], c="green", s=50, label="HNSW k‑NN")
    for idx, (x, y) in zip(approx, nn_vecs):
        plt.text(x, y, str(idx), fontsize=8)

    plt.legend()
    plt.title("HNSW approximate nearest neighbours (k=5)")
    plt.show()
