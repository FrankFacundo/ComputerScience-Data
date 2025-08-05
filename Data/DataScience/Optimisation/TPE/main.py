"""
Tree‑structured Parzen Estimator (TPE) optimisation
===================================================

A *from‑scratch* implementation of the sequential model‐based optimisation
algorithm described in:

"Algorithms for Hyper‑Parameter Optimisation" (Bergstra et al., 2011)

This version:
-------------
* depends **only on NumPy** for the TPE core (all maths done with NumPy).
* is intentionally concise (< 200 LOC) so that students can read and tweak it.
* supports **continuous and integer** search‑space dimensions.
* provides a minimal public API: ``TPE(...).optimise(n_iter)`` returning the
  best (params, loss) pair.
* includes a *worked example* that tunes a ``DecisionTreeClassifier`` on the
  Iris dataset using scikit‑learn (an external lib, but *not* required for the
  optimiser itself).

Usage (TL;DR)
-------------
>>> from tpe_numpy import TPE, demo_decision_tree
>>> best_params, best_loss = demo_decision_tree()
>>> print(best_params, -best_loss)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class SearchDimension:
    """Metadata for one hyper‑parameter."""

    low: float
    high: float
    kind: str  # 'int' | 'float'


# -----------------------------------------------------------------------------
# Core optimiser
# -----------------------------------------------------------------------------
class TPE:
    """Tree‑structured Parzen Estimator optimiser implemented with NumPy only."""

    def __init__(
        self,
        space: Dict[str, Tuple[float, float, str]],
        objective: Callable[[Dict[str, float]], float],
        *,
        n_initial: int = 10,
        gamma: float = 0.15,
        n_candidates: int = 100,
        random_state: int | None = None,
    ) -> None:
        """Create a new optimiser.

        Parameters
        ----------
        space
            Mapping ``name -> (low, high, kind)`` where *kind* is ``'int'`` or
            ``'float'``.
        objective
            Function to minimise: takes a *params dict* returns a scalar loss.
        n_initial
            Number of random evaluations before TPE kicks in.
        gamma
            Fraction of the observations considered *\"good\"* (typically
            0.1 – 0.3).  Smaller ⇒ greedier selection pressure.
        n_candidates
            Number of candidates sampled from the *good* model at each
            iteration.
        random_state
            Seed for NumPy's RNG to ensure deterministic behaviour.
        """
        self.objective = objective
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.n_initial = n_initial
        self.rng = np.random.RandomState(random_state)

        # ----- Search‑space bookkeeping ------------------------------------- #
        self.names: List[str] = list(space.keys())
        self.bounds = np.array([[lo, hi] for lo, hi, _ in space.values()])
        self.kinds = [kind for *_, kind in space.values()]

        # ----- Storage ------------------------------------------------------- #
        self.history: List[Tuple[Dict[str, float], float]] = []  # [(params, loss)]

    # --------------------------------------------------------------------- #
    # Internal helpers (all NumPy‑based)
    # --------------------------------------------------------------------- #
    def _random_point(self) -> Dict[str, float]:
        u = self.rng.rand(len(self.names))
        vec = self.bounds[:, 0] + u * (self.bounds[:, 1] - self.bounds[:, 0])
        return {
            n: (int(round(v)) if k == "int" else float(v))
            for n, v, k in zip(self.names, vec, self.kinds)
        }

    def _vectorise(self, plist: List[Dict[str, float]]) -> np.ndarray:
        mat = np.zeros((len(plist), len(self.names)))
        for r, p in enumerate(plist):
            for c, n in enumerate(self.names):
                mat[r, c] = p[n]
        return mat

    # ---------- Parzen estimator (Gaussian mixture with fixed σ) ---------- #
    def _parzen(self, arr: np.ndarray):
        sigma = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
        return arr, sigma  # means + (shared) std per dimension

    def _pdf(self, x: np.ndarray, means: np.ndarray, sigma: np.ndarray) -> float:
        norm = 1.0 / np.sqrt(2.0 * np.pi * sigma**2)
        probs = norm * np.exp(-0.5 * ((x - means) / sigma) ** 2)
        return float(np.mean(np.prod(probs, axis=1)) + 1e-12)

    def _sample_kde(self, means: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        c = self.rng.randint(len(means))
        z = self.rng.normal(means[c], sigma)
        for d in range(z.size):
            lo, hi = self.bounds[d]
            z[d] = np.clip(z[d], lo, hi)
            if self.kinds[d] == "int":
                z[d] = int(round(z[d]))
        return z

    def _candidate(
        self,
        good_m: np.ndarray,
        good_s: np.ndarray,
        bad_m: np.ndarray,
        bad_s: np.ndarray,
    ) -> Dict[str, float]:
        best, best_ratio = None, -np.inf
        for _ in range(self.n_candidates):
            z = self._sample_kde(good_m, good_s)
            l = self._pdf(z, good_m, good_s)
            g = self._pdf(z, bad_m, bad_s)
            r = l / g
            if r > best_ratio:
                best_ratio, best = r, z
        return {
            n: (int(best[i]) if self.kinds[i] == "int" else float(best[i]))
            for i, n in enumerate(self.names)
        }

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def optimise(self, n_iter: int = 50, *, verbose: bool = False):
        # -- Bootstrap with pure random search
        while len(self.history) < self.n_initial:
            p = self._random_point()
            loss = self.objective(p)
            self.history.append((p, loss))

        # -- Sequential model‑based loop
        for t in range(n_iter):
            losses = np.array([l for _, l in self.history])
            order = np.argsort(losses)
            n_good = max(1, int(np.ceil(self.gamma * len(losses))))
            good_idx, bad_idx = order[:n_good], order[n_good:]
            good, bad = (
                [self.history[i][0] for i in good_idx],
                [self.history[i][0] for i in bad_idx],
            )
            good_m, sigma = self._parzen(self._vectorise(good))
            bad_m, _ = self._parzen(self._vectorise(bad))
            cand = self._candidate(good_m, sigma, bad_m, sigma)
            loss = self.objective(cand)
            self.history.append((cand, loss))
            if verbose:
                print(f"Iter {t+1:02d}: loss={loss:.4f}, best={losses.min():.4f}")
        return min(self.history, key=lambda x: x[1])


# -----------------------------------------------------------------------------
# Demonstration (optional – requires scikit‑learn & matplotlib)
# -----------------------------------------------------------------------------


def demo_decision_tree():  # pragma: no cover – illustrative only
    """Tune a DecisionTree on Iris; plot convergence & explored depths."""
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris(return_X_y=True)
    X, y = iris

    def loss(p: Dict[str, float]):
        m = DecisionTreeClassifier(
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        acc = cross_val_score(m, X, y, cv=cv).mean()
        return -acc  # minimise

    space = {
        "max_depth": (1, 10, "int"),
        "min_samples_split": (2, 20, "int"),
        "min_samples_leaf": (1, 10, "int"),
    }

    opt = TPE(space, loss, n_initial=15, random_state=0)
    best_params, best_loss = opt.optimise(n_iter=50)

    losses = np.array([l for _, l in opt.history])
    cum_best = np.minimum.accumulate(losses)
    depths = [p["max_depth"] for p, _ in opt.history]
    acc = -losses

    plt.figure()
    plt.plot(cum_best)
    plt.xlabel("iteration")
    plt.ylabel("best loss")
    plt.title("TPE convergence")
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.scatter(depths, acc)
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.title("Explored depths")
    plt.tight_layout()
    plt.show()

    return best_params, best_loss


if __name__ == "__main__":
    demo_decision_tree()
