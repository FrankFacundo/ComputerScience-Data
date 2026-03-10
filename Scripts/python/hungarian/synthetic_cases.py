from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SyntheticCase:
    name: str
    description: str
    cost: np.ndarray
    row_labels: list[str]
    col_labels: list[str]
    row_positions: np.ndarray | None = None
    col_positions: np.ndarray | None = None


def _labels(prefix: str, size: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(size)]


def classic_case() -> SyntheticCase:
    cost = np.array(
        [
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2],
        ],
        dtype=float,
    )
    return SyntheticCase(
        name="Classic 3x3",
        description="The standard textbook example: small enough to inspect by hand, but nontrivial.",
        cost=cost,
        row_labels=_labels("worker_", 3),
        col_labels=_labels("job_", 3),
    )


def anti_greedy_case() -> SyntheticCase:
    cost = np.array(
        [
            [3, 5, 1, 3],
            [9, 3, 3, 2],
            [8, 7, 1, 1],
            [5, 8, 7, 1],
        ],
        dtype=float,
    )
    return SyntheticCase(
        name="Greedy Trap",
        description=(
            "A row-by-row greedy rule grabs the cheap entries in columns 2 and 3 too early and "
            "ends at cost 15. The Hungarian algorithm finds the diagonal assignment with cost 8."
        ),
        cost=cost,
        row_labels=_labels("worker_", 4),
        col_labels=_labels("job_", 4),
    )


def tie_heavy_case() -> SyntheticCase:
    cost = np.array(
        [
            [1, 1, 4, 4],
            [1, 1, 4, 4],
            [4, 4, 1, 1],
            [4, 4, 1, 1],
        ],
        dtype=float,
    )
    return SyntheticCase(
        name="Tie-Heavy 4x4",
        description=(
            "A matrix with several optimal assignments. It is useful for understanding why the "
            "algorithm reasons about zero reduced-cost edges rather than about a single path."
        ),
        cost=cost,
        row_labels=_labels("worker_", 4),
        col_labels=_labels("job_", 4),
    )


def rectangular_case() -> SyntheticCase:
    cost = np.array(
        [
            [2, 8, 7],
            [6, 4, 3],
            [5, 8, 1],
            [9, 7, 6],
            [4, 6, 8],
        ],
        dtype=float,
    )
    return SyntheticCase(
        name="Rectangular 5x3",
        description=(
            "Five workers compete for three jobs. The notebook pads the matrix with two dummy jobs, "
            "so two workers can stay idle at zero additional cost."
        ),
        cost=cost,
        row_labels=_labels("worker_", 5),
        col_labels=_labels("job_", 3),
    )


def block_case(seed: int = 7) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    group_size = 3
    groups = 2
    size = groups * group_size
    cost = np.full((size, size), 8.0)
    for group in range(groups):
        start = group * group_size
        stop = start + group_size
        cost[start:stop, start:stop] = 1.5
    cost += rng.uniform(0.0, 1.5, size=(size, size))
    return SyntheticCase(
        name="Block Preferences",
        description=(
            "Workers and jobs belong to two groups. Costs are low inside a group and high outside, "
            "which makes the equality graph develop a visible block structure."
        ),
        cost=cost,
        row_labels=_labels("worker_", size),
        col_labels=_labels("job_", size),
    )


def geometric_case(size: int = 6, seed: int = 7) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    row_positions = rng.normal(loc=(0.0, 0.0), scale=(1.0, 0.6), size=(size, 2))
    true_perm = rng.permutation(size)
    col_positions = row_positions[true_perm] + rng.normal(loc=(0.35, -0.2), scale=0.18, size=(size, 2))
    pairwise_diff = row_positions[:, None, :] - col_positions[None, :, :]
    cost = np.linalg.norm(pairwise_diff, axis=2)
    return SyntheticCase(
        name="Geometric Matching",
        description=(
            "Costs come from Euclidean distance between worker and job coordinates. "
            "This turns the assignment into a spatial matching problem."
        ),
        cost=cost,
        row_labels=_labels("worker_", size),
        col_labels=_labels("job_", size),
        row_positions=row_positions,
        col_positions=col_positions,
    )


def random_case(size: int = 5, seed: int = 7) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 15, size=(size, size)).astype(float)
    return SyntheticCase(
        name=f"Random {size}x{size}",
        description="A generic random instance for experimentation.",
        cost=cost,
        row_labels=_labels("worker_", size),
        col_labels=_labels("job_", size),
    )


def default_case_suite(seed: int = 7) -> dict[str, SyntheticCase]:
    cases = [
        classic_case(),
        anti_greedy_case(),
        tie_heavy_case(),
        rectangular_case(),
        block_case(seed=seed),
        geometric_case(size=6, seed=seed),
        random_case(size=5, seed=seed),
    ]
    return {case.name: case for case in cases}


__all__ = [
    "SyntheticCase",
    "anti_greedy_case",
    "block_case",
    "classic_case",
    "default_case_suite",
    "geometric_case",
    "random_case",
    "rectangular_case",
    "tie_heavy_case",
]
