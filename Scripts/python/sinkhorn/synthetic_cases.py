from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .algorithm import pairwise_squared_distance
except ImportError:
    from algorithm import pairwise_squared_distance


@dataclass(slots=True)
class SinkhornCase:
    name: str
    description: str
    source_weights: np.ndarray
    target_weights: np.ndarray
    cost_matrix: np.ndarray
    source_labels: list[str]
    target_labels: list[str]
    epsilon_hint: float
    source_positions: np.ndarray | None = None
    target_positions: np.ndarray | None = None


def _labels(prefix: str, size: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(size)]


def classic_1d_case() -> SinkhornCase:
    source_positions = np.array([0.0, 1.0, 2.0, 3.0])
    target_positions = np.array([0.0, 1.0, 2.0, 3.0])
    source_weights = np.array([0.45, 0.25, 0.20, 0.10])
    target_weights = np.array([0.10, 0.20, 0.30, 0.40])
    cost_matrix = pairwise_squared_distance(source_positions, target_positions)
    return SinkhornCase(
        name="Classic 1D Shift",
        description=(
            "A small one-dimensional transport problem. Mass starts concentrated on the left "
            "and must gradually move to the right."
        ),
        source_weights=source_weights,
        target_weights=target_weights,
        cost_matrix=cost_matrix,
        source_labels=_labels("source_", source_weights.size),
        target_labels=_labels("target_", target_weights.size),
        epsilon_hint=0.35,
        source_positions=source_positions,
        target_positions=target_positions,
    )


def bimodal_1d_case() -> SinkhornCase:
    source_positions = np.linspace(0.0, 5.0, 6)
    target_positions = np.linspace(0.0, 5.0, 6)
    source_weights = np.array([0.28, 0.22, 0.04, 0.04, 0.18, 0.24])
    target_weights = np.array([0.05, 0.15, 0.30, 0.25, 0.15, 0.10])
    cost_matrix = pairwise_squared_distance(source_positions, target_positions)
    return SinkhornCase(
        name="Bimodal to Center",
        description=(
            "The source has two peaks. The target asks the mass to concentrate toward the center, "
            "which makes the coupling visibly diffuse when epsilon is large."
        ),
        source_weights=source_weights,
        target_weights=target_weights,
        cost_matrix=cost_matrix,
        source_labels=_labels("source_", source_weights.size),
        target_labels=_labels("target_", target_weights.size),
        epsilon_hint=0.45,
        source_positions=source_positions,
        target_positions=target_positions,
    )


def rectangular_case() -> SinkhornCase:
    source_positions = np.array([0.0, 1.0, 2.0, 3.5, 5.0])
    target_positions = np.array([0.4, 2.2, 3.7, 5.1])
    source_weights = np.array([0.18, 0.20, 0.24, 0.16, 0.22])
    target_weights = np.array([0.22, 0.28, 0.20, 0.30])
    cost_matrix = pairwise_squared_distance(source_positions, target_positions)
    return SinkhornCase(
        name="Rectangular 5x4",
        description=(
            "The number of source and target bins does not need to match. "
            "Optimal transport only needs equal total mass."
        ),
        source_weights=source_weights,
        target_weights=target_weights,
        cost_matrix=cost_matrix,
        source_labels=_labels("source_", source_weights.size),
        target_labels=_labels("target_", target_weights.size),
        epsilon_hint=0.55,
        source_positions=source_positions,
        target_positions=target_positions,
    )


def geometric_2d_case(seed: int = 7) -> SinkhornCase:
    rng = np.random.default_rng(seed)
    source_positions = rng.normal(loc=(-0.7, 0.0), scale=(0.35, 0.55), size=(5, 2))
    target_positions = rng.normal(loc=(0.7, 0.1), scale=(0.40, 0.45), size=(5, 2))
    source_weights = np.array([0.12, 0.18, 0.22, 0.28, 0.20])
    target_weights = np.array([0.18, 0.16, 0.20, 0.24, 0.22])
    cost_matrix = pairwise_squared_distance(source_positions, target_positions)
    return SinkhornCase(
        name="Geometric 2D Clouds",
        description=(
            "Source and target masses live on two point clouds in the plane. "
            "The cost matrix comes from squared Euclidean distance."
        ),
        source_weights=source_weights,
        target_weights=target_weights,
        cost_matrix=cost_matrix,
        source_labels=_labels("source_", source_weights.size),
        target_labels=_labels("target_", target_weights.size),
        epsilon_hint=0.30,
        source_positions=source_positions,
        target_positions=target_positions,
    )


def random_case(size_source: int = 6, size_target: int = 5, seed: int = 7) -> SinkhornCase:
    rng = np.random.default_rng(seed)
    source_positions = np.sort(rng.uniform(0.0, 6.0, size=size_source))
    target_positions = np.sort(rng.uniform(0.0, 6.0, size=size_target))
    source_weights = rng.uniform(0.1, 1.0, size=size_source)
    target_weights = rng.uniform(0.1, 1.0, size=size_target)
    source_weights = source_weights / source_weights.sum()
    target_weights = target_weights / target_weights.sum()
    cost_matrix = pairwise_squared_distance(source_positions, target_positions)
    return SinkhornCase(
        name="Random 1D",
        description="A generic random balanced transport problem for experimentation.",
        source_weights=source_weights,
        target_weights=target_weights,
        cost_matrix=cost_matrix,
        source_labels=_labels("source_", size_source),
        target_labels=_labels("target_", size_target),
        epsilon_hint=0.40,
        source_positions=source_positions,
        target_positions=target_positions,
    )


def default_case_suite(seed: int = 7) -> dict[str, SinkhornCase]:
    cases = [
        classic_1d_case(),
        bimodal_1d_case(),
        rectangular_case(),
        geometric_2d_case(seed=seed),
        random_case(seed=seed),
    ]
    return {case.name: case for case in cases}


__all__ = [
    "SinkhornCase",
    "bimodal_1d_case",
    "classic_1d_case",
    "default_case_suite",
    "geometric_2d_case",
    "random_case",
    "rectangular_case",
]
