from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

EPS = 1e-12


@dataclass(slots=True)
class SinkhornTraceFrame:
    iteration: int
    stage: str
    note: str
    epsilon: float
    kernel: np.ndarray
    u: np.ndarray
    v: np.ndarray
    coupling: np.ndarray
    row_sum: np.ndarray
    col_sum: np.ndarray
    row_error: np.ndarray
    col_error: np.ndarray
    row_l1_error: float
    col_l1_error: float
    transport_cost: float
    entropy_term: float
    regularized_objective: float
    dual_f: np.ndarray
    dual_g: np.ndarray


@dataclass(slots=True)
class SinkhornResult:
    source_weights: np.ndarray
    target_weights: np.ndarray
    cost_matrix: np.ndarray
    epsilon: float
    kernel: np.ndarray
    u: np.ndarray
    v: np.ndarray
    coupling: np.ndarray
    converged: bool
    iterations_run: int
    tol: float
    trace: list[SinkhornTraceFrame]
    row_l1_history: np.ndarray
    col_l1_history: np.ndarray
    transport_cost_history: np.ndarray
    regularized_objective_history: np.ndarray
    transport_cost: float
    entropy_term: float
    regularized_objective: float


def as_cost_matrix(cost_matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("cost_matrix must be non-empty")
    return cost


def as_probability_vector(
    weights: Sequence[float] | np.ndarray,
    *,
    normalize: bool = True,
    name: str = "weights",
) -> np.ndarray:
    vector = np.asarray(weights, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(vector < 0):
        raise ValueError(f"{name} must be non-negative")
    total_mass = float(vector.sum())
    if total_mass <= 0:
        raise ValueError(f"{name} must have positive total mass")
    if normalize:
        vector = vector / total_mass
    return vector


def pairwise_squared_distance(
    source_points: Sequence[Sequence[float]] | np.ndarray,
    target_points: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.ndim == 1:
        source = source[:, None]
    if target.ndim == 1:
        target = target[:, None]
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("Points must be one- or two-dimensional arrays")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target points must share the same ambient dimension")
    diff = source[:, None, :] - target[None, :, :]
    return np.sum(diff * diff, axis=2)


def gibbs_kernel(cost_matrix: Sequence[Sequence[float]] | np.ndarray, epsilon: float) -> np.ndarray:
    cost = as_cost_matrix(cost_matrix)
    if epsilon <= 0:
        raise ValueError("epsilon must be strictly positive")
    return np.exp(-cost / epsilon)


def coupling_from_scalings(kernel: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[:, None] * kernel * v[None, :]


def transport_cost(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    coupling: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    cost = as_cost_matrix(cost_matrix)
    plan = as_cost_matrix(coupling)
    if cost.shape != plan.shape:
        raise ValueError("cost_matrix and coupling must have the same shape")
    return float(np.sum(cost * plan))


def entropy_term(coupling: Sequence[Sequence[float]] | np.ndarray) -> float:
    plan = as_cost_matrix(coupling)
    mask = plan > 0
    return float(np.sum(plan[mask] * (np.log(plan[mask]) - 1.0)))


def regularized_ot_objective(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    coupling: Sequence[Sequence[float]] | np.ndarray,
    epsilon: float,
) -> float:
    return transport_cost(cost_matrix, coupling) + epsilon * entropy_term(coupling)


def _build_trace_frame(
    *,
    iteration: int,
    stage: str,
    note: str,
    epsilon: float,
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    cost_matrix: np.ndarray,
    kernel: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> SinkhornTraceFrame:
    coupling = coupling_from_scalings(kernel, u, v)
    row_sum = coupling.sum(axis=1)
    col_sum = coupling.sum(axis=0)
    row_error = row_sum - source_weights
    col_error = col_sum - target_weights
    cost_value = transport_cost(cost_matrix, coupling)
    entropy_value = entropy_term(coupling)
    return SinkhornTraceFrame(
        iteration=iteration,
        stage=stage,
        note=note,
        epsilon=epsilon,
        kernel=kernel.copy(),
        u=u.copy(),
        v=v.copy(),
        coupling=coupling,
        row_sum=row_sum,
        col_sum=col_sum,
        row_error=row_error,
        col_error=col_error,
        row_l1_error=float(np.abs(row_error).sum()),
        col_l1_error=float(np.abs(col_error).sum()),
        transport_cost=cost_value,
        entropy_term=entropy_value,
        regularized_objective=cost_value + epsilon * entropy_value,
        dual_f=epsilon * np.log(np.maximum(u, EPS)),
        dual_g=epsilon * np.log(np.maximum(v, EPS)),
    )


def sinkhorn(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    source_weights: Sequence[float] | np.ndarray,
    target_weights: Sequence[float] | np.ndarray,
    *,
    epsilon: float = 0.4,
    max_iterations: int = 200,
    tol: float = 1e-9,
    normalize: bool = True,
    record_trace: bool = True,
) -> SinkhornResult:
    cost = as_cost_matrix(cost_matrix)
    source = as_probability_vector(source_weights, normalize=normalize, name="source_weights")
    target = as_probability_vector(target_weights, normalize=normalize, name="target_weights")

    if cost.shape != (source.shape[0], target.shape[0]):
        raise ValueError(
            "cost_matrix shape must match len(source_weights) x len(target_weights)"
        )
    if not normalize and not np.isclose(source.sum(), target.sum(), atol=tol, rtol=0.0):
        raise ValueError("source_weights and target_weights must have the same total mass")
    if epsilon <= 0:
        raise ValueError("epsilon must be strictly positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    kernel = gibbs_kernel(cost, epsilon)
    u = np.ones_like(source)
    v = np.ones_like(target)

    trace: list[SinkhornTraceFrame] = []
    row_l1_history: list[float] = []
    col_l1_history: list[float] = []
    transport_cost_history: list[float] = []
    regularized_objective_history: list[float] = []

    if record_trace:
        trace.append(
            _build_trace_frame(
                iteration=0,
                stage="Initialize Kernel",
                note=(
                    "Build the Gibbs kernel K = exp(-C / epsilon). "
                    "Start with scaling vectors u = 1 and v = 1."
                ),
                epsilon=epsilon,
                source_weights=source,
                target_weights=target,
                cost_matrix=cost,
                kernel=kernel,
                u=u,
                v=v,
            )
        )

    converged = False
    iterations_run = 0

    for iteration in range(1, max_iterations + 1):
        u = source / np.maximum(kernel @ v, EPS)
        if record_trace:
            trace.append(
                _build_trace_frame(
                    iteration=iteration,
                    stage="Update u",
                    note=(
                        "Scale rows: u = a / (K v). "
                        "After this update, the row sums of P = diag(u) K diag(v) match the source weights exactly."
                    ),
                    epsilon=epsilon,
                    source_weights=source,
                    target_weights=target,
                    cost_matrix=cost,
                    kernel=kernel,
                    u=u,
                    v=v,
                )
            )

        v = target / np.maximum(kernel.T @ u, EPS)
        frame = _build_trace_frame(
            iteration=iteration,
            stage="Update v",
            note=(
                "Scale columns: v = b / (K^T u). "
                "After this update, the column sums match the target weights exactly."
            ),
            epsilon=epsilon,
            source_weights=source,
            target_weights=target,
            cost_matrix=cost,
            kernel=kernel,
            u=u,
            v=v,
        )

        row_l1_history.append(frame.row_l1_error)
        col_l1_history.append(frame.col_l1_error)
        transport_cost_history.append(frame.transport_cost)
        regularized_objective_history.append(frame.regularized_objective)

        if record_trace:
            trace.append(frame)

        iterations_run = iteration
        if max(frame.row_l1_error, frame.col_l1_error) < tol:
            converged = True
            break

    final_frame = _build_trace_frame(
        iteration=iterations_run,
        stage="Final Coupling",
        note=(
            "Assemble the final transport plan P = diag(u) K diag(v). "
            "If the marginal errors are tiny, this is the Sinkhorn solution."
        ),
        epsilon=epsilon,
        source_weights=source,
        target_weights=target,
        cost_matrix=cost,
        kernel=kernel,
        u=u,
        v=v,
    )
    if record_trace:
        trace.append(final_frame)

    return SinkhornResult(
        source_weights=source,
        target_weights=target,
        cost_matrix=cost,
        epsilon=epsilon,
        kernel=kernel,
        u=u.copy(),
        v=v.copy(),
        coupling=final_frame.coupling,
        converged=converged,
        iterations_run=iterations_run,
        tol=tol,
        trace=trace,
        row_l1_history=np.asarray(row_l1_history, dtype=float),
        col_l1_history=np.asarray(col_l1_history, dtype=float),
        transport_cost_history=np.asarray(transport_cost_history, dtype=float),
        regularized_objective_history=np.asarray(regularized_objective_history, dtype=float),
        transport_cost=final_frame.transport_cost,
        entropy_term=final_frame.entropy_term,
        regularized_objective=final_frame.regularized_objective,
    )


__all__ = [
    "EPS",
    "SinkhornResult",
    "SinkhornTraceFrame",
    "as_cost_matrix",
    "as_probability_vector",
    "coupling_from_scalings",
    "entropy_term",
    "gibbs_kernel",
    "pairwise_squared_distance",
    "regularized_ot_objective",
    "sinkhorn",
    "transport_cost",
]
