from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Sequence

import numpy as np

EPS = 1e-12


@dataclass(slots=True)
class TraceFrame:
    augmentation: int
    stage: str
    note: str
    row_potential: np.ndarray
    col_potential: np.ndarray
    reduced_cost: np.ndarray
    matching: list[tuple[int, int]]
    used_rows: np.ndarray | None = None
    used_columns: np.ndarray | None = None
    active_row: int | None = None
    active_column: int | None = None
    delta: float | None = None
    root_row: int | None = None
    current_column: int | None = None
    next_column: int | None = None
    owner_by_column: np.ndarray | None = None
    predecessor_column: np.ndarray | None = None
    min_slack_by_column: np.ndarray | None = None


@dataclass(slots=True)
class HungarianResult:
    original_cost: np.ndarray
    padded_cost: np.ndarray
    assignment: list[tuple[int, int]]
    full_assignment: list[tuple[int, int]]
    row_to_col: np.ndarray
    objective_cost: float
    real_cost: float
    row_potential: np.ndarray
    col_potential: np.ndarray
    reduced_cost: np.ndarray
    trace: list[TraceFrame]
    pad_value: float

    @property
    def dual_objective(self) -> float:
        return float(self.row_potential.sum() + self.col_potential.sum())


def as_cost_matrix(cost_matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("cost_matrix must be non-empty")
    return cost


def pad_cost_matrix(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    pad_value: float = 0.0,
) -> np.ndarray:
    cost = as_cost_matrix(cost_matrix)
    size = max(cost.shape)
    padded = np.full((size, size), float(pad_value), dtype=float)
    padded[: cost.shape[0], : cost.shape[1]] = cost
    return padded


def row_reduction(cost_matrix: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = as_cost_matrix(cost_matrix)
    row_offsets = cost.min(axis=1)
    reduced = cost - row_offsets[:, None]
    return row_offsets, reduced


def column_reduction(cost_matrix: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = as_cost_matrix(cost_matrix)
    col_offsets = cost.min(axis=0)
    reduced = cost - col_offsets[None, :]
    return col_offsets, reduced


def initial_dual_from_reductions(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_potential, row_reduced = row_reduction(cost_matrix)
    col_potential, reduced = column_reduction(row_reduced)
    return row_potential, col_potential, reduced


def reduced_cost_matrix(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    row_potential: Sequence[float] | np.ndarray,
    col_potential: Sequence[float] | np.ndarray,
) -> np.ndarray:
    cost = as_cost_matrix(cost_matrix)
    u = np.asarray(row_potential, dtype=float)
    v = np.asarray(col_potential, dtype=float)
    if u.shape[0] != cost.shape[0] or v.shape[0] != cost.shape[1]:
        raise ValueError("Potential dimensions must match the cost matrix")
    return cost - u[:, None] - v[None, :]


def assignment_cost(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    assignment: Sequence[tuple[int, int]],
) -> float:
    cost = as_cost_matrix(cost_matrix)
    return float(sum(cost[row, col] for row, col in assignment))


def greedy_row_assignment(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[list[tuple[int, int]], float]:
    cost = as_cost_matrix(cost_matrix)
    if cost.shape[0] > cost.shape[1]:
        raise ValueError("greedy_row_assignment needs at least as many columns as rows")

    used_columns: set[int] = set()
    assignment: list[tuple[int, int]] = []
    total = 0.0

    for row in range(cost.shape[0]):
        available = [col for col in range(cost.shape[1]) if col not in used_columns]
        col = min(available, key=lambda candidate: cost[row, candidate])
        used_columns.add(col)
        assignment.append((row, col))
        total += float(cost[row, col])

    return assignment, total


def brute_force_assignment(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    pad_value: float = 0.0,
    max_size: int = 8,
) -> dict[str, object]:
    original = as_cost_matrix(cost_matrix)
    padded = pad_cost_matrix(original, pad_value=pad_value)
    size = padded.shape[0]

    if size > max_size:
        raise ValueError(f"Brute force is capped at size {max_size}, got {size}")

    best_perm: tuple[int, ...] | None = None
    best_objective = float("inf")

    for perm in permutations(range(size)):
        objective = float(padded[np.arange(size), perm].sum())
        if objective < best_objective - EPS:
            best_objective = objective
            best_perm = perm

    if best_perm is None:
        raise RuntimeError("Failed to find a brute-force assignment")

    full_assignment = [(row, best_perm[row]) for row in range(size)]
    assignment = [
        (row, col)
        for row, col in full_assignment
        if row < original.shape[0] and col < original.shape[1]
    ]
    row_to_col = np.array(
        [
            col if col < original.shape[1] else -1
            for col in best_perm[: original.shape[0]]
        ],
        dtype=int,
    )

    return {
        "assignment": assignment,
        "full_assignment": full_assignment,
        "row_to_col": row_to_col,
        "objective_cost": best_objective,
        "real_cost": assignment_cost(original, assignment),
    }


def _matching_from_owner(column_owner: np.ndarray) -> list[tuple[int, int]]:
    return sorted(
        (int(column_owner[col] - 1), int(col - 1))
        for col in range(1, column_owner.shape[0])
        if column_owner[col] != 0
    )


def _tree_rows(column_owner: np.ndarray, used_columns: np.ndarray, size: int) -> np.ndarray:
    rows = np.zeros(size, dtype=bool)
    for col in range(used_columns.shape[0]):
        if used_columns[col] and column_owner[col] != 0:
            rows[column_owner[col] - 1] = True
    return rows


def _public_owner_by_column(column_owner: np.ndarray) -> np.ndarray:
    return np.array(
        [int(owner - 1) if owner != 0 else -1 for owner in column_owner[1:]],
        dtype=int,
    )


def _public_predecessor_column(way: np.ndarray) -> np.ndarray:
    return np.array(
        [int(previous - 1) if previous != 0 else -1 for previous in way[1:]],
        dtype=int,
    )


def _build_trace_frame(
    augmentation: int,
    stage: str,
    note: str,
    padded_cost: np.ndarray,
    row_potential: np.ndarray,
    col_potential: np.ndarray,
    column_owner: np.ndarray,
    way: np.ndarray | None = None,
    min_slack: np.ndarray | None = None,
    used_columns: np.ndarray | None = None,
    root_row: int | None = None,
    current_column: int | None = None,
    next_column: int | None = None,
    active_row: int | None = None,
    active_column: int | None = None,
    delta: float | None = None,
) -> TraceFrame:
    reduced = reduced_cost_matrix(padded_cost, row_potential[1:], col_potential[1:])
    rows = None
    cols = None
    if used_columns is not None:
        cols = used_columns[1:].copy()
        rows = _tree_rows(column_owner, used_columns, padded_cost.shape[0])

    predecessor = None if way is None else _public_predecessor_column(way)
    slack = None if min_slack is None else min_slack[1:].copy()

    return TraceFrame(
        augmentation=augmentation,
        stage=stage,
        note=note,
        row_potential=row_potential[1:].copy(),
        col_potential=col_potential[1:].copy(),
        reduced_cost=reduced,
        matching=_matching_from_owner(column_owner),
        used_rows=rows,
        used_columns=cols,
        active_row=active_row,
        active_column=active_column,
        delta=delta,
        root_row=root_row,
        current_column=None if current_column in (None, 0) else current_column - 1,
        next_column=None if next_column in (None, 0) else next_column - 1,
        owner_by_column=_public_owner_by_column(column_owner),
        predecessor_column=predecessor,
        min_slack_by_column=slack,
    )


def hungarian(
    cost_matrix: Sequence[Sequence[float]] | np.ndarray,
    pad_value: float = 0.0,
    record_trace: bool = True,
) -> HungarianResult:
    original = as_cost_matrix(cost_matrix)
    padded = pad_cost_matrix(original, pad_value=pad_value)
    size = padded.shape[0]

    row_init, col_init, _ = initial_dual_from_reductions(padded)
    row_potential = np.zeros(size + 1, dtype=float)
    col_potential = np.zeros(size + 1, dtype=float)
    row_potential[1:] = row_init
    col_potential[1:] = col_init

    column_owner = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    trace: list[TraceFrame] = []

    if record_trace:
        trace.append(
            _build_trace_frame(
                augmentation=0,
                stage="Row and Column Reduction",
                note=(
                    "Subtract row minima, then column minima. "
                    "The reduced costs stay non-negative and zeros become candidate edges."
                ),
                padded_cost=padded,
                row_potential=row_potential,
                col_potential=col_potential,
                column_owner=column_owner,
                root_row=None,
            )
        )

    for augmentation in range(1, size + 1):
        column_owner[0] = augmentation
        min_slack = np.full(size + 1, float("inf"))
        used_columns = np.zeros(size + 1, dtype=bool)
        way.fill(0)
        current_column = 0

        if record_trace:
            trace.append(
                _build_trace_frame(
                    augmentation=augmentation,
                    stage="Start Augmentation",
                    note=f"Search for an augmenting path that inserts row {augmentation - 1}.",
                    padded_cost=padded,
                    row_potential=row_potential,
                    col_potential=col_potential,
                    column_owner=column_owner,
                    way=way,
                    min_slack=min_slack,
                    used_columns=used_columns,
                    root_row=augmentation - 1,
                    current_column=current_column,
                    active_row=augmentation - 1,
                )
            )

        while True:
            used_columns[current_column] = True
            row = column_owner[current_column]
            delta = float("inf")
            next_column = 0

            for col in range(1, size + 1):
                if used_columns[col]:
                    continue
                slack = padded[row - 1, col - 1] - row_potential[row] - col_potential[col]
                if slack < min_slack[col] - EPS:
                    min_slack[col] = slack
                    way[col] = current_column
                if min_slack[col] < delta - EPS:
                    delta = min_slack[col]
                    next_column = col

            if record_trace:
                note = (
                    f"Expand the alternating tree from row {row - 1}. "
                    f"Update slack values for uncovered columns and pick column {next_column - 1} "
                    f"with smallest slack delta={delta:.3f}."
                )
                trace.append(
                    _build_trace_frame(
                        augmentation=augmentation,
                        stage="Scan Uncovered Columns",
                        note=note,
                        padded_cost=padded,
                        row_potential=row_potential,
                        col_potential=col_potential,
                        column_owner=column_owner,
                        way=way,
                        min_slack=min_slack,
                        used_columns=used_columns,
                        root_row=augmentation - 1,
                        current_column=current_column,
                        next_column=next_column,
                        active_row=row - 1,
                        active_column=next_column - 1,
                    )
                )

            if not np.isfinite(delta):
                raise RuntimeError("No augmenting path found; the input matrix may be invalid.")

            for col in range(size + 1):
                if used_columns[col]:
                    row_potential[column_owner[col]] += delta
                    col_potential[col] -= delta
                else:
                    min_slack[col] -= delta

            if record_trace:
                trace.append(
                    _build_trace_frame(
                        augmentation=augmentation,
                        stage="Dual Update",
                        note=(
                            f"Shift the dual variables by delta={delta:.3f}. "
                            "This preserves feasibility and creates at least one new zero edge."
                        ),
                        padded_cost=padded,
                        row_potential=row_potential,
                        col_potential=col_potential,
                        column_owner=column_owner,
                        way=way,
                        min_slack=min_slack,
                        used_columns=used_columns,
                        root_row=augmentation - 1,
                        current_column=current_column,
                        next_column=next_column,
                        active_row=row - 1,
                        active_column=next_column - 1,
                        delta=delta,
                    )
                )

            current_column = next_column
            if column_owner[current_column] == 0:
                break

        while True:
            previous_column = way[current_column]
            column_owner[current_column] = column_owner[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

        if record_trace:
            trace.append(
                _build_trace_frame(
                    augmentation=augmentation,
                    stage="Augment Matching",
                    note=f"Augment along the alternating path. The matching now contains {augmentation} edges.",
                    padded_cost=padded,
                    row_potential=row_potential,
                    col_potential=col_potential,
                    column_owner=column_owner,
                    way=way,
                    min_slack=min_slack,
                    root_row=augmentation - 1,
                    active_row=augmentation - 1,
                )
            )

    full_assignment = [(row, col) for row, col in _matching_from_owner(column_owner)]
    row_to_col = np.full(original.shape[0], -1, dtype=int)
    assignment: list[tuple[int, int]] = []

    for row, col in full_assignment:
        if row < original.shape[0]:
            if col < original.shape[1]:
                assignment.append((row, col))
                row_to_col[row] = col
            else:
                row_to_col[row] = -1

    reduced = reduced_cost_matrix(padded, row_potential[1:], col_potential[1:])
    objective_cost = assignment_cost(padded, full_assignment)
    real_cost = assignment_cost(original, assignment)

    return HungarianResult(
        original_cost=original.copy(),
        padded_cost=padded,
        assignment=assignment,
        full_assignment=full_assignment,
        row_to_col=row_to_col,
        objective_cost=objective_cost,
        real_cost=real_cost,
        row_potential=row_potential[1:].copy(),
        col_potential=col_potential[1:].copy(),
        reduced_cost=reduced,
        trace=trace,
        pad_value=float(pad_value),
    )


def certificate_summary(result: HungarianResult, tol: float = 1e-9) -> dict[str, float | bool]:
    dual_feasible = bool(np.all(result.reduced_cost >= -tol))
    matched_edges_are_tight = bool(
        all(abs(result.reduced_cost[row, col]) <= tol for row, col in result.full_assignment)
    )
    gap = float(result.objective_cost - result.dual_objective)
    return {
        "dual_feasible": dual_feasible,
        "matched_edges_are_tight": matched_edges_are_tight,
        "objective_cost": result.objective_cost,
        "dual_objective": result.dual_objective,
        "gap": gap,
    }


__all__ = [
    "EPS",
    "HungarianResult",
    "TraceFrame",
    "as_cost_matrix",
    "assignment_cost",
    "brute_force_assignment",
    "certificate_summary",
    "column_reduction",
    "greedy_row_assignment",
    "hungarian",
    "initial_dual_from_reductions",
    "pad_cost_matrix",
    "reduced_cost_matrix",
    "row_reduction",
]
