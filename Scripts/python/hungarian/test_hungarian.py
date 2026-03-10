from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from algorithm import brute_force_assignment, certificate_summary, hungarian
from synthetic_cases import anti_greedy_case, classic_case, rectangular_case


def test_classic_case_matches_bruteforce() -> None:
    case = classic_case()
    result = hungarian(case.cost)
    brute = brute_force_assignment(case.cost)
    assert np.isclose(result.objective_cost, brute["objective_cost"])
    assert sorted(result.assignment) == sorted(brute["assignment"])


def test_rectangular_case_matches_bruteforce() -> None:
    case = rectangular_case()
    result = hungarian(case.cost)
    brute = brute_force_assignment(case.cost)
    assert np.isclose(result.objective_cost, brute["objective_cost"])
    assert np.array_equal(result.row_to_col, brute["row_to_col"])


def test_dual_certificate_is_tight() -> None:
    result = hungarian(classic_case().cost)
    certificate = certificate_summary(result)
    assert certificate["dual_feasible"] is True
    assert certificate["matched_edges_are_tight"] is True
    assert abs(float(certificate["gap"])) < 1e-9


def test_random_small_matrices_match_bruteforce() -> None:
    rng = np.random.default_rng(42)
    for _ in range(20):
        n_rows = int(rng.integers(2, 6))
        n_cols = int(rng.integers(2, 6))
        cost = rng.integers(0, 12, size=(n_rows, n_cols)).astype(float)
        result = hungarian(cost)
        brute = brute_force_assignment(cost)
        assert np.isclose(result.objective_cost, brute["objective_cost"])


def test_greedy_trap_is_strictly_worse_than_hungarian() -> None:
    case = anti_greedy_case()
    result = hungarian(case.cost)
    assert result.objective_cost == 8.0


def test_trace_contains_implementation_state() -> None:
    result = hungarian(classic_case().cost, record_trace=True)
    padded_size = max(classic_case().cost.shape)

    scan_frame = next(frame for frame in result.trace if frame.stage == "Scan Uncovered Columns")
    assert scan_frame.owner_by_column is not None
    assert scan_frame.predecessor_column is not None
    assert scan_frame.min_slack_by_column is not None
    assert scan_frame.owner_by_column.shape == (padded_size,)
    assert scan_frame.predecessor_column.shape == (padded_size,)
    assert scan_frame.min_slack_by_column.shape == (padded_size,)
