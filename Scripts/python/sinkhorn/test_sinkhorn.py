from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from algorithm import sinkhorn
from synthetic_cases import classic_1d_case, rectangular_case


def test_classic_case_matches_marginals() -> None:
    case = classic_1d_case()
    result = sinkhorn(case.cost_matrix, case.source_weights, case.target_weights, epsilon=0.35, max_iterations=200)
    assert result.converged is True
    assert np.allclose(result.coupling.sum(axis=1), result.source_weights, atol=1e-8)
    assert np.allclose(result.coupling.sum(axis=0), result.target_weights, atol=1e-8)


def test_rectangular_case_converges() -> None:
    case = rectangular_case()
    result = sinkhorn(case.cost_matrix, case.source_weights, case.target_weights, epsilon=0.55, max_iterations=250)
    assert result.converged is True
    assert result.coupling.shape == case.cost_matrix.shape
    assert np.all(result.coupling >= 0.0)


def test_random_balanced_cases_converge() -> None:
    rng = np.random.default_rng(42)
    for _ in range(12):
        n_source = int(rng.integers(3, 7))
        n_target = int(rng.integers(3, 7))
        source = rng.uniform(0.1, 1.0, size=n_source)
        target = rng.uniform(0.1, 1.0, size=n_target)
        source /= source.sum()
        target /= target.sum()
        cost = rng.uniform(0.0, 4.0, size=(n_source, n_target))
        result = sinkhorn(cost, source, target, epsilon=0.6, max_iterations=250)
        assert np.allclose(result.coupling.sum(axis=1), result.source_weights, atol=1e-8)
        assert np.allclose(result.coupling.sum(axis=0), result.target_weights, atol=1e-8)


def test_trace_stages_enforce_expected_marginals() -> None:
    case = classic_1d_case()
    result = sinkhorn(case.cost_matrix, case.source_weights, case.target_weights, epsilon=0.35, max_iterations=80)
    update_u_frame = next(frame for frame in result.trace if frame.stage == "Update u")
    update_v_frame = next(frame for frame in result.trace if frame.stage == "Update v")
    assert np.allclose(update_u_frame.row_sum, result.source_weights, atol=1e-10)
    assert np.allclose(update_v_frame.col_sum, result.target_weights, atol=1e-10)
