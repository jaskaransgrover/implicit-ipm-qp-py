"""Tests for the Maros-Meszaros loader and its l/u -> Ax>=b, Cx=d conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from implicit_ipm_qp.benchmarks import load_maros_meszaros
from implicit_ipm_qp.solvers.implicit import solve_implicit

# The committed subset lives here relative to the repo root.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "maros_meszaros"


def test_loads_qptest_with_correct_shapes() -> None:
    qp = load_maros_meszaros("QPTEST", _DATA_DIR)
    # QPTEST is a 2-variable problem.
    assert qp.n == 2
    # Q must be square (n x n) and exactly symmetric after symmetrization.
    assert qp.Q.shape == (qp.n, qp.n)
    np.testing.assert_allclose(qp.Q, qp.Q.T, atol=0.0)
    # q has length n; constraint blocks have n columns and matching row counts.
    assert qp.q.shape == (qp.n,)
    assert qp.A.shape[1] == qp.n
    assert qp.b.shape[0] == qp.A.shape[0]
    assert qp.C.shape[1] == qp.n
    assert qp.d.shape[0] == qp.C.shape[0]


def test_missing_problem_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_maros_meszaros("DOES_NOT_EXIST", _DATA_DIR)


def test_all_subset_problems_load() -> None:
    # Every committed subset problem must load and pass QP's own validation.
    names = ["QPTEST", "HS21", "HS35", "HS52", "HS53", "TAME", "ZECEVIC2", "CVXQP1_S"]
    for name in names:
        qp = load_maros_meszaros(name, _DATA_DIR)
        # Dimensions are internally consistent.
        assert qp.A.shape[1] == qp.n
        assert qp.C.shape[1] == qp.n
        # Q is symmetric.
        np.testing.assert_allclose(qp.Q, qp.Q.T, atol=1e-12)


@pytest.mark.parametrize("name", ["QPTEST", "HS21", "HS35", "HS52", "HS53"])
def test_solver_converges_on_small_problems(name: str) -> None:
    # The implicit solver should converge on these small, well-behaved problems.
    qp = load_maros_meszaros(name, _DATA_DIR)
    result = solve_implicit(qp)
    assert result.converged
    # At the solution, primal feasibility holds: Ax >= b (small tolerance).
    if qp.A.shape[0] > 0:
        assert np.all(qp.A @ result.x - qp.b >= -1e-4)
    # Equality feasibility: Cx = d.
    if qp.C.shape[0] > 0:
        np.testing.assert_allclose(qp.C @ result.x, qp.d, atol=1e-4)
