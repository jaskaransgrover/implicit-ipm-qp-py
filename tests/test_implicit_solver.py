"""Correctness tests for the implicit interior-point QP solver.

Each test checks a property derivable from the QP optimality conditions, not
from any reference implementation:
  - exact recovery of a known minimizer when constraints are inactive,
  - the KKT conditions at the returned point for an active-constraint problem,
  - monotone decrease of the duality gap (the defining IPM behavior),
  - exactness of the starting-point retraction inversion v = lambda - mu/lambda.
"""

from __future__ import annotations

import itertools

import numpy as np

from i2pd.qp import QP
from i2pd.retraction import b_mu
from i2pd.solvers.implicit import _starting_point, solve_implicit


def _empty_equalities(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A (0, n) matrix and length-0 vector: a QP with no equality constraints."""
    return np.zeros((0, n)), np.zeros(0)


def test_recovers_unconstrained_minimizer() -> None:
    # min 1/2 xᵀQx + qᵀx with inequalities that are slack at the optimum.
    # The true minimizer is x* = -Q^{-1} q. We place a loose box so Ax >= b
    # holds strictly at x*, leaving all constraints inactive.
    q_mat = np.array([[2.0, 0.0], [0.0, 3.0]])
    q_vec = np.array([-2.0, -6.0])
    x_star = np.linalg.solve(q_mat, -q_vec)  # = [1, 2]

    # Inequalities  x >= -10  and  -x >= -10  (i.e. -10 <= x <= 10): inactive.
    a_mat = np.vstack([np.eye(2), -np.eye(2)])
    b_vec = np.array([-10.0, -10.0, -10.0, -10.0])
    c_mat, d_vec = _empty_equalities(2)

    qp = QP(Q=q_mat, q=q_vec, A=a_mat, b=b_vec, C=c_mat, d=d_vec)
    result = solve_implicit(qp)

    assert result.converged
    np.testing.assert_allclose(result.x, x_star, rtol=1e-5, atol=1e-5)


def test_kkt_conditions_with_active_constraint() -> None:
    # min 1/2(x1^2 + x2^2)  s.t.  x1 + x2 >= 2.
    # Unconstrained min is 0; the constraint is active, optimum at (1, 1).
    q_mat = np.eye(2)
    q_vec = np.zeros(2)
    a_mat = np.array([[1.0, 1.0]])
    b_vec = np.array([2.0])
    c_mat, d_vec = _empty_equalities(2)

    qp = QP(Q=q_mat, q=q_vec, A=a_mat, b=b_vec, C=c_mat, d=d_vec)
    result = solve_implicit(qp)

    assert result.converged
    x, lam, s = result.x, result.lam, result.s

    # Stationarity: Qx + q - Aᵀλ = 0.
    stat = q_mat @ x + q_vec - a_mat.T @ lam
    np.testing.assert_allclose(stat, np.zeros(2), atol=1e-4)
    # Primal feasibility and slack definition: Ax - b - s = 0, s >= 0.
    np.testing.assert_allclose(a_mat @ x - b_vec - s, np.zeros(1), atol=1e-4)
    assert np.all(s >= -1e-8)
    # Dual feasibility: λ >= 0.
    assert np.all(lam >= -1e-8)
    # Complementarity: λᵀs ≈ 0.
    assert float(lam @ s) <= 1e-4
    # Known optimum.
    np.testing.assert_allclose(x, np.array([1.0, 1.0]), rtol=1e-4, atol=1e-4)


def test_duality_gap_decreases_monotonically() -> None:
    q_mat = np.eye(2)
    q_vec = np.array([1.0, 1.0])
    a_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    b_vec = np.array([0.5, 0.5])
    c_mat, d_vec = _empty_equalities(2)

    qp = QP(Q=q_mat, q=q_vec, A=a_mat, b=b_vec, C=c_mat, d=d_vec)
    result = solve_implicit(qp)

    gaps = result.gap_history
    assert len(gaps) >= 2
    # Each recorded gap is no larger than the previous one (allow tiny slack).
    for earlier, later in itertools.pairwise(gaps):
        assert later <= earlier + 1e-9


def test_starting_point_retraction_is_exact() -> None:
    # At the initial point, lambda must equal b_mu(v) with mu = (lambdaᵀs)/m.
    # This verifies the inversion v = lambda - mu/lambda is exact.
    q_mat = np.eye(2)
    q_vec = np.array([1.0, 1.0])
    a_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
    b_vec = np.array([0.5, 0.5])
    c_mat, d_vec = _empty_equalities(2)

    qp = QP(Q=q_mat, q=q_vec, A=a_mat, b=b_vec, C=c_mat, d=d_vec)
    _x, lam, _gamma, s, v = _starting_point(qp)

    m = lam.shape[0]
    mu = float(lam @ s) / m
    np.testing.assert_allclose(lam, b_mu(v, mu), rtol=1e-10, atol=1e-12)
