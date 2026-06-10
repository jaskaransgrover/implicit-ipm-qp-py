"""Tests for Ruiz equilibration."""

from __future__ import annotations

import numpy as np
import pytest

from i2pd.qp import QP
from i2pd.scaling import ruiz_equilibrate


def _badly_scaled_qp() -> QP:
    rng = np.random.default_rng(0)
    n = 5
    # Build an SPD Q with wildly varying scales.
    root = rng.standard_normal((n, n))
    scales = np.diag([1e-3, 1.0, 1e2, 1e1, 1e-1])
    Q = scales @ (root @ root.T + n * np.eye(n)) @ scales
    Q = 0.5 * (Q + Q.T)
    q = rng.standard_normal(n) * 1e2
    A = rng.standard_normal((4, n)) * np.array([1e3, 1.0, 1e-2, 1e1])[:, None]
    b = rng.standard_normal(4)
    C = rng.standard_normal((2, n))
    d = rng.standard_normal(2)
    return QP(Q=Q, q=q, A=A, b=b, C=C, d=d)


def test_does_not_mutate_input() -> None:
    qp = _badly_scaled_qp()
    Q_before = qp.Q.copy()
    ruiz_equilibrate(qp)
    np.testing.assert_array_equal(qp.Q, Q_before)


def test_row_and_col_norms_approach_one() -> None:
    qp = _badly_scaled_qp()
    res = ruiz_equilibrate(qp, max_iters=20)
    scaled = res.qp

    # Stack the columns the way the algorithm does: Q, A, C share the variable
    # (column) scaling. Each variable's column inf-norm should be ~1.
    col_stack = np.vstack([scaled.Q, scaled.A, scaled.C])
    col_norms = np.max(np.abs(col_stack), axis=0)
    np.testing.assert_allclose(col_norms, np.ones(qp.n), atol=0.2)

    # Each inequality / equality row inf-norm should be ~1.
    a_row_norms = np.max(np.abs(scaled.A), axis=1)
    c_row_norms = np.max(np.abs(scaled.C), axis=1)
    np.testing.assert_allclose(a_row_norms, np.ones(qp.m), atol=0.2)
    np.testing.assert_allclose(c_row_norms, np.ones(qp.p), atol=0.2)


def test_scaling_vectors_have_right_shapes() -> None:
    qp = _badly_scaled_qp()
    res = ruiz_equilibrate(qp)
    assert res.d_scale.shape == (qp.n,)
    assert res.ea_scale.shape == (qp.m,)
    assert res.ec_scale.shape == (qp.p,)


def test_rejects_nonpositive_iters() -> None:
    with pytest.raises(ValueError, match="max_iters must be positive"):
        ruiz_equilibrate(_badly_scaled_qp(), max_iters=0)
