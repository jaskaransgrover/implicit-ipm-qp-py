"""Tests for the QP data structure and its validation."""

from __future__ import annotations

import numpy as np
import pytest

from i2pd.qp import QP


def _toy_qp() -> QP:
    # The paper's synthetic 2D problem: Q = I_2, q = 0, four inequalities,
    # no equality constraints.
    Q = np.eye(2)
    q = np.zeros(2)
    A = np.array([[1.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    b = np.array([0.65, -0.1, -0.85, -0.8])
    C = np.zeros((0, 2))
    d = np.zeros(0)
    return QP(Q=Q, q=q, A=A, b=b, C=C, d=d)


def test_dimensions() -> None:
    qp = _toy_qp()
    assert qp.n == 2
    assert qp.m == 4
    assert qp.p == 0


def test_accepts_no_equality_constraints() -> None:
    # p = 0 is valid; C is (0, n), d is (0,).
    qp = _toy_qp()
    assert qp.C.shape == (0, 2)
    assert qp.d.shape == (0,)


def test_rejects_nonsquare_Q() -> None:
    with pytest.raises(ValueError, match="square"):
        QP(
            Q=np.ones((2, 3)),
            q=np.zeros(3),
            A=np.zeros((0, 3)),
            b=np.zeros(0),
            C=np.zeros((0, 3)),
            d=np.zeros(0),
        )


def test_rejects_asymmetric_Q() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        QP(
            Q=np.array([[1.0, 2.0], [0.0, 1.0]]),
            q=np.zeros(2),
            A=np.zeros((0, 2)),
            b=np.zeros(0),
            C=np.zeros((0, 2)),
            d=np.zeros(0),
        )


def test_rejects_mismatched_b() -> None:
    with pytest.raises(ValueError, match="b must have shape"):
        QP(
            Q=np.eye(2),
            q=np.zeros(2),
            A=np.zeros((3, 2)),
            b=np.zeros(2),  # wrong: should be length 3
            C=np.zeros((0, 2)),
            d=np.zeros(0),
        )
