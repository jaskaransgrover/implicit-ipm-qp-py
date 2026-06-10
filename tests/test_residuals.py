"""Tests for QP residual functions, checked against hand computation."""

from __future__ import annotations

import numpy as np

from implicit_ipm_qp.qp import (
    QP,
    duality_gap,
    residual_complementarity,
    residual_equality,
    residual_inequality,
    residual_stationarity,
)


def _tiny_qp() -> QP:
    # n=2, m=2, p=1. Small enough to verify residuals by hand.
    Q = np.array([[2.0, 0.0], [0.0, 4.0]])
    q = np.array([1.0, 1.0])
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    C = np.array([[1.0, 1.0]])
    d = np.array([1.0])
    return QP(Q=Q, q=q, A=A, b=b, C=C, d=d)


def test_stationarity_hand_computed() -> None:
    qp = _tiny_qp()
    x = np.array([1.0, 2.0])
    lam = np.array([0.5, 0.5])
    gamma = np.array([1.0])
    # Q x + q = [2*1+1, 4*2+1] = [3, 9]
    # A^T lam = [0.5, 0.5];  C^T gamma = [1, 1]
    # r_x = [3,9] - [0.5,0.5] - [1,1] = [1.5, 7.5]
    expected = np.array([1.5, 7.5])
    np.testing.assert_allclose(residual_stationarity(qp, x, lam, gamma), expected)


def test_inequality_hand_computed() -> None:
    qp = _tiny_qp()
    x = np.array([1.0, 2.0])
    s = np.array([0.3, 0.4])
    # A x - b - s = [1,2] - [0,0] - [0.3,0.4] = [0.7, 1.6]
    np.testing.assert_allclose(residual_inequality(qp, x, s), np.array([0.7, 1.6]))


def test_equality_hand_computed() -> None:
    qp = _tiny_qp()
    x = np.array([1.0, 2.0])
    # C x - d = (1+2) - 1 = 2
    np.testing.assert_allclose(residual_equality(qp, x), np.array([2.0]))


def test_complementarity_is_elementwise() -> None:
    lam = np.array([1.0, 2.0, 3.0])
    s = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(residual_complementarity(lam, s), np.array([4.0, 10.0, 18.0]))


def test_duality_gap() -> None:
    lam = np.array([1.0, 2.0, 3.0])
    s = np.array([4.0, 5.0, 6.0])
    # 1*4 + 2*5 + 3*6 = 32
    assert duality_gap(lam, s) == 32.0
