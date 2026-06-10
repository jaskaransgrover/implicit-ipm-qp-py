"""Convex quadratic program (QP) data structure.

Problem form (matching the paper):

    minimize    (1/2) x^T Q x + q^T x
    subject to  Ax >= b (inequality constraints, m of them)
                Cx = d (equality constraints, p of them)

with Q symmetric positive semidefinite. This module defines a validated,
immutable container for the problem data. Raw NumPy arrays are converted into
a typed ``QP`` here, at the boundary, and validated once; downstream solver
code can then assume shapes and types are correct .
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QP:
    """A validated convex QP instance.

    Attributes:
        Q: (n, n) symmetric positive-semidefinite objective matrix.
        q: (n,) linear objective vector.
        A: (m, n) inequality constraint matrix (A x >= b).
        b: (m,) inequality right-hand side.
        C: (p, n) equality constraint matrix (C x = d).
        d: (p,) equality right-hand side.
    """

    Q: FloatArray
    q: FloatArray
    A: FloatArray
    b: FloatArray
    C: FloatArray
    d: FloatArray

    @property
    def n(self) -> int:
        """Number of primal variables."""
        return int(self.q.shape[0])

    @property
    def m(self) -> int:
        """Number of inequality constraints."""
        return int(self.b.shape[0])

    @property
    def p(self) -> int:
        """Number of equality constraints."""
        return int(self.d.shape[0])

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Check that all shapes are mutually consistent and Q is symmetric.

        Raises:
            ValueError: If any dimension disagrees or Q is not square/symmetric.
        """
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError(f"Q must be square 2D, got shape {self.Q.shape}")

        n = self.Q.shape[0]
        if self.q.shape != (n,):
            raise ValueError(f"q must have shape ({n},), got {self.q.shape}")

        if self.A.ndim != 2 or self.A.shape[1] != n:
            raise ValueError(f"A must have shape (m, {n}), got {self.A.shape}")
        if self.b.shape != (self.A.shape[0],):
            raise ValueError(f"b must have shape ({self.A.shape[0]},), got {self.b.shape}")

        if self.C.ndim != 2 or self.C.shape[1] != n:
            raise ValueError(f"C must have shape (p, {n}), got {self.C.shape}")
        if self.d.shape != (self.C.shape[0],):
            raise ValueError(f"d must have shape ({self.C.shape[0]},), got {self.d.shape}")

        if not np.allclose(self.Q, self.Q.T):
            raise ValueError("Q must be symmetric")


def residual_stationarity(
    qp: QP,
    x: FloatArray,
    lam: FloatArray,
    gamma: FloatArray,
) -> FloatArray:
    """Stationarity residual r_x = Q x + q - A^T lambda - C^T gamma."""
    return qp.Q @ x + qp.q - qp.A.T @ lam - qp.C.T @ gamma


def residual_inequality(qp: QP, x: FloatArray, s: FloatArray) -> FloatArray:
    """Primal inequality residual r_i = A x - b - s (with s >= 0)."""
    return qp.A @ x - qp.b - s


def residual_equality(qp: QP, x: FloatArray) -> FloatArray:
    """Primal equality residual r_e = C x - d."""
    return qp.C @ x - qp.d


def residual_complementarity(lam: FloatArray, s: FloatArray) -> FloatArray:
    """Complementarity residual r_c = lambda (elementwise *) s."""
    return lam * s


def duality_gap(lam: FloatArray, s: FloatArray) -> float:
    """Total duality gap eta = lambda^T s."""
    return float(lam @ s)
