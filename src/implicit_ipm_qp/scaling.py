"""Ruiz equilibration for QP data.

Implements the QP-specific Ruiz scaling used in the reference implementation
(Arrizabalaga & Manchester, citing Ruiz 2001). Every problem instance is
equilibrated before solving to improve conditioning.

Three diagonal scalings are computed and accumulated over a fixed number of
iterations:

  * ``d_scale``  (length n): scales primal variables (columns of Q, A, C and,
    by symmetry, rows of Q).
  * ``ea_scale`` (length m): scales inequality rows (rows of A, and b).
  * ``ec_scale`` (length p): scales equality rows (rows of C, and d).

Per iteration, each scaling factor is the inverse square root of the relevant
max-absolute (infinity) norm, clamped to 1 where the norm is numerically zero.
This is a pure function: inputs are not mutated; scaled copies are returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from implicit_ipm_qp.qp import QP, FloatArray

_TINY: float = 1e-12


@dataclass(frozen=True)
class RuizScaling:
    """Result of Ruiz equilibration.

    Attributes:
        qp: The equilibrated QP.
        d_scale: (n,) accumulated primal-variable scaling.
        ea_scale: (m,) accumulated inequality-row scaling.
        ec_scale: (p,) accumulated equality-row scaling.
    """

    qp: QP
    d_scale: FloatArray
    ea_scale: FloatArray
    ec_scale: FloatArray


def _inv_sqrt_or_one(norms: FloatArray) -> FloatArray:
    """Return 1/sqrt(norm), or 1.0 where norm <= _TINY (avoid div-by-zero)."""
    return np.where(norms <= _TINY, 1.0, 1.0 / np.sqrt(norms))


def ruiz_equilibrate(qp: QP, max_iters: int = 10) -> RuizScaling:
    """Equilibrate a QP using Ruiz scaling.

    Args:
        qp: The problem to scale (not mutated).
        max_iters: Number of Ruiz iterations.

    Returns:
        A ``RuizScaling`` holding the scaled QP and the three accumulated
        scaling vectors.

    Raises:
        ValueError: If ``max_iters`` is not positive.
    """
    if max_iters <= 0:
        raise ValueError(f"max_iters must be positive, got {max_iters}")

    n, m, p = qp.n, qp.m, qp.p

    Q = qp.Q.astype(np.float64, copy=True)
    A = qp.A.astype(np.float64, copy=True)
    C = qp.C.astype(np.float64, copy=True)

    d_scale = np.ones(n, dtype=np.float64)
    ea_scale = np.ones(m, dtype=np.float64)
    ec_scale = np.ones(p, dtype=np.float64)

    def col_max_abs(matrix: FloatArray, width: int) -> FloatArray:
        if matrix.shape[0] == 0:
            return np.zeros(width, dtype=np.float64)
        return np.asarray(np.max(np.abs(matrix), axis=0), dtype=np.float64)

    def row_max_abs(matrix: FloatArray) -> FloatArray:
        if matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(np.max(np.abs(matrix), axis=1), dtype=np.float64)

    for _ in range(max_iters):
        # Column (variable) norms across Q, A, C.
        delta = col_max_abs(Q, n)
        if m > 0:
            delta = np.maximum(delta, col_max_abs(A, n))
        if p > 0:
            delta = np.maximum(delta, col_max_abs(C, n))

        inv_sqrt_delta = _inv_sqrt_or_one(delta)
        inv_sqrt_epsA = _inv_sqrt_or_one(row_max_abs(A))
        inv_sqrt_epsC = _inv_sqrt_or_one(row_max_abs(C))

        # Accumulate.
        d_scale *= inv_sqrt_delta
        if m > 0:
            ea_scale *= inv_sqrt_epsA
        if p > 0:
            ec_scale *= inv_sqrt_epsC

        # Apply to matrices: Q <- Dd Q Dd ; A <- Dea A Dd ; C <- Dec C Dd.
        Q = inv_sqrt_delta[:, None] * Q * inv_sqrt_delta[None, :]
        if m > 0:
            A = inv_sqrt_epsA[:, None] * A * inv_sqrt_delta[None, :]
        if p > 0:
            C = inv_sqrt_epsC[:, None] * C * inv_sqrt_delta[None, :]

    q = d_scale * qp.q
    b = ea_scale * qp.b
    d = ec_scale * qp.d

    scaled = QP(Q=Q, q=q, A=A, b=b, C=C, d=d)
    return RuizScaling(qp=scaled, d_scale=d_scale, ea_scale=ea_scale, ec_scale=ec_scale)
