"""Loader for the Maros-Meszaros convex QP test set.

The .mat files are the standard 138-problem set, sourced from
qpsolvers/maros_meszaros_qpbenchmark (https://github.com/qpsolvers/maros_meszaros_qpbenchmark),
which themselves derive from sif2mat.m in the proxqp_benchmark project. Each
file stores a problem in the double-sided form

    min  1/2 xᵀ P x + qᵀ x + r   s.t.   l <= A x <= u

with the infinity constant set to 1e20. We convert this into the form our
solver expects:

    min  1/2 xᵀ Q x + qᵀ x   s.t.   Ax >= b,  Cx = d.

Conversion rule (matching the canonical qpbenchmark conversion):
  * a row with l_i == u_i (within tol) becomes an equality  A_i x = l_i   -> (C, d)
  * otherwise, a finite l_i gives an inequality  A_i x >= l_i
              a finite u_i gives an inequality -A_i x >= -u_i
  * rows with an infinite bound on that side are dropped (no constraint).
The constant offset r does not affect the minimizer and is discarded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as spio

from implicit_ipm_qp.qp import QP

# Bounds at or beyond this magnitude are treated as infinite (no constraint).
_INF_THRESHOLD = 9e19
# A row whose lower and upper bounds are this close is treated as an equality.
_EQ_TOL = 1e-10


def load_maros_meszaros(name: str, data_dir: str | Path) -> QP:
    """Load a Maros-Meszaros problem by name and convert it to our QP form.

    Args:
        name: Problem name without extension, e.g. "QPTEST".
        data_dir: Directory containing the .mat files.

    Returns:
        The problem as a QP with Ax >= b inequalities and Cx = d equalities.
    """
    path = Path(data_dir) / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(f"no such problem file: {path}")

    mat = spio.loadmat(str(path))

    # P, A may be sparse; densify to plain float64 arrays for our NumPy solver.
    p_mat = _to_dense(mat["P"])
    a_all = _to_dense(mat["A"])
    q_vec = np.asarray(mat["q"], dtype=np.float64).ravel()
    lo = np.asarray(mat["l"], dtype=np.float64).ravel()
    up = np.asarray(mat["u"], dtype=np.float64).ravel()

    # The stored P is the upper (or lower) triangle convention; symmetrize so
    # Q is exactly symmetric, which our solver and KKT derivation assume.
    q_sym = 0.5 * (p_mat + p_mat.T)

    eq_rows: list[int] = []
    ge_rows: list[tuple[int, float, int]] = []  # (row, bound, sign): A_i x >= sign*bound
    for i in range(a_all.shape[0]):
        li, ui = lo[i], up[i]
        if abs(ui - li) <= _EQ_TOL and abs(li) < _INF_THRESHOLD:
            eq_rows.append(i)
        else:
            if li > -_INF_THRESHOLD:  # finite lower:  A_i x >= l_i
                ge_rows.append((i, li, +1))
            if ui < _INF_THRESHOLD:  # finite upper: -A_i x >= -u_i
                ge_rows.append((i, ui, -1))

    n = q_sym.shape[0]

    # Equality block C x = d.
    if eq_rows:
        c_mat = a_all[eq_rows, :].astype(np.float64)
        d_vec = lo[eq_rows].astype(np.float64)
    else:
        c_mat = np.zeros((0, n), dtype=np.float64)
        d_vec = np.zeros(0, dtype=np.float64)

    # Inequality block A x >= b (build with the proper sign per row).
    if ge_rows:
        a_rows = np.vstack([sign * a_all[i, :] for (i, _b, sign) in ge_rows]).astype(np.float64)
        b_vec = np.array([sign * bnd for (_i, bnd, sign) in ge_rows], dtype=np.float64)
    else:
        a_rows = np.zeros((0, n), dtype=np.float64)
        b_vec = np.zeros(0, dtype=np.float64)

    return QP(Q=q_sym, q=q_vec, A=a_rows, b=b_vec, C=c_mat, d=d_vec)


def _to_dense(mat: object) -> np.ndarray:
    """Return a dense float64 2D array from a possibly-sparse loadmat entry.

    loadmat returns sparse matrices for some problems; those expose .todense().
    """
    todense = getattr(mat, "todense", None)
    dense = np.asarray(todense()) if callable(todense) else np.asarray(mat)
    return np.asarray(dense, dtype=np.float64)
