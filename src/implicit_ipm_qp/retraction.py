"""Softplus retraction map for the implicit interior-point method.

The retraction map b_mu enforces complementarity by construction. Following
Arrizabalaga & Manchester (Definition 1, Eq. 15), the unique map satisfying

    b_mu(v) * b_mu(-v) = mu          (complementarity, 14a)
    db_mu(v) + db_mu(-v) = 1         (structural symmetry, 14b)
    0 < db_mu(v) <= 1                (bounded derivative, 14c)

is the softplus map

    b_mu(v)  = (v + sqrt(v^2 + 4 mu)) / 2
    db_mu(v) = (1 + v / sqrt(v^2 + 4 mu)) / 2

These are scalar functions applied elementwise to v in R^m, with mu a single
positive scalar shared across all components. Pure functions: inputs are not
mutated.
"""

from __future__ import annotations

import numpy as np

from implicit_ipm_qp.qp import FloatArray


def b_mu(v: FloatArray, mu: float) -> FloatArray:
    """Softplus retraction map, applied elementwise.

    Args:
        v: Auxiliary variable, shape (m,).
        mu: Barrier parameter, a single positive scalar.

    Returns:
        ``(v + sqrt(v**2 + 4*mu)) / 2``, shape (m,), strictly positive.

    Raises:
        ValueError: If ``mu`` is not strictly positive.
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be strictly positive, got {mu}")
    return np.asarray(0.5 * (v + np.sqrt(v * v + 4.0 * mu)), dtype=np.float64)


def db_mu(v: FloatArray, mu: float) -> FloatArray:
    """Derivative of the softplus retraction map, applied elementwise.

    Args:
        v: Auxiliary variable, shape (m,).
        mu: Barrier parameter, a single positive scalar.

    Returns:
        ``(1 + v / sqrt(v**2 + 4*mu)) / 2``, shape (m,), in the range (0, 1].

    Raises:
        ValueError: If ``mu`` is not strictly positive.
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be strictly positive, got {mu}")
    root = np.sqrt(v * v + 4.0 * mu)
    return np.asarray(0.5 * (1.0 + v / root), dtype=np.float64)
