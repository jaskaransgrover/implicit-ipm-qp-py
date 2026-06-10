"""Property tests for the softplus retraction map (Definition 1)."""

from __future__ import annotations

import numpy as np

from i2pd.retraction import b_mu, db_mu


def _v_grid() -> np.ndarray:
    # A spread of values including negative, near-zero, and large.
    return np.array([-50.0, -3.0, -0.1, 0.0, 0.1, 3.0, 50.0])


def test_complementarity_identity() -> None:
    # (14a): b_mu(v) * b_mu(-v) = mu, for every component and any mu.
    v = _v_grid()
    for mu in (1e-8, 1e-3, 1.0, 10.0):
        prod = b_mu(v, mu) * b_mu(-v, mu)
        np.testing.assert_allclose(prod, np.full_like(v, mu), rtol=1e-7, atol=1e-15)


def test_derivative_sums_to_one() -> None:
    # (14b): db_mu(v) + db_mu(-v) = 1.
    v = _v_grid()
    for mu in (1e-8, 1e-3, 1.0, 10.0):
        total = db_mu(v, mu) + db_mu(-v, mu)
        np.testing.assert_allclose(total, np.ones_like(v), rtol=1e-12, atol=0.0)


def test_derivative_bounds() -> None:
    # (14c): 0 < db_mu(v) <= 1.
    v = _v_grid()
    for mu in (1e-8, 1e-3, 1.0, 10.0):
        d = db_mu(v, mu)
        assert np.all(d > 0.0)
        assert np.all(d <= 1.0)


def test_b_mu_is_positive() -> None:
    # b_mu: R -> R_+, strictly positive everywhere.
    v = _v_grid()
    for mu in (1e-8, 1e-3, 1.0, 10.0):
        assert np.all(b_mu(v, mu) > 0.0)


def test_db_mu_matches_finite_difference() -> None:
    # db_mu is the analytic derivative of b_mu. Verify against central differences.
    v = _v_grid()
    mu = 0.7
    h = 1e-6
    fd = (b_mu(v + h, mu) - b_mu(v - h, mu)) / (2.0 * h)
    np.testing.assert_allclose(db_mu(v, mu), fd, rtol=1e-6, atol=1e-7)


def test_works_on_scalars() -> None:
    np.testing.assert_allclose(b_mu(np.array([0.0]), 1.0), [1.0])
    np.testing.assert_allclose(db_mu(np.array([0.0]), 1.0), [0.5])
