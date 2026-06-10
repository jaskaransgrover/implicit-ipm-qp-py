"""Tests for the seeding helper."""

from __future__ import annotations

import numpy as np
import pytest

from implicit_ipm_qp.utils.seeding import seed_everything


def test_seed_everything_is_deterministic() -> None:
    rng_a = seed_everything(123)
    a = rng_a.standard_normal(5)

    rng_b = seed_everything(123)
    b = rng_b.standard_normal(5)

    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ() -> None:
    a = seed_everything(1).standard_normal(5)
    b = seed_everything(2).standard_normal(5)
    assert not np.array_equal(a, b)


def test_negative_seed_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
