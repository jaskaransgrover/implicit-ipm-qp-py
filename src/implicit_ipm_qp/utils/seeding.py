"""Deterministic seeding for reproducible runs and tests."""

from __future__ import annotations

import random

import numpy as np

DEFAULT_SEED: int = 0


def seed_everything(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Seed all relevant RNGs and return a NumPy Generator.

    Seeds Python's ``random`` module and NumPy's *legacy* global RNG (for any
    code that still uses ``np.random.*`` free functions), and returns a fresh
    ``np.random.Generator`` that callers should thread explicitly through their
    code. Preferring an explicit generator keeps randomness from hiding in
    global state.

    Args:
        seed: Non-negative integer seed.

    Returns:
        A seeded ``np.random.Generator`` (PCG64) for explicit use.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
