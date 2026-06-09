"""Smoke test: the package imports and exposes a version."""

from __future__ import annotations

import i2pd


def test_package_imports() -> None:
    assert isinstance(i2pd.__version__, str)
    assert i2pd.__version__ != ""
