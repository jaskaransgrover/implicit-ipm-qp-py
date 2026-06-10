"""Smoke test: the package imports and exposes a version."""

from __future__ import annotations

import implicit_ipm_qp


def test_package_imports() -> None:
    assert isinstance(implicit_ipm_qp.__version__, str)
    assert implicit_ipm_qp.__version__ != ""
