"""Tests for the structlog setup."""

from __future__ import annotations

import structlog

from i2pd.utils.logging import get_logger


def test_get_logger_emits_structured_event() -> None:
    log = get_logger("test")
    with structlog.testing.capture_logs() as captured:
        log.info("solver_started", iteration=0, mu=1.0)

    assert len(captured) == 1
    event = captured[0]
    assert event["event"] == "solver_started"
    assert event["iteration"] == 0
    assert event["mu"] == 1.0
    assert event["log_level"] == "info"


def test_get_logger_returns_usable_logger() -> None:
    log = get_logger(__name__)
    # Accepts structured kwargs without raising.
    log.debug("debug_event", value=42)
