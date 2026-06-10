"""Structured logging using ``structlog``.

Every module should call :func:`get_logger` with its ``__name__`` and use the
returned bound logger. The application entry point calls :func:`configure_logging`
exactly once to choose between human-readable (development) and JSON
(production / cloud / CI) output.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from structlog.types import Processor


def configure_logging(
    level: str = "INFO",
    *,
    pretty: bool = True,
) -> None:
    """Configure the global ``structlog`` logger.

    Call exactly once at process start.

    Args:
        level: Standard logging level name (``"DEBUG"``, ``"INFO"``, etc.).
        pretty: If True, render coloured human-readable output to stderr.
            If False, emit single-line JSON suitable for log aggregators.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level.upper(),
    )

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if pretty:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()],
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a bound logger for ``name`` (typically ``__name__``).

    The return type is ``Any`` because ``structlog`` produces a different
    bound-logger class depending on configuration; pinning it more tightly
    would require runtime gymnastics.
    """
    return structlog.get_logger(name)
