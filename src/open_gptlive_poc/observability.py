"""Application logging configuration using Loguru."""

from __future__ import annotations

import logging
import os
import sys

from loguru import logger


_configured = False


class _InterceptHandler(logging.Handler):
    """Forward framework and dependency logs into the Loguru sink."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.bind(stdlib_logger=record.name).opt(exception=record.exc_info).log(level, record.getMessage())


def configure_logging() -> None:
    """Set one JSON stderr sink and route standard-library logs through it."""
    global _configured
    if _configured:
        return

    logger.remove()
    level = os.getenv("GPTLIVE_LOG_LEVEL", "INFO").upper()
    sink_options = {
        "level": level,
        "serialize": True,
        "backtrace": False,
        "diagnose": False,
        "enqueue": True,
    }
    logger.add(sys.stderr, **sink_options)
    log_file = os.getenv("GPTLIVE_LOG_FILE")
    if log_file:
        logger.add(log_file, rotation="10 MB", retention="7 days", **sink_options)
    handler = _InterceptHandler()
    logging.basicConfig(handlers=[handler], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        framework_logger = logging.getLogger(name)
        framework_logger.handlers = [handler]
        framework_logger.propagate = False
        framework_logger.setLevel(0)
    _configured = True
