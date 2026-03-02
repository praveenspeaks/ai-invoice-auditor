"""
Centralized logging utility for the AI Invoice Auditor pipeline.
Log level and output path are driven by rules.yaml logging section.
"""

import logging
import os
from pathlib import Path


def get_logger(name: str, log_file: str | None = None, log_level: str | None = None) -> logging.Logger:
    """
    Return a named logger configured with both file and console handlers.

    If called multiple times with the same name, returns the existing logger
    (handlers are not duplicated).

    Args:
        name:      Logger name — use __name__ in each module.
        log_file:  Override log file path. Falls back to rules.yaml, then default.
        log_level: Override level string. Falls back to rules.yaml, then INFO.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers have been added yet
    if logger.handlers:
        return logger

    # Resolve level
    resolved_level = _resolve_level(log_level)
    logger.setLevel(resolved_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(resolved_level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    resolved_path = _resolve_log_path(log_file)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def _resolve_level(override: str | None) -> int:
    """Resolve log level: override → rules.yaml → INFO."""
    if override:
        return logging.getLevelName(override.upper())
    try:
        from core.config import get_rules
        return logging.getLevelName(get_rules().logging.log_level.upper())
    except Exception:
        return logging.INFO


def _resolve_log_path(override: str | None) -> Path:
    """Resolve log file path: override → rules.yaml → ./logs/invoice_auditor.log."""
    if override:
        return Path(override)
    try:
        from core.config import get_rules
        return Path(get_rules().logging.log_file)
    except Exception:
        return Path("./logs/invoice_auditor.log")
