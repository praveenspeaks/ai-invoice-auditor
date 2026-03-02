"""
Unit tests for core/logger.py — centralized logging setup.
Target coverage: core/logger.py ≥ 75%
"""

import logging
import os
from pathlib import Path

import pytest

from core.logger import get_logger, _resolve_level, _resolve_log_path


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_loggers():
    """Remove any handlers added during tests to avoid cross-test pollution."""
    yield
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("test."):
            logger = logging.getLogger(name)
            logger.handlers.clear()


# ── get_logger ─────────────────────────────────────────────────────────────────

class TestGetLogger:
    def test_returns_logger_instance(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = get_logger("test.basic", log_file=log_file, log_level="DEBUG")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = get_logger("test.named", log_file=log_file, log_level="INFO")
        assert logger.name == "test.named"

    def test_has_two_handlers(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = get_logger("test.handlers", log_file=log_file, log_level="INFO")
        assert len(logger.handlers) == 2

    def test_has_stream_handler(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = get_logger("test.stream", log_file=log_file, log_level="INFO")
        types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in types

    def test_has_file_handler(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        logger = get_logger("test.file", log_file=log_file, log_level="INFO")
        types = [type(h) for h in logger.handlers]
        assert logging.FileHandler in types

    def test_log_file_created(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "app.log")
        get_logger("test.create", log_file=log_file, log_level="INFO")
        assert Path(log_file).exists()

    def test_log_file_parent_dirs_created(self, tmp_path):
        log_file = str(tmp_path / "a" / "b" / "c" / "app.log")
        get_logger("test.mkdirs", log_file=log_file, log_level="INFO")
        assert Path(log_file).parent.exists()

    def test_second_call_same_name_no_duplicate_handlers(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        get_logger("test.dedup", log_file=log_file, log_level="INFO")
        logger2 = get_logger("test.dedup", log_file=log_file, log_level="INFO")
        assert len(logger2.handlers) == 2  # still only 2, not 4

    def test_writes_to_log_file(self, tmp_path):
        log_file = str(tmp_path / "write.log")
        logger = get_logger("test.write", log_file=log_file, log_level="INFO")
        logger.info("hello from test")
        content = Path(log_file).read_text(encoding="utf-8")
        assert "hello from test" in content

    def test_debug_level_set(self, tmp_path):
        log_file = str(tmp_path / "debug.log")
        logger = get_logger("test.debug_level", log_file=log_file, log_level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_warning_level_set(self, tmp_path):
        log_file = str(tmp_path / "warn.log")
        logger = get_logger("test.warn_level", log_file=log_file, log_level="WARNING")
        assert logger.level == logging.WARNING


# ── _resolve_level ─────────────────────────────────────────────────────────────

class TestResolveLevel:
    def test_explicit_info(self):
        assert _resolve_level("INFO") == logging.INFO

    def test_explicit_debug(self):
        assert _resolve_level("DEBUG") == logging.DEBUG

    def test_explicit_warning(self):
        assert _resolve_level("WARNING") == logging.WARNING

    def test_explicit_error(self):
        assert _resolve_level("ERROR") == logging.ERROR

    def test_lowercase_accepted(self):
        assert _resolve_level("info") == logging.INFO

    def test_none_falls_back(self):
        # Falls back to rules.yaml or INFO — just verify it returns a valid int level
        level = _resolve_level(None)
        assert isinstance(level, int)
        assert level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)


# ── _resolve_log_path ──────────────────────────────────────────────────────────

class TestResolveLogPath:
    def test_explicit_path_returned(self):
        result = _resolve_log_path("/tmp/custom.log")
        assert result == Path("/tmp/custom.log")

    def test_none_returns_path_object(self):
        result = _resolve_log_path(None)
        assert isinstance(result, Path)

    def test_none_uses_rules_or_default(self):
        result = _resolve_log_path(None)
        assert result.suffix == ".log"
