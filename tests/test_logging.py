import logging

import pytest

from backend.core.logging import configure_logging, get_logger


def test_get_logger_returns_logger():
    logger = get_logger("ETRIS.test")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "ETRIS.test"


def test_configure_logging_accepts_valid_level():
    configure_logging("DEBUG")

    logger = get_logger("ETRIS.test")
    assert logger.isEnabledFor(logging.DEBUG)


def test_configure_logging_rejects_invalid_level():
    with pytest.raises(TypeError, match="Invalid logging level"):
        configure_logging("INVALID")


def test_configure_logging_creates_log_file(tmp_path):
    log_file = tmp_path / "test.log"

    configure_logging("INFO", log_file)

    logger = get_logger("ETRIS.file_test")
    logger.info("test log message")

    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_file.exists()
    assert "test log message" in log_file.read_text(encoding="utf-8")