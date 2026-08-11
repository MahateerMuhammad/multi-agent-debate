import json
import logging

from app.core.logging import JSONFormatter, get_logger, setup_logging


def test_json_formatter() -> None:
    """Test custom JSON log formatter output."""
    formatter = JSONFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test message"
    assert "timestamp" in parsed


def test_setup_logging() -> None:
    """Test logger initialization."""
    setup_logging(log_level="DEBUG", log_format="console")
    logger = get_logger("app.test")
    assert logger.name == "app.test"
