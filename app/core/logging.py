import json
import logging
import logging.config
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured application logs."""
    def format(self, record: logging.LogRecord) -> str:
        from asgi_correlation_id import correlation_id

        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "request_id": correlation_id.get() or "system",
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    """Configure structured logging for the application."""
    formatter_name = "json" if log_format.lower() == "json" else "console"

    logging_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "app.core.logging.JSONFormatter",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": formatter_name,
            },
        },
        "root": {
            "level": log_level.upper(),
            "handlers": ["default"],
        },
        "loggers": {
            "uvicorn": {
                "level": log_level.upper(),
                "handlers": ["default"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": log_level.upper(),
                "handlers": ["default"],
                "propagate": False,
            },
            "app": {
                "level": log_level.upper(),
                "handlers": ["default"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)
