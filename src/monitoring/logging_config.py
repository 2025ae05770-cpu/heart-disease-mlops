"""Structured (JSON-lines) logging for the API: stdout + optional rotating file."""

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level=None, log_file=None):
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_file = log_file if log_file is not None else os.getenv("LOG_FILE", "")

    logger = logging.getLogger("heart_api")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    stream = logging.StreamHandler()
    stream.setFormatter(JsonFormatter())
    logger.addHandler(stream)

    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("file_logging_disabled", extra={"log_file": str(log_file), "error": str(exc)})
    return logger
