from __future__ import annotations

import json
import logging
from typing import Any

try:
    import structlog  # type: ignore
except Exception:  # pragma: no cover
    structlog = None


class _JsonFallbackLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def _emit(self, level: str, event: str, **kwargs: Any) -> None:
        payload = {'event': event, 'level': level, **kwargs}
        self._logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(payload, ensure_ascii=False))

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit('info', event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit('warning', event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit('error', event, **kwargs)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")
    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def get_logger(name: str):
    if structlog is not None:
        return structlog.get_logger(name)
    return _JsonFallbackLogger(name)
