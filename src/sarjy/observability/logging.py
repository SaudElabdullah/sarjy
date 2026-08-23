from __future__ import annotations

import logging
import sys

import structlog

# httpx logs every request line at INFO, query string included — which for the
# OpenWeatherMap fallback means `appid=<key>` in the logs of any deployment
# running at INFO. Nothing we log ourselves carries a URL, so silencing these
# two below WARNING costs no diagnostics we actually rely on.
_NOISY_HTTP_LOGGERS = ("httpx", "httpcore")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    for name in _NOISY_HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            # Render `log.exception(...)` tracebacks into an "exception" field;
            # without this the JSON renderer emits `"exc_info": true` and drops
            # the traceback entirely.
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
