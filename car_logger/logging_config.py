"""structlog setup: JSON logs to stdout so journalctl/systemd capture them."""

import logging
import sys

import structlog


def configure_logging(level="INFO"):
    # Resolve once, use for BOTH stdlib and structlog filtering — otherwise
    # LOG_LEVEL=DEBUG would change stdlib but structlog would stay at INFO.
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout,
                        level=resolved)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(resolved),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name):
    return structlog.get_logger(name)
