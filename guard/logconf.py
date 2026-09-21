"""Shared terminal logging setup for the guard CLI and API service."""

import logging
import os
import sys


def setup_logging() -> None:
    level_name = os.environ.get("GUARD_LOG_LEVEL", "INFO").upper()
    guard_logger = logging.getLogger("guard")
    guard_logger.setLevel(getattr(logging, level_name, logging.INFO))
    if not guard_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [guard] %(message)s", "%H:%M:%S")
        )
        guard_logger.addHandler(handler)
    guard_logger.propagate = False
