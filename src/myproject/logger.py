"""Simple logging helper writing to file controlled by env vars."""
from __future__ import annotations

import logging
import os
from logging import Logger
from typing import Optional


def _parse_level(level_str: Optional[str]) -> int:
    if level_str is None:
        return logging.INFO
    try:
        lvl = int(level_str)
    except Exception:
        return logging.INFO
    if lvl <= 0:
        return logging.NOTSET
    if lvl == 1:
        return logging.INFO
    return logging.DEBUG


def get_logger(name: str = "myproject") -> Logger:
    log_file = os.environ.get("LOG_FILE", "myproject.log")
    level_env = os.environ.get("LOG_LEVEL")
    level = _parse_level(level_env)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
