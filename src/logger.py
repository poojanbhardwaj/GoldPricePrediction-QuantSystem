"""
logger.py — Centralized Logging Configuration
==============================================
Provides a unified logger for every module in the project.
Supports colored console output + rotating file logs.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello from my module")
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# ── Optional colorlog (falls back gracefully if not installed) ──
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# ── ANSI color codes for manual fallback ─────────────────────────
RESET = "\033[0m"
COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
}


class ColorFormatter(logging.Formatter):
    """Manual ANSI-color formatter (used when colorlog is absent)."""

    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, RESET)
        formatter = logging.Formatter(
            fmt=f"{color}{self.FORMAT}{RESET}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


def setup_logger(
    name: str = "GoldPrediction",
    level: str = "INFO",
    log_file: Optional[str] = "logs/gold_prediction.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create and configure a logger with:
      - Colored console output
      - Rotating file handler

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    level : str
        Logging level string: DEBUG | INFO | WARNING | ERROR | CRITICAL.
    log_file : str | None
        Path to the rotating log file. Pass None to disable file logging.
    max_bytes : int
        Max size of a single log file before rotation (default 10 MB).
    backup_count : int
        Number of backup log files to keep.

    Returns
    -------
    logging.Logger
        Fully configured logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)

    # Prevent duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(numeric_level)
    logger.propagate = False

    datefmt = "%Y-%m-%d %H:%M:%S"

    # ── Console Handler ───────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if HAS_COLORLOG:
        log_colors = {
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_magenta",
        }
        fmt = "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s"
        color_formatter = colorlog.ColoredFormatter(fmt, datefmt=datefmt, log_colors=log_colors)
        console_handler.setFormatter(color_formatter)
    else:
        console_handler.setFormatter(ColorFormatter())

    logger.addHandler(console_handler)

    # ── File Handler ──────────────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        plain_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt=datefmt,
        )
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "GoldPrediction") -> logging.Logger:
    """
    Convenience wrapper — returns a logger configured from config.yaml
    (or sensible defaults when config is unavailable).

    Parameters
    ----------
    name : str
        Module name, typically passed as ``__name__``.

    Returns
    -------
    logging.Logger
    """
    try:
        from src.config_loader import ConfigLoader
        cfg = ConfigLoader().get("logging", {})
        return setup_logger(
            name=name,
            level=cfg.get("level", "INFO"),
            log_file=cfg.get("log_file", "logs/gold_prediction.log"),
            max_bytes=cfg.get("max_bytes", 10 * 1024 * 1024),
            backup_count=cfg.get("backup_count", 5),
        )
    except Exception:
        # Fallback if config hasn't been initialized yet
        return setup_logger(name=name)


# ── Module-level default logger ───────────────────────────────────
logger = get_logger(__name__)


if __name__ == "__main__":
    log = get_logger("TestLogger")
    log.debug("This is a DEBUG message")
    log.info("Logger initialized successfully ✓")
    log.warning("This is a WARNING")
    log.error("This is an ERROR")
    log.critical("This is CRITICAL")
