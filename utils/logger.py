"""
Logging utilities for training and evaluation.

Supports both file-based and console logging with colorized output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "clip_lora_elr",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_color: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        use_color: Whether to use colorized console output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Avoid duplicate handlers

    # Formatters
    if use_color:
        try:
            import colorlog
            console_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        except ImportError:
            use_color = False

    if not use_color:
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerMixin:
    """Mixin that adds logging capability to any class."""

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = setup_logger(self.__class__.__name__)
        return self._logger
