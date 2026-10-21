"""A singleton application logger configured by Dynaconf, plus an execution-logging decorator."""

from __future__ import annotations

import functools
import logging
import reprlib
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

from config.settings import StageConfig, load_logging

T = TypeVar("T", bound=Callable[..., Any])

# Arguments and return values are logged at DEBUG; keep a tensor or a model to one short line.
_short = reprlib.Repr()
_short.maxstring = 120
_short.maxother = 120
_short.maxlist = _short.maxtuple = _short.maxdict = 8


class AppLogger:
    """One process-wide logger whose level and format come from configs/logging.toml."""

    _instance: AppLogger | None = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, logger_name: str | None = None) -> AppLogger:
        """Create the single instance on first use, safely across threads."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, logger_name: str | None = None) -> None:
        """Configure the logger once; later calls return the same, configured instance."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.settings: StageConfig = load_logging()
            self.logger_name: str = logger_name or self.settings.app_name
            self.logger = logging.getLogger(self.logger_name)
            self._configure_logger()
            self._initialized = True

    def _configure_logger(self) -> None:
        """Attach one console handler with the configured level and format."""
        level = getattr(logging, self.settings.log_level)
        self.logger.handlers.clear()
        self.logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(self.settings.log_format, self.settings.date_format)
        )
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def get_logger(self) -> logging.Logger:
        """The configured ``logging.Logger``."""
        return self.logger

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.logger.critical(message)


def log_execution(func: T) -> T:
    """Log a call's entry, duration, failure or result, and exit."""
    name = func.__qualname__  # "Trainer.step" for methods, "main" for functions

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = AppLogger().get_logger()
        logger.info("Entering: %s", name)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Args: %s, kwargs: %s", _short.repr(args), _short.repr(kwargs))
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            duration = time.perf_counter() - start
            logger.error("%s failed in %.4fs: %s", name, duration, e)
            logger.debug("Exception: %s: %s", type(e).__name__, e)
            raise
        else:
            duration = time.perf_counter() - start
            logger.info("%s completed in %.4fs", name, duration)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Return: %s", _short.repr(result))
            return result
        finally:
            logger.debug("Exiting: %s", name)

    return cast(T, wrapper)
