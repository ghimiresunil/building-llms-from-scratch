"""Logging utilities: the shared ``logger`` and the ``log_execution`` decorator."""

from src.decorators.logging_service import AppLogger, log_execution

app_logger = AppLogger()
logger = app_logger.get_logger()

__all__ = ["AppLogger", "app_logger", "log_execution", "logger"]
