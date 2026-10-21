"""The Dynaconf logging settings, the AppLogger singleton and the log_execution decorator."""

from __future__ import annotations

import logging
import os
import threading

import pytest

from config.settings import ConfigError, load_logging
from src.decorators import AppLogger, log_execution, logger


@pytest.fixture(autouse=True)
def _no_blfs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BLFS_* variable in the developer's shell must not change these tests."""
    for key in list(os.environ):
        if key.startswith("BLFS_"):
            monkeypatch.delenv(key)


class _Records(logging.Handler):
    """Collect formatted records so a test can read what was logged."""

    def __init__(self) -> None:
        """Start with no records."""
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record as ``LEVEL message``."""
        self.lines.append(f"{record.levelname} {record.getMessage()}")


@pytest.fixture
def records() -> _Records:
    """Attach a collector at DEBUG to the shared logger for one test."""
    handler = _Records()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def test_logging_settings_load_from_toml() -> None:
    cfg = load_logging()
    assert (cfg.app_name, cfg.log_level) == ("building-llms-from-scratch", "INFO")
    assert load_logging(env="test").log_level == "DEBUG"
    assert load_logging(env="smoke").log_level == "INFO"  # no smoke table: use [default]


def test_log_level_from_env_var_any_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLFS_LOG_LEVEL", "warning")
    assert load_logging().log_level == "WARNING"


def test_bad_log_level_names_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLFS_LOG_LEVEL", "loud")
    with pytest.raises(ConfigError, match="environment variable BLFS_LOG_LEVEL"):
        load_logging()


def test_app_logger_is_one_instance_across_threads() -> None:
    seen: list[AppLogger] = []
    threads = [threading.Thread(target=lambda: seen.append(AppLogger())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(x) for x in seen}) == 1 and seen[0] is AppLogger()
    assert AppLogger().get_logger() is logger
    assert len(logger.handlers) >= 1 and logger.propagate is False


def test_log_execution_names_methods_and_returns_the_result(records: _Records) -> None:
    class Trainer:
        @log_execution
        def step(self, x: int) -> int:
            return x * 2

    assert Trainer().step(21) == 42
    text = "\n".join(records.lines)
    assert "Entering: " in text and "Trainer.step completed in" in text
    assert "Return: 42" in text and "Exiting: " in text


def test_log_execution_logs_and_reraises_failures(records: _Records) -> None:
    @log_execution
    def load_batch() -> None:
        raise ValueError("bad batch")

    with pytest.raises(ValueError, match="bad batch"):
        load_batch()
    assert any(
        line.startswith("ERROR") and "failed in" in line and "bad batch" in line
        for line in records.lines
    )


def test_debug_arguments_are_shortened(records: _Records) -> None:
    @log_execution
    def consume(values: list[int]) -> int:
        return len(values)

    consume(list(range(10_000)))
    args_line = next(line for line in records.lines if line.startswith("DEBUG Args:"))
    assert len(args_line) < 300 and "..." in args_line


def test_log_execution_keeps_the_function_identity() -> None:
    @log_execution
    def documented() -> None:
        """A docstring that must survive the decorator."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring that must survive the decorator."
