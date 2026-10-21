"""Dynaconf settings: every shipped file loads, and bad values fail early and clearly."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from config.settings import (
    PRESETS,
    STAGES,
    ConfigError,
    apply_preset,
    convert,
    load_stage,
    schema,
)
from src.post_training.cli import parse_stage


@pytest.fixture(autouse=True)
def _no_blfs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BLFS_* variable in the developer's shell must not change what these tests see."""
    for key in list(os.environ):
        if key.startswith("BLFS_"):
            monkeypatch.delenv(key)


@pytest.mark.parametrize("env", ["default", "smoke", "test"])
@pytest.mark.parametrize("stage", STAGES)
def test_every_shipped_stage_loads(env: str, stage: str) -> None:
    cfg = load_stage(stage, env=env)
    assert set(cfg.to_dict()) == set(schema(stage))
    assert not any(isinstance(v, str) and v.startswith("/ephemeral") for v in cfg.to_dict().values())


def test_smoke_shrinks_the_model_and_never_overwrites_real_checkpoints() -> None:
    for stage in STAGES:
        cfg = load_stage(stage, env="smoke")
        assert cfg.out_ckpt.startswith("models/smoke/") and cfg.device == "cpu" and cfg.amp_dtype is None
        assert cfg.n_embed == 128


def test_test_variant_is_tiny() -> None:
    cfg = load_stage("dpo", env="test")
    assert (cfg.vocab_size, cfg.context_length, cfg.max_len) == (256, 64, 64)


def _file(tmp_path: Path, text: str, name: str = "sft.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_unknown_keys_fail_with_a_suggestion(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown key 'batchsize' for the sft stage \\(did you mean 'batch_size'\\?\\)"):
        load_stage("sft", config_path=_file(tmp_path, "[default]\nbatchsize = 4\n"))


def test_wrong_values_name_the_setting_and_the_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"loss_type must be one of \["):
        load_stage("dpo", overrides={"loss_type": "dop"})
    with pytest.raises(ConfigError, match="sft.toml: lr must be a number"):
        load_stage("sft", config_path=_file(tmp_path, '[default]\nlr = "fast"\n'))
    with pytest.raises(ConfigError, match="divisible"):
        load_stage("sft", overrides={"n_embed": 30, "n_head": 4})
    with pytest.raises(ConfigError, match="beta must be positive"):
        load_stage("dpo", overrides={"beta": "-1"})
    with pytest.raises(ConfigError, match="max_len"):
        load_stage("reward", overrides={"max_len": 4096})


def test_variant_values_name_their_table(tmp_path: Path) -> None:
    path = _file(tmp_path, '[default]\nlr = 1e-5\n\n[default.smoke]\nlr = "fast"\n')
    with pytest.raises(ConfigError, match=r"sft.toml \[default.smoke\]: lr must be a number"):
        load_stage("sft", config_path=path, env="smoke")


def test_top_level_variants_point_to_the_right_layout(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"put it under \[default.smoke\]"):
        load_stage("sft", config_path=_file(tmp_path, "[default]\nlr = 1e-5\n\n[smoke]\nlr = 2e-5\n"))
    with pytest.raises(ConfigError, match=r"under the \[default\] table"):
        load_stage("sft", config_path=_file(tmp_path, "lr = 1e-5\n"))


@pytest.mark.parametrize(
    ("key", "value", "kind", "expected"),
    [("batch_size", "4", int, 4), ("batch_size", 2000.0, int, 2000), ("train_steps", "50_000", int, 50_000),
     ("lr", "1e-5", float, 1e-5), ("lr", 3, float, 3.0), ("compile", "true", bool, True),
     ("compile", "off", bool, False), ("n_kv_head", "none", int, None), ("n_kv_head", "8", int, 8),
     ("clip_high", None, float, None), ("adv_norm", "none", str, "none")],
)
def test_convert_accepts_sensible_spellings(key: str, value: object, kind: type, expected: object) -> None:
    assert convert(key, value, kind) == expected


@pytest.mark.parametrize(("value", "kind"), [(True, int), (1, bool), ("1.5", int), ("abc", float), (3, str)])
def test_convert_rejects_the_rest(value: object, kind: type) -> None:
    with pytest.raises(ConfigError):
        convert("x", value, kind)


def test_blfs_variables_override_the_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLFS_LR", "2e-5")
    monkeypatch.setenv("BLFS_COMPILE", "true")
    monkeypatch.setenv("BLFS_MAX_LEN", "64")  # a DPO/reward setting: SFT simply skips it
    cfg = load_stage("sft", env="smoke")  # the variable also beats the [default.smoke] table
    assert (cfg.lr, cfg.compile) == (2e-5, True)
    assert load_stage("sft", overrides={"lr": "3e-5"}).lr == 3e-5  # the command line beats both


def test_bad_variables_name_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLFS_LR", "fast")
    with pytest.raises(ConfigError, match="environment variable BLFS_LR: lr must be a number"):
        load_stage("sft")


def test_blfs_env_picks_the_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLFS_ENV", "smoke")
    assert load_stage("sft").n_embed == 128
    assert load_stage("sft", env="default").n_embed == 1024  # an explicit --env wins


def test_unknown_variants_fail_with_a_suggestion() -> None:
    with pytest.raises(ConfigError, match="unknown environment 'smok' \\(did you mean 'smoke'\\?\\)"):
        load_stage("sft", env="smok")


def test_local_files_are_merged_in(tmp_path: Path) -> None:
    stage = _file(tmp_path, '[default]\nlr = 1e-5\nepochs = 3\n\n[default.smoke]\nmax_steps = 10\n')
    _file(tmp_path, "[default]\nlr = 7e-6\n\n[default.smoke]\nbatch_size = 2\n", name="sft.local.toml")
    assert (load_stage("sft", config_path=stage).lr, load_stage("sft", config_path=stage).epochs) == (7e-6, 3)
    smoke = load_stage("sft", config_path=stage, env="smoke")
    assert (smoke.max_steps, smoke.batch_size, smoke.n_embed) == (10, 2, 128)  # sub-tables merge, not replace


def test_environment_variables_in_values_are_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIG_DISK", "/mnt/big")
    cfg = load_stage("sft", config_path=_file(tmp_path, '[default]\nout_ckpt = "$BIG_DISK/models/sft.pt"\n'))
    assert cfg.out_ckpt == "/mnt/big/models/sft.pt"


def test_missing_files_are_an_error() -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_stage("sft", config_path="configs/does_not_exist.toml")


def test_cli_values_are_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["train_sft.py", "--env", "smoke", "--lr", "2e-5", "--compile", "true", "--amp_dtype", "none",
            "--max_steps", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    cfg, _ = parse_stage("sft")
    assert (cfg.lr, cfg.compile, cfg.amp_dtype, cfg.max_steps, cfg.n_embed) == (2e-5, True, None, 3, 128)
    monkeypatch.setattr(sys, "argv", ["train_dpo.py", "--loss_type", "dop"])
    with pytest.raises(SystemExit):
        parse_stage("dpo")


def test_print_config_round_trips(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                  tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["train_grpo.py", "--env", "smoke", "--print-config"])
    with pytest.raises(SystemExit):
        parse_stage("grpo")
    printed = capsys.readouterr().out
    assert printed.startswith("[default]") and 'clip_high = "none"' in printed
    assert load_stage("grpo", config_path=_file(tmp_path, printed, "grpo.toml")) == load_stage("grpo", env="smoke")


def test_presets_come_from_their_sub_tables() -> None:
    assert {"tiny", "student", "small", "13m", "77m", "3b"} <= set(PRESETS)
    assert apply_preset({"n_embed": 1, "keep": True}, "13m") == {**PRESETS["13m"], "keep": True}
    assert "train_path" not in PRESETS["tiny"] or PRESETS["tiny"]["train_path"].startswith("data/tiny")
