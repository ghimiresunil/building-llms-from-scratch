"""Every setting in the repo, loaded from the TOML files in configs/ with Dynaconf."""

from __future__ import annotations

import difflib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dynaconf import Dynaconf, ValidationError, Validator, get_history

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
STAGES = ("pretrain", "sft", "reward", "dpo", "ppo", "grpo")

ENVVAR_PREFIX = "BLFS"
ENV_VARIABLE = "BLFS_ENV"
DEFAULT_ENV = "default"
_NO_SWITCHER = "_BLFS_INTERNAL_ENV_SWITCHER"  # we pick the variant ourselves, so Dynaconf's switcher stays idle

# Settings whose shipped value is "none": their real type, since TOML has no null.
NULLABLE: dict[str, type] = {
    "amp_dtype": str,
    "n_kv_head": int,
    "kv_latent_dim": int,
    "sliding_window": int,
    "clip_high": float,
}

CHOICES: dict[str, list[str]] = {
    "arch": ["classic", "modern"],
    "attention": ["gqa", "mla"],
    "amp_dtype": ["bf16", "fp16"],
    "optimizer": ["adamw", "muon"],
    "lr_schedule": ["cosine", "wsd", "linear"],
    "loss_type": ["dpo", "ipo", "simpo", "orpo", "kto"],
    "reward_source": ["verifier", "rm"],
    "adv_norm": ["std", "none"],
    "loss_agg": ["token-mean", "seq-mean-token-mean", "seq-mean-token-sum-norm"],
    "ratio_level": ["token", "sequence"],
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
}

_POSITIVE = (
    "vocab_size",
    "context_length",
    "n_embed",
    "n_head",
    "n_blocks",
    "batch_size",
    "grad_accum",
    "n_kv_head",
    "group_size",
)
_TRUE = {"true", "1", "yes", "y", "on"}
_FALSE = {"false", "0", "no", "n", "off"}
_NONE = {"none", "null"}


class ConfigError(ValueError):
    """A setting is unknown, has the wrong type, or is out of range."""


class StageConfig(SimpleNamespace):
    """A stage's resolved settings, read as attributes (``cfg.lr``)."""

    def to_dict(self) -> dict[str, Any]:
        """The settings as a plain dict."""
        return dict(vars(self))

    def replace(self, **changes: Any) -> StageConfig:
        """A copy with some settings changed (not re-validated)."""
        return StageConfig(**{**vars(self), **changes})


# --- reading files --------------------------------------------------------------------------


def display_path(path: str | os.PathLike[str]) -> str:
    """A short, forward-slash path for messages."""
    p = Path(path)
    try:
        p = p.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    return p.as_posix()


def find_file(path: str | os.PathLike[str]) -> Path:
    """Resolve a config path from the working dir first, then from the repo root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    from_root = REPO_ROOT / p
    return from_root.resolve() if from_root.exists() else p.resolve()


def local_variant(path: Path) -> Path:
    """``configs/sft.toml`` -> ``configs/sft.local.toml``, which Dynaconf loads automatically."""
    return path.with_name(f"{path.stem}.local{path.suffix}")


def _read_default(path: Path) -> dict[str, Any]:
    """The ``[default]`` table of one file, after checking the file's layout."""
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(
            f"{display_path(path)}: could not parse the file ({e})"
        ) from None
    for key, value in data.items():
        if key in ("default", "global") and isinstance(value, dict):
            continue
        hint = f"[default.{key}]" if isinstance(value, dict) else "the [default] table"
        raise ConfigError(
            f"{display_path(path)}: '{key}' sits at the top level; put it under {hint}"
        )
    return dict(data.get("default", {}))


def _settings(files: list[Path], *, env_vars: bool) -> Dynaconf:
    """A Dynaconf object over ``files`` (plus their .local.toml files and, optionally, BLFS_* vars)."""
    for f in files:
        for candidate in (f, local_variant(f)):
            if candidate.exists():
                _read_default(candidate)
    options: dict[str, Any] = dict(
        settings_files=[str(f) for f in files],
        environments=True,
        env=DEFAULT_ENV,
        env_switcher=_NO_SWITCHER,
        envvar_prefix=ENVVAR_PREFIX,
        merge_enabled=True,
        root_path=str(REPO_ROOT),
    )
    if env_vars:
        options.update(load_dotenv=True, dotenv_path=str(REPO_ROOT / ".env"))
    else:
        options.update(loaders=[])
    return Dynaconf(**options)


def _split(
    settings: Dynaconf,
) -> tuple[dict[str, tuple[Any, str]], dict[str, dict[str, tuple[Any, str]]]]:
    """Top-level ``{key: (value, source)}`` and every ``[default.NAME]`` variant, with sources."""
    top: dict[str, tuple[Any, str]] = {}
    variants: dict[str, dict[str, tuple[Any, str]]] = {}
    for upper, value in settings.as_dict().items():
        history = [
            h
            for h in get_history(settings, key=upper)
            if h.get("loader") != "set_method"
        ]
        if not history or upper.startswith("_"):
            continue
        key = upper.lower()
        if isinstance(value, dict):
            table: dict[str, tuple[Any, str]] = {}
            for sub, sub_value in value.items():
                owners = [
                    h
                    for h in history
                    if isinstance(h.get("value"), dict) and sub in h["value"]
                ]
                where = display_path(owners[-1]["identifier"]) if owners else "settings"
                table[str(sub).lower()] = (sub_value, f"{where} [default.{key}]")
            variants[key] = table
            continue
        last = history[-1]
        if str(last.get("loader", "")).startswith("env"):
            top[key] = (value, f"environment variable {ENVVAR_PREFIX}_{upper}")
        else:
            top[key] = (value, display_path(last["identifier"]))
    return top, variants


def resolve_env(env: str | None) -> str:
    """The variant to use: an explicit value, else ``$BLFS_ENV``, else ``default``."""
    return (env or os.environ.get(ENV_VARIABLE) or DEFAULT_ENV).strip().lower()


def _pick_variant(env: str, variants: dict[str, Any], files: list[Path]) -> None:
    """Fail clearly when the requested ``[default.NAME]`` table doesn't exist."""
    if env == DEFAULT_ENV or env in variants:
        return
    known = sorted({DEFAULT_ENV, *variants})
    hint = difflib.get_close_matches(env, known, n=1)
    suggestion = f" (did you mean '{hint[0]}'?)" if hint else ""
    raise ConfigError(
        f"unknown environment '{env}'{suggestion}; "
        f"{', '.join(display_path(f) for f in files)} define: {', '.join(known)}"
    )


# --- types and checks -----------------------------------------------------------------------


def schema(stage: str) -> dict[str, tuple[Any, type]]:
    """``{setting: (shipped default, type)}`` for a stage, read from the shipped TOML files."""
    if stage not in STAGES:
        raise ConfigError(f"unknown stage '{stage}'; choose one of {', '.join(STAGES)}")
    shipped = {
        **_read_default(CONFIG_DIR / "base.toml"),
        **_read_default(CONFIG_DIR / f"{stage}.toml"),
    }
    out: dict[str, tuple[Any, type]] = {}
    for key, value in shipped.items():
        if isinstance(value, dict) or key.startswith("_"):
            continue
        kind = NULLABLE.get(key, type(value))
        out[key] = (None if key in NULLABLE and value == "none" else value, kind)
    return out


def type_label(key: str, kind: type) -> str:
    """A short name for a setting's type, for help texts and errors."""
    if key in CHOICES:
        return (
            "{" + ",".join(CHOICES[key] + (["none"] if key in NULLABLE else [])) + "}"
        )
    return kind.__name__ + (" | none" if key in NULLABLE else "")


def convert(key: str, value: Any, kind: type) -> Any:
    """Turn a value from a file, a variable or the command line into ``kind``, or raise."""
    if key in NULLABLE and (
        value is None or (isinstance(value, str) and value.strip().lower() in _NONE)
    ):
        return None
    if isinstance(value, str) and kind is not str:
        text = value.strip()
        if kind is bool and text.lower() in _TRUE | _FALSE:
            return text.lower() in _TRUE
        try:
            value = tomllib.loads(f"v = {text}")["v"]
        except tomllib.TOMLDecodeError:
            pass
    if kind is bool and isinstance(value, bool):
        return value
    if kind is int and not isinstance(value, bool):
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    if kind is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if kind is str and isinstance(value, str):
        return os.path.expandvars(value)
    names = {
        bool: "true or false",
        int: "an integer",
        float: "a number",
        str: "a string",
    }
    raise ConfigError(f"{key} must be {names.get(kind, kind.__name__)}, got {value!r}")


def _validators(keys: set[str]) -> list[Validator]:
    """Dynaconf validators for the choices and ranges of the given settings."""
    out = [
        Validator(
            k,
            is_in=v + ([None] if k in NULLABLE else []),
            messages={
                "operations": "{name} must be one of " + str(v) + ", got {value!r}"
            },
        )
        for k, v in CHOICES.items()
        if k in keys
    ]
    positive = "{name} must be positive, got {value}"
    out += [
        Validator(
            k, gt=0, messages={"operations": positive}, when=Validator(k, ne=None)
        )
        for k in _POSITIVE
        if k in keys
    ]
    out += [
        Validator(k, gte=0, messages={"operations": "{name} must be >= 0, got {value}"})
        for k in ("n_experts", "lora_rank", "min_lr")
        if k in keys
    ]
    out += [
        Validator(
            k, gt=0, messages={"operations": "{name} must be positive, got {value}"}
        )
        for k in ("beta", "clip")
        if k in keys
    ]
    if "clip_high" in keys:
        out.append(
            Validator(
                "clip_high",
                gt=0,
                when=Validator("clip_high", ne=None),
                messages={"operations": "clip_high must be positive, got {value}"},
            )
        )
    if "label_smoothing" in keys:
        out.append(
            Validator(
                "label_smoothing",
                gte=0.0,
                lt=0.5,
                messages={
                    "operations": "label_smoothing must be in [0, 0.5), got {value}"
                },
            )
        )
    if "group_size" in keys:
        out.append(
            Validator(
                "group_size",
                gte=2,
                messages={
                    "operations": "group_size must be at least 2 (the group is the baseline)"
                },
            )
        )
    return out


def _cross_checks(v: dict[str, Any]) -> list[str]:
    """Rules that involve two settings at once."""
    problems = []
    if v["n_embed"] % v["n_head"]:
        problems.append(
            f"n_embed ({v['n_embed']}) must be divisible by n_head ({v['n_head']})"
        )
    if v.get("n_kv_head") and v["n_head"] % v["n_kv_head"]:
        problems.append(
            f"n_head ({v['n_head']}) must be divisible by n_kv_head ({v['n_kv_head']})"
        )
    if v.get("n_experts") and not 1 <= v["moe_top_k"] <= v["n_experts"]:
        problems.append("moe_top_k must be between 1 and n_experts")
    if "max_len" in v and not 16 <= v["max_len"] <= v["context_length"]:
        problems.append(
            f"max_len ({v['max_len']}) must be between 16 and context_length ({v['context_length']})"
        )
    if "min_lr" in v and "lr" in v and v["min_lr"] > v["lr"]:
        problems.append("need min_lr <= lr")
    if (
        "warmup_steps" in v
        and "train_steps" in v
        and v["warmup_steps"] >= v["train_steps"]
    ):
        problems.append("warmup_steps must be smaller than train_steps")
    return problems


# --- the public entry points ----------------------------------------------------------------


def _merge(
    top: dict[str, tuple[Any, str]],
    variants: dict[str, dict[str, tuple[Any, str]]],
    chosen: str,
    overrides: dict[str, Any] | None,
    types: dict[str, type],
    owner: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Layer files < [default.ENV] < BLFS_* < overrides, then convert every value."""
    merged = {k: kv for k, kv in top.items() if not kv[1].startswith("environment")}
    merged.update(variants.get(chosen, {}) if chosen != DEFAULT_ENV else {})
    merged.update({k: kv for k, kv in top.items() if kv[1].startswith("environment")})
    merged.update(
        {k: (v, "command line") for k, v in (overrides or {}).items() if v is not None}
    )
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key, (value, source) in merged.items():
        if key not in types:
            if source.startswith("environment"):
                continue  # BLFS_* variables are shared; this one belongs elsewhere
            hint = difflib.get_close_matches(key, types, n=1)
            suggestion = f" (did you mean '{hint[0]}'?)" if hint else ""
            raise ConfigError(f"{source}: unknown key '{key}' for {owner}{suggestion}")
        try:
            values[key] = convert(key, value, types[key])
        except ConfigError as e:
            raise ConfigError(f"{source}: {e}") from None
        sources[key] = source
    return values, sources


def _validate(values: dict[str, Any], sources: dict[str, str]) -> None:
    """Run the Dynaconf validators for these settings and raise the first failure."""
    checker = Dynaconf(environments=False, envvar_prefix="_BLFS_NO_ENV", loaders=[])
    checker.update(values)
    checker.validators.register(*_validators(set(values)))
    try:
        checker.validators.validate_all()
    except ValidationError as e:
        first, message = e.details[0]
        source = sources.get(first.names[0], "settings")
        raise ConfigError(f"{source}: {message}") from None


def load_stage(
    stage: str,
    *,
    config_path: str | os.PathLike[str] | None = None,
    env: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> StageConfig:
    """Resolve a stage: base < stage < --config < .local < [default.ENV] < BLFS_* < CLI."""
    types = {k: kind for k, (_, kind) in schema(stage).items()}
    shipped = [CONFIG_DIR / "base.toml", CONFIG_DIR / f"{stage}.toml"]
    files = list(shipped)
    stage_file = shipped[1]
    # your own file goes on top of the shipped ones, so it can be partial
    if config_path is not None:
        stage_file = find_file(config_path)
        if not stage_file.exists():
            raise ConfigError(f"{display_path(stage_file)}: config file not found")
        sibling = stage_file.with_name("base.toml")
        files += [
            f.resolve()
            for f in (sibling, stage_file)
            if f.exists() and f.resolve() not in files
        ]

    top, variants = _split(_settings(files, env_vars=True))
    chosen = resolve_env(env)
    _pick_variant(chosen, variants, files)
    values, sources = _merge(
        top, variants, chosen, overrides, types, f"the {stage} stage"
    )
    missing = sorted(set(types) - set(values))
    if missing:
        raise ConfigError(
            f"{display_path(stage_file)}: missing settings {', '.join(missing)}"
        )
    _validate(values, sources)
    problems = _cross_checks(values)
    if problems:
        origin = ", ".join(sorted(set(sources.values())))
        raise ConfigError(f"{problems[0]} (from {origin})")
    return StageConfig(**values)


def load_logging(
    *, env: str | None = None, overrides: dict[str, Any] | None = None
) -> StageConfig:
    """Resolve configs/logging.toml < .local.toml < [default.ENV] < BLFS_* < overrides."""
    path = CONFIG_DIR / "logging.toml"
    types = {
        k: type(v)
        for k, v in _read_default(path).items()
        if not isinstance(v, dict) and not k.startswith("_")
    }
    top, variants = _split(_settings([path], env_vars=True))
    chosen = resolve_env(env)
    if chosen not in variants:  # BLFS_ENV=smoke is for training; logging may lack it
        chosen = DEFAULT_ENV
    values, sources = _merge(top, variants, chosen, overrides, types, "logging")
    values["log_level"] = str(values["log_level"]).upper()
    _validate(values, sources)
    return StageConfig(**values)


def _classic() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The ``[default]`` table of configs/presets.toml and its ``[default.NAME]`` presets."""

    def plain(v: Any) -> Any:
        return None if v == "none" else v

    top, variants = _split(_settings([CONFIG_DIR / "presets.toml"], env_vars=False))
    return (
        {k: plain(v) for k, (v, _) in top.items()},
        {
            name: {k: plain(v) for k, (v, _) in table.items()}
            for name, table in variants.items()
        },
    )


_CLASSIC_DEFAULT, PRESETS = _classic()


def apply_preset(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a copy of ``config`` with preset ``name`` applied."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose one of {sorted(PRESETS)}")
    return {**config, **PRESETS[name]}


def __getattr__(name: str) -> Any:
    """``default_config``: the classic script's settings, with the device picked on first use."""
    if name == "default_config":
        from src.device import resolve_device

        value = {**_CLASSIC_DEFAULT, "device": resolve_device("auto")}
        globals()["default_config"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
