"""Command-line flags for every setting of a stage, resolved through config.settings."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

from config.settings import (
    CHOICES,
    NULLABLE,
    ConfigError,
    StageConfig,
    load_stage,
    schema,
    type_label,
)


def to_toml(values: dict[str, Any], table: str = "default") -> str:
    """Render flat settings as a TOML table (``None`` becomes ``"none"``)."""
    lines = [f"[{table}]"] + [
        f"{k} = {json.dumps('none' if v is None else v)}" for k, v in values.items()
    ]
    return "\n".join(lines) + "\n"


def parse_stage(
    stage: str, extra: dict | None = None
) -> tuple[StageConfig, SimpleNamespace]:
    """Parse ``--field value`` overrides plus ``--config``/``--env``/``--print-config`` for one stage."""
    fields = schema(stage)
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, (default, kind) in fields.items():
        choices = CHOICES.get(key)
        if choices and key in NULLABLE:
            choices = [*choices, "none"]
        p.add_argument(
            f"--{key}",
            type=str,
            default=None,
            help=f"default: {default}",
            metavar=None if choices else type_label(key, kind),
            choices=choices,
        )
    for flag, kwargs in (extra or {}).items():
        p.add_argument(flag, **kwargs)
    p.add_argument(
        "--config",
        default=None,
        help=f"settings file to load (default: configs/{stage}.toml)",
    )
    p.add_argument(
        "--env",
        default=None,
        help="[default.NAME] table to apply, e.g. smoke (default: $BLFS_ENV)",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved settings as TOML and exit",
    )
    args = vars(p.parse_args())
    overrides = {k: v for k, v in args.items() if k in fields and v is not None}
    extras = {
        k: v
        for k, v in args.items()
        if k not in fields and k not in ("config", "env", "print_config")
    }
    try:
        cfg = load_stage(
            stage, config_path=args["config"], env=args["env"], overrides=overrides
        )
    except ConfigError as e:
        p.error(str(e))
    if args["print_config"]:
        print(to_toml(cfg.to_dict()), end="")
        raise SystemExit(0)
    return cfg, SimpleNamespace(**extras)
