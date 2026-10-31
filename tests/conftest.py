"""Test-wide setup: check tensor shapes and types at runtime."""

from __future__ import annotations

import os

CHECKED_MODULES = [
    "src.inference",
    "src.models.lora",
    "src.models.modern",
    "src.optim",
    "src.post_training.distill",
    "src.post_training.dpo",
    "src.post_training.grpo",
]

if os.environ.get("SHAPE_CHECKS", "1") != "0":
    try:
        from jaxtyping import install_import_hook

        import tests._typecheck  # noqa: F401  (jaxtyping looks the checker up as tests -> _typecheck)
    except (
        ImportError
    ):  # beartype is a dev dependency; plain runs still work without it
        pass
    else:
        # Installed for the whole session (no `with`), before any test imports these modules.
        # tests/_typecheck.py is beartype with PEP 484 int-for-float promotion, like mypy.
        install_import_hook(CHECKED_MODULES, "tests._typecheck.typechecker")
