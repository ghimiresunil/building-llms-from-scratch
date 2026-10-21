"""Build a model from any config: stage settings, a plain dict, or a checkpoint's cfg."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import torch.nn as nn

from src.models.modern import ModernConfig, ModernTransformer
from src.models.transformer import Transformer

ARCHITECTURES = ("classic", "modern")

# Either architecture. Both expose forward / forward_hidden / lm_head / context_length / generate,
# which is all the post-training code relies on.
LanguageModel = Transformer | ModernTransformer


def config_as_dict(cfg: Any) -> dict[str, Any]:
    """A plain dict view of a dataclass instance, a mapping, or a namespace."""
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    return dict(vars(cfg))


def build_model(cfg: Any) -> LanguageModel:
    """Construct a fresh, randomly initialized model described by ``cfg``."""
    values = config_as_dict(cfg)
    arch = values.get("arch") or "classic"
    if arch == "classic":
        return Transformer(
            n_head=values["n_head"],
            n_embed=values["n_embed"],
            context_length=values["context_length"],
            vocab_size=values["vocab_size"],
            N_BLOCKS=values["n_blocks"],
        )
    if arch == "modern":
        return ModernTransformer(ModernConfig.from_mapping(values))
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
