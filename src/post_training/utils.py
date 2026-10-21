"""Shared post-training helpers: model building, frozen copies, checkpoints, masked means."""

from __future__ import annotations

import contextlib
import copy
import random
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.checkpoint import (
    atomic_save,
    load_checkpoint,
    load_model_weights,
    model_state_from_checkpoint,
    strip_wrapper_prefixes,
    unwrap_model,
)
from src.models.factory import LanguageModel, build_model
from src.models.modern import ModernTransformer


def amp_autocast(amp_dtype: str | None, device: str):
    """Return a bf16 autocast context on CUDA when requested, else a no-op context."""
    if (
        amp_dtype == "bf16"
        and str(device).startswith("cuda")
        and torch.cuda.is_available()
    ):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --- Reproducibility ---------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (incl. CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Model construction ------------------------------------------------------


def _cfg_get(cfg: Any, key: str) -> Any:
    """Read a field from either a dataclass/object (attr) or a dict."""
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def build_model_from_config(cfg: Any) -> LanguageModel:
    """Build a fresh model from a config with the standard size keys (and ``arch``)."""
    return build_model(cfg)


# Kept for code that imported the old private name; it now also strips torch.compile's
# ``_orig_mod.`` prefix (issue #36).
_strip_ddp_prefix = strip_wrapper_prefixes


def load_backbone_from_ckpt(cfg: Any, ckpt_path: str, device: str) -> LanguageModel:
    """Build a model from ``cfg`` and load backbone weights from any stage's checkpoint."""
    model = build_model_from_config(cfg)
    state = model_state_from_checkpoint(load_checkpoint(ckpt_path, map_location="cpu"))
    load_model_weights(model, state, source=ckpt_path)
    return model.to(device)


def unwrap(model: nn.Module) -> nn.Module:
    """Return the plain model behind DDP and torch.compile wrappers (or the model itself)."""
    return unwrap_model(model)


def moe_balance_loss(model: nn.Module) -> torch.Tensor | float:
    """The Mixture-of-Experts balancing loss of the last forward pass, times its coefficient."""
    inner = unwrap(model)
    if isinstance(inner, ModernTransformer) and inner.aux_loss is not None:
        return inner.config.moe_aux_loss_coef * inner.aux_loss
    return 0.0


def make_frozen_copy(model: nn.Module, device: str | None = None) -> nn.Module:
    """Deep-copy a model, put it in eval mode, and disable all gradients."""
    ref = copy.deepcopy(unwrap(model))
    if device is not None:
        ref = ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# --- Masked reductions -------------------------------------------------------


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over positions where ``mask`` is truthy (safe if mask empty)."""
    mask = mask.to(values.dtype)
    total = (values * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return total / count


def masked_mean_per_row(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row (B,) mean of ``values`` (B,T) over masked positions."""
    mask = mask.to(values.dtype)
    total = (values * mask).sum(dim=-1)
    count = mask.sum(dim=-1).clamp(min=1.0)
    return total / count


def gather_last(values: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """Pick each row's value at its last real token, given ``seq_lengths``."""
    idx = (seq_lengths - 1).clamp(min=0).long()
    return values[torch.arange(values.size(0), device=values.device), idx]


# --- Checkpoint I/O ----------------------------------------------------------


def _cfg_to_dict(cfg: Any) -> Any:
    """A plain dict for the checkpoint, so it loads without importing the config code."""
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return asdict(cfg) if is_dataclass(cfg) and not isinstance(cfg, type) else cfg


def save_stage_ckpt(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    stage: str,
    cfg: Any,
    step: int,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Save a checkpoint with the model and optimizer state plus stage, cfg, step and metrics."""
    payload = {
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
        "stage": stage,
        "cfg": _cfg_to_dict(cfg),
        "step": step,
        "metrics": metrics or {},
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if extra:
        payload.update(extra)
    atomic_save(payload, path)
