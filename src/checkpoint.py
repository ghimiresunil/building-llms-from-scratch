"""Checkpoint helpers shared by every training and inference script."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Mapping
from typing import Any, TypeVar

import torch
import torch.nn as nn
from torch.torch_version import TorchVersion

V = TypeVar("V")

# Prefixes added by DistributedDataParallel / DataParallel and by torch.compile.
WRAPPER_PREFIXES: tuple[str, ...] = ("module.", "_orig_mod.")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the plain module behind any mix of DDP / DataParallel / torch.compile wrappers."""
    while True:
        inner = getattr(model, "_orig_mod", None)  # torch.compile's OptimizedModule
        if isinstance(inner, nn.Module):
            model = inner
            continue
        if isinstance(model, nn.parallel.DistributedDataParallel | nn.DataParallel):
            model = model.module
            continue
        return model


def strip_wrapper_prefixes(
    state_dict: Mapping[str, V], extra_prefixes: tuple[str, ...] = ()
) -> dict[str, V]:
    """Remove leading wrapper prefixes from every key, in any order and any nesting depth."""
    prefixes = WRAPPER_PREFIXES + tuple(extra_prefixes)
    cleaned: dict[str, V] = {}
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = value
    return cleaned


TRUST_ENV = "BLFS_TRUST_CHECKPOINTS"


def _trusted_by_env() -> bool:
    return os.environ.get(TRUST_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def load_checkpoint(
    path: str | os.PathLike[str],
    map_location: Any = "cpu",
    *,
    trusted: bool | None = None,
) -> dict[str, Any]:
    """Load a checkpoint dict with PyTorch's safe loader, which refuses to run pickled code.

    Our checkpoints hold only tensors, numbers, strings, lists and dicts, so
    ``weights_only=True`` reads them. A file that needs full unpickling (an old checkpoint
    with custom objects, or a file from someone else) is refused unless you trust it:
    pass ``trusted=True`` or set ``BLFS_TRUST_CHECKPOINTS=1``. Unpickling can run arbitrary
    code, so only do that for files you made yourself.
    """
    try:
        # Older checkpoints stored torch.__version__, a harmless str subclass; allow it.
        with torch.serialization.safe_globals([TorchVersion]):
            return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as err:
        if not (trusted if trusted is not None else _trusted_by_env()):
            raise RuntimeError(
                f"{os.fspath(path)} contains Python objects that the safe loader refuses "
                f"({str(err).splitlines()[0]}). Loading it fully could run arbitrary code. "
                f"If you made this file yourself, set {TRUST_ENV}=1 to load it anyway."
            ) from None
        return torch.load(path, map_location=map_location, weights_only=False)


def model_state_from_checkpoint(
    checkpoint: Mapping[str, Any], extra_prefixes: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return the model weights of a checkpoint with wrapper prefixes removed."""
    state = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    cleaned = strip_wrapper_prefixes(state, extra_prefixes)
    return {k: v for k, v in cleaned.items() if not k.endswith(".tril")}


def model_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return the model settings stored in a checkpoint."""
    cfg = checkpoint.get("cfg") or checkpoint.get("config") or {}
    return dict(cfg)


def load_model_weights(
    model: nn.Module, state: Mapping[str, torch.Tensor], *, source: str = "checkpoint"
) -> None:
    """Copy matching weights from ``state`` into ``model``."""
    own_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in own_keys}
    missing, _ = model.load_state_dict(filtered, strict=False)
    param_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    missing_params = [k for k in missing if k in param_names]
    if missing_params:
        raise RuntimeError(
            f"{source} is missing {len(missing_params)} model parameters "
            f"(for example {missing_params[:3]}). Check that the checkpoint was saved from a "
            "model with the same architecture and size."
        )


def atomic_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """``torch.save`` to a temporary file first, then rename, so a crash never leaves half a file."""
    path = os.fspath(path)
    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target_dir,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        delete=False,
    ) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
