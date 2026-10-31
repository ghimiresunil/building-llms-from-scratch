"""Checkpoints load with PyTorch's safe unpickler unless you explicitly trust the file."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from src.checkpoint import TRUST_ENV, load_checkpoint


class Suspicious:
    """Stands in for any custom object; unpickling one could run code."""


def test_plain_checkpoints_load_safely(tmp_path) -> None:
    path = tmp_path / "ok.pt"
    payload = {
        "model_state_dict": OrderedDict(w=torch.ones(2)),
        "cfg": {"lr": 1e-3, "x": None},
        "metrics": {"loss": 1.5},
        "pytorch_version": torch.__version__,
    }  # a TorchVersion, as old files stored
    torch.save(payload, path)
    loaded = load_checkpoint(path)
    assert (
        torch.equal(loaded["model_state_dict"]["w"], torch.ones(2))
        and loaded["cfg"]["x"] is None
    )


def test_custom_objects_need_explicit_trust(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "custom.pt"
    torch.save({"thing": Suspicious()}, path)
    monkeypatch.delenv(TRUST_ENV, raising=False)
    with pytest.raises(RuntimeError, match=TRUST_ENV):
        load_checkpoint(path)
    assert isinstance(load_checkpoint(path, trusted=True)["thing"], Suspicious)
    monkeypatch.setenv(TRUST_ENV, "1")
    assert isinstance(load_checkpoint(path)["thing"], Suspicious)
    with pytest.raises(RuntimeError):
        load_checkpoint(path, trusted=False)  # an explicit False beats the variable
