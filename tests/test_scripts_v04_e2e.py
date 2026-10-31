"""The 0.4 scripts end to end: distillation, long-context eval, curation, paged serving.

None of these steps needs the tiktoken download, so they also run offline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import zstandard as zstd

from src.models.modern import ModernTransformer
from src.post_training.inference import load_model_from_ckpt

pytestmark = pytest.mark.slow
REPO = Path(__file__).resolve().parents[1]


def run(script: str, *args: object) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        [sys.executable, f"scripts/{script}", *map(str, args)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
    )
    assert r.returncode == 0, (
        f"{script} failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    )
    return r.stdout


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("v04_data")
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 512, 30_000, dtype=np.int32)
    for name in ("pile_train.h5", "pile_dev.h5"):
        with h5py.File(d / name, "w") as f:
            f.create_dataset("tokens", data=tokens)
    rows = tokens[: 64 * 64].reshape(64, 64)
    mask = (rng.random(rows.shape) > 0.5).astype(np.int32)
    for name in ("sft_packed.h5", "sft_dev_packed.h5"):
        with h5py.File(d / name, "w") as f:
            f.create_dataset("tokens", data=rows)
            f.create_dataset("loss_mask", data=mask)
    words = (
        "the river carried small boats past the old mill while children watched".split()
    )
    docs = [" ".join(rng.choice(words, 80)) for _ in range(20)]
    docs += docs[:5] + ["too short"] * 5
    (d / "raw").mkdir()
    with zstd.open(d / "raw" / "part.jsonl.zst", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps({"text": t}) + "\n" for t in docs)
    return d


def test_v04_scripts(data: Path, tmp_path: Path) -> None:
    common = [
        "--env",
        "smoke",
        "--device",
        "cpu",
        "--arch",
        "modern",
        "--n_kv_head",
        "2",
        "--context_length",
        "64",
        "--vocab_size",
        "512",
        "--log_dir",
        tmp_path / "logs",
    ]
    pre = [
        "--train_path",
        data / "pile_train.h5",
        "--dev_path",
        data / "pile_dev.h5",
        "--eval_steps",
        100,
        "--eval_iters",
        1,
    ]
    run(
        "pretrain_base.py",
        *common,
        *pre,
        "--out_ckpt",
        tmp_path / "teacher.pt",
        "--n_blocks",
        3,
        "--train_steps",
        6,
    )
    run(
        "pretrain_base.py",
        *common,
        *pre,
        "--out_ckpt",
        tmp_path / "student.pt",
        "--n_embed",
        64,
        "--train_steps",
        4,
    )

    distill = [
        *common,
        "--n_embed",
        64,
        "--teacher_ckpt",
        tmp_path / "teacher.pt",
        "--max_steps",
        3,
    ]
    out = run(
        "train_distill.py",
        *distill,
        "--student_ckpt",
        tmp_path / "student.pt",
        "--data_format",
        "pretrain",
        "--data_path",
        data / "pile_train.h5",
        "--dev_path",
        data / "pile_dev.h5",
        "--out_ckpt",
        tmp_path / "distill.pt",
    )
    assert "smaller" in out
    run(
        "train_distill.py",
        *distill,
        "--student_ckpt",
        "none",
        "--distill_loss",
        "jsd",
        "--data_path",
        data / "sft_packed.h5",
        "--dev_path",
        data / "sft_dev_packed.h5",
        "--out_ckpt",
        tmp_path / "distill_sft.pt",
    )
    for name in ("distill", "distill_sft"):
        model = load_model_from_ckpt(str(tmp_path / f"{name}.pt"), "cpu")
        assert isinstance(model, ModernTransformer) and model.config.n_embed == 64

    out = run(
        "eval_long_context.py",
        "--ckpt",
        tmp_path / "teacher.pt",
        "--data",
        data / "pile_dev.h5",
        "--lengths",
        64,
        128,
        "--batches",
        1,
        "--batch_size",
        2,
    )
    assert all(s in out for s in ("none", "linear", "ntk", "yarn"))

    out = run(
        "curate_data.py", "--input", data / "raw", "--out_dir", tmp_path / "curated"
    )
    with zstd.open(
        tmp_path / "curated" / "curated.jsonl.zst", "rt", encoding="utf-8"
    ) as f:
        assert len(f.readlines()) == 20  # 5 duplicates and 5 short docs removed

    out = run(
        "benchmark_paged.py",
        "--ckpt",
        tmp_path / "teacher.pt",
        "--requests",
        4,
        "--max_prompt",
        20,
        "--max_new",
        10,
        "--block_size",
        4,
        "--num_blocks",
        32,
    )
    assert "continuous batching" in out
