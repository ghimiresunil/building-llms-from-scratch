"""Measure loss at longer contexts than a model was trained on, with each RoPE scaling.

Train a modern model at a short context, then see how each method holds up past it:

    python scripts/eval_long_context.py --ckpt models/base_pretrained.pt \
        --data data/pile_dev.h5 --lengths 256 512 1024

``none`` just runs the model on longer inputs (the loss usually explodes); ``linear``,
``ntk`` and ``yarn`` stretch the rotary embeddings by ``max(lengths) / trained context``,
with no extra training. A short fine-tune at the long length (``pretrain_base.py --init_from`` with
``--context_length 1024 --rope_scaling yarn --rope_factor 4``) closes most of what's left.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # run from the repo without installing

import argparse
import json
import os

import numpy as np
import torch

from src.checkpoint import load_checkpoint, model_config_from_checkpoint
from src.models.modern.rope import ROPE_SCALINGS
from src.post_training.inference import load_model_from_ckpt


@torch.no_grad()
def eval_loss(
    model, tokens: np.ndarray, length: int, n_batches: int, batch_size: int, device
):
    """Mean next-token loss over fixed, evenly spaced windows of ``length`` tokens."""
    n_windows = (len(tokens) - 1) // length
    if n_windows < 1:
        raise SystemExit(f"--data has too few tokens for length {length}")
    count = min(n_windows, n_batches * batch_size)
    starts = np.linspace(0, (n_windows - 1) * length, count).astype(np.int64)
    total = 0.0
    for i in range(0, count, batch_size):
        rows = np.stack(
            [tokens[s : s + length + 1] for s in starts[i : i + batch_size]]
        )
        x = torch.tensor(rows[:, :-1], dtype=torch.long, device=device)
        y = torch.tensor(rows[:, 1:], dtype=torch.long, device=device)
        _, loss = model(x, y)
        total += loss.item() * x.size(0)
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--ckpt", required=True, help="a checkpoint of an arch='modern' model"
    )
    p.add_argument(
        "--data", required=True, help="tokenized HDF5 file (a 'tokens' array)"
    )
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument(
        "--scalings", nargs="+", default=list(ROPE_SCALINGS), choices=ROPE_SCALINGS
    )
    p.add_argument(
        "--factor", type=float, default=None, help="default: longest / trained"
    )
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--max_tokens", type=int, default=5_000_000, help="tokens read from --data"
    )
    p.add_argument("--report", default=None, help="append the results to this JSONL")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    stored = model_config_from_checkpoint(load_checkpoint(args.ckpt))
    if stored.get("arch") != "modern":
        raise SystemExit(
            "RoPE scaling needs a modern-architecture checkpoint (arch = modern)"
        )
    trained = int(stored.get("rope_original_context") or stored["context_length"])
    longest = max(args.lengths)
    factor = args.factor or max(1.0, longest / trained)

    import h5py

    with h5py.File(args.data, "r") as f:
        ds = f["tokens"]
        tokens = np.asarray(ds[: min(len(ds), args.max_tokens)])

    print(f"trained at {trained} tokens; stretching by {factor:g} to reach {longest}\n")
    header = f"{'scaling':<10}" + "".join(f"{L:>12}" for L in args.lengths)
    print(header + "\n" + "-" * len(header))
    results = {}
    for scaling in args.scalings:
        overrides = {
            "context_length": max(longest, trained),
            "rope_scaling": scaling,
            "rope_factor": factor if scaling != "none" else 1.0,
            "rope_original_context": trained,
        }
        model = load_model_from_ckpt(args.ckpt, args.device, overrides=overrides)
        losses = [
            eval_loss(model, tokens, L, args.batches, args.batch_size, args.device)
            for L in args.lengths
        ]
        results[scaling] = dict(zip(map(str, args.lengths), losses))
        print(f"{scaling:<10}" + "".join(f"{loss:>12.3f}" for loss in losses))
        del model
    print("\nmean next-token loss; lower is better")

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "a") as f:
            f.write(
                json.dumps(
                    {"ckpt": args.ckpt, "trained": trained, "factor": factor, **results}
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
