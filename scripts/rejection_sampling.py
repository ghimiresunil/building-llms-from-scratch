"""Sample N answers per GSM8K question: report pass@1, pass@N and maj@N, and build RFT data.

Examples
--------
Measure how much test-time compute buys (no files written)::

    python scripts/rejection_sampling.py --ckpt models/sft.pt --n_samples 8 --limit 200 \
        --split test --no_write

Rejection-sampling fine-tuning: keep the verified answers, then run SFT on them::

    python scripts/rejection_sampling.py --ckpt models/sft.pt --n_samples 8 --keep correct
    python scripts/train_sft.py --pretrained_ckpt models/sft.pt \
        --data_path data/rft_packed.h5 --dev_path data/rft_dev_packed.h5 --out_ckpt models/rft.pt

Sequence-level distillation: keep everything a (bigger) teacher writes::

    python scripts/rejection_sampling.py --ckpt models/teacher.pt --n_samples 1 --keep all \
        --prefix distill
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

import h5py
import numpy as np
import torch

from src.post_training.chat_template import encode_chat, encode_prompt
from src.post_training.evaluation import batched_generate, load_gsm8k_eval
from src.post_training.inference import load_model_from_ckpt
from src.post_training.rewards import extract_answer, gsm8k_gold_answer
from src.post_training.sft import pack_examples
from src.post_training.test_time import answers_match, majority_vote, pass_at_k


def write_packed(examples, context_length: int, path: str) -> int:
    tokens, masks = pack_examples(examples, context_length)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("tokens", data=tokens)
        f.create_dataset("loss_mask", data=masks)
    return tokens.shape[0]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=1000, help="questions to use (0 = all)")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--micro_batch", type=int, default=32)
    p.add_argument(
        "--keep",
        choices=["correct", "all"],
        default="correct",
        help="correct = rejection sampling (RFT); all = sequence-level distillation",
    )
    p.add_argument(
        "--max_per_question",
        type=int,
        default=4,
        help="cap kept answers per question, deduplicated, so easy questions don't dominate",
    )
    p.add_argument("--out_dir", default="data")
    p.add_argument(
        "--prefix", default="rft", help="writes <prefix>_packed.h5 and _dev_packed.h5"
    )
    p.add_argument("--dev_frac", type=float, default=0.02)
    p.add_argument("--no_write", action="store_true", help="only report the metrics")
    p.add_argument(
        "--report", default=None, help="append the metrics row to this JSONL"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = load_model_from_ckpt(args.ckpt, args.device)
    qa = load_gsm8k_eval(args.split, limit=args.limit or None)
    n = args.n_samples
    prompts = [encode_prompt([{"role": "user", "content": q}]) for q, _ in qa]
    print(f"{len(qa)} questions x {n} samples from {args.ckpt} (T={args.temperature})")
    responses = batched_generate(
        model,
        [pr for pr in prompts for _ in range(n)],
        args.max_new_tokens,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        micro_batch=args.micro_batch,
    )

    examples: list[tuple[list[int], list[int]]] = []
    p1 = pn = maj = 0.0
    for i, (q, ans) in enumerate(qa):
        gold = gsm8k_gold_answer(ans)
        group = responses[i * n : (i + 1) * n]
        parsed = [extract_answer(r) for r in group]
        correct = [answers_match(a, gold) for a in parsed]
        c = sum(correct)
        p1 += pass_at_k(n, c, 1)
        pn += pass_at_k(n, c, n)
        maj += float(answers_match(majority_vote(parsed), gold))

        seen: set[str] = set()
        for resp, ok in zip(group, correct):
            if args.keep == "correct" and not ok:
                continue
            if resp in seen or len(seen) >= args.max_per_question:
                continue
            seen.add(resp)
            ids, mask = encode_chat(
                [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": resp},
                ]
            )
            if len(ids) <= model.context_length:
                examples.append((ids, mask))

    m = len(qa)
    row = {
        "ckpt": args.ckpt,
        "split": args.split,
        "n_questions": m,
        "n_samples": n,
        "pass@1": p1 / m,
        f"pass@{n}": pn / m,
        f"maj@{n}": maj / m,
        "kept": len(examples),
    }
    print(
        f"pass@1 {100 * row['pass@1']:.1f}% | pass@{n} {100 * row[f'pass@{n}']:.1f}% | "
        f"maj@{n} {100 * row[f'maj@{n}']:.1f}% | kept {len(examples)} answers"
    )
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "a") as f:
            f.write(json.dumps(row) + "\n")

    if args.no_write:
        return
    if not examples:
        raise SystemExit(
            "no answers kept; try more samples, a higher temperature or --keep all"
        )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(examples)
    n_dev = max(1, int(len(examples) * args.dev_frac))
    ctx_len = model.context_length
    for name, part in (("packed", examples[n_dev:]), ("dev_packed", examples[:n_dev])):
        path = os.path.join(args.out_dir, f"{args.prefix}_{name}.h5")
        rows = write_packed(part, ctx_len, path)
        print(f"  wrote {rows} packed rows x {ctx_len} -> {path}")
        if rows == 0:
            print(f"  (fewer than {ctx_len} tokens in this split; use more questions)")


if __name__ == "__main__":
    main()
