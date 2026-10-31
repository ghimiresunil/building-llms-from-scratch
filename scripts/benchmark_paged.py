"""Continuous batching with a paged KV cache vs. one request at a time.

    python scripts/benchmark_paged.py                       # a random modern model, CPU-sized
    python scripts/benchmark_paged.py --ckpt models/sft.pt  # any arch='modern', attention='gqa' checkpoint

Prints throughput for both, how full the decode batches were, and how much KV memory the
paged cache actually used compared with a contiguous cache sized for the same batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # run from the repo without installing

import argparse
import time

import torch

from src.inference.paged import PagedEngine
from src.models.modern import ModernConfig, ModernTransformer


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--ckpt", default=None, help="default: a randomly initialized small model"
    )
    p.add_argument("--requests", type=int, default=24)
    p.add_argument("--min_prompt", type=int, default=8)
    p.add_argument("--max_prompt", type=int, default=96)
    p.add_argument("--min_new", type=int, default=8)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--max_batch", type=int, default=8)
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--num_blocks", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.ckpt:
        from src.post_training.inference import load_model_from_ckpt

        model = load_model_from_ckpt(args.ckpt, args.device)
        if not isinstance(model, ModernTransformer):
            raise SystemExit("the paged engine needs an arch='modern' checkpoint")
    else:
        cfg = ModernConfig(
            vocab_size=4096,
            context_length=512,
            n_embed=256,
            n_head=8,
            n_kv_head=2,
            n_blocks=4,
        )
        model = ModernTransformer(cfg).to(args.device).eval()

    g = torch.Generator().manual_seed(args.seed)
    vocab = model.config.vocab_size
    work = []
    for _ in range(args.requests):
        L = int(torch.randint(args.min_prompt, args.max_prompt + 1, (1,), generator=g))
        n = int(torch.randint(args.min_new, args.max_new + 1, (1,), generator=g))
        work.append((torch.randint(0, vocab, (L,), generator=g).tolist(), n))
    new_tokens = sum(n for _, n in work)

    t0 = time.perf_counter()
    with torch.no_grad():
        for prompt, n in work:
            model.generate(torch.tensor([prompt], device=args.device), n, top_k=1)
    sequential = time.perf_counter() - t0

    engine = PagedEngine(
        model,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_batch=args.max_batch,
        greedy=True,
    )
    for prompt, n in work:
        engine.add_request(prompt, n)
    t0 = time.perf_counter()
    engine.run()
    paged = time.perf_counter() - t0

    s = engine.stats
    per_token = engine.cache.k[:, 0, :, 0].numel() * 2 * engine.cache.k.element_size()
    contiguous = args.max_batch * model.context_length * per_token
    used = s.peak_blocks * args.block_size * per_token
    print(
        f"{args.requests} requests, {new_tokens} new tokens, max_batch {args.max_batch}\n"
    )
    print(f"{'one at a time':<26}{new_tokens / sequential:>10.1f} tok/s")
    print(
        f"{'continuous batching':<26}{new_tokens / paged:>10.1f} tok/s   "
        f"({sequential / paged:.1f}x)"
    )
    print(f"{'mean decode batch':<26}{s.mean_batch:>10.1f}")
    print(f"{'preemptions':<26}{s.preemptions:>10}")
    print(
        f"{'KV memory, contiguous':<26}{contiguous / 2**20:>10.1f} MiB  "
        f"({args.max_batch} x {model.context_length} slots reserved up front)"
    )
    print(
        f"{'KV memory, paged (peak)':<26}{used / 2**20:>10.1f} MiB  "
        f"({s.peak_blocks} blocks of {args.block_size})"
    )


if __name__ == "__main__":
    main()
