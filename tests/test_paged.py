"""The paged KV cache and continuous-batching engine (src/inference/paged.py)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.inference.paged import BlockAllocator, PagedEngine, PagedKVCache
from src.models.modern import ModernConfig, ModernTransformer

BASE = ModernConfig(
    vocab_size=61, context_length=48, n_embed=32, n_head=4, n_blocks=2, n_kv_head=2
)
VARIANTS = {
    "gqa": BASE,
    "mha": replace(BASE, n_kv_head=None),
    "window": replace(BASE, sliding_window=6),
    "moe": replace(BASE, n_experts=4, moe_top_k=2),
    "gated": replace(BASE, attn_gate=True, qk_norm=False),
    "yarn": replace(BASE, rope_scaling="yarn", rope_factor=2.0),
}


def _model(cfg: ModernConfig) -> ModernTransformer:
    torch.manual_seed(0)
    return ModernTransformer(cfg).eval()


@torch.no_grad()
def _greedy_reference(model: ModernTransformer, prompt: list[int], n: int) -> list[int]:
    """Greedy decoding with no cache at all: the slowest, most obviously correct version."""
    idx = torch.tensor([prompt])
    out: list[int] = []
    for _ in range(n):
        token = int(model(idx)[0][0, -1].argmax())
        out.append(token)
        idx = torch.cat([idx, torch.tensor([[token]])], dim=1)
    return out


def _prompts() -> list[tuple[list[int], int]]:
    g = torch.Generator().manual_seed(1)
    lengths = [3, 9, 1, 14, 6, 11]
    new = [7, 3, 12, 5, 9, 4]
    return [
        (torch.randint(0, 61, (L,), generator=g).tolist(), n)
        for L, n in zip(lengths, new)
    ]


def test_allocator_hands_out_and_takes_back_blocks() -> None:
    alloc = BlockAllocator(4)
    got = [alloc.allocate() for _ in range(4)]
    assert sorted(got) == [0, 1, 2, 3] and alloc.n_free == 0
    with pytest.raises(RuntimeError):
        alloc.allocate()
    alloc.free(got[:2])
    assert alloc.n_free == 2 and alloc.n_used == 2


def test_cache_write_then_gather_round_trips() -> None:
    cache = PagedKVCache(1, num_blocks=6, block_size=4, n_kv_heads=2, head_dim=3)
    tables = [[5, 1], [2]]
    positions = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 0, 0]])
    blocks, offsets = cache.slots(tables, positions)
    assert blocks.tolist() == [5, 5, 5, 5, 1, 2, 2, 2, 2, 2]
    assert offsets.tolist() == [0, 1, 2, 3, 0, 0, 1, 2, 0, 0]
    k = torch.arange(5 * 2 * 3, dtype=torch.float32).view(5, 2, 3)
    cache.write(0, blocks[:5], offsets[:5], k, -k)
    k_all, v_all, valid = cache.gather(0, [tables[0]], torch.tensor([5]))
    assert torch.equal(k_all[0, :, :5].transpose(0, 1), k)
    assert torch.equal(v_all[0, :, :5].transpose(0, 1), -k)
    assert valid[0].tolist() == [True] * 5 + [False] * 3


@pytest.mark.parametrize("name", list(VARIANTS))
def test_engine_matches_uncached_greedy_decoding(name: str) -> None:
    model = _model(VARIANTS[name])
    engine = PagedEngine(model, num_blocks=40, block_size=4, max_batch=3, greedy=True)
    ids = {engine.add_request(p, n): (p, n) for p, n in _prompts()}
    outputs = engine.run()
    for rid, (prompt, n) in ids.items():
        assert outputs[rid] == _greedy_reference(model, prompt, n), f"request {rid}"
    assert engine.stats.mean_batch > 1  # requests really did share decode steps
    assert engine.cache.allocator.n_free == 40  # every block came back


def test_preemption_recomputes_and_still_matches() -> None:
    model = _model(BASE)
    # 7 blocks of 4 tokens: far too few for six requests at once, so some get preempted
    engine = PagedEngine(model, num_blocks=7, block_size=4, max_batch=6, greedy=True)
    ids = {engine.add_request(p, n): (p, n) for p, n in _prompts()}
    outputs = engine.run()
    assert engine.stats.preemptions > 0
    assert engine.stats.peak_blocks <= 7
    for rid, (prompt, n) in ids.items():
        assert outputs[rid] == _greedy_reference(model, prompt, n)


def test_stop_tokens_and_context_limit_end_requests() -> None:
    model = _model(BASE)
    prompt = [1, 2, 3]
    first = _greedy_reference(model, prompt, 1)[0]
    engine = PagedEngine(
        model, num_blocks=20, block_size=4, greedy=True, stop_tokens=[first]
    )
    rid = engine.add_request(prompt, 10)
    engine.run()
    assert engine.finished[rid].finish_reason == "stop" and engine.finished[
        rid
    ].output == [first]

    engine = PagedEngine(model, num_blocks=20, block_size=4, greedy=True)
    rid = engine.add_request(list(range(45)), 50)
    engine.run()
    assert engine.finished[rid].finish_reason == "context"
    assert len(engine.finished[rid].tokens) == model.context_length


def test_sampling_is_reproducible_with_a_generator() -> None:
    model = _model(BASE)
    runs = []
    for _ in range(2):
        engine = PagedEngine(
            model,
            num_blocks=40,
            block_size=4,
            temperature=0.9,
            top_k=20,
            generator=torch.Generator().manual_seed(3),
        )
        for p, n in _prompts():
            engine.add_request(p, n)
        runs.append(engine.run())
    assert runs[0] == runs[1]


def test_rejects_what_it_cannot_serve() -> None:
    with pytest.raises(ValueError, match="gqa"):
        PagedEngine(_model(replace(BASE, attention="mla", kv_latent_dim=8, rope_dim=4)))
    engine = PagedEngine(_model(BASE), num_blocks=2, block_size=4)
    with pytest.raises(ValueError, match="more KV-cache blocks"):
        engine.add_request([1] * 20, 5)
    with pytest.raises(ValueError, match="at least one token"):
        engine.add_request([], 5)
