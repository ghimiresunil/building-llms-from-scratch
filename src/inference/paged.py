"""A paged KV cache and a continuous-batching engine: the two ideas behind vLLM.

``KVCache`` in ``src/models/modern/kv_cache.py`` reserves ``context_length`` slots for every
sequence up front. Most requests stop long before that, so most of that memory is wasted,
and a batch can only start and finish together.

Paging fixes the memory. The cache becomes a pool of fixed-size *blocks* (say 16 tokens).
Each sequence owns a *block table*, a list of block ids, and takes a new block only when it
has filled its last one. Waste is at most one partly filled block per sequence.

Continuous batching fixes the scheduling. Every step, finished sequences leave the batch
and give their blocks back, and waiting requests join as soon as there is room. When the
pool runs dry mid-generation, the newest request is *preempted*: its blocks are freed and it
goes back to the queue, to be recomputed later.

A production engine reads the blocks directly inside its attention kernel. Here we gather
them into a padded tensor and call ``scaled_dot_product_attention``, which is slower but
shows exactly the same bookkeeping. Only grouped-query attention (``attention = "gqa"``) is
supported.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from src.inference.sampling import sample_next_token
from src.models.modern.attention import GroupedQueryAttention
from src.models.modern.block import ModernBlock
from src.models.modern.model import ModernTransformer
from src.models.modern.rope import apply_rope


class BlockAllocator:
    """Hands out block ids from a free list."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def n_used(self) -> int:
        return self.num_blocks - len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("out of KV-cache blocks")
        return self._free.pop()

    def free(self, blocks: Sequence[int]) -> None:
        self._free.extend(reversed(blocks))


class PagedKVCache:
    """Keys and values for every layer, stored in ``num_blocks`` blocks of ``block_size``."""

    def __init__(
        self,
        n_layers: int,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.block_size = block_size
        shape = (n_layers, num_blocks, n_kv_heads, block_size, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.allocator = BlockAllocator(num_blocks)

    def blocks_for(self, n_tokens: int) -> int:
        """How many blocks ``n_tokens`` tokens need."""
        return math.ceil(n_tokens / self.block_size)

    def slots(
        self, tables: Sequence[Sequence[int]], positions: Int[Tensor, "batch seq"]
    ) -> tuple[Int[Tensor, " n"], Int[Tensor, " n"]]:
        """Map every ``(row, position)`` to its ``(block id, offset in block)``."""
        bs = self.block_size
        width = max(len(t) for t in tables)
        table = torch.tensor(
            [list(t) + [0] * (width - len(t)) for t in tables],
            dtype=torch.long,
            device=positions.device,
        )
        block = table.gather(1, positions // bs)
        return block.reshape(-1), (positions % bs).reshape(-1)

    def write(
        self,
        layer: int,
        blocks: Int[Tensor, " n"],
        offsets: Int[Tensor, " n"],
        k_new: Float[Tensor, "n heads dim"],
        v_new: Float[Tensor, "n heads dim"],
    ) -> None:
        """Store one key/value per token at its slot."""
        self.k[layer, blocks, :, offsets] = k_new.to(self.k.dtype)
        self.v[layer, blocks, :, offsets] = v_new.to(self.v.dtype)

    def gather(
        self,
        layer: int,
        tables: Sequence[Sequence[int]],
        lengths: Int[Tensor, " batch"],
    ) -> tuple[
        Float[Tensor, "batch heads total dim"],
        Float[Tensor, "batch heads total dim"],
        Bool[Tensor, "batch total"],
    ]:
        """Read every sequence's keys/values back as one padded batch, plus a validity mask."""
        width = max(len(t) for t in tables)
        idx = torch.tensor(
            [list(t) + [0] * (width - len(t)) for t in tables],
            dtype=torch.long,
            device=self.k.device,
        )
        B, H, D = idx.size(0), self.k.size(2), self.k.size(4)
        k = self.k[layer, idx]  # (B, width, H, bs, D)
        v = self.v[layer, idx]
        k = k.permute(0, 2, 1, 3, 4).reshape(B, H, width * self.block_size, D)
        v = v.permute(0, 2, 1, 3, 4).reshape(B, H, width * self.block_size, D)
        pos = torch.arange(width * self.block_size, device=self.k.device)
        valid = pos[None, :] < lengths[:, None].to(self.k.device)
        return k, v, valid


@dataclass
class Request:
    """One generation request and its progress."""

    rid: int
    prompt: list[int]
    max_new_tokens: int
    output: list[int] = field(default_factory=list)
    table: list[int] = field(default_factory=list)
    cached: int = 0  # tokens whose keys/values are in the cache
    finish_reason: str | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output


@dataclass
class EngineStats:
    steps: int = 0
    prefills: int = 0
    decode_tokens: int = 0
    preemptions: int = 0
    peak_blocks: int = 0
    batch_sizes: list[int] = field(default_factory=list)

    @property
    def mean_batch(self) -> float:
        return sum(self.batch_sizes) / max(1, len(self.batch_sizes))


class PagedEngine:
    """Serve many requests at once with a paged cache and continuous batching."""

    def __init__(
        self,
        model: ModernTransformer,
        *,
        num_blocks: int = 256,
        block_size: int = 16,
        max_batch: int = 16,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
        stop_tokens: Sequence[int] = (),
        generator: torch.Generator | None = None,
    ) -> None:
        if not isinstance(model, ModernTransformer) or model.config.attention != "gqa":
            raise ValueError(
                "PagedEngine needs a ModernTransformer with attention='gqa'"
            )
        self.model = model.eval()
        cfg = model.config
        ref = model.token_embed.weight
        self.cache = PagedKVCache(
            cfg.n_blocks,
            num_blocks,
            block_size,
            cfg.kv_heads,
            cfg.head_dim,
            device=ref.device,
            dtype=ref.dtype,
        )
        self.max_batch = max_batch
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.greedy = greedy
        self.stop_tokens = set(stop_tokens)
        self.generator = generator
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: dict[int, Request] = {}
        self.stats = EngineStats()
        self._next_id = 0

    # ------------------------------------------------------------------ public API
    def add_request(self, prompt: Sequence[int], max_new_tokens: int) -> int:
        """Queue a prompt; returns its request id."""
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        longest = min(len(prompt) + max_new_tokens, self.model.context_length)
        if self.cache.blocks_for(longest) > self.cache.allocator.num_blocks:
            raise ValueError("request needs more KV-cache blocks than the whole pool")
        req = Request(self._next_id, list(prompt), max_new_tokens)
        self._next_id += 1
        self.waiting.append(req)
        return req.rid

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def run(self) -> dict[int, list[int]]:
        """Step until every request has finished; returns ``{request id: new tokens}``."""
        while self.has_work():
            self.step()
        return {rid: r.output for rid, r in sorted(self.finished.items())}

    @torch.no_grad()
    def step(self) -> list[Request]:
        """Admit what fits, then decode one token for every running request."""
        done: list[Request] = []
        self._admit(done)
        if self.running:
            self._decode(done)
        self.stats.steps += 1
        return done

    # ------------------------------------------------------------------ scheduling
    def _admit(self, done: list[Request]) -> None:
        while self.waiting and len(self.running) < self.max_batch:
            req = self.waiting[0]
            # the prompt (plus anything generated before a preemption) and one more token
            need = self.cache.blocks_for(len(req.tokens) + 1)
            if need > self.cache.allocator.n_free:
                return
            self.waiting.popleft()
            req.table = [self.cache.allocator.allocate() for _ in range(need)]
            self._track_peak()
            logits = self._forward([req], [req.tokens])
            req.cached = len(req.tokens)
            self.stats.prefills += 1
            self._append(req, logits[0:1], done)
            if req.finish_reason is None:
                self.running.append(req)

    def _make_room(self) -> None:
        """Give every running request a slot for its next token, preempting if needed."""
        for req in list(self.running):
            if req not in self.running:  # preempted while making room for another one
                continue
            while self.cache.blocks_for(req.cached + 1) > len(req.table):
                if self.cache.allocator.n_free == 0:
                    victim = self.running[-1]
                    self._preempt(victim)
                    if victim is req:
                        break
                    continue
                req.table.append(self.cache.allocator.allocate())
        self._track_peak()

    def _preempt(self, req: Request) -> None:
        self.running.remove(req)
        self.cache.allocator.free(req.table)
        req.table, req.cached = [], 0
        self.waiting.appendleft(req)  # first in line once blocks free up
        self.stats.preemptions += 1

    def _decode(self, done: list[Request]) -> None:
        self._make_room()
        batch = list(self.running)
        if not batch:
            return
        self.stats.batch_sizes.append(len(batch))
        logits = self._forward(batch, [[r.output[-1]] for r in batch])
        for i, req in enumerate(batch):
            req.cached += 1
            self.stats.decode_tokens += 1
            self._append(req, logits[i : i + 1], done)
        self.running = [r for r in self.running if r.finish_reason is None]

    def _append(self, req: Request, logits: Tensor, done: list[Request]) -> None:
        token = int(
            sample_next_token(
                logits,
                self.temperature,
                self.top_k,
                self.top_p,
                greedy=self.greedy,
                generator=self.generator,
            ).item()
        )
        req.output.append(token)
        if token in self.stop_tokens:
            req.finish_reason = "stop"
        elif len(req.output) >= req.max_new_tokens:
            req.finish_reason = "length"
        elif len(req.tokens) >= self.model.context_length:
            req.finish_reason = "context"
        if req.finish_reason is not None:
            self.cache.allocator.free(req.table)
            req.table = []
            self.finished[req.rid] = req
            done.append(req)

    def _track_peak(self) -> None:
        self.stats.peak_blocks = max(
            self.stats.peak_blocks, self.cache.allocator.n_used
        )

    # ------------------------------------------------------------------ the model
    def _forward(
        self, reqs: Sequence[Request], new_tokens: Sequence[Sequence[int]]
    ) -> Float[Tensor, "batch vocab"]:
        """Run the new tokens of every request (all the same length) through the model."""
        model, cache = self.model, self.cache
        device = model.token_embed.weight.device
        T = len(new_tokens[0])
        idx = torch.tensor(
            [list(t) for t in new_tokens], dtype=torch.long, device=device
        )
        starts = torch.tensor([r.cached for r in reqs], dtype=torch.long, device=device)
        positions = starts[:, None] + torch.arange(T, device=device)[None, :]  # (B, T)
        tables = [r.table for r in reqs]
        blocks, offsets = cache.slots(tables, positions)
        cos = model.rope_cos[positions].unsqueeze(
            1
        )  # (B, 1, T, half): one row per sequence
        sin = model.rope_sin[positions].unsqueeze(1)

        x = model.token_embed(idx)
        for layer, module in enumerate(model.blocks):
            assert isinstance(module, ModernBlock)
            block, attn = module, module.attn
            assert isinstance(attn, GroupedQueryAttention)
            x = x + self._attention(
                attn,
                block.attn_norm(x),
                cos,
                sin,
                layer,
                tables,
                positions,
                blocks,
                offsets,
            )
            x = x + block.mlp(block.mlp_norm(x))
        return model.lm_head(model.final_norm(x[:, -1, :]))

    def _attention(
        self,
        attn: GroupedQueryAttention,
        x: Float[Tensor, "batch seq embed"],
        cos: Tensor,
        sin: Tensor,
        layer: int,
        tables: Sequence[Sequence[int]],
        positions: Int[Tensor, "batch seq"],
        blocks: Int[Tensor, " n"],
        offsets: Int[Tensor, " n"],
    ) -> Float[Tensor, "batch seq embed"]:
        B, T, _ = x.shape
        H, Hkv, D = attn.n_head, attn.n_kv_head, attn.head_dim
        q = attn.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q, k = attn.q_norm(q), attn.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # write this step's keys/values into their blocks, then read the whole history back
        self.cache.write(
            layer,
            blocks,
            offsets,
            k.transpose(1, 2).reshape(B * T, Hkv, D),
            v.transpose(1, 2).reshape(B * T, Hkv, D),
        )
        lengths = positions[:, -1] + 1
        k_all, v_all, valid = self.cache.gather(layer, tables, lengths)

        k_pos = torch.arange(k_all.size(2), device=x.device)
        allowed = valid[:, None, :] & (k_pos[None, None, :] <= positions[:, :, None])
        if attn.window is not None:
            allowed &= k_pos[None, None, :] > positions[:, :, None] - attn.window
        if Hkv != H:
            k_all = k_all.repeat_interleave(H // Hkv, dim=1)
            v_all = v_all.repeat_interleave(H // Hkv, dim=1)
        y = F.scaled_dot_product_attention(
            q, k_all.to(q.dtype), v_all.to(q.dtype), attn_mask=allowed[:, None]
        )
        y = y.transpose(1, 2).reshape(B, T, H * D)
        if attn.gate is not None:
            y = y * torch.sigmoid(attn.gate(x))
        return attn.o_proj(y)
