<!-- omit in toc -->
# Paged KV Cache and Continuous Batching

[Fast Inference](inference.md) made *one* sequence fast with a KV cache. Serving many users at once raises two new problems, and the two ideas on this page, from vLLM (Kwon et al., 2023), solve them. The code is [`src/inference/paged.py`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/src/inference/paged.py).

## Problem 1: the cache wastes memory

`KVCache` reserves `context_length` slots for every sequence before generating anything, because it can't know how long the answer will be. Most answers are far shorter, so most of that memory sits empty, and memory is what limits how many requests fit on a GPU.

**Paging** borrows the idea from operating systems. The cache becomes a pool of fixed-size *blocks*, say 16 tokens each. Every sequence owns a *block table*, a list of block ids, and takes a new block only when its last one is full. At most one partly filled block per sequence is wasted.

```text
pool:      [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ][ 7 ] ...
request A:  table = [5, 1]      tokens 0-15 in block 5, 16-20 in block 1
request B:  table = [2]         tokens 0-9 in block 2
```

Token $m$ of a sequence lives in block `table[m // block_size]` at offset `m % block_size`. That's the whole address calculation (`PagedKVCache.slots`).

## Problem 2: batches finish together

With a plain batch, every request waits for the longest one, and nothing new can start until the batch is done. **Continuous batching** schedules per step instead:

1. finished requests leave and return their blocks to the pool;
2. waiting requests join as soon as there's a free batch slot and enough blocks for their prompt (they get a *prefill*);
3. every running request decodes one token, all in one batched forward pass.

Because requests in a batch now have different lengths, each row gets its own positions (for RoPE) and its own attention mask.

## When the pool runs dry

A request might need a new block when none are free. The engine then *preempts* the newest running request: frees its blocks and puts it back at the front of the queue. When it is readmitted, its prompt and everything it generated so far are prefilled again (recomputation). The result is identical; only time is lost. The tests check exactly that: with a tiny pool that forces preemptions, every output still matches plain greedy decoding token for token.

## What a real engine does differently

Here, each layer gathers a sequence's blocks into a padded tensor and calls `scaled_dot_product_attention`. vLLM's *PagedAttention* kernel reads the blocks directly, never building that copy. The bookkeeping, which is the part worth understanding, is the same. Production engines add prefix sharing (several requests pointing at the same blocks), chunked prefill and swapping to CPU memory.

## Try it

```bash
python scripts/benchmark_paged.py                        # random model, runs on a CPU
python scripts/benchmark_paged.py --ckpt models/sft.pt   # any modern GQA checkpoint
```

It prints tokens per second for one-request-at-a-time generation and for the engine, the mean decode batch size, the number of preemptions, and the KV memory a contiguous cache would reserve against what the paged cache actually used. Shrink `--num_blocks` to watch preemption kick in.

```python
from src.inference.paged import PagedEngine

engine = PagedEngine(model, num_blocks=256, block_size=16, max_batch=16, top_p=0.95)
ids = [engine.add_request(prompt_ids, max_new_tokens=200) for prompt_ids in prompts]
outputs = engine.run()            # {request id: generated token ids}
print(engine.stats.mean_batch, engine.stats.preemptions, engine.stats.peak_blocks)
```

It supports grouped-query attention (including multi-query and plain multi-head), sliding windows, MoE and scaled RoPE. MLA stores a latent rather than keys and values, so it isn't supported.

## References

- Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.*
- Yu et al. (2022). *Orca: A Distributed Serving System for Transformer-Based Generative Models* (iteration-level scheduling).
