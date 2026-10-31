<!-- omit in toc -->
# Post-Training & Alignment: Overview

When I first trained this transformer from scratch, it could *continue* text but it couldn't
*follow instructions* or *reason*. That's what post-training fixes. This `docs/` folder walks
through the whole journey I built on top of the base model, every stage written from scratch
in plain PyTorch (no `trl`, no `peft`, no `transformers`), trained on real public datasets, and
runnable on a single GPU or scaled across multiple GPUs with DDP.

If you are new to LLM training internals, start with the
**[LLM Foundations](foundations/README.md)** section before reading the stage pages. It explains the
token shapes, decoder-only Transformer, attention masks, objectives, optimization loop, and generation
mechanics that every later page relies on.

Two newer sections sit next to the pipeline:

- **[Laptop Track](student/README.md)**: no GPU needed. Prepare TinyStories, train your own BPE
  tokenizer and a small model on a CPU in minutes, and generate stories with it.
- **[Modern LLM](modern/README.md)**: what changed between the 2017 Transformer and today's open
  models (RoPE, RMSNorm, SwiGLU, grouped-query and latent attention, Mixture of Experts, the KV
  cache), plus Muon, LoRA, the newer preference losses, the GRPO follow-ups and fast inference.
  Every stage of the pipeline can train either model.

## Recommended reading order

1. **Foundations first**:
   [Tokenization](foundations/tokenization.md) ->
   [Transformer](foundations/transformer.md) ->
   [Attention](foundations/attention.md) ->
   [Objectives](foundations/objectives.md) ->
   [Optimization](foundations/optimization.md) ->
   [Generation](foundations/generation.md).
2. **Then the full pipeline**:
   [Data](01_data_pipeline.md) ->
   [Pretraining](02_pretraining.md) ->
   [SFT](03_sft.md) ->
   [Reward Model](04_reward_model.md) ->
   [DPO](05_dpo.md) ->
   [PPO](06_ppo.md) ->
   [GRPO](07_grpo.md), then the extras:
   [Data curation](data_curation.md),
   [Distillation](10_distillation.md) and
   [Test-time compute](11_test_time_compute.md).
3. **Finally run and inspect**:
   [Evaluation](08_evaluation.md), [Inference / Chat](09_inference.md), and the
   [command cheatsheet](howto/commands.md).
4. **Then go modern**: [the modern model](modern/README.md), [optimizers](modern/optimizers.md),
   [preference losses](modern/preference.md), [RL for reasoning](modern/rl_reasoning.md),
   [LoRA](modern/lora.md), [fast inference](modern/inference.md),
   [context extension](modern/context_extension.md) and [paged serving](modern/paged_serving.md).

The pipeline mirrors how modern aligned/reasoning models are actually built:

![Post-training pipeline](diagrams/pipeline.svg)

## The stages, in order

| # | Stage | What it teaches the model | Doc |
|---|---|---|---|
| 1 | **Pretraining** | language itself (next-token prediction on the Pile) | [02_pretraining.md](02_pretraining.md) |
| 2 | **SFT** | to follow instructions & produce the `<think>/<answer>` format | [03_sft.md](03_sft.md) |
| 3 | **Reward Model** | to score which answer humans prefer | [04_reward_model.md](04_reward_model.md) |
| 4 | **DPO / ORPO / KTO** | to prefer better answers *without* an RL loop | [05_dpo.md](05_dpo.md) |
| 5 | **PPO** | to maximize a reward (RM or verifier) with the classic RLHF loop | [06_ppo.md](06_ppo.md) |
| 6 | **GRPO / RLVR** | to reason, using verifiable rewards (DeepSeek-R1 style) | [07_grpo.md](07_grpo.md) |
| 7 | **Distillation** | to imitate a bigger teacher's token distributions | [10_distillation.md](10_distillation.md) |
| | **Data curation** | which raw text to throw away before pretraining | [data_curation.md](data_curation.md) |
| | **Test-time compute** | pass@k, majority voting, rejection-sampling fine-tuning | [11_test_time_compute.md](11_test_time_compute.md) |
| | **Data pipeline** | how every dataset above is downloaded & preprocessed | [01_data_pipeline.md](01_data_pipeline.md) |
| | **Evaluation** | how I measure GSM8K accuracy across all stages | [08_evaluation.md](08_evaluation.md) |
| | **Inference / chat** | how to actually talk to any checkpoint | [09_inference.md](09_inference.md) |

## The one design rule: *wrap, don't rewrite*

Everything here sits on top of the original [`Transformer`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/src/models/transformer.py). I changed the
educational model in exactly **one** place: I added a [`forward_hidden`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/src/models/transformer.py#L68)
method that returns the final hidden states the `lm_head` consumes. Every post-training head (a value
head for PPO, a scalar reward head for the reward model) and every RL log-prob computation composes
*around* that one method, so the from-scratch model you already understand stays intact.

## Reading the diagrams

Every diagram uses the same colours and shapes, so once you know them you can read any page.

![How to read the diagrams](diagrams/legend.svg)

They are drawn from code: [diagrams/README.md](diagrams/README.md) explains how to change one.

## Run the whole thing

Once the base model has pretrained ([02_pretraining.md](02_pretraining.md)), the entire chain is one script:

```bash
bash scripts/run_posttraining.sh          # SFT -> RM -> DPO -> PPO -> GRPO -> eval table
```

See [POST_TRAINING.md](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/POST_TRAINING.md) for the condensed command reference.
