<!-- omit in toc -->
# Test-Time Compute and Rejection Sampling

Greedy decoding asks the model for its single best guess. But a model that's right 30% of the time on its first try is often right on *some* try out of eight. This chapter is about using that: spending more compute when answering, and turning the good samples into training data.

The helpers are in [`src/post_training/test_time.py`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/src/post_training/test_time.py), and [`scripts/rejection_sampling.py`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/scripts/rejection_sampling.py) runs everything on GSM8K.

## Three numbers

Sample $n$ answers per question with temperature, and count how many, $c$, are correct.

**pass@k** asks: if you drew $k$ of those samples, would at least one be right? The unbiased estimate (Chen et al., 2021) is

$$
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

pass@1 is the plain accuracy. pass@$n$ is a ceiling: no way of choosing among the samples can beat it.

**maj@k** (self-consistency, Wang et al., 2022) takes the most common final answer. It needs no verifier, so it works even when you don't know the right answer. Different wrong reasoning paths tend to land on different wrong numbers, while correct ones agree.

**best-of-N** scores each sample with a verifier or a reward model and keeps the top one. With a perfect verifier it reaches pass@$n$.

```bash
python scripts/rejection_sampling.py --ckpt models/sft.pt --split test --limit 200 \
    --n_samples 8 --no_write
# prints pass@1, pass@8 and maj@8 for the 200 questions
```

The gap between pass@1 and pass@8 is exactly the room reinforcement learning tries to close: GRPO moves probability toward answers the model can already find sometimes.

## Rejection-sampling fine-tuning

The simplest way to close that gap needs no RL at all. Keep only the samples the verifier accepts, then run ordinary SFT on them (RFT, Yuan et al. 2023; STaR, Zelikman et al. 2022):

```bash
python scripts/rejection_sampling.py --ckpt models/sft.pt --n_samples 8 --keep correct
python scripts/train_sft.py --pretrained_ckpt models/sft.pt \
    --data_path data/rft_packed.h5 --dev_path data/rft_dev_packed.h5 --out_ckpt models/rft.pt
python scripts/eval_post_training.py --ckpt models/rft.pt --label rft --append logs/table.jsonl
```

`--max_per_question` (default 4) caps and deduplicates the kept answers, so the easy questions, which produce many correct samples, don't drown out the hard ones.

Put `rft` in your evaluation table next to `grpo`. RFT is the baseline any RL method should beat. If GRPO doesn't, something in the RL setup is wrong.

## Sequence-level distillation

With `--keep all`, nothing is filtered: you get every answer a model wrote. Run it on a big teacher and fine-tune a small student on the result, and you have sequence-level distillation:

```bash
python scripts/rejection_sampling.py --ckpt models/teacher.pt --n_samples 1 --keep all --prefix distill
```

See [Knowledge Distillation](10_distillation.md) for the token-level version.

## References

- Chen et al. (2021). *Evaluating Large Language Models Trained on Code* (pass@k).
- Wang et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.*
- Zelikman et al. (2022). *STaR: Bootstrapping Reasoning With Reasoning.*
- Yuan et al. (2023). *Scaling Relationship on Learning Mathematical Reasoning with Large Language Models* (RFT).
