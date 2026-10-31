<!-- omit in toc -->
# Knowledge Distillation

Most of the small open models you can download weren't trained only on text. They were trained to imitate a bigger model. That's distillation, and it's one of the cheapest ways to make a small model better than its size suggests.

The code is [`src/post_training/distill.py`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/src/post_training/distill.py) and the trainer is [`scripts/train_distill.py`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/scripts/train_distill.py).

## Why a teacher helps

A normal training example says "the next token is *cat*", and nothing else. A trained teacher says much more: *cat* 70%, *kitten* 20%, *dog* 5%, and almost nothing for *carburetor*. Hinton called that extra information "dark knowledge". It tells the student which mistakes are reasonable, so every token carries more signal than a one-hot label.

The student gets two losses, mixed by `distill_alpha`:

$$
\mathcal{L} = \alpha \, \tau^2 \, D\big(p_T^{(\tau)},\, p_S^{(\tau)}\big) + (1 - \alpha)\, \mathrm{CE}(p_S, y)
$$

$\tau$ is the temperature (`distill_temperature`). Dividing the logits by $\tau > 1$ flattens both distributions, so the small probabilities, where the dark knowledge lives, actually show up in the loss. The $\tau^2$ keeps the gradient the same size whatever temperature you pick.

## Three choices of divergence

| `distill_loss` | Divergence | Behavior |
|---|---|---|
| `forward_kl` | $\mathrm{KL}(p_T \,\|\, p_S)$ | Mode covering: the student must put mass everywhere the teacher does. The classic choice. |
| `reverse_kl` | $\mathrm{KL}(p_S \,\|\, p_T)$ | Mode seeking: the student is punished for mass where the teacher has none, so it commits to the teacher's main answers. Often better when the student is much smaller (MiniLLM). |
| `jsd` | generalized Jensen-Shannon, weight `jsd_beta` | Bounded by $\ln 2$, so a confidently wrong student can't blow up the loss. `jsd_beta` near 0 acts like forward KL, near 1 like reverse KL (GKD). |

Here is the core of it:

```python
log_s = F.log_softmax(student_logits / temperature, dim=-1)
log_t = F.log_softmax(teacher_logits / temperature, dim=-1)
if kind == "forward_kl":
    div = _kl(log_t, log_s)          # sum p_t * (log p_t - log p_s)
elif kind == "reverse_kl":
    div = _kl(log_s, log_t)
return div * temperature**2
```

## Running it

The student's size comes from `configs/base.toml`, like every other stage. The teacher is rebuilt from the settings stored in its own checkpoint, so it can be any size. The only requirement is the same tokenizer, which the script checks.

```bash
# distill an SFT teacher into a smaller student, on the SFT data (loss on the replies only)
python scripts/train_distill.py --teacher_ckpt models/sft_big.pt --student_ckpt models/base_pretrained.pt

# distill during pretraining instead, on raw text
python scripts/train_distill.py --data_format pretrain \
    --data_path data/pile_train.h5 --dev_path data/pile_dev.h5 --student_ckpt none

# try reverse KL or JSD
python scripts/train_distill.py --distill_loss reverse_kl
python scripts/train_distill.py --distill_loss jsd --jsd_beta 0.5
```

`--env smoke` gives a tiny run that finishes in seconds. The log shows the divergence (`div`) and the plain cross-entropy (`ce`) separately, so you can see which one is moving.

## Things to watch

**The teacher has to be better than the student would be on its own.** Distillation pulls the student toward the teacher. If the teacher is undertrained, it pulls the student down. Always run the same student with `--distill_alpha 0` (plain training) as a baseline.

**Teacher forward passes cost compute.** Every student step also runs the teacher, with no gradients. That's usually cheaper than training the student for more steps, but it isn't free.

**Sequence-level distillation needs no new loss.** Instead of matching distributions token by token, let the teacher *write* answers and fine-tune the student on them (Kim & Rush, 2016). [`scripts/rejection_sampling.py --keep all`](11_test_time_compute.md) generates that data, and `train_sft.py` does the rest.

## References

- Hinton, Vinyals & Dean (2015). *Distilling the Knowledge in a Neural Network.*
- Kim & Rush (2016). *Sequence-Level Knowledge Distillation.*
- Gu et al. (2023). *MiniLLM: Knowledge Distillation of Large Language Models.*
- Agarwal et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (GKD).
