<!-- omit in toc -->
# Context Extension: Linear, NTK and YaRN

Training at long context is expensive, because attention costs grow with the square of the length. So models are usually pretrained short and stretched afterwards. Llama 2 went from 4K to 32K this way, and Qwen and DeepSeek use YaRN to reach 128K. This page shows why naive stretching fails and what the three standard fixes do.

The code is in [`src/models/modern/rope.py`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/src/models/modern/rope.py). Read [Rotary Embeddings](rope.md) first.

## Why longer inputs break a RoPE model

RoPE rotates each pair of dimensions $i$ by the angle $m\,\theta_i$ at position $m$. The fast pairs (small $i$) spin many times within the training context, so the model has seen every angle they can take. The slow pairs turn less than once. At positions past the training length $L$, those slow pairs reach angles the model *never saw*, and attention scores become garbage.

## The three fixes

Each one changes the rotation speeds $\theta_i$ so that a context $s$ times longer maps back onto known angles. $s$ is `rope_factor`.

**Linear (position interpolation).** Divide every speed by $s$. Position $sL$ now has the angles position $L$ used to have. Simple, but it also slows the fast pairs, which encode the order of nearby words, so short-range precision suffers.

**NTK-aware.** Raise the base instead: $\theta' = \theta \cdot s^{d/(d-2)}$. The fastest pair is unchanged and the slowest is slowed by exactly $s$, with a smooth transition between them.

**YaRN.** Treat pairs by how often they rotate within the original context:

- pairs that turn more than `beta_fast` = 32 times are left alone;
- pairs that turn less than `beta_slow` = 1 time are fully interpolated (divided by $s$);
- pairs in between get a linear blend.

Stretching also makes attention distributions flatter, so YaRN sharpens them again by scaling both cos and sin by $0.1 \ln s + 1$.

```python
base = rope_frequencies(dim, theta)
ramp = yarn_ramp(dim, theta, original_context)        # 0 = keep, 1 = interpolate
freqs = base * (1.0 - ramp) + (base / factor) * ramp
return freqs, yarn_attention_factor(factor)          # 0.1 * ln(factor) + 1
```

## Using it

Three settings in `configs/base.toml` control it, for `arch = "modern"`:

```toml
rope_scaling = "yarn"           # "none" | "linear" | "ntk" | "yarn"
rope_factor = 4.0               # 4x the trained context
rope_original_context = 256     # what it was trained at; "none" = context_length / rope_factor
```

They are stored in every checkpoint, so later stages, `chat.py` and the evaluation scripts rebuild the same tables.

To see the effect without training anything, compare the methods on held-out text:

```bash
python scripts/eval_long_context.py --ckpt models/base_pretrained.pt \
    --data data/pile_dev.h5 --lengths 256 512 1024
```

It prints the mean loss at each length for every method. With `none` the loss usually climbs past the trained length. The other three hold it down, YaRN usually the most.

Stretching without any training already helps, but a short fine-tune at the new length helps much more. YaRN reports good results after a few hundred steps:

```bash
python scripts/pretrain_base.py --arch modern --context_length 1024 \
    --rope_scaling yarn --rope_factor 4 --rope_original_context 256 \
    --init_from models/base_pretrained.pt --train_steps 400 --warmup_steps 40 \
    --out_ckpt models/base_1k.pt
```

## References

- Chen et al. (2023). *Extending Context Window of Large Language Models via Positional Interpolation.*
- bloc97 (2023). *NTK-Aware Scaled RoPE.* (Reddit post that introduced the method.)
- Peng et al. (2023). *YaRN: Efficient Context Window Extension of Large Language Models.*
