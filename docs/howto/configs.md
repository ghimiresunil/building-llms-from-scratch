# Configure with Dynaconf

All settings live in TOML files under
[`configs/`](https://github.com/ghimiresunil/building-llms-from-scratch/tree/main/configs). One
Python module,
[`config/settings.py`](https://github.com/ghimiresunil/building-llms-from-scratch/blob/main/config/settings.py),
loads them with [Dynaconf](https://www.dynaconf.com/) and checks them. There are no defaults hidden in
Python: what you see in the TOML files is what a run uses.

```
configs/
  base.toml        # model shape + runtime, shared by every post-training stage
  pretrain.toml    # one file per stage, holding only that stage's settings
  sft.toml  reward.toml  dpo.toml  ppo.toml  grpo.toml
  presets.toml     # the model sizes for scripts/train_transformer.py --preset
  logging.toml     # log level and format, read by src/decorators
```

## `[default]` and its sub-tables

Every file has one `[default]` table listing **every** setting with its normal value. Variations
are sub-tables of it, named `[default.NAME]`, and they only list what they change:

```toml
# configs/sft.toml
[default]
pretrained_ckpt = "models/base_pretrained.pt"
lr = 1e-5
epochs = 3
batch_size = 16
max_steps = -1

[default.smoke]
pretrained_ckpt = "models/smoke/base_pretrained.pt"
max_steps = 10
batch_size = 4
```

| Sub-table | Picked with | What it's for |
|---|---|---|
| `[default.smoke]` | `--env smoke` or `BLFS_ENV=smoke` | A tiny model on the CPU, a handful of steps. Checks a stage in seconds. |
| `[default.test]` | `env="test"` in Python | The even smaller model the unit tests build. |
| `[default.tiny]`, `[default.13m]`, ... (in `presets.toml`) | `--preset tiny` | Model sizes for the classic pretraining script. |

Because `[default]` is the complete list, a key that isn't in it is treated as a typo:
`unknown key 'batchsize' for the sft stage (did you mean 'batch_size'?)`. A top-level `[smoke]`
table is rejected with a hint to write `[default.smoke]` instead.

TOML has no `null`, so optional settings are written as `"none"`, for example `n_kv_head = "none"`
or `amp_dtype = "none"`.

## The order things are applied

For a post-training stage, a later layer wins over an earlier one:

```
configs/base.toml          [default]
configs/<stage>.toml       [default]
configs/*.local.toml       your own tweaks (git-ignored), merged in by Dynaconf
[default.ENV]              from all of the above, when you pass --env ENV
BLFS_* environment vars    BLFS_LR=2e-5, also read from a .env file
command-line flags         --lr 2e-5
```

Dynaconf merges sub-tables across files instead of replacing them, so a
`[default.smoke]` table in `sft.local.toml` only changes the keys you write in it.

After merging, every value is converted to the type of its shipped default (so `--lr 2e-5` becomes
a float) and checked with Dynaconf validators: allowed choices such as `loss_type`, positive sizes,
and ranges such as `label_smoothing`. A few rules involve two settings at once, for example
`n_embed` must be divisible by `n_head`, and are checked right after. Errors name where the bad value
came from:

```text
configs/sft.local.toml: lr must be a number, got 'fast'
configs/sft.toml [default.smoke]: batch_size must be positive, got 0
environment variable BLFS_BETA: beta must be positive, got -1.0
```

## Three ways to change a value

=== "A local file"

    ```toml
    # configs/sft.local.toml  (git-ignored, merged automatically)
    [default]
    lr = 2e-5

    [default.smoke]
    batch_size = 2
    ```

=== "An environment variable"

    ```bash
    BLFS_LR=2e-5 BLFS_BATCH_SIZE=8 python scripts/train_sft.py
    # or the same lines in .env (see .env.example)
    ```

    `BLFS_*` variables are shared by every stage, so one that doesn't exist for the stage you're
    running (`BLFS_MAX_LEN` during SFT) is skipped.

=== "The command line"

    ```bash
    python scripts/train_sft.py --lr 2e-5 --batch_size 8
    ```

## Seeing the result

```bash
python scripts/train_sft.py --env smoke --print-config
```

prints the merged settings as a `[default]` TOML table and exits. The output is a valid settings
file you can save and pass back with `--config`.

## In Python

```python
from config.settings import load_stage

cfg = load_stage("sft", env="smoke", overrides={"lr": 2e-5})
cfg.lr            # attribute access
cfg.to_dict()     # a plain dict
```

The classic script's settings are `config.settings.default_config` (the `[default]` table of
`presets.toml`, plus the detected device), and `config.settings.PRESETS` holds the
`[default.NAME]` presets. To add a size of your own, put a `[default.mine]` table in
`configs/presets.local.toml` and run `python scripts/train_transformer.py --preset mine`.
Presets ignore `BLFS_*` variables, so `--preset 13m` always builds the same model.

## Logging settings

`configs/logging.toml` sets the log level, the message format and the logger's name for every
script. It follows the same rules as the other files: a `[default]` table, a `logging.local.toml`
for your own defaults, and `BLFS_LOG_LEVEL=DEBUG` (in any case) for a single run. An unknown level
fails with the usual message. It has no `[default.smoke]` table, so `--env smoke` simply keeps the
`[default]` logging.

```python
from config.settings import load_logging

load_logging().log_level   # "INFO", or whatever BLFS_LOG_LEVEL says
```
