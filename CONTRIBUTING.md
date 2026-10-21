# Contributing

Thanks for helping. This repo is a teaching project first, so the bar for a change is: does
it stay easy to read, and is it correct? Small, focused pull requests are the easiest to
review and merge.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). If you don't have it yet, the
[install page](https://docs.astral.sh/uv/getting-started/installation/) is one command.

```bash
git clone https://github.com/ghimiresunil/building-llms-from-scratch.git
cd building-llms-from-scratch
uv sync                          # .venv with the package (editable) + the dev tools
uv run pre-commit install        # the commit hooks and the commit-message check
uv run pytest -q -m "not slow"   # the fast tests, CPU only
uv run pytest -q                 # everything, including the end-to-end run of every script
```

Adding a dependency? Use `uv add <package>` (or `uv add --group dev <package>` for a tool) so
`pyproject.toml` and `uv.lock` change together, and commit both. CI fails if the lockfile is
out of date.

No uv? Plain pip still works: `python -m venv .venv`, activate it, then
`pip install -e . pytest beartype ruff mypy`.

On Linux the default PyTorch wheel bundles CUDA (about 2.5 GB). If you only need the CPU build,
see [docs/howto/hardware.md](docs/howto/hardware.md).

## Branches and commit messages

Commits straight to `main` (or `master`) are blocked by a pre-commit hook. Work on a branch and
open a pull request:

```bash
git switch -c feat/sliding-window-docs
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/), and the
scope is required:

```text
type(scope): what changed, in the imperative
```

The allowed types are `feat`, `fix`, `chore`, `refactor`, `docs`, `style`, `test`, `perf`, `ci`,
`build` and `revert`. A few examples:

```text
feat(grpo): add a --filter_groups flag for DAPO dynamic sampling
fix(settings): report the right file for errors in [default.smoke]
docs(readme): explain the BLFS_ environment variables
```

`updated stuff` and `feat: add config` (no scope) are both rejected.

## Pre-commit hooks

`uv run pre-commit install` sets up two git hooks. Before each commit they check for large files
(over 500 KB; `uv.lock` is exempt), merge-conflict markers, leftover `breakpoint()` calls, private
keys, invalid YAML/JSON/TOML and stray whitespace; keep `uv.lock` in sync; and run Ruff, which lints
and formats the code at 88 characters. When writing the commit message, they check its format.

If a hook fixes a file, the commit stops so you can look at the change: `git add` it and commit
again. To run everything by hand:

```bash
uv run pre-commit run --all-files
```

## Checks

CI also runs these three checks on Linux, Windows and macOS:

```bash
uv run ruff check .   # lint (pyflakes, pycodestyle, bugbear, pyupgrade, import order)
uv run mypy           # types, configured in pyproject.toml
uv run pytest -q      # tests, CPU only
```

Some guidelines that keep the code consistent:

- **No GPU needed for tests.** Use tiny models (a few thousand parameters) and generated data.
  Test a property when you can (causality, a cached forward equals a full one, a loss is zero
  at its target) rather than a hard-coded number.
- **Type everything new.** New functions have annotations; tensor arguments use
  [jaxtyping](https://github.com/patrick-kidger/jaxtyping) shapes such as
  `Float[Tensor, "batch seq vocab"]`. `tests/conftest.py` checks these shapes at runtime.
- **Settings live in TOML.** A new hyperparameter is a key in the `[default]` table of the
  matching `configs/<stage>.toml` (plus `[default.smoke]` if the tiny run needs a different
  value). If it has a fixed set of values, add them to `CHOICES` in `config/settings.py`; if it
  can be unset, write `"none"` and add it to `NULLABLE`.
- **One-line docstrings.** Every module, class and function gets a single-line docstring.
- **Never commit personal settings.** `*.local.toml` and `.env` are git-ignored on purpose.
- **Keep the classic model simple.** `src/models/` is what the README teaches line by line.
  New architecture ideas go in `src/models/modern/` or behind a flag.
- **Write like the README.** Plain words, short sentences, explain why before how.

## Pull requests

- One topic per pull request, with a short description of what changed and why.
- Add or update tests for the behavior you changed, and the docs if a command or flag changed.
- Commit messages follow the existing style: `fix(area): what was wrong`,
  `feat(area): what is new`, `docs: ...`.

## Reporting a bug

Please include the command you ran, the full error, and the output of:

```bash
uv run python -c "import torch, sys; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
```
