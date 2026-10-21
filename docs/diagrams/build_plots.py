"""Draw the docs' data plots as SVG from the repo's own code: run ``python docs/diagrams/build_plots.py``."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from svgkit import GRID, INK, MUTED, PAPER, ROLES  # noqa: E402

COBALT, TEAL, CRIMSON, MUSTARD, OLIVE = (
    ROLES[r][1] for r in ("model", "data", "loss", "reward", "result")
)
PLOTS: dict[str, Callable[[], plt.Figure]] = {}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Segoe UI",
            "Roboto",
            "Helvetica Neue",
            "Arial",
            "DejaVu Sans",
        ],
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
)


def plot(name: str) -> Callable[[Callable[[], plt.Figure]], Callable[[], plt.Figure]]:
    """Register a plot function under its file name."""

    def wrap(fn: Callable[[], plt.Figure]) -> Callable[[], plt.Figure]:
        PLOTS[name] = fn
        return fn

    return wrap


@plot("lr_schedules")
def lr_schedules() -> plt.Figure:
    """The three schedules from src/optim/schedules.py with one warmup and one peak."""
    from src.optim.schedules import lr_at

    steps = list(range(0, 10_000, 10))
    kw = dict(warmup_steps=500, max_steps=10_000, lr=3e-4, min_lr=3e-5)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, color in (("cosine", COBALT), ("wsd", MUSTARD), ("linear", TEAL)):
        ax.plot(
            steps,
            [lr_at(name, s, **kw) for s in steps],
            color=color,
            lw=2.2,
            label=f"--lr_schedule {name}",
        )
    ax.set_title("Three learning-rate schedules, same warmup and peak")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="lower left", bbox_to_anchor=(0.1, 0.02))
    ax.annotate(
        "WSD stays flat, so any checkpoint\nhere can still be decayed later",
        xy=(6200, 3e-4),
        xytext=(5000, 2.25e-4),
        color=MUTED,
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=MUTED),
    )
    return fig


@plot("rope")
def rope() -> plt.Figure:
    """How fast each RoPE pair rotates, from src/models/modern/rope.py."""
    import math

    from src.models.modern.rope import rope_cache, rope_frequencies

    n = 96
    cos, _ = rope_cache(dim=64, max_len=n)
    speed = rope_frequencies(64)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for pair, color in ((0, CRIMSON), (3, MUSTARD), (6, TEAL), (10, COBALT)):
        period = 2 * math.pi / float(speed[pair])
        style = dict(marker="o", ms=3, lw=1.2) if pair == 0 else dict(lw=2.2)
        ax.plot(
            range(n),
            cos[:, pair],
            color=color,
            label=f"pair {pair}: a turn every {period:.0f} positions",
            **style,
        )
    ax.set_title("RoPE: every pair of dimensions turns at its own speed")
    ax.set_xlabel("position in the sequence")
    ax.set_ylabel("cos(position x speed)")
    ax.legend(loc="lower left", ncol=2, fontsize=10)
    ax.set_ylim(-1.55, 1.2)
    return fig


@plot("kv_cache_memory")
def kv_cache_memory() -> plt.Figure:
    """Cache bytes per generated token for the post-training base size, measured with KVCache."""
    from config.settings import load_stage
    from src.models.modern import ModernConfig
    from src.models.modern.kv_cache import KVCache

    base = load_stage("pretrain").to_dict()
    known = {f for f in ModernConfig.__dataclass_fields__}
    rows = []
    for label, extra in (
        ("multi-head (16 KV heads)", {"n_kv_head": 16}),
        ("GQA, 4 KV heads", {"n_kv_head": 4}),
        ("multi-query, 1 KV head", {"n_kv_head": 1}),
        ("MLA latent", {"attention": "mla"}),
    ):
        cfg = ModernConfig(
            **{
                k: v
                for k, v in {**base, "arch": "modern", **extra}.items()
                if k in known
            }
        )
        if cfg.attention == "mla":
            heads, k_dim, v_dim = 1, cfg.latent_dim, cfg.mla_rope_dim
        else:
            heads, k_dim, v_dim = cfg.kv_heads, cfg.head_dim, cfg.head_dim
        cache = KVCache(cfg.n_blocks, 1, 1, heads, k_dim, v_dim, dtype=torch.bfloat16)
        rows.append((label, cache.bytes_per_token() / 1024))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = [MUTED, COBALT, TEAL, MUSTARD]
    ax.barh(
        [r[0] for r in rows][::-1],
        [r[1] for r in rows][::-1],
        color=colors[::-1],
        height=0.55,
    )
    for y, (_, kib) in enumerate(rows[::-1]):
        ax.text(kib + 2, y, f"{kib:.0f} KiB", va="center", color=INK, fontsize=10)
    ax.set_title(
        f"Cache per generated token ({base['n_blocks']} layers, n_embed {base['n_embed']}, bf16)"
    )
    ax.set_xlabel("KiB per token")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.18)
    return fig


@plot("attention_masks")
def attention_masks() -> plt.Figure:
    """Causal and sliding-window masks from src/models/modern/attention.py."""
    from matplotlib.colors import ListedColormap

    from src.models.modern.attention import attention_mask

    T = 12
    cpu = torch.device("cpu")
    causal_mask, is_causal = attention_mask(
        T, T, 0, None, cpu
    )  # the full-prompt case uses SDPA's own flag
    assert causal_mask is None and is_causal
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    window, _ = attention_mask(T, T, 0, 4, cpu)
    assert window is not None
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    cmap = ListedColormap([PAPER, COBALT])
    for ax, mask, title in (
        (axes[0], causal, "Causal"),
        (axes[1], window, "Sliding window, --sliding_window 4"),
    ):
        ax.imshow(mask.int(), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
        ax.set_xticks(range(0, T, 3))
        ax.set_yticks(range(0, T, 3))
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.suptitle(
        "Which earlier positions each query may read (filled = allowed)",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    return fig


@plot("muon_singular_values")
def muon_singular_values() -> plt.Figure:
    """Singular values of a gradient-like matrix before and after src/optim/muon.newton_schulz."""
    from src.optim.muon import newton_schulz

    torch.manual_seed(0)
    low_rank = torch.randn(256, 8) @ torch.randn(8, 512) * 3.0
    G = low_rank + 0.3 * torch.randn(256, 512)
    before = torch.linalg.svdvals(G)
    after = torch.linalg.svdvals(newton_schulz(G).float())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(
        before / before.max(),
        color=MUTED,
        lw=2.2,
        label="momentum, as is (scaled to 1)",
    )
    ax.plot(after, color=COBALT, lw=2.2, label="after 5 Newton-Schulz steps")
    ax.set_title("Muon lifts the weak directions of an update toward 1")
    ax.set_xlabel("singular value index, largest first")
    ax.set_ylabel("singular value")
    ax.legend(loc="upper right")
    return fig


def main() -> None:
    """Write every registered plot (or just the names given on the command line)."""
    for name in sys.argv[1:] or list(PLOTS):
        fig = PLOTS[name]()
        fig.tight_layout()
        out = HERE / f"{name}.svg"
        fig.savefig(out, facecolor=PAPER, metadata={"Date": None})
        # matplotlib leaves trailing spaces; strip them so the pre-commit hooks stay clean
        lines = out.read_text(encoding="utf-8").splitlines()
        out.write_text(
            "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
        )
        plt.close(fig)
        print(f"wrote docs/diagrams/{name}.svg")


if __name__ == "__main__":
    main()
