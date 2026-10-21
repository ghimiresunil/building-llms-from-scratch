"""Draw every diagram in docs/diagrams as SVG: run ``python docs/diagrams/build_diagrams.py``."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from svgkit import ROLES, Diagram  # noqa: E402

OUT = Path(__file__).resolve().parent
DIAGRAMS: dict[str, Callable[[], Diagram]] = {}


def diagram(name: str) -> Callable[[Callable[[], Diagram]], Callable[[], Diagram]]:
    """Register a diagram function under its file name."""

    def wrap(fn: Callable[[], Diagram]) -> Callable[[], Diagram]:
        DIAGRAMS[name] = fn
        return fn

    return wrap


@diagram("pipeline")
def pipeline() -> Diagram:
    """The whole journey, from raw text to an evaluated reasoning model."""
    d = Diagram(
        1200,
        520,
        "From raw text to a reasoning model",
        "One Transformer backbone all the way through. Each stage changes the data and the loss, not the model.",
    )
    text = d.card(120, 60, "Raw text", "data", sub="The Pile, TinyStories", stack=True)
    tok = d.card(310, 60, "Tokenizer", "model", sub="text to ids")
    pre = d.card(510, 60, "Pretraining", "model", sub="next-token loss")
    base = d.card(710, 60, "Base model", "result", sub="continues text")
    sft = d.card(910, 60, "SFT", "model", sub="answers in chat format")
    d.arrow(text, tok)
    d.arrow(tok, pre, "ids", label_dy=-10)
    d.arrow(pre, base)
    d.arrow(base, sft)

    d.group(250, 150, 760, 260, "Alignment: use one or several", "reward")
    lanes = (215, 290, 365)
    rm = d.card(
        470, lanes[0], "Reward model", "reward", sub="learns to score answers", w=210
    )
    ppo = d.card(760, lanes[0], "PPO", "model", sub="RL with a value head", w=210)
    dpo = d.card(
        615, lanes[1], "DPO family", "model", sub="DPO, IPO, SimPO, ORPO, KTO", w=500
    )
    grpo = d.card(
        615,
        lanes[2],
        "GRPO",
        "model",
        sub="RL against a math checker, no value head",
        w=500,
    )
    d.arrow(rm, ppo, "scores", label_dy=-10)
    bus_in, bus_out = 300, 935
    d.arrow(sft, (bus_in, lanes[0]), via=[(910, 125), (bus_in, 125)], head=False)
    d.arrow((bus_in, lanes[0]), (bus_in, lanes[2]), head=False)
    for lane, node in zip(lanes, (rm, dpo, grpo)):
        d.arrow((bus_in, lane), node, into="l")
    d.arrow((bus_out, lanes[0]), (bus_out, lanes[2]), head=False)
    for lane, node in zip(lanes, (ppo, dpo, grpo)):
        d.arrow(node, (bus_out, lane), out="r", head=False)
    ev = d.card(
        1105, lanes[1], "Evaluate\nand chat", "result", sub="GSM8K accuracy", w=150
    )
    d.arrow((bus_out, lanes[1]), ev, into="l")
    for node, k in ((text, 1), (tok, 2), (pre, 3), (sft, 4), (ev, 6)):
        d.badge(node, k)
    d.badge(rm, 5)
    return d


@diagram("tokenization")
def tokenization() -> Diagram:
    """Text to pieces to ids, computed with the repo's own BPE tokenizer."""
    from src.tokenizer.bpe import BPETokenizer

    corpus = (
        "The cat sat on the mat. The dog sat on the log. A cat and a dog sat together. "
        "Then the cat sat on the mat again, and the dog sat by the door. "
    ) * 20
    bpe = BPETokenizer.train(corpus, 300)
    sentence = "The cat sat on the mat."
    ids = bpe.encode_ordinary(sentence)
    pieces = [
        bpe.token_bytes(i).decode("utf-8", "replace").replace(" ", "\u2423")
        for i in ids
    ]

    d = Diagram(
        1100,
        430,
        "Text becomes numbers",
        "The tokenizer splits text into pieces it learned from data, then swaps each piece for its id.",
    )
    gap = 118
    src = d.card(
        550, 40, f"\u201c{sentence}\u201d", "data", w=gap * len(pieces) - 10, bold=False
    )
    start = 550 - gap * (len(pieces) - 1) / 2
    for k, (piece, i) in enumerate(zip(pieces, ids)):
        x = start + gap * k
        pc = d.chip(x, 150, piece, "data", w=96, size=15)
        idc = d.chip(x, 250, str(i), "model", w=70, size=15)
        d.arrow((x, 40 + src.h / 2), pc, into="t", color="#8A96AB")
        d.arrow(pc, idc, into="t")
    d.label(70, 155, "pieces", 13, anchor="start", weight=600)
    d.label(70, 255, "ids", 13, anchor="start", weight=600)
    d.label(
        550,
        300,
        f"\u2423 marks a space. Vocabulary: {bpe.n_vocab} entries, trained with src/tokenizer/bpe.py "
        "on a few sentences.",
        13,
    )
    return d


@diagram("transformer_block")
def transformer_block() -> Diagram:
    """One pre-norm Transformer block with its two residual paths."""
    d = Diagram(
        820,
        830,
        "One Transformer block",
        "Normalise, transform, add back. Twice: once to mix positions, once to think per position.",
    )
    cx, skip = 410, 230
    x_in = d.card(cx, 20, "x", "data", sub="from the block below", w=200)
    ln1 = d.card(cx, 125, "LayerNorm", "plain", pill=True, w=150, size=14)
    attn = d.card(cx, 210, "Multi-head attention", "model", sub="causal mask", w=240)
    p1 = d.plus(cx, 295)
    ln2 = d.card(cx, 380, "LayerNorm", "plain", pill=True, w=150, size=14)
    mlp = d.card(cx, 465, "MLP", "model", sub="4x wider inside, ReLU", w=240)
    p2 = d.plus(cx, 550)
    out = d.card(cx, 635, "x", "result", sub="to the next block", w=200)
    for a_, b_ in (
        (x_in, ln1),
        (ln1, attn),
        (attn, p1),
        (p1, ln2),
        (ln2, mlp),
        (mlp, p2),
        (p2, out),
    ):
        d.arrow(a_, b_)
    d.label(cx + 14, 80, "(B, T, C)", 12, mono=True, anchor="start")
    d.label(cx + 14, 600, "(B, T, C)", 12, mono=True, anchor="start")
    d.arrow(
        (cx, 72),
        p1,
        via=[(skip, 72), (skip, 295)],
        into="l",
        label="skip",
        label_at=0.45,
        mono=False,
    )
    d.arrow(
        (cx, 330),
        p2,
        via=[(skip, 330), (skip, 550)],
        into="l",
        label="skip",
        label_at=0.5,
        mono=False,
    )
    d.label(580, 205, "each position reads", 13, anchor="start")
    d.label(580, 223, "the ones before it", 13, anchor="start")
    d.label(580, 460, "each position on", 13, anchor="start")
    d.label(580, 478, "its own", 13, anchor="start")
    d.label(
        cx,
        705,
        "x = x + attn(ln1(x));   x = x + mlp(ln2(x))",
        14,
        mono=True,
        color="#1F2A44",
    )
    return d


@diagram("training_loop")
def training_loop() -> Diagram:
    """The pretraining loop: batch, forward, loss, backward, step."""
    d = Diagram(
        1140,
        490,
        "The pretraining loop",
        "Predict every next token, measure the miss, nudge the weights. Repeat for train_steps.",
        top=118,
    )
    data = d.card(
        120, 60, "Token file", "data", sub="one long array of ids", stack=True
    )
    batch = d.card(350, 60, "Random windows", "data", sub="batch_size x context")
    model = d.card(610, 60, "Transformer", "model", sub="forward pass")
    loss = d.card(900, 60, "Cross-entropy", "loss", sub="vs the next token")
    back = d.card(
        900, 250, "Backward", "plain", pill=True, sub="loss.backward()", w=180
    )
    step = d.card(610, 250, "Optimizer step", "model", sub="AdamW or Muon")
    lr = d.card(
        350, 250, "Learning rate", "plain", pill=True, sub="warmup, then decay", w=200
    )
    d.arrow(data, batch)
    d.arrow(batch, model, "(B, T)", label_dy=-10)
    d.arrow(model, loss, "(B, T, V) logits", label_dy=-10)
    d.arrow(
        (350, 60 - 30),
        (900, 60 - 30),
        via=[(350, 0), (900, 0)],
        label="targets: the same window shifted by one",
        color="#2B7A76",
        dashed=True,
    )
    d.arrow(loss, back, "one number", label_dy=4)
    d.arrow(back, step, "gradients", label_dy=-10)
    d.arrow(lr, step)
    d.arrow(step, model, "new weights", out="t", into="b", dashed=True, label_dy=4)
    d.label(
        610,
        340,
        "Every eval_steps, the same loss on the dev file tells you if the model generalises.",
        13,
    )
    return d


@diagram("reward_model")
def reward_model() -> Diagram:
    """Bradley-Terry reward model training on preference pairs."""
    d = Diagram(
        1140,
        430,
        "Training a reward model",
        "Both answers go through the same network; the loss only cares which one scores higher.",
    )
    chosen = d.card(
        150, 50, "Prompt + chosen", "data", sub="the answer people preferred", w=250
    )
    rejected = d.card(
        150, 230, "Prompt + rejected", "data", sub="the other answer", w=250
    )
    d.card(
        470,
        140,
        "SFT backbone\n+ reward head",
        "model",
        sub="shared weights",
        w=220,
        h=150,
    )
    rc = d.chip(730, 50, "r chosen", "reward", size=14)
    rr = d.chip(730, 230, "r rejected", "reward", size=14)
    loss = d.card(
        960,
        140,
        "Bradley-Terry loss",
        "loss",
        sub="-log \u03c3(r chosen - r rejected)",
        w=260,
    )
    d.arrow(chosen, (360, 90), into="l", via=[(320, 50), (320, 90)])
    d.arrow(rejected, (360, 190), into="l", via=[(320, 230), (320, 190)])
    d.arrow((580, 90), rc, via=[(640, 90), (640, 50)], into="l")
    d.arrow((580, 190), rr, via=[(640, 190), (640, 230)], into="l")
    d.arrow(rc, loss, into="t", out="r")
    d.arrow(rr, loss, into="b", out="r")
    d.label(470, 245, "reads each answer at its last real token", 12)
    d.label(
        570,
        320,
        "Accuracy on held-out pairs: how often r chosen > r rejected. 50% is a coin flip.",
        13,
    )
    return d


@diagram("grpo")
def grpo() -> Diagram:
    """GRPO: sample a group, score it, push answers above the group mean up."""
    d = Diagram(
        1200,
        450,
        "GRPO: learn from a group of tries",
        "No value network. Each answer is judged against the other answers to the same question.",
    )
    prompt = d.card(105, 60, "Math question", "data", sub="from GSM8K")
    policy = d.card(330, 60, "Policy", "model", sub="the model being trained")
    group = d.card(610, 60, "G answers", "data", sub="same question", stack=True)
    check = d.card(
        840, 60, "Answer checker", "reward", sub="right number? right format?"
    )
    rewards = d.card(1075, 60, "Rewards", "reward", sub="one per answer", w=160)
    adv = d.card(1075, 250, "Advantages", "reward", sub="(r - group mean) / std", w=200)
    loss = d.card(720, 250, "Clipped loss + KL", "loss", sub="like PPO, per token")
    ref = d.card(330, 250, "Reference model", "frozen", sub="the SFT model, frozen")
    d.arrow(prompt, policy)
    d.arrow(policy, group, "sample G times", label_dy=-12, mono=False)
    d.arrow(group, check)
    d.arrow(check, rewards)
    d.arrow(rewards, adv)
    d.arrow(adv, loss)
    d.arrow(ref, loss, "KL penalty", label_dy=-10, mono=False)
    d.arrow(
        loss,
        policy,
        out="t",
        into="b",
        via=[(720, 160), (330, 160)],
        dashed=True,
        label="update",
        label_at=0.5,
        label_dy=-8,
        mono=False,
    )
    return d


@diagram("data_pipeline")
def data_pipeline() -> Diagram:
    """Four kinds of training data, each with its own script and file format."""
    d = Diagram(
        1180,
        490,
        "Four kinds of training data",
        "Every stage reads integers from disk; only the shape of the records changes.",
        top=122,
    )
    rows = [
        (
            "The Pile",
            "plain web text",
            "prepare_pretrain_data.py",
            "pile_train.h5",
            "one flat array of ids",
            "Pretraining",
        ),
        (
            "Alpaca, Dolly, GSM8K",
            "instructions + answers",
            "prepare_sft_data.py",
            "sft_packed.h5",
            "rows of ids + loss mask",
            "SFT",
        ),
        (
            "HH-RLHF, UltraFeedback",
            "two answers, one preferred",
            "prepare_preference_data.py",
            "preferences.jsonl",
            "prompt, chosen, rejected",
            "Reward model, DPO",
        ),
        (
            "GSM8K",
            "questions with answers",
            "prepare_rl_prompts.py",
            "rl_prompts_train.jsonl",
            "prompt, gold",
            "PPO, GRPO",
        ),
    ]
    for k, (src, src_sub, script, file, file_sub, stage) in enumerate(rows):
        y = 30 + 88 * k
        a = d.card(140, y, src, "data", sub=src_sub, w=230)
        b = d.card(430, y, script, "plain", pill=True, w=250, size=14)
        c = d.card(735, y, file, "data", sub=file_sub, stack=True, w=240)
        e = d.card(1035, y, stage, "model", w=190)
        d.arrow(a, b)
        d.arrow(b, c)
        d.arrow(c, e)
    for x, text in (
        (140, "public dataset"),
        (430, "script in scripts/"),
        (735, "file in data/"),
        (1035, "used by"),
    ):
        d.label(x, -10, text, 13, weight=600)
    return d


@diagram("sft")
def sft() -> Diagram:
    """SFT: the same next-token loss, counted only on the assistant's tokens."""
    d = Diagram(
        1180,
        450,
        "Supervised fine-tuning",
        "Train on conversations, but only grade the model on the words the assistant says.",
    )
    conv = d.card(130, 50, "Conversation", "data", sub="user turn + assistant turn")
    tmpl = d.card(
        370, 50, "Chat template", "plain", pill=True, sub="role markers", w=190
    )
    d.arrow(conv, tmpl)
    d.label(470, 18, "tokens", 13, anchor="start", weight=600)
    d.label(470, 63, "mask", 13, anchor="start", weight=600)
    tokens = ["user", "What", "is", "2+2", "?", "asst", "It", "is", "4", ".", "end"]
    graded = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    for k, (t, g) in enumerate(zip(tokens, graded)):
        x = 575 + 54 * k
        role = "result" if g else "data"
        d.chip(x, 12, t, role, w=48, size=13)
        d.chip(x, 57, str(g), role, w=48, size=13)
    d.label(710, 100, "user turn and role markers: not graded", 12)
    d.label(980, 100, "the answer and its end: graded", 12)
    d.arrow(tmpl, (545, 35), into="l")
    model = d.card(370, 230, "Transformer", "model", sub="starts from the base model")
    ce = d.card(720, 230, "Cross-entropy per token", "loss", w=260)
    mean = d.card(
        1060, 230, "Masked mean", "loss", sub="sum(ce x mask) / sum(mask)", w=220
    )
    d.arrow((370, 75), model, out="b", into="t", label="(B, T) ids", label_dy=4)
    d.arrow(model, ce, "(B, T, V)", label_dy=-10)
    d.arrow(ce, mean, "(B, T)", label_dy=-10)
    d.arrow((1115, 75), (1115, 230 - 30), label="mask", mono=False, label_dy=4)
    return d


@diagram("dpo")
def dpo() -> Diagram:
    """DPO: compare the policy with a frozen reference on both answers."""
    d = Diagram(
        1180,
        470,
        "Direct preference optimisation",
        "No reward model and no sampling: just make the preferred answer more likely than before.",
    )
    pair = d.card(
        120, 160, "Preference pair", "data", sub="prompt, chosen, rejected", w=210
    )
    pol = d.card(420, 60, "Policy", "model", sub="being trained")
    ref = d.card(420, 260, "Reference", "frozen", sub="the SFT model, frozen")
    d.arrow(pair, pol, out="r", into="l")
    d.arrow(pair, ref, out="r", into="l")
    p_c = d.chip(680, 30, "log p(chosen)", "model", size=13, w=150)
    p_r = d.chip(680, 90, "log p(rejected)", "model", size=13, w=150)
    r_c = d.chip(680, 230, "log p(chosen)", "frozen", size=13, w=150)
    r_r = d.chip(680, 290, "log p(rejected)", "frozen", size=13, w=150)
    for a_, b_ in ((pol, p_c), (pol, p_r), (ref, r_c), (ref, r_r)):
        d.arrow(a_, b_, out="r", into="l")
    gap = d.card(
        930,
        160,
        "Margin",
        "reward",
        sub="how much more the policy\nprefers chosen than ref does",
        w=240,
    )
    for c in (p_c, p_r, r_c, r_r):
        d.arrow(c, (810, 160), out="r", into="l", head=False, color="#8A96AB")
    d.arrow((810, 160), gap, into="l")
    loss = d.card(
        930,
        300,
        "-log \u03c3(\u03b2 x margin)",
        "loss",
        sub="--loss_type dpo, ipo, simpo, orpo, kto",
        mono_sub=True,
        w=290,
    )
    d.arrow(gap, loss, into="t")
    return d


@diagram("ppo")
def ppo() -> Diagram:
    """PPO: roll out, score, estimate advantages with a value head, take clipped steps."""
    d = Diagram(
        1200,
        500,
        "PPO: reinforcement learning with a critic",
        "Collect answers, score them, work out which tokens helped, then take small, clipped steps.",
    )
    prompts = d.card(100, 50, "Prompts", "data", sub="GSM8K questions")
    actor = d.card(
        320, 50, "Policy + value head", "model", sub="one shared backbone", w=230
    )
    roll = d.card(625, 50, "Rollouts", "data", sub="answers + log-probs", stack=True)
    score = d.card(870, 50, "Reward", "reward", sub="checker or reward model")
    kl = d.card(
        870, 220, "Per-token KL", "frozen", sub="vs the frozen SFT model", dashed=True
    )
    gae = d.card(1090, 135, "GAE", "reward", sub="advantages per token", w=170)
    loss = d.card(625, 300, "Clipped policy loss\n+ value loss", "loss", w=240)
    d.arrow(prompts, actor)
    d.arrow(actor, roll, "generate", mono=False, label_dy=-12)
    d.arrow(roll, score)
    d.arrow(roll, kl, out="b", into="l")
    d.arrow(score, gae, out="r", into="t")
    d.arrow(
        kl,
        gae,
        out="r",
        into="b",
        label="penalty",
        mono=False,
        label_at=0.3,
        label_dy=-8,
    )
    d.arrow(gae, loss, out="b", into="r", via=[(1090, 300)])
    d.arrow(
        loss,
        actor,
        out="l",
        into="b",
        via=[(320, 300)],
        dashed=True,
        label="ppo_epochs updates",
        mono=False,
        label_at=0.25,
        label_dy=-8,
    )
    d.label(
        625,
        375,
        "Clipping keeps the new policy within a small ratio of the one that wrote the answers.",
        13,
    )
    return d


@diagram("evaluation")
def evaluation() -> Diagram:
    """One benchmark, the same decoding, every checkpoint."""
    d = Diagram(
        1180,
        400,
        "Scoring every stage the same way",
        "Greedy answers to held-out GSM8K questions, checked by the same function the RL stages use.",
    )
    names = ["base_pretrained", "sft", "dpo", "ppo", "grpo"]
    for k, n in enumerate(names):
        d.chip(125, 10 + 52 * k, f"{n}.pt", "result", w=170, size=13)
    gen = d.card(400, 115, "Greedy decoding", "model", sub="--limit questions")
    parse = d.card(
        650,
        115,
        "Extract the answer",
        "plain",
        pill=True,
        sub="<answer> tag, then fallbacks",
        w=250,
    )
    cmp_ = d.card(865, 115, "Match gold?", "reward", sub="1 or 0", w=150)
    table = d.card(
        1060,
        115,
        "One row per stage",
        "result",
        sub="logs/table.jsonl",
        mono_sub=True,
        w=200,
    )
    for k in range(5):
        d.arrow(
            (210, 10 + 52 * k),
            (310, 115),
            via=[(255, 10 + 52 * k), (255, 115)],
            head=k == 2,
            color="#8A96AB" if k != 2 else "#1F2A44",
        )
    d.arrow(gen, parse)
    d.arrow(parse, cmp_)
    d.arrow(cmp_, table)
    d.label(
        590,
        270,
        "Small models score low in absolute terms; the story is how the number moves.",
        13,
    )
    return d


@diagram("inference")
def inference() -> Diagram:
    """Chatting with a checkpoint: the model rebuilds itself from the file."""
    d = Diagram(
        1180,
        420,
        "Talking to a checkpoint",
        top=118,
        caption="The checkpoint stores its own settings, so chat.py needs nothing but the file.",
    )
    ck = d.card(110, 60, "Checkpoint", "result", sub="weights + cfg", stack=True)
    build = d.card(330, 60, "Rebuild the model", "model", sub="classic or modern")
    tmpl = d.card(
        580,
        60,
        "Chat template",
        "plain",
        pill=True,
        sub="your message + <|assistant|>",
        w=260,
    )
    loop = d.card(860, 60, "Next-token loop", "model", sub="temperature, top-p, top-k")
    reply = d.card(860, 240, "Reply", "result", sub="decoded until <|endoftext|>")
    d.arrow(ck, build)
    d.arrow(build, tmpl)
    d.arrow(tmpl, loop, "ids", label_dy=-10)
    d.arrow(loop, reply, "ids", label_dy=4)
    d.arrow(
        loop,
        (860, 60 - loop.h / 2),
        out="r",
        via=[(1040, 60), (1040, -5), (860, -5)],
        label="append the new token, repeat",
        mono=False,
        label_at=0.5,
        label_dy=-8,
    )
    d.label(
        400,
        240,
        "--raw skips the template, for a base model that only continues text.",
        13,
    )
    return d


@diagram("student_track")
def student_track() -> Diagram:
    """The laptop track in three commands, with the real model size."""
    d = Diagram(
        1240,
        300,
        "A language model on a laptop",
        "Three commands, no GPU. Model sizes from scripts/model_report.py.",
    )
    a = d.card(
        185,
        70,
        "prepare_tiny_data.py",
        "data",
        sub="TinyStories + a 4,096-token BPE",
        w=290,
    )
    b = d.card(
        625,
        70,
        "train_transformer.py --preset tiny",
        "model",
        sub="369K parameters with --arch modern",
        w=380,
    )
    c = d.card(
        1065, 70, "generate_text.py", "result", sub="writes a short story", w=240
    )
    d.arrow(a, b, "data/tiny/*.h5", label_dy=32)
    d.arrow(b, c, "models/tiny.pt", label_dy=32)
    for n, k in ((a, 1), (b, 2), (c, 3)):
        d.badge(n, k)
    d.label(
        625,
        160,
        "Next sizes up: --preset student (2.6M) and --preset small (12.2M), modern architecture.",
        13,
    )
    return d


@diagram("settings")
def settings() -> Diagram:
    """How a stage's settings are layered by Dynaconf and then checked."""
    d = Diagram(
        1180,
        610,
        "Where a setting comes from",
        "Dynaconf stacks the sources, later ones winning, then config/settings.py checks every value.",
        top=124,
    )
    layers = [
        ("configs/base.toml", "shared model + runtime"),
        ("configs/<stage>.toml", "that stage's settings"),
        ("--config my.toml", "optional, partial is fine"),
        ("*.local.toml", "your tweaks, git-ignored"),
        ("[default.smoke]", "with --env smoke"),
        ("BLFS_* variables", "or a .env file"),
        ("--field value", "the command line"),
    ]
    for k, (name, sub) in enumerate(layers):
        role = "data" if k < 5 else "plain"
        d.card(240, 15 + 62 * k, name, role, sub=sub, w=300, size=14)
    d.arrow((60, 0), (60, 15 + 62 * 6 + 10), color="#8A96AB")
    d.label(60, -14, "first", 12)
    d.label(60, 15 + 62 * 6 + 30, "last, wins", 12)
    checks = d.card(
        640,
        200,
        "Checks",
        "model",
        sub="known key?\nright type?\nDynaconf validators\ncross-setting rules",
        w=230,
        h=170,
    )
    d.arrow((390, 200), checks, into="l", label="merged", mono=False, label_dy=-10)
    ok = d.card(
        980,
        120,
        "StageConfig",
        "result",
        sub="cfg.lr, cfg.to_dict()",
        mono_sub=True,
        w=230,
    )
    bad = d.card(
        980, 290, "Clear error", "loss", sub="names the file or variable", w=230
    )
    d.arrow(checks, ok, out="r", into="l")
    d.arrow(checks, bad, out="r", into="l")
    return d


@diagram("modern_block")
def modern_block() -> Diagram:
    """The modern block: the same residual shape with today's parts inside."""
    d = Diagram(
        960,
        830,
        "The modern block",
        "Same skeleton as the classic block. Almost every part inside it has been swapped.",
    )
    cx, skip = 400, 220
    x_in = d.card(cx, 20, "x", "data", sub="from the block below", w=200)
    n1 = d.card(cx, 125, "RMSNorm", "plain", pill=True, w=150, size=14)
    attn = d.card(
        cx, 210, "Attention", "model", sub="RoPE on q and k, GQA or MLA", w=260
    )
    p1 = d.plus(cx, 295)
    n2 = d.card(cx, 380, "RMSNorm", "plain", pill=True, w=150, size=14)
    mlp = d.card(cx, 465, "SwiGLU", "model", sub="or Mixture of Experts", w=260)
    p2 = d.plus(cx, 550)
    out = d.card(cx, 635, "x", "result", sub="to the next block", w=200)
    for a_, b_ in (
        (x_in, n1),
        (n1, attn),
        (attn, p1),
        (p1, n2),
        (n2, mlp),
        (mlp, p2),
        (p2, out),
    ):
        d.arrow(a_, b_)
    d.arrow(
        (cx, 72),
        p1,
        via=[(skip, 72), (skip, 295)],
        into="l",
        label="skip",
        label_at=0.45,
        mono=False,
    )
    d.arrow(
        (cx, 330),
        p2,
        via=[(skip, 330), (skip, 550)],
        into="l",
        label="skip",
        label_at=0.5,
        mono=False,
    )
    notes = [
        (125, "LayerNorm without the mean:", "cheaper, works as well"),
        (210, "word order as rotations,", "fewer key/value heads"),
        (465, "a gated MLP, or a router", "that picks k experts per token"),
    ]
    for y, first, second in notes:
        d.label(560, y - 4, first, 13, anchor="start")
        d.label(560, y + 14, second, 13, anchor="start")
    d.label(
        cx, 705, "--arch modern   (src/models/modern/)", 14, mono=True, color="#1F2A44"
    )
    return d


@diagram("gqa")
def gqa() -> Diagram:
    """Multi-head, grouped-query and multi-query attention side by side."""
    d = Diagram(
        1180,
        420,
        "Sharing keys and values between heads",
        "Every query head still exists. What changes is how many key/value heads they read from.",
    )
    panels = [
        ("Multi-head", 8, "--n_kv_head 8"),
        ("Grouped-query", 2, "--n_kv_head 2"),
        ("Multi-query", 1, "--n_kv_head 1"),
    ]
    for k, (name, kv, flag) in enumerate(panels):
        x0 = 40 + 380 * k
        d.label(x0 + 160, 0, name, 15, color="#1F2A44", weight=700)
        d.label(x0 + 160, 20, flag, 12, mono=True)
        qx = [x0 + 20 + 40 * i for i in range(8)]
        for x in qx:
            d.chip(x, 70, "q", "model", w=30, size=13)
        per = 8 // kv
        kvx = [sum(qx[j * per : (j + 1) * per]) / per for j in range(kv)]
        for x in kvx:
            d.chip(x, 200, "kv", "data", w=30 if kv == 8 else 40, size=13)
        for i, x in enumerate(qx):
            d.arrow(
                (x, 84),
                (kvx[i // per], 186),
                via=[(x, 135), (kvx[i // per], 135)],
                head=False,
                color="#8A96AB",
            )
        d.label(x0 + 160, 250, f"KV cache per token: {kv}/8 of multi-head", 13)
    return d


@diagram("kv_cache")
def kv_cache() -> Diagram:
    """Generation with a KV cache: each step computes one token and reuses the rest."""
    d = Diagram(
        1240,
        440,
        "Generating with a KV cache",
        "Each step runs only the newest token through the model and reuses every key and value before it.",
    )
    new = d.card(110, 120, "Newest token", "data", sub="position t", w=170)
    proj = d.card(340, 120, "q, k, v", "model", sub="for this token only", w=180)
    d.arrow(new, proj)
    d.label(745, 18, "cache: keys and values of every earlier token", 13)
    for i in range(6):
        d.chip(560 + 62 * i, 50, f"k{i + 1} v{i + 1}", "data", w=56, size=12)
    last = d.chip(560 + 62 * 6, 50, "k7 v7", "result", w=56, size=12)
    d.arrow(
        proj,
        (last.x, 64),
        out="r",
        via=[(last.x, 120)],
        into="b",
        label="append k, v",
        mono=False,
        label_at=0.45,
        label_dy=-6,
    )
    att = d.card(
        1080, 230, "Attention", "model", sub="q against the whole cache", w=230
    )
    d.arrow(
        (last.x + 28, 50),
        att,
        via=[(1080, 50)],
        into="t",
        label="read all",
        mono=False,
        label_at=0.3,
        label_dy=-8,
    )
    d.arrow(
        proj,
        att,
        out="b",
        via=[(340, 230)],
        into="l",
        label="q",
        label_at=0.6,
        label_dy=-6,
    )
    d.arrow(att, (1080, 320), out="b", label="next token", mono=False, label_dy=4)
    return d


@diagram("moe")
def moe() -> Diagram:
    """Mixture of Experts: a router picks k experts per token."""
    d = Diagram(
        1180,
        590,
        "Mixture of Experts",
        "A router sends each token to its top k experts. More parameters, about the same work per token.",
        top=112,
    )
    tok = d.card(110, 160, "Token vector", "data", w=170)
    router = d.card(320, 160, "Router", "model", sub="a score per expert", w=170)
    d.arrow(tok, router)
    picked = {1, 5}
    for i in range(8):
        y = 10 + 43 * i
        role = "model" if i in picked else "frozen"
        e = d.card(
            600,
            y,
            f"Expert {i + 1}",
            role,
            w=150,
            h=34,
            size=13,
            dashed=i not in picked,
        )
        if i in picked:
            d.arrow(router, e, out="r", into="l", via=[(460, 160), (460, y)])
    shared = d.card(
        600, 410, "Shared expert", "model", sub="every token, optional", w=200
    )
    d.arrow(tok, shared, out="b", into="l", via=[(110, 410)])
    mix = d.card(
        880, 160, "Weighted sum", "plain", pill=True, sub="router weights", w=190
    )
    d.arrow((675, 10 + 43), mix, via=[(760, 53), (760, 150)], into="l")
    d.arrow((675, 10 + 43 * 5), mix, via=[(760, 225), (760, 170)], into="l")
    d.arrow(shared, mix, out="r", into="b")
    out = d.card(1080, 160, "Output", "result", w=130)
    d.arrow(mix, out)
    d.label(320, 230, "--n_experts 8 --moe_top_k 2", 12, mono=True)
    d.label(320, 250, "plus a small loss that keeps", 12)
    d.label(320, 267, "experts evenly used", 12)
    return d


@diagram("grpo_variants")
def grpo_variants() -> Diagram:
    """What each 2025 follow-up changes in GRPO, and the flag for it."""
    d = Diagram(
        1180,
        500,
        "GRPO and its 2025 follow-ups",
        "Each fix is a flag on scripts/train_grpo.py, and they combine freely.",
        top=110,
    )
    cards = [
        (
            "GRPO",
            "the starting point",
            "advantage = (r - mean) / std,\nclipped per-token ratio",
            "default settings",
        ),
        (
            "Dr. GRPO",
            "removes two length and difficulty biases",
            "no std division, a fixed\nlength normaliser",
            "--adv_norm none --loss_agg seq-mean-token-sum-norm",
        ),
        (
            "DAPO",
            "keeps exploration alive",
            "a looser upper clip, and drops\ngroups where every answer agrees",
            "--clip_high 0.28 --filter_groups true",
        ),
        (
            "GSPO",
            "steadier updates",
            "one importance ratio for the\nwhole answer, not per token",
            "--ratio_level sequence",
        ),
    ]
    for k, (name, why, what, flag) in enumerate(cards):
        x = 300 + 580 * (k % 2)
        y = 70 + 175 * (k // 2)
        role = "plain" if k == 0 else "reward"
        d.card(x, y - 12, name, role, sub=why + "\n" + what, w=540, h=150)
        d.label(x, y + 48, flag, 12, mono=True, color="#1F2A44")
    return d


@diagram("lora")
def lora() -> Diagram:
    """LoRA: a frozen weight plus a trainable low-rank update."""
    d = Diagram(
        1000,
        470,
        "LoRA: train a small side path",
        "The big matrix stays frozen. Two thin matrices learn the change, then get merged back in.",
    )
    x = d.card(130, 160, "x", "data", sub="input, size k", w=150)
    w = d.card(470, 60, "W", "frozen", sub="d x k, frozen", w=200, h=80)
    a = d.card(400, 270, "A", "model", sub="r x k", w=120)
    b = d.card(590, 270, "B", "model", sub="d x r, starts at 0", w=170)
    plus = d.plus(780, 160)
    out = d.card(900, 160, "y", "result", w=100)
    d.arrow(x, w, out="r", into="l", via=[(250, 160), (250, 60)])
    d.arrow(x, a, out="r", into="l", via=[(250, 160), (250, 270)])
    d.arrow(a, b, "r", label_dy=-10)
    d.arrow(w, plus, out="r", into="t")
    d.arrow(
        b,
        plus,
        out="r",
        into="b",
        label="x alpha / r",
        mono=False,
        label_at=0.4,
        label_dy=-8,
    )
    d.arrow(plus, out)
    d.label(
        500,
        345,
        "--lora_rank 16: r is tiny next to d and k, so few parameters train.",
        13,
    )
    d.label(
        500,
        365,
        "At save time W becomes W + (alpha / r) B A, a normal checkpoint again.",
        13,
    )
    return d


@diagram("speculative")
def speculative() -> Diagram:
    """Speculative decoding: draft cheaply, verify in one pass, keep the agreeing prefix."""
    d = Diagram(
        1180,
        440,
        "Speculative decoding",
        "A small model guesses a few tokens; the big model checks them all at once.",
    )
    draft = d.card(
        110, 40, "Draft model", "frozen", sub="small and fast", dashed=False, w=170
    )
    xs = [380 + 70 * i for i in range(4)]
    for i, x in enumerate(xs):
        d.chip(x, 40, f"t{i + 1}", "plain", w=56, size=13)
    d.arrow(
        draft, (xs[0] - 28, 40), into="l", label="proposes k", mono=False, label_dy=-12
    )
    target = d.card(
        870, 40, "Big model", "model", sub="one forward pass over all k", w=250
    )
    d.arrow(
        (xs[-1] + 28, 40), target, into="l", label="check", mono=False, label_dy=-12
    )
    verdict = [
        ("result", "kept"),
        ("result", "kept"),
        ("loss", "rejected"),
        ("frozen", "dropped"),
    ]
    for i, (x, (role, word)) in enumerate(zip(xs, verdict)):
        d.chip(x, 165, f"t{i + 1}", role, w=56, size=13)
        d.label(x, 205, word, 12)
    d.chip(xs[2], 245, "t3'", "model", w=56, size=13)
    d.arrow((xs[2], 215), (xs[2], 231), color="#8A96AB")
    d.label(xs[2] + 40, 250, "resampled from the big model", 12, anchor="start")
    d.arrow(target, (xs[-1] + 28, 165), out="b", into="r", via=[(870, 165)])
    d.label(870, 290, "The output follows the big model's distribution exactly.", 13)
    d.label(870, 310, "A better draft only means fewer passes.", 13)
    return d


@diagram("bpe")
def bpe() -> Diagram:
    """BPE training, with the first merges the repo's tokenizer learns on a toy corpus."""
    from src.tokenizer.bpe import BPETokenizer

    corpus = (
        "The cat sat on the mat. The dog sat on the log. A cat and a dog sat together. "
        "Then the cat sat on the mat again, and the dog sat by the door. "
    ) * 20
    tok = BPETokenizer.train(corpus, 262)

    def show(i: int) -> str:
        return tok.token_bytes(i).decode("utf-8", "replace").replace(" ", "\u2423")

    d = Diagram(
        1180,
        460,
        "Training a BPE tokenizer",
        top=110,
        caption="Start from bytes. Repeatedly glue the most frequent neighbouring pair into a new token.",
    )
    steps = [
        d.card(200, 40 + 95 * k, t, r, sub=s, w=240)
        for k, (t, r, s) in enumerate(
            [
                ("Text as bytes", "data", "256 starting tokens"),
                ("Count neighbour pairs", "model", "across the whole corpus"),
                ("Merge the top pair", "model", "it gets the next id"),
            ]
        )
    ]
    d.arrow(steps[0], steps[1])
    d.arrow(steps[1], steps[2])
    d.arrow(
        steps[2], steps[1], out="l", into="l", via=[(50, 230), (50, 135)], head=True
    )
    d.label(58, 187, "repeat", 12, anchor="start")
    d.label(200, 300, "until vocab_size is reached", 12)
    d.label(
        700,
        0,
        "The first merges learned on a toy corpus (src/tokenizer/bpe.py)",
        13,
        weight=600,
    )
    for k, (a, b) in enumerate(tok.merges[:6]):
        y = 40 + 50 * k
        new_id = 256 + k
        d.chip(520, y, show(a), "data", w=70)
        d.label(570, y + 5, "+", 15)
        d.chip(620, y, show(b), "data", w=70)
        d.arrow((660, y), (740, y), color="#8A96AB")
        d.chip(800, y, show(new_id), "result", w=90)
        d.label(870, y + 5, f"id {new_id}", 12, mono=True, anchor="start")
    return d


@diagram("legend")
def legend() -> Diagram:
    """How to read the diagrams: what each colour and shape means."""
    d = Diagram(
        1180,
        400,
        "How to read these diagrams",
        "Colour says what a thing is. Shape says how it is used.",
    )
    roles = [
        ("data", "Data", "text, tokens, datasets"),
        ("model", "Computation", "a network or a step"),
        ("loss", "Loss", "what training minimises"),
        ("reward", "Reward", "scores and advantages"),
        ("frozen", "Frozen", "a copy that never trains"),
        ("result", "Result", "checkpoints, answers"),
    ]
    assert {r for r, _, _ in roles} == set(ROLES) - {"plain"}, (
        "list every role from svgkit.ROLES"
    )
    for k, (role, name, sub) in enumerate(roles):
        d.card(110 + 190 * k, 30, name, role, sub=sub, w=176)
    d.card(130, 170, "Stored data", "data", stack=True, sub="a file on disk", w=180)
    d.card(360, 170, "Small step", "plain", pill=True, sub="one operation", w=180)
    n = d.card(590, 170, "Ordered step", "model", sub="when order matters", w=180)
    d.badge(n, 1)
    d.arrow((760, 170), (980, 170), "(B, T, C)", label_dy=-10)
    d.label(870, 200, "tensor shape on an arrow", 12)
    d.arrow((760, 240), (980, 240), dashed=True)
    d.label(870, 262, "dashed: an update or a side input", 12)
    return d


@diagram("next_token_shift")
def next_token_shift() -> Diagram:
    """Inputs and targets are the same window, one position apart."""
    from src.tokenizer.bpe import BPETokenizer

    corpus = (
        "The cat sat on the mat. The dog sat on the log. A cat and a dog sat together. "
        "Then the cat sat on the mat again, and the dog sat by the door. "
    ) * 20
    bpe = BPETokenizer.train(corpus, 300)
    pieces = [
        bpe.token_bytes(i).decode("utf-8", "replace").replace(" ", "\u2423")
        for i in bpe.encode_ordinary("The cat sat on the mat.")
    ]
    d = Diagram(
        1140,
        400,
        "One window, two jobs",
        "Read context_length + 1 tokens. The first T are the input, the last T are the answers.",
    )
    gap = 120
    xs = [200 + gap * k for k in range(len(pieces))]
    for x, piece in zip(xs, pieces):
        d.chip(x, 20, piece, "data", w=96, size=14)
    d.label(40, 25, "window", 12, anchor="start", weight=600)
    for k, x in enumerate(xs[:-1]):
        d.chip(x, 120, pieces[k], "model", w=96, size=14)
        d.chip(x + gap, 220, pieces[k + 1], "result", w=96, size=14)
        d.arrow((x, 134), (x + gap, 206), straight=True, color="#8A96AB")
    d.label(40, 125, "input x", 12, anchor="start", weight=600)
    d.label(40, 225, "target y", 12, anchor="start", weight=600)
    d.label(
        590,
        285,
        "Position i sees x\u2080 \u2026 x\u1d62 and is graded on y\u1d62, the token that came next.",
        13,
    )
    return d


@diagram("ddp_accumulation")
def ddp_accumulation() -> Diagram:
    """Gradient accumulation under DDP: sync only on the last microstep."""
    d = Diagram(
        1180,
        430,
        "Accumulate locally, sync once",
        "With --grad_accum 3 on two GPUs, gradients only travel between GPUs after the last microstep.",
    )
    for g, y in enumerate((40, 180)):
        d.label(40, y + 5, f"GPU {g}", 13, anchor="start", weight=600)
        prev = None
        for m in range(3):
            last = m == 2
            node = d.card(
                230 + 200 * m,
                y,
                f"microstep {m + 1}",
                "reward" if last else "model",
                w=180,
                sub="backward, then sync" if last else "backward in no_sync()",
            )
            if prev is not None:
                d.arrow(prev, node)
            prev = node
        d.arrow(prev, (790, 110), out="r", into="l", via=[(740, y), (740, 110)])
    reduce_ = d.card(
        880, 110, "All-reduce", "reward", sub="average the gradients", w=180
    )
    step = d.card(1080, 110, "Optimizer step", "result", sub="same on every GPU", w=170)
    d.arrow(reduce_, step)
    d.label(
        430,
        270,
        "Every microstep runs forward and backward on its own batch. Only the last one talks to the other GPU.",
        13,
    )
    return d


@diagram("transformer_model")
def transformer_model() -> Diagram:
    """The decoder-only Transformer's forward pass with every tensor shape."""
    d = Diagram(
        1300,
        380,
        "The forward pass, end to end",
        "Ids become vectors, pass through N identical blocks, and come out as a score for every word.",
    )
    ids = d.card(90, 80, "Token ids", "data", sub="(B, T)", mono_sub=True, w=130)
    tok = d.card(
        330, 30, "Token embedding", "model", sub="one row per vocabulary entry", w=240
    )
    pos = d.card(
        330, 130, "Position embedding", "model", sub="one row per position", w=240
    )
    add = d.plus(500, 80)
    blocks = d.card(
        700, 80, "N blocks", "model", sub="attention + MLP", stack=True, w=170
    )
    norm = d.card(900, 80, "LayerNorm", "plain", pill=True, w=140, size=14)
    head = d.card(1070, 80, "lm_head", "model", sub="C to vocab", w=140)
    logits = d.card(1225, 80, "Logits", "result", sub="(B, T, V)", mono_sub=True, w=110)
    d.arrow(ids, tok, out="r", into="l")
    d.arrow(ids, pos, out="r", into="l")
    d.arrow(tok, add, out="r", into="t")
    d.arrow(pos, add, out="r", into="b")
    d.arrow(add, blocks, "(B, T, C)", label_dy=-12)
    d.arrow(blocks, norm)
    d.arrow(norm, head)
    d.arrow(head, logits)
    d.label(
        650,
        220,
        "forward_hidden() stops after the LayerNorm. The reward and value heads reuse exactly that.",
        13,
    )
    return d


def main() -> None:
    """Write every registered diagram (or just the names given on the command line)."""
    names = sys.argv[1:] or list(DIAGRAMS)
    for name in names:
        (OUT / f"{name}.svg").write_text(DIAGRAMS[name]().svg(), encoding="utf-8")
        print(f"wrote docs/diagrams/{name}.svg")


if __name__ == "__main__":
    main()
