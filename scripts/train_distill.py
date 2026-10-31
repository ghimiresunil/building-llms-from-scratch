"""Knowledge distillation: train a small student on a bigger teacher's token distributions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # run from the repo without installing

import contextlib
import math
import os
import time

import torch

from data_loader.data_loader import get_batch_iterator
from data_loader.sft_dataset import get_sft_batch_iterator
from src.decorators import log_execution
from src.post_training.cli import parse_stage
from src.post_training.distill import distillation_loss, shift_for_next_token
from src.post_training.distributed import cleanup, ddp_setup, ddp_wrap, reduce_scalar
from src.post_training.inference import load_model_from_ckpt
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer, cosine_lr, set_lr
from src.post_training.utils import (
    amp_autocast,
    build_model_from_config,
    load_backbone_from_ckpt,
    moe_balance_loss,
    save_stage_ckpt,
    set_seed,
    unwrap,
)


def batches(cfg, ctx, path: str, *, shuffle: bool, infinite: bool):
    """Yield ``(inputs, targets, mask)`` with targets already shifted by one."""
    if cfg.data_format == "pretrain":
        it = get_batch_iterator(path, cfg.batch_size, cfg.context_length, ctx.device)
        for xb, yb in it:
            yield xb, yb, torch.ones_like(yb)
        return
    it = get_sft_batch_iterator(
        path,
        cfg.batch_size,
        device=ctx.device,
        rank=ctx.rank,
        world_size=ctx.world_size,
        shuffle=shuffle,
        infinite=infinite,
    )
    for tokens, mask, _ in it:
        yield tokens, None, mask


def step_loss(student, teacher, cfg, xb, yb, mask):
    """Run both models on one batch and return ``(total, divergence, ce)``."""
    with torch.no_grad():
        t_logits, _ = teacher(xb)
    s_logits, _ = student(xb)
    if yb is None:  # packed SFT rows: shift so position t predicts token t + 1
        s_logits, yb, mask = shift_for_next_token(s_logits, xb, mask)
        t_logits = t_logits[:, :-1, :]
    return distillation_loss(
        s_logits,
        t_logits,
        yb,
        mask,
        kind=cfg.distill_loss,
        temperature=cfg.distill_temperature,
        alpha=cfg.distill_alpha,
        beta=cfg.jsd_beta,
    )


@torch.no_grad()
def eval_dev(student, teacher, cfg, ctx, max_batches: int = 50) -> dict[str, float]:
    if not os.path.exists(cfg.dev_path):
        return {"dev_div": float("nan"), "dev_ce": float("nan")}
    student.eval()
    div_sum = ce_sum = 0.0
    n = 0
    for xb, yb, mask in batches(cfg, ctx, cfg.dev_path, shuffle=False, infinite=False):
        with amp_autocast(cfg.amp_dtype, ctx.device):
            _, div, ce = step_loss(student, teacher, cfg, xb, yb, mask)
        div_sum += div.item()
        ce_sum += ce.item()
        n += 1
        if n >= max_batches:
            break
    student.train()
    return {"dev_div": div_sum / max(1, n), "dev_ce": ce_sum / max(1, n)}


@log_execution
def main():
    cfg, _ = parse_stage("distill")
    ctx = ddp_setup(cfg.device)
    set_seed(cfg.seed + ctx.rank)

    teacher = load_model_from_ckpt(cfg.teacher_ckpt, ctx.device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    if cfg.student_ckpt is None:
        student = build_model_from_config(cfg).to(ctx.device)
    else:
        student = load_backbone_from_ckpt(cfg, cfg.student_ckpt, ctx.device)
    if teacher.lm_head.out_features != student.lm_head.out_features:
        raise SystemExit(
            f"teacher vocab {teacher.lm_head.out_features} != student vocab "
            f"{student.lm_head.out_features}: both must use the same tokenizer"
        )
    if ctx.is_main:
        n_t = sum(p.numel() for p in teacher.parameters())
        n_s = sum(p.numel() for p in unwrap(student).parameters())
        print(
            f"distilling {cfg.teacher_ckpt} ({n_t / 1e6:.1f}M) -> student "
            f"({n_s / 1e6:.1f}M, {n_t / n_s:.1f}x smaller) | loss={cfg.distill_loss} "
            f"tau={cfg.distill_temperature} alpha={cfg.distill_alpha}"
        )
    if cfg.compile:
        student = torch.compile(student)
    student = ddp_wrap(student, ctx)
    optimizer = configure_optimizer(unwrap(student), cfg.lr, cfg.weight_decay)
    logger = (
        MetricsLogger(
            "distill",
            cfg.log_dir,
            use_wandb=cfg.use_wandb,
            wandb_project=cfg.wandb_project,
        )
        if ctx.is_main
        else None
    )

    train_it = batches(cfg, ctx, cfg.data_path, shuffle=True, infinite=True)
    student.train()
    t0 = time.perf_counter()
    for step in range(cfg.max_steps):
        lr = cosine_lr(
            step,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            lr=cfg.lr,
            min_lr=cfg.min_lr,
        )
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        div_sum = ce_sum = 0.0
        for micro in range(cfg.grad_accum):
            xb, yb, mask = next(train_it)
            sync = micro == cfg.grad_accum - 1 or not ctx.enabled
            with (
                contextlib.nullcontext() if sync else student.no_sync(),
                amp_autocast(cfg.amp_dtype, ctx.device),
            ):
                total, div, ce = step_loss(student, teacher, cfg, xb, yb, mask)
                loss = (total + moe_balance_loss(student)) / cfg.grad_accum
            loss.backward()
            div_sum += div.item() / cfg.grad_accum
            ce_sum += ce.item() / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        optimizer.step()

        if ctx.is_main and step % 20 == 0:
            dt = time.perf_counter() - t0
            t0 = time.perf_counter()
            print(
                f"step {step}/{cfg.max_steps} | div {div_sum:.4f} | ce {ce_sum:.4f} "
                f"| ppl {math.exp(min(20, ce_sum)):.2f} | lr {lr:.2e} | {dt:.1f}s/20"
            )
            if logger:
                logger.log(step, {"train_div": div_sum, "train_ce": ce_sum, "lr": lr})

        if step > 0 and step % cfg.eval_steps == 0:
            ev = eval_dev(student, teacher, cfg, ctx)
            ev = {k: reduce_scalar(v, ctx) for k, v in ev.items()}
            if ctx.is_main:
                print(f"  [eval] step {step} | {ev}")
                if logger:
                    logger.log(step, ev)

        if ctx.is_main and step > 0 and step % cfg.save_every == 0:
            save_stage_ckpt(
                cfg.out_ckpt, student, optimizer, stage="distill", cfg=cfg, step=step
            )

    if ctx.is_main:
        ev = eval_dev(unwrap(student), teacher, cfg, ctx)
        save_stage_ckpt(
            cfg.out_ckpt,
            student,
            optimizer,
            stage="distill",
            cfg=cfg,
            step=cfg.max_steps,
            metrics=ev,
        )
        print(f"Done distillation. {ev} -> {cfg.out_ckpt}")
        if logger:
            logger.close()
    cleanup(ctx)


if __name__ == "__main__":
    main()
