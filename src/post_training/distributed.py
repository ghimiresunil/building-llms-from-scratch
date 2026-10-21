"""Optional, from-scratch multi-GPU helpers (pure ``torch.distributed`` -- no accelerate)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.device import configure_cpu_threads, resolve_device
from src.models.modern.moe import MoE


@dataclass
class DDPContext:
    rank: int
    local_rank: int
    world_size: int
    device: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def ddp_setup(device: str = "auto") -> DDPContext:
    """Initialize the process group if launched under torchrun; otherwise single-process."""
    device = resolve_device(device)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        if device == "cuda":
            torch.cuda.set_device(0)
        elif device == "cpu":
            configure_cpu_threads()
        return DDPContext(rank=0, local_rank=0, world_size=1, device=device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        dev = f"cuda:{local_rank}"
    else:
        dev = "cpu"
    return DDPContext(
        rank=rank, local_rank=local_rank, world_size=world_size, device=dev
    )


def ddp_wrap(
    model: torch.nn.Module, ctx: DDPContext, find_unused_parameters: bool = False
) -> torch.nn.Module:
    """Wrap a model in DDP when running multi-GPU; return it unchanged otherwise."""
    if not ctx.enabled:
        return model
    device_ids = [ctx.local_rank] if ctx.device.startswith("cuda") else None
    find_unused_parameters = find_unused_parameters or any(
        isinstance(m, MoE) for m in model.modules()
    )
    return DDP(
        model, device_ids=device_ids, find_unused_parameters=find_unused_parameters
    )


def is_main_process(ctx: DDPContext) -> bool:
    return ctx.is_main


def reduce_scalar(value: float, ctx: DDPContext, average: bool = True) -> float:
    """All-reduce a python scalar across ranks (sum or mean). No-op when single-process."""
    if not ctx.enabled:
        return value
    t = torch.tensor([value], device=ctx.device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if average:
        t /= ctx.world_size
    return t.item()


def barrier(ctx: DDPContext) -> None:
    if ctx.enabled:
        dist.barrier()


def cleanup(ctx: DDPContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def shard_indices(num_items: int, ctx: DDPContext) -> range:
    """Return this rank's strided slice of ``range(num_items)`` for data sharding."""
    if not ctx.enabled:
        return range(num_items)
    return range(ctx.rank, num_items, ctx.world_size)
