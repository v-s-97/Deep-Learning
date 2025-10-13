from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def setup_distributed(backend: str = "nccl") -> DistributedContext:
    """
    Initialize torch distributed if ranks are provided through the environment.
    Returns a context describing the current process.
    """
    if not dist.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(False, 0, 0, 1, device)

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size_env > 1 and "RANK" in os.environ and "LOCAL_RANK" in os.environ
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return DistributedContext(True, rank, local_rank, world_size, device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(False, 0, 0, 1, device)


def is_rank0() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log_rank0(msg: str, logger: Optional[Callable[[str], None]] = None) -> None:
    if not is_rank0():
        return
    fn = logger if logger is not None else print
    fn(msg)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
