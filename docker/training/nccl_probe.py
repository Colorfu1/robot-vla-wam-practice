"""One worker in the robot-practice NCCL all-reduce verification."""

from __future__ import annotations

import os

import torch
from torch import distributed


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    distributed.init_process_group("nccl")
    value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    distributed.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"NCCL all-reduce mismatch on rank {rank}: {value.item()} != {expected}")
    distributed.barrier()
    if rank == 0:
        print(f"NCCL all-reduce passed for {world_size} processes")
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
