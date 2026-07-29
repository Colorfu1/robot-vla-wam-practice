"""Launch E5 Mode-BCChunk training on an explicit number of GPUs."""

from __future__ import annotations

from collections.abc import Sequence

from .distributed_training import launch_multi_gpu
from .train_mode_bc_chunk import resolve_training


def main(argv: Sequence[str] | None = None) -> int:
    return launch_multi_gpu(
        argv,
        description=__doc__,
        policy_type="mode_bc_chunk",
        single_gpu_script="scripts/experiments/pusht/train_mode_bc_chunk.sh",
        resolve_training=resolve_training,
    )


if __name__ == "__main__":
    raise SystemExit(main())
