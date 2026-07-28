"""Launch explicit Diffusion Policy PushT training on multiple visible CUDA GPUs."""

from __future__ import annotations

from collections.abc import Sequence

from .distributed_training import launch_multi_gpu
from .train_diffusion import resolve_training


def main(argv: Sequence[str] | None = None) -> int:
    return launch_multi_gpu(
        argv,
        description=__doc__ or "Launch Diffusion multi-GPU training",
        policy_type="diffusion",
        single_gpu_script="scripts/experiments/pusht/train_diffusion.sh",
        resolve_training=resolve_training,
    )


if __name__ == "__main__":
    raise SystemExit(main())
