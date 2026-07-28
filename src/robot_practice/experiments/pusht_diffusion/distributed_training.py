"""Single-machine Accelerate launcher shared by explicit PushT model entries."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .training_common import (
    DistributedProcessContext,
    ResolvedTraining,
    TrainArguments,
    training_worker_arguments,
)

TrainingResolver = Callable[
    [Sequence[str] | None],
    tuple[TrainArguments, ResolvedTraining],
]


class _UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class MultiGpuLaunchPlan:
    """Fully resolved multi-GPU invocation."""

    num_gpus: int
    available_cuda_devices: int
    arguments: TrainArguments
    resolved: ResolvedTraining
    command: tuple[str, ...]

    @property
    def process_context(self) -> DistributedProcessContext:
        return DistributedProcessContext(rank=0, local_rank=0, world_size=self.num_gpus)

    def summary(self) -> dict[str, Any]:
        summary = self.resolved.summary(self.process_context)
        summary["launcher"] = {
            "type": "accelerate_multi_gpu",
            "requested_gpus": self.num_gpus,
            "available_cuda_devices": self.available_cuda_devices,
            "worker_module": self.resolved.entry_module,
        }
        return summary


def _num_gpu_option_count(arguments: Sequence[str]) -> int:
    return sum(
        argument == "--num-gpus" or argument.startswith("--num-gpus=") for argument in arguments
    )


def _parse_num_gpus(
    raw_arguments: Sequence[str],
    *,
    description: str,
    single_gpu_script: str,
) -> tuple[int, tuple[str, ...], _UsageParser]:
    parser = _UsageParser(
        description=description,
        epilog=(
            "All remaining options are the same as the single-GPU training entry. "
            f"Use {single_gpu_script} for one GPU."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of currently visible CUDA devices to use, starting at logical device 0.",
    )
    if _num_gpu_option_count(raw_arguments) > 1:
        parser.error("duplicate argument: --num-gpus")
    launcher, remaining = parser.parse_known_args(raw_arguments)
    if launcher.num_gpus < 2:
        parser.error(f"--num-gpus must be at least 2; use {single_gpu_script} for one GPU")
    return launcher.num_gpus, tuple(remaining), parser


def build_accelerate_command(
    resolved: ResolvedTraining,
    *,
    num_gpus: int,
    python_executable: str | None = None,
) -> tuple[str, ...]:
    """Build a deterministic Accelerate command independent of user defaults."""

    executable = python_executable or sys.executable
    return (
        executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        f"--num_processes={num_gpus}",
        "--num_machines=1",
        "--gpu_ids=all",
        "--mixed_precision=no",
        "--dynamo_backend=no",
        "--main_process_port=0",
        "-m",
        resolved.entry_module,
        *training_worker_arguments(resolved),
    )


def resolve_multi_gpu_launch(
    argv: Sequence[str] | None,
    *,
    description: str,
    policy_type: str,
    single_gpu_script: str,
    resolve_training: TrainingResolver,
) -> MultiGpuLaunchPlan:
    """Resolve and validate a model-specific multi-GPU launch without spawning."""

    raw = tuple(sys.argv[1:] if argv is None else argv)
    num_gpus, worker_arguments, parser = _parse_num_gpus(
        raw,
        description=description,
        single_gpu_script=single_gpu_script,
    )
    try:
        arguments, resolved = resolve_training(worker_arguments)
    except RuntimeError as exc:
        parser.error(str(exc))
    if resolved.policy_type != policy_type:
        raise TypeError(f"Expected resolved policy {policy_type!r}, got {resolved.policy_type!r}")
    if arguments.requested_device not in {"auto", "cuda"}:
        parser.error("multi-GPU training only supports --device=auto or --device=cuda")
    if not torch.cuda.is_available():
        parser.error("multi-GPU training requires CUDA, but CUDA is not available")
    available_cuda_devices = torch.cuda.device_count()
    if num_gpus > available_cuda_devices:
        parser.error(
            f"--num-gpus={num_gpus} requested, but only "
            f"{available_cuda_devices} CUDA device(s) are visible"
        )
    if resolved.device != "cuda":
        parser.error(f"multi-GPU training resolved an invalid device: {resolved.device}")

    return MultiGpuLaunchPlan(
        num_gpus=num_gpus,
        available_cuda_devices=available_cuda_devices,
        arguments=arguments,
        resolved=resolved,
        command=build_accelerate_command(resolved, num_gpus=num_gpus),
    )


def print_multi_gpu_plan(plan: MultiGpuLaunchPlan) -> None:
    print("Command:")
    print(shlex.join(plan.command))
    print("Resolved configuration:")
    print(json.dumps(plan.summary(), indent=2, sort_keys=True))


def launch_multi_gpu(
    argv: Sequence[str] | None,
    *,
    description: str,
    policy_type: str,
    single_gpu_script: str,
    resolve_training: TrainingResolver,
) -> int:
    """Validate, display, and execute one single-machine DDP run."""

    plan = resolve_multi_gpu_launch(
        argv,
        description=description,
        policy_type=policy_type,
        single_gpu_script=single_gpu_script,
        resolve_training=resolve_training,
    )
    if plan.arguments.print_command:
        print_multi_gpu_plan(plan)
        return 0

    training = plan.summary()["training"]
    print(f"Launching {plan.resolved.policy_type} with Accelerate DDP", flush=True)
    print(f"Visible CUDA devices: {plan.available_cuda_devices}", flush=True)
    print(f"World size: {training['world_size']}", flush=True)
    print(f"Per-GPU batch size: {training['per_gpu_batch_size']}", flush=True)
    print(f"Global batch size: {training['global_batch_size']}", flush=True)
    print(f"Output directory: {plan.resolved.output_dir}", flush=True)
    completed = subprocess.run(plan.command, check=False)
    return completed.returncode


__all__ = [
    "MultiGpuLaunchPlan",
    "build_accelerate_command",
    "launch_multi_gpu",
    "print_multi_gpu_plan",
    "resolve_multi_gpu_launch",
]
