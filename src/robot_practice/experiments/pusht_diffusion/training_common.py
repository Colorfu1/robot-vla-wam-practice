"""Shared argument parsing, config construction, and validation for PushT training."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_MAX_STEPS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRAIN_SEED,
    SMOKE_STEPS,
    default_output_root,
    default_split_path,
    load_json_object,
)
from .dataset_split import EpisodeSplit, load_split
from .model_configs import model_config_fingerprint
from .provenance import sha256_file
from .route_field_data import ROUTE_FIELD_SCHEMA_VERSION

_POLICY_TYPES = {
    "bc_chunk",
    "diffusion",
    "mode_bc_chunk",
    "mode_diffusion",
    "route_field_bc_chunk",
    "route_field_diffusion",
}
_REQUIRED_INITIALIZATION_FILES = {
    "config.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_preprocessor.json",
    "train_config.json",
}


class _UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class TrainArguments:
    mode: str
    steps: int
    train_seed: int
    split_file: Path
    output_dir: Path | None
    batch_size: int
    num_workers: int
    requested_device: str
    print_command: bool


@dataclass(frozen=True)
class DistributedProcessContext:
    """Distributed rank metadata supplied by Accelerate."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"WORLD_SIZE must be positive, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"RANK must be in [0, {self.world_size}), got {self.rank}")
        if not 0 <= self.local_rank < self.world_size:
            raise ValueError(f"LOCAL_RANK must be in [0, {self.world_size}), got {self.local_rank}")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> DistributedProcessContext:
        source = os.environ if environment is None else environment

        def parse(name: str, default: int) -> int:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

        return cls(
            rank=parse("RANK", 0),
            local_rank=parse("LOCAL_RANK", 0),
            world_size=parse("WORLD_SIZE", 1),
        )


@dataclass(frozen=True)
class ResolvedTraining:
    entry_module: str
    model_config_module: str
    policy_type: str
    policy_fingerprint: str
    device: str
    mode: str
    split: EpisodeSplit
    episodes: tuple[int, ...]
    dataset_root: Path
    initialization_root: Path | None
    output_dir: Path
    config: TrainPipelineConfig
    prototype_manifest: Path | None = None
    route_manifest: Path | None = None

    def summary(
        self,
        process_context: DistributedProcessContext | None = None,
    ) -> dict[str, Any]:
        policy = self.config.policy
        if policy is None:
            raise RuntimeError("Resolved training config has no policy")
        context = process_context or DistributedProcessContext.from_environment()
        per_gpu_batch_size = self.config.batch_size
        summary = {
            "entry_module": self.entry_module,
            "model": {
                "config_module": self.model_config_module,
                "type": self.policy_type,
                "fingerprint": self.policy_fingerprint,
                "device": self.device,
                "vision_backbone": getattr(policy, "vision_backbone", None),
                "down_dims": list(getattr(policy, "down_dims", ())),
                "n_obs_steps": getattr(policy, "n_obs_steps", None),
                "horizon": getattr(policy, "horizon", None),
                "n_action_steps": getattr(policy, "n_action_steps", None),
            },
            "dataset": {
                "repo_id": DATASET_REPO_ID,
                "revision": DATASET_REVISION,
                "root": str(self.dataset_root),
                "episodes": list(self.episodes),
                "split_manifest_sha256": self.split.manifest_sha256,
            },
            "training": {
                "mode": self.mode,
                "batch_size": per_gpu_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "world_size": context.world_size,
                "global_batch_size": per_gpu_batch_size * context.world_size,
                "steps": self.config.steps,
                "seed": self.config.seed,
                "num_workers": self.config.num_workers,
                "save_freq": self.config.save_freq,
                "initialization": (
                    str(self.initialization_root)
                    if self.initialization_root is not None
                    else "random_from_train_seed"
                ),
                "initialization_seed": (
                    None if self.initialization_root is not None else self.config.seed
                ),
                "output_dir": str(self.output_dir),
            },
        }
        if self.prototype_manifest is not None:
            summary["prototype_conditioning"] = {
                "manifest": str(self.prototype_manifest),
                "num_prototypes": getattr(policy, "num_prototypes", None),
                "mode_embedding_dim": getattr(policy, "mode_embedding_dim", None),
                "condition_dropout_prob": getattr(
                    policy,
                    "condition_dropout_prob",
                    None,
                ),
                "null_prototype_id": getattr(policy, "num_prototypes", None),
            }
        if self.route_manifest is not None:
            summary["route_field_conditioning"] = {
                "manifest": str(self.route_manifest),
                "grid_size": getattr(policy, "route_grid_size", None),
                "field_channels": getattr(policy, "route_field_channels", None),
                "embedding_dim": getattr(policy, "route_embedding_dim", None),
            }
        return summary


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _reject_duplicate_options(parser: argparse.ArgumentParser, arguments: Sequence[str]) -> None:
    seen: set[str] = set()
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        key = argument.split("=", maxsplit=1)[0]
        if key in seen:
            parser.error(f"duplicate argument: {key}")
        seen.add(key)


def parse_train_arguments(
    argv: Sequence[str] | None = None,
    *,
    description: str,
) -> TrainArguments:
    """Parse the shared, model-independent PushT training interface."""

    raw = tuple(sys.argv[1:] if argv is None else argv)
    parser = _UsageParser(description=description)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--steps", type=_positive_integer)
    parser.add_argument("--train-seed", type=_nonnegative_integer, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--split-file", type=Path, default=default_split_path())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=_positive_integer)
    parser.add_argument("--num-workers", type=_nonnegative_integer, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--print-command", action="store_true")
    _reject_duplicate_options(parser, raw)
    args = parser.parse_args(raw)
    if args.output_dir is not None:
        candidate = args.output_dir.expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            _reject_mode_mixing(candidate.resolve(), args.mode)
        except ValueError as exc:
            parser.error(str(exc))
    steps = (
        args.steps
        if args.steps is not None
        else (SMOKE_STEPS if args.mode == "smoke" else DEFAULT_MAX_STEPS)
    )
    batch_size = (
        args.batch_size if args.batch_size is not None else (8 if args.mode == "smoke" else 64)
    )
    return TrainArguments(
        mode=args.mode,
        steps=steps,
        train_seed=args.train_seed,
        split_file=args.split_file.expanduser().resolve(),
        output_dir=args.output_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        requested_device=args.device,
        print_command=args.print_command,
    )


def resolve_device(requested: str) -> str:
    """Resolve an explicit LeRobot policy device."""

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    return requested


def resolve_training_output_dir(arguments: TrainArguments, policy_type: str) -> Path:
    """Resolve one run directory before any distributed workers are created."""

    if arguments.output_dir is not None:
        output = arguments.output_dir.expanduser()
        if not output.is_absolute():
            output = Path.cwd() / output
        return output.resolve()
    timestamp = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    return (default_output_root() / arguments.mode / policy_type / timestamp).resolve()


def _reject_mode_mixing(output_dir: Path, mode: str) -> None:
    other_mode = "pilot" if mode == "smoke" else "smoke"
    forbidden = (default_output_root() / other_mode).resolve()
    try:
        output_dir.relative_to(forbidden)
    except ValueError:
        return
    raise ValueError(f"{mode} training output cannot use the {other_mode} root: {output_dir}")


def build_train_pipeline_config(
    *,
    arguments: TrainArguments,
    policy: PreTrainedConfig,
    policy_type: str,
    entry_module: str,
    model_config_module: str,
    prototype_manifest: Path | None = None,
    route_manifest: Path | None = None,
    episodes_override: Sequence[int] | None = None,
    random_initialization: bool = False,
) -> ResolvedTraining:
    """Resolve experiment paths and construct LeRobot's official config object."""

    if policy_type not in _POLICY_TYPES:
        raise ValueError(f"Unsupported PushT training policy: {policy_type}")
    if policy.type != policy_type:
        raise TypeError(f"Expected policy.type={policy_type!r}, got {policy.type!r}")
    split = load_split(arguments.split_file)
    mode_episodes = split.episodes_for_mode(arguments.mode)
    episodes = (
        mode_episodes
        if episodes_override is None
        else tuple(int(episode) for episode in episodes_override)
    )
    if not episodes:
        raise ValueError(f"No episodes remain for {policy_type} {arguments.mode} training")
    if len(episodes) != len(set(episodes)):
        raise ValueError("Training episode override contains duplicates")
    invalid_episodes = sorted(set(episodes) - set(mode_episodes))
    if invalid_episodes:
        raise ValueError(
            f"Training episode override is outside the {arguments.mode} split: {invalid_episodes}"
        )
    output_dir = resolve_training_output_dir(arguments, policy_type)
    _reject_mode_mixing(output_dir, arguments.mode)
    dataset_root = (default_output_root() / "dataset_views" / arguments.mode).resolve()
    if random_initialization:
        initialization_root = None
        policy.pretrained_path = None
    else:
        initialization_parent = (
            default_output_root()
            / "initialization"
            / arguments.mode
            / f"seed_{arguments.train_seed}"
        )
        if policy_type in {"mode_bc_chunk", "mode_diffusion"}:
            initialization_parent /= "e5_mode_conditioned"
        elif policy_type in {"route_field_bc_chunk", "route_field_diffusion"}:
            initialization_parent /= "route_field_7x7"
        initialization_root = (initialization_parent / policy_type).resolve()
        policy.pretrained_path = initialization_root
    fingerprint = model_config_fingerprint(policy)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=DATASET_REPO_ID,
            root=str(dataset_root),
            episodes=list(episodes),
            revision=DATASET_REVISION,
            use_imagenet_stats=False,
        ),
        policy=policy,
        output_dir=output_dir,
        job_name=f"pusht_{policy_type}_{arguments.mode}",
        seed=arguments.train_seed,
        num_workers=arguments.num_workers,
        batch_size=arguments.batch_size,
        steps=arguments.steps,
        eval_freq=0,
        log_freq=min(arguments.steps, 100),
        save_checkpoint=True,
        save_freq=min(arguments.steps, DEFAULT_SAVE_FREQUENCY),
        wandb=WandBConfig(enable=False),
    )
    return ResolvedTraining(
        entry_module=entry_module,
        model_config_module=model_config_module,
        policy_type=policy_type,
        policy_fingerprint=fingerprint,
        device=str(policy.device),
        mode=arguments.mode,
        split=split,
        episodes=episodes,
        dataset_root=dataset_root,
        initialization_root=initialization_root,
        output_dir=output_dir,
        config=config,
        prototype_manifest=prototype_manifest,
        route_manifest=route_manifest,
    )


def _require_equal(payload: dict[str, Any], key: str, expected: Any, *, source: Path) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise ValueError(f"{source}: expected {key}={expected!r}, got {actual!r}")


def validate_training_inputs(
    resolved: ResolvedTraining,
    *,
    check_run_destination: bool = True,
) -> None:
    """Validate every external artifact before LeRobot creates the run."""

    if check_run_destination:
        if resolved.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {resolved.output_dir}")
        temporary_log = resolved.output_dir.with_name(f"{resolved.output_dir.name}.train.log")
        if temporary_log.exists():
            raise FileExistsError(f"Previous failed training log exists: {temporary_log}")
    view_manifest = resolved.dataset_root / "view_manifest.json"
    if not view_manifest.is_file():
        raise FileNotFoundError(
            f"Dataset view is missing: {resolved.dataset_root}. Run prepare_dataset.sh first."
        )
    is_mode_policy = resolved.policy_type in {"mode_bc_chunk", "mode_diffusion"}
    if is_mode_policy != (resolved.prototype_manifest is not None):
        raise ValueError("Mode-conditioned policies require exactly one prototype manifest")
    is_route_policy = resolved.policy_type in {
        "route_field_bc_chunk",
        "route_field_diffusion",
    }
    if is_route_policy != (resolved.route_manifest is not None):
        raise ValueError("Route-field policies require exactly one route manifest")
    if resolved.prototype_manifest is not None and resolved.route_manifest is not None:
        raise ValueError("A policy cannot use prototype and route-field manifests together")
    prototype: dict[str, Any] | None = None
    if resolved.prototype_manifest is not None:
        prototype = load_json_object(resolved.prototype_manifest)
        _require_equal(
            prototype,
            "schema_version",
            1,
            source=resolved.prototype_manifest,
        )
        _require_equal(
            prototype,
            "status",
            "complete",
            source=resolved.prototype_manifest,
        )
        _require_equal(
            prototype,
            "split_manifest_sha256",
            resolved.split.manifest_sha256,
            source=resolved.prototype_manifest,
        )
        _require_equal(
            prototype,
            "selected_k",
            resolved.config.policy.num_prototypes,
            source=resolved.prototype_manifest,
        )
    if resolved.route_manifest is not None:
        route = load_json_object(resolved.route_manifest)
        _require_equal(
            route,
            "schema_version",
            ROUTE_FIELD_SCHEMA_VERSION,
            source=resolved.route_manifest,
        )
        _require_equal(route, "status", "complete", source=resolved.route_manifest)
        _require_equal(
            route,
            "split_manifest_sha256",
            resolved.split.manifest_sha256,
            source=resolved.route_manifest,
        )
        grid = route.get("grid")
        field = route.get("field")
        if not isinstance(grid, dict) or not isinstance(field, dict):
            raise TypeError(f"{resolved.route_manifest}: grid and field must be objects")
        _require_equal(
            grid,
            "size",
            resolved.config.policy.route_grid_size,
            source=resolved.route_manifest,
        )
        channels = field.get("channels")
        if not isinstance(channels, list):
            raise TypeError(f"{resolved.route_manifest}: field.channels must be an array")
        if len(channels) != resolved.config.policy.route_field_channels:
            raise ValueError(
                f"{resolved.route_manifest}: expected "
                f"{resolved.config.policy.route_field_channels} channels, got {channels}"
            )
        training = route.get("training")
        if not isinstance(training, dict):
            raise TypeError(f"{resolved.route_manifest}: training must be an object")
        eligible_episodes = {int(value) for value in training.get("episode_ids", ())}
        invalid = sorted(set(resolved.episodes) - eligible_episodes)
        if invalid:
            raise ValueError(
                f"Resolved training includes episodes without route fields: {invalid}"
            )
    if resolved.initialization_root is None:
        if resolved.config.policy.pretrained_path is not None:
            raise ValueError(
                "Randomly initialized training must not set policy.pretrained_path"
            )
        return
    missing = sorted(
        name
        for name in _REQUIRED_INITIALIZATION_FILES
        if not (resolved.initialization_root / name).is_file()
    )
    if missing:
        raise FileNotFoundError(
            f"Paired initialization is incomplete at {resolved.initialization_root}: {missing}"
        )
    manifest_path = resolved.initialization_root.parent / "initialization_manifest.json"
    manifest = load_json_object(manifest_path)
    _require_equal(manifest, "schema_version", 2, source=manifest_path)
    _require_equal(manifest, "status", "complete", source=manifest_path)
    _require_equal(manifest, "mode", resolved.mode, source=manifest_path)
    _require_equal(manifest, "train_seed", resolved.config.seed, source=manifest_path)
    _require_equal(manifest, "episodes", list(resolved.episodes), source=manifest_path)
    _require_equal(
        manifest,
        "split_manifest_sha256",
        resolved.split.manifest_sha256,
        source=manifest_path,
    )
    _require_equal(
        manifest,
        "dataset_view_manifest_sha256",
        sha256_file(view_manifest),
        source=manifest_path,
    )
    fingerprints = manifest.get("model_config_fingerprints")
    if not isinstance(fingerprints, dict):
        raise TypeError(f"{manifest_path}: model_config_fingerprints must be an object")
    if fingerprints.get(resolved.policy_type) != resolved.policy_fingerprint:
        raise ValueError(
            f"{manifest_path}: {resolved.policy_type} config fingerprint does not match the "
            "current Python model config; regenerate paired initialization"
        )
    model_files = manifest.get("model_files")
    if not isinstance(model_files, dict):
        raise TypeError(f"{manifest_path}: model_files must be an object")
    model_path = resolved.initialization_root / "model.safetensors"
    _require_equal(
        model_files,
        resolved.policy_type,
        sha256_file(model_path),
        source=manifest_path,
    )
    checkpoint_config = load_json_object(resolved.initialization_root / "config.json")
    _require_equal(
        checkpoint_config,
        "type",
        resolved.policy_type,
        source=resolved.initialization_root / "config.json",
    )
    if resolved.prototype_manifest is not None:
        _require_equal(
            manifest,
            "prototype_manifest_sha256",
            sha256_file(resolved.prototype_manifest),
            source=manifest_path,
        )


def training_worker_arguments(resolved: ResolvedTraining) -> list[str]:
    """Return the canonical arguments accepted by a model-specific worker."""

    arguments = [
        f"--mode={resolved.mode}",
        f"--steps={resolved.config.steps}",
        f"--train-seed={resolved.config.seed}",
        f"--split-file={resolved.split.source_path}",
        f"--output-dir={resolved.output_dir}",
        f"--batch-size={resolved.config.batch_size}",
        f"--num-workers={resolved.config.num_workers}",
        f"--device={resolved.device}",
    ]
    if resolved.prototype_manifest is not None:
        arguments.append(f"--prototype-manifest={resolved.prototype_manifest}")
    if resolved.route_manifest is not None:
        arguments.append(f"--route-manifest={resolved.route_manifest}")
    return arguments


def _resolved_command(resolved: ResolvedTraining) -> str:
    arguments = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "robot-practice",
        "python",
        "-m",
        resolved.entry_module,
        *training_worker_arguments(resolved),
    ]
    return shlex.join(arguments)


def print_training_plan(
    resolved: ResolvedTraining,
    process_context: DistributedProcessContext | None = None,
) -> None:
    print("Command:")
    print(_resolved_command(resolved))
    print("Resolved configuration:")
    print(json.dumps(resolved.summary(process_context), indent=2, sort_keys=True))


__all__ = [
    "DistributedProcessContext",
    "ResolvedTraining",
    "TrainArguments",
    "build_train_pipeline_config",
    "parse_train_arguments",
    "print_training_plan",
    "resolve_device",
    "resolve_training_output_dir",
    "training_worker_arguments",
    "validate_training_inputs",
]
