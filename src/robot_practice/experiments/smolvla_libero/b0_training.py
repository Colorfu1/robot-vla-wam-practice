"""Contracts and runtime adapters for SmolVLA LIBERO B0 training."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TextIO

from .data_preparation import (
    EpisodeCatalogEntry,
    load_episode_catalog,
    sha256_file,
)

SCHEMA_VERSION = 1
DATASET_REPO_ID = "lerobot/libero"
BASE_POLICY_REPO_ID = "lerobot/smolvla_base"
EXCLUDED_TARGET_TASK_INDICES = (10, 17, 18)
EXPECTED_TASKS = 37
EXPECTED_TRAIN_EPISODES = 1412
EXPECTED_TRAIN_FRAMES = 236470
EXPECTED_TOTAL_EPISODES = 1693
EXPECTED_TOTAL_FRAMES = 273465
EXPECTED_STATE_DIM = 8
EXPECTED_ACTION_DIM = 7
EXPECTED_FPS = 10
CAMERA_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

FORMAL_WORLD_SIZE = 8
FORMAL_BATCH_SIZE = 8
FORMAL_GLOBAL_BATCH_SIZE = 64
FORMAL_STEPS = 30000
FORMAL_SAVE_FREQ = 5000
FORMAL_LEARNING_RATE = 1e-4
FORMAL_TRAIN_SEED = 1000
FORMAL_MIXED_PRECISION = "bf16"
FORMAL_LOG_FREQ = 100

OPTIMIZER_PRESET = {
    "lr": 1e-4,
    "weight_decay": 1e-10,
    "grad_clip_norm": 10,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
}
SCHEDULER_PRESET = {
    "num_warmup_steps": 1000,
    "num_decay_steps": 30000,
    "peak_lr": 1e-4,
    "decay_lr": 2.5e-6,
}

REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_BASE_POLICY_REFERENCE = "policies/smolvla_or_pi0/smolvla_base"
LOCAL_BASE_POLICY_PATH = REPO_ROOT / LOCAL_BASE_POLICY_REFERENCE
OFFLINE_VLM_ASSETS_DIRNAME = "vlm_assets"
LOCAL_VLM_ASSETS_REFERENCE = f"{LOCAL_BASE_POLICY_REFERENCE}/{OFFLINE_VLM_ASSETS_DIRNAME}"
_REPOSITORY_PATH_ANCHORS = frozenset(
    {
        "configs",
        "datasets",
        "docker",
        "docs",
        "envs",
        "eval",
        "outputs",
        "policies",
        "reports",
        "scripts",
        "src",
        "tests",
    }
)
REQUIRED_OFFLINE_VLM_ASSET_FILES = (
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_STEP_NAME = re.compile(r"^[0-9]{6}$")
_FRESH_ONLY_OPTIONS = {
    "--mode",
    "--base-policy",
    "--dataset-root",
    "--split-file",
    "--batch-size",
    "--steps",
    "--save-freq",
    "--learning-rate",
    "--train-seed",
    "--num-workers",
    "--output-dir",
    "--overwrite",
}
_REQUIRED_POLICY_FILES = {
    "config.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_preprocessor.json",
    "train_config.json",
}
_REQUIRED_TRAINING_STATE_FILES = {
    "optimizer_param_groups.json",
    "optimizer_state.safetensors",
    "rng_state.safetensors",
    "scheduler_state.json",
    "training_step.json",
}


class _UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class BackgroundSplit:
    source: Path
    sha256: str
    tasks: tuple[int, ...]
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    train_frames: int
    validation_frames: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class DatasetAudit:
    dataset_root: Path
    total_episodes: int
    total_frames: int
    train_episodes: int
    train_frames: int
    fps: int
    features: dict[str, Any]


@dataclass(frozen=True)
class FreshArguments:
    mode: str
    base_policy: str
    dataset_root: Path
    split_file: Path
    num_processes: int
    batch_size: int
    steps: int
    save_freq: int
    learning_rate: float
    train_seed: int
    num_workers: int
    mixed_precision: str
    output_dir: Path
    overwrite: bool = False


@dataclass(frozen=True)
class ResumeArguments:
    resume_from: Path
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class LaunchInspection:
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class ResumeContext:
    checkpoint_dir: Path
    run_dir: Path
    config_path: Path
    training_state_dir: Path
    saved_step: int
    attempt_id: int
    previous_metrics_tail_step: int | None
    run_config: dict[str, Any]
    split: BackgroundSplit
    dataset_audit: DatasetAudit


@dataclass(frozen=True)
class BasePolicyIdentity:
    requested: str
    source_type: str
    resolved_revision: str | None
    files_sha256: dict[str, str]


@dataclass
class MetricsSink:
    stream: TextIO
    attempt_id: int
    initial_step: int
    started_monotonic: float

    def write(self, payload: Mapping[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("attempt_id", self.attempt_id)
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {source}")
    return payload


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(destination)
    return destination


def _canonical_option(argument: str) -> str | None:
    if not argument.startswith("--"):
        return None
    return argument.split("=", maxsplit=1)[0]


def _reject_duplicate_options(parser: argparse.ArgumentParser, arguments: Sequence[str]) -> None:
    seen: set[str] = set()
    for argument in arguments:
        option = _canonical_option(argument)
        if option is None:
            continue
        if option in seen:
            parser.error(f"duplicate argument: {option}")
        seen.add(option)


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> FreshArguments | ResumeArguments | LaunchInspection:
    raw = tuple(sys.argv[1:] if argv is None else argv)
    parser = _UsageParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"))
    parser.add_argument("--base-policy", default=LOCAL_BASE_POLICY_REFERENCE)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--num-processes", type=_positive_integer)
    parser.add_argument("--batch-size", type=_positive_integer)
    parser.add_argument("--steps", type=_positive_integer)
    parser.add_argument("--save-freq", type=_positive_integer)
    parser.add_argument("--learning-rate", type=_positive_float)
    parser.add_argument("--train-seed", type=_nonnegative_integer)
    parser.add_argument("--num-workers", type=_nonnegative_integer, default=4)
    parser.add_argument("--mixed-precision", choices=("bf16",))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate --output-dir so a fresh run replaces previous artifacts",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--inspect-launch", action="store_true", help=argparse.SUPPRESS)
    _reject_duplicate_options(parser, raw)
    namespace = parser.parse_args(raw)
    specified = {_canonical_option(argument) for argument in raw}

    if namespace.resume_from is not None:
        invalid = sorted(option for option in _FRESH_ONLY_OPTIONS if option in specified)
        if invalid:
            parser.error(f"resume does not accept fresh-run options: {', '.join(invalid)}")
        if namespace.inspect_launch:
            if "--num-processes" in specified or "--mixed-precision" in specified:
                parser.error("resume launch settings are restored from run_config.json")
            world_size, mixed_precision = inspect_resume_launch(namespace.resume_from)
            return LaunchInspection(world_size, mixed_precision)
        if namespace.num_processes is None or namespace.mixed_precision is None:
            parser.error("resume runtime requires internal --num-processes and --mixed-precision")
        return ResumeArguments(
            resume_from=resolve_repository_path(namespace.resume_from),
            num_processes=namespace.num_processes,
            mixed_precision=namespace.mixed_precision,
        )

    if namespace.mode is None:
        parser.error("--mode is required for a fresh run")
    if namespace.dataset_root is None:
        parser.error("--dataset-root is required for a fresh run")
    if namespace.split_file is None:
        parser.error("--split-file is required for a fresh run")
    if namespace.output_dir is None:
        parser.error("--output-dir is required for a fresh run")

    formal = namespace.mode == "formal"
    arguments = FreshArguments(
        mode=namespace.mode,
        base_policy=normalize_base_policy_reference(namespace.base_policy),
        dataset_root=resolve_repository_path(namespace.dataset_root),
        split_file=resolve_repository_path(namespace.split_file),
        num_processes=namespace.num_processes
        if namespace.num_processes is not None
        else (FORMAL_WORLD_SIZE if formal else 1),
        batch_size=namespace.batch_size
        if namespace.batch_size is not None
        else (FORMAL_BATCH_SIZE if formal else 1),
        steps=namespace.steps if namespace.steps is not None else (FORMAL_STEPS if formal else 2),
        save_freq=namespace.save_freq
        if namespace.save_freq is not None
        else (FORMAL_SAVE_FREQ if formal else 2),
        learning_rate=namespace.learning_rate
        if namespace.learning_rate is not None
        else FORMAL_LEARNING_RATE,
        train_seed=namespace.train_seed if namespace.train_seed is not None else FORMAL_TRAIN_SEED,
        num_workers=namespace.num_workers,
        mixed_precision=namespace.mixed_precision or FORMAL_MIXED_PRECISION,
        output_dir=resolve_repository_path(namespace.output_dir),
        overwrite=namespace.overwrite,
    )
    validate_fresh_contract(arguments)
    if namespace.inspect_launch:
        return LaunchInspection(arguments.num_processes, arguments.mixed_precision)
    return arguments


def validate_fresh_contract(arguments: FreshArguments) -> None:
    if arguments.output_dir.exists() and not arguments.overwrite:
        raise FileExistsError(f"Output directory already exists: {arguments.output_dir}")
    if arguments.base_policy == BASE_POLICY_REPO_ID:
        raise ValueError(
            "SmolVLA B0 training is offline-only; materialize the local base policy "
            "with scripts/experiments/smolvla_libero/materialize_base_policy.sh"
        )
    if arguments.save_freq > arguments.steps:
        raise ValueError("save frequency cannot exceed total steps")
    if arguments.learning_rate != FORMAL_LEARNING_RATE:
        raise ValueError(f"B0 requires learning_rate={FORMAL_LEARNING_RATE}")
    if arguments.train_seed != FORMAL_TRAIN_SEED:
        raise ValueError(f"B0 requires train_seed={FORMAL_TRAIN_SEED}")
    if arguments.mode != "formal":
        return
    expected = {
        "base_policy": LOCAL_BASE_POLICY_REFERENCE,
        "num_processes": FORMAL_WORLD_SIZE,
        "batch_size": FORMAL_BATCH_SIZE,
        "steps": FORMAL_STEPS,
        "save_freq": FORMAL_SAVE_FREQ,
        "learning_rate": FORMAL_LEARNING_RATE,
        "train_seed": FORMAL_TRAIN_SEED,
        "mixed_precision": FORMAL_MIXED_PRECISION,
    }
    for field, value in expected.items():
        actual = getattr(arguments, field)
        if actual != value:
            raise ValueError(f"Formal B0 requires {field}={value!r}, got {actual!r}")
    if arguments.batch_size * arguments.num_processes != FORMAL_GLOBAL_BATCH_SIZE:
        raise ValueError("Formal B0 requires global batch size 64")


def _integer_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise TypeError(f"{field} must be a list of integers")
    return [int(item) for item in value]


def load_background_split(path: str | Path) -> BackgroundSplit:
    source = resolve_repository_path(path)
    payload = load_json_object(source)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported background split schema")
    if payload.get("dataset_repo_id") != DATASET_REPO_ID:
        raise ValueError("Background split dataset repo ID mismatch")
    if payload.get("split_seed") != FORMAL_TRAIN_SEED:
        raise ValueError("Background split seed mismatch")
    excluded = _integer_list(
        payload.get("excluded_target_task_indices"),
        field="excluded_target_task_indices",
    )
    if tuple(excluded) != EXCLUDED_TARGET_TASK_INDICES:
        raise ValueError("Background split target exclusions mismatch")
    expected_tasks = set(range(EXPECTED_TASKS + len(EXCLUDED_TARGET_TASK_INDICES)))
    expected_tasks.difference_update(EXCLUDED_TARGET_TASK_INDICES)
    tasks_payload = payload.get("tasks")
    if not isinstance(tasks_payload, dict):
        raise TypeError("Background split tasks must be an object")

    tasks: list[int] = []
    train_episodes: list[int] = []
    validation_episodes: list[int] = []
    train_frames = 0
    validation_frames = 0
    for raw_task_index, task_payload in sorted(
        tasks_payload.items(), key=lambda item: int(item[0])
    ):
        if not isinstance(task_payload, dict):
            raise TypeError(f"Task {raw_task_index} split must be an object")
        task_index = int(raw_task_index)
        if task_payload.get("task_index") != task_index:
            raise ValueError(f"Task key and task_index disagree for {raw_task_index}")
        if task_index in EXCLUDED_TARGET_TASK_INDICES:
            raise ValueError(f"Target task {task_index} appears in B0 split")
        task_train = _integer_list(
            task_payload.get("train_episode_indices"),
            field=f"tasks.{task_index}.train_episode_indices",
        )
        task_validation = _integer_list(
            task_payload.get("validation_episode_indices"),
            field=f"tasks.{task_index}.validation_episode_indices",
        )
        if task_train != sorted(task_train) or task_validation != sorted(task_validation):
            raise ValueError(f"Task {task_index} episode lists must be sorted")
        if len(task_train) != len(set(task_train)):
            raise ValueError(f"Task {task_index} contains duplicate train episodes")
        if len(task_validation) != len(set(task_validation)):
            raise ValueError(f"Task {task_index} contains duplicate validation episodes")
        if set(task_train) & set(task_validation):
            raise ValueError(f"Task {task_index} train/validation episodes overlap")
        task_train_frames = task_payload.get("train_frames")
        task_validation_frames = task_payload.get("validation_frames")
        if not isinstance(task_train_frames, int) or not isinstance(task_validation_frames, int):
            raise TypeError(f"Task {task_index} frame counts must be integers")
        tasks.append(task_index)
        train_episodes.extend(task_train)
        validation_episodes.extend(task_validation)
        train_frames += task_train_frames
        validation_frames += task_validation_frames

    if len(tasks) != len(set(tasks)):
        raise ValueError("Background split contains duplicate normalized task indices")
    if set(tasks) != expected_tasks:
        raise ValueError("Background split task indices do not match the 37-task contract")
    if len(train_episodes) != len(set(train_episodes)):
        raise ValueError("An episode appears in multiple B0 train tasks")
    if len(validation_episodes) != len(set(validation_episodes)):
        raise ValueError("An episode appears in multiple B0 validation tasks")
    if set(train_episodes) & set(validation_episodes):
        raise ValueError("B0 train and validation episodes overlap globally")
    if len(train_episodes) != EXPECTED_TRAIN_EPISODES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_EPISODES} B0 train episodes, got {len(train_episodes)}"
        )
    if train_frames != EXPECTED_TRAIN_FRAMES:
        raise ValueError(f"Expected {EXPECTED_TRAIN_FRAMES} B0 train frames, got {train_frames}")

    aggregate = payload.get("aggregate")
    expected_aggregate = {
        "num_tasks": len(tasks),
        "train_episodes": len(train_episodes),
        "validation_episodes": len(validation_episodes),
        "train_frames": train_frames,
        "validation_frames": validation_frames,
    }
    if aggregate != expected_aggregate:
        raise ValueError("Background split aggregate does not match task records")
    return BackgroundSplit(
        source=source,
        sha256=sha256_file(source),
        tasks=tuple(tasks),
        train_episodes=tuple(sorted(train_episodes)),
        validation_episodes=tuple(sorted(validation_episodes)),
        train_frames=train_frames,
        validation_frames=validation_frames,
        payload=payload,
    )


def validate_split_against_catalog(
    split: BackgroundSplit,
    catalog: Mapping[int, EpisodeCatalogEntry],
) -> None:
    task_payloads = split.payload["tasks"]
    observed_train_frames = 0
    observed_validation_frames = 0
    for raw_task_index, task_payload in task_payloads.items():
        task_index = int(raw_task_index)
        for field, frame_field in (
            ("train_episode_indices", "train_frames"),
            ("validation_episode_indices", "validation_frames"),
        ):
            frame_count = 0
            for episode_index in task_payload[field]:
                entry = catalog.get(int(episode_index))
                if entry is None:
                    raise ValueError(f"Split episode {episode_index} is absent from dataset")
                if entry.task_index != task_index:
                    raise ValueError(
                        f"Episode {episode_index} belongs to task {entry.task_index}, "
                        f"not split task {task_index}"
                    )
                frame_count += entry.length
            if frame_count != task_payload[frame_field]:
                raise ValueError(
                    f"Task {task_index} {field} frame count mismatch: "
                    f"{frame_count} != {task_payload[frame_field]}"
                )
            if field == "train_episode_indices":
                observed_train_frames += frame_count
            else:
                observed_validation_frames += frame_count
    if observed_train_frames != split.train_frames:
        raise ValueError("Dataset B0 train frames disagree with split")
    if observed_validation_frames != split.validation_frames:
        raise ValueError("Dataset B0 validation frames disagree with split")


def validate_dataset(
    dataset_root: str | Path,
    split: BackgroundSplit,
) -> DatasetAudit:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = resolve_repository_path(dataset_root)
    catalog, _ = load_episode_catalog(root)
    validate_split_against_catalog(split, catalog)
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=root)
    if int(metadata.total_episodes) != EXPECTED_TOTAL_EPISODES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_EPISODES} dataset episodes, got {metadata.total_episodes}"
        )
    if int(metadata.total_frames) != EXPECTED_TOTAL_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_FRAMES} dataset frames, got {metadata.total_frames}"
        )
    required_features = {
        "observation.images.image": ("video", (256, 256, 3)),
        "observation.images.image2": ("video", (256, 256, 3)),
        "observation.state": ("float32", (EXPECTED_STATE_DIM,)),
        "action": ("float32", (EXPECTED_ACTION_DIM,)),
    }
    for key, (expected_dtype, expected_shape) in required_features.items():
        feature = metadata.features.get(key)
        if not isinstance(feature, dict):
            raise TypeError(f"Dataset feature is missing or malformed: {key}")
        if feature.get("dtype") != expected_dtype:
            raise ValueError(f"Dataset feature {key} dtype mismatch")
        if tuple(feature.get("shape", ())) != expected_shape:
            raise ValueError(f"Dataset feature {key} shape mismatch")
    if int(metadata.fps) != EXPECTED_FPS:
        raise ValueError(f"Expected dataset fps={EXPECTED_FPS}, got {metadata.fps}")
    return DatasetAudit(
        dataset_root=root,
        total_episodes=int(metadata.total_episodes),
        total_frames=int(metadata.total_frames),
        train_episodes=len(split.train_episodes),
        train_frames=split.train_frames,
        fps=int(metadata.fps),
        features={
            key: {
                "dtype": metadata.features[key]["dtype"],
                "shape": list(metadata.features[key]["shape"]),
            }
            for key in required_features
        },
    )


def configure_smolvla_policy(
    pretrained_path: str | Path,
    *,
    device: str,
    learning_rate: float,
):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    resolved_pretrained_path = resolve_repository_path(pretrained_path)
    policy = PreTrainedConfig.from_pretrained(resolved_pretrained_path)
    if not isinstance(policy, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(policy).__name__}")
    policy.pretrained_path = Path(portable_path(resolved_pretrained_path))
    policy.vlm_model_name = portable_path(
        validate_local_vlm_assets(resolved_pretrained_path / OFFLINE_VLM_ASSETS_DIRNAME)
    )
    # The outer SmolVLA model.safetensors already contains every VLM tensor.
    # Construct the nested VLM from local config only, then let the outer
    # checkpoint load all weights exactly once.
    policy.load_vlm_weights = False
    policy.device = device
    policy.use_amp = False
    policy.use_peft = False
    policy.push_to_hub = False
    policy.repo_id = None
    policy.n_obs_steps = 1
    policy.chunk_size = 50
    policy.n_action_steps = 10
    policy.freeze_vision_encoder = True
    policy.train_expert_only = True
    policy.train_state_proj = True
    policy.optimizer_lr = learning_rate
    policy.optimizer_betas = tuple(OPTIMIZER_PRESET["betas"])
    policy.optimizer_eps = float(OPTIMIZER_PRESET["eps"])
    policy.optimizer_weight_decay = float(OPTIMIZER_PRESET["weight_decay"])
    policy.optimizer_grad_clip_norm = float(OPTIMIZER_PRESET["grad_clip_norm"])
    policy.scheduler_warmup_steps = int(SCHEDULER_PRESET["num_warmup_steps"])
    policy.scheduler_decay_steps = int(SCHEDULER_PRESET["num_decay_steps"])
    policy.scheduler_decay_lr = float(SCHEDULER_PRESET["decay_lr"])
    policy.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(EXPECTED_STATE_DIM,),
        ),
        "observation.images.camera1": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 256, 256),
        ),
        "observation.images.camera2": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 256, 256),
        ),
    }
    policy.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(EXPECTED_ACTION_DIM,),
        )
    }
    validate_policy_preset(policy)
    return policy


def _asdict(value: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(value):
        raise TypeError(f"Expected dataclass config, got {type(value).__name__}")
    return dataclasses.asdict(value)


def validate_policy_preset(
    policy: Any,
    *,
    expected_learning_rate: float | None = None,
) -> None:
    if policy.type != "smolvla":
        raise ValueError(f"Expected policy type smolvla, got {policy.type}")
    expected_fields = {
        "use_peft": False,
        "use_amp": False,
        "push_to_hub": False,
        "freeze_vision_encoder": True,
        "train_expert_only": True,
        "train_state_proj": True,
        "load_vlm_weights": False,
        "n_obs_steps": 1,
        "chunk_size": 50,
        "n_action_steps": 10,
    }
    for field, expected in expected_fields.items():
        actual = getattr(policy, field)
        if actual != expected:
            raise ValueError(f"SmolVLA B0 requires {field}={expected!r}, got {actual!r}")
    optimizer = _asdict(policy.get_optimizer_preset())
    scheduler = _asdict(policy.get_scheduler_preset())
    expected_optimizer = dict(OPTIMIZER_PRESET)
    expected_scheduler = dict(SCHEDULER_PRESET)
    if expected_learning_rate is not None:
        # Diagnostic runs may scale the learning rate while every other
        # optimizer/scheduler field stays frozen.
        expected_optimizer["lr"] = float(expected_learning_rate)
        expected_scheduler["peak_lr"] = float(expected_learning_rate)
    if optimizer != expected_optimizer:
        raise ValueError(f"Optimizer preset mismatch: {optimizer}")
    if scheduler != expected_scheduler:
        raise ValueError(f"Scheduler preset mismatch: {scheduler}")
    actual_inputs = {
        key: (getattr(feature.type, "value", str(feature.type)), tuple(feature.shape))
        for key, feature in policy.input_features.items()
    }
    expected_inputs = {
        "observation.state": ("STATE", (EXPECTED_STATE_DIM,)),
        "observation.images.camera1": ("VISUAL", (3, 256, 256)),
        "observation.images.camera2": ("VISUAL", (3, 256, 256)),
    }
    if actual_inputs != expected_inputs:
        raise ValueError(f"SmolVLA B0 input feature mismatch: {actual_inputs}")
    actual_outputs = {
        key: (getattr(feature.type, "value", str(feature.type)), tuple(feature.shape))
        for key, feature in policy.output_features.items()
    }
    expected_outputs = {"action": ("ACTION", (EXPECTED_ACTION_DIM,))}
    if actual_outputs != expected_outputs:
        raise ValueError(f"SmolVLA B0 output feature mismatch: {actual_outputs}")


def build_train_config(
    arguments: FreshArguments,
    split: BackgroundSplit,
    policy: Any,
):
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    optimizer = policy.get_optimizer_preset()
    scheduler = policy.get_scheduler_preset()
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=DATASET_REPO_ID,
            root=portable_path(arguments.dataset_root),
            episodes=list(split.train_episodes),
            use_imagenet_stats=True,
        ),
        policy=policy,
        output_dir=Path(portable_path(arguments.output_dir)),
        job_name="smolvla_libero_b0_domain_adaptation",
        seed=arguments.train_seed,
        num_workers=arguments.num_workers,
        batch_size=arguments.batch_size,
        steps=arguments.steps,
        eval_freq=0,
        log_freq=min(arguments.steps, FORMAL_LOG_FREQ),
        save_checkpoint=True,
        save_freq=arguments.save_freq,
        use_policy_training_preset=True,
        optimizer=optimizer,
        scheduler=scheduler,
        wandb=WandBConfig(enable=False),
        rename_map=dict(CAMERA_RENAME_MAP),
    )


def _hash_identity_files(root: Path) -> dict[str, str]:
    candidates = {
        path
        for pattern in (
            "config.json",
            "model*.safetensors",
            "model*.index.json",
            "policy_preprocessor*.json",
            "policy_preprocessor*.safetensors",
            "policy_postprocessor*.json",
            "policy_postprocessor*.safetensors",
            f"{OFFLINE_VLM_ASSETS_DIRNAME}/*",
        )
        for path in root.glob(pattern)
        if path.is_file()
    }
    if not any(path.name == "config.json" for path in candidates):
        raise FileNotFoundError(f"Base policy config.json is missing: {root}")
    if not any(path.name.startswith("model") for path in candidates):
        raise FileNotFoundError(f"Base policy weights are missing: {root}")
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(candidates)}


def normalize_base_policy_reference(reference: str) -> str:
    if reference == BASE_POLICY_REPO_ID:
        return reference
    return portable_path(resolve_repository_path(reference))


def validate_local_vlm_assets(root: str | Path) -> Path:
    resolved = resolve_repository_path(root)
    if not resolved.is_dir():
        raise FileNotFoundError(
            "Local SmolVLA base policy is missing offline VLM assets: "
            f"{resolved}. Run scripts/experiments/smolvla_libero/"
            "materialize_base_policy.sh on a machine with the cached "
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct snapshot."
        )
    symlink = next((path for path in resolved.rglob("*") if path.is_symlink()), None)
    if symlink is not None:
        raise ValueError(
            "Offline VLM assets must be self-contained; found symlink: "
            f"{symlink.relative_to(resolved)}"
        )
    missing = [
        filename
        for filename in REQUIRED_OFFLINE_VLM_ASSET_FILES
        if not (resolved / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Offline VLM assets are incomplete; missing {', '.join(missing)} in {resolved}"
        )

    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(resolved, local_files_only=True)
    if getattr(config, "model_type", None) != "smolvlm":
        raise ValueError(
            "Offline VLM assets must describe a SmolVLM model, got "
            f"{getattr(config, 'model_type', None)!r}"
        )
    AutoProcessor.from_pretrained(resolved, local_files_only=True)
    return resolved


def validate_local_base_policy(root: str | Path) -> Path:
    resolved = resolve_repository_path(root)
    if not resolved.is_dir():
        raise FileNotFoundError(
            "Local SmolVLA base policy is missing: "
            f"{resolved}. Run scripts/experiments/smolvla_libero/"
            "materialize_base_policy.sh before offline training."
        )
    symlink = next((path for path in resolved.rglob("*") if path.is_symlink()), None)
    if symlink is not None:
        raise ValueError(
            "Local SmolVLA base policy must be self-contained; found symlink: "
            f"{symlink.relative_to(resolved)}"
        )
    _hash_identity_files(resolved)
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (resolved / filename).is_file():
            raise FileNotFoundError(f"Local SmolVLA base policy is missing {filename}: {resolved}")
    validate_local_vlm_assets(resolved / OFFLINE_VLM_ASSETS_DIRNAME)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    policy = PreTrainedConfig.from_pretrained(resolved)
    if not isinstance(policy, SmolVLAConfig):
        raise TypeError(f"Local base policy must be a SmolVLA config, got {type(policy).__name__}")
    return resolved


def resolve_base_policy(
    reference: str,
    *,
    main_process_first: Callable[[], Any] | None = None,
) -> tuple[Path, str | None]:
    candidate = resolve_repository_path(reference)

    @contextmanager
    def no_op_context() -> Iterator[None]:
        yield

    context_factory = main_process_first or no_op_context
    with context_factory():
        if reference == BASE_POLICY_REPO_ID:
            raise ValueError(
                "SmolVLA B0 training is offline-only; Hub base policies are not allowed"
            )
        else:
            resolved = validate_local_base_policy(candidate)
    revision = (
        resolved.name
        if resolved.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", resolved.name)
        else None
    )
    return resolved, revision


def identify_base_policy(
    requested: str,
    resolved_path: Path,
    resolved_revision: str | None,
) -> BasePolicyIdentity:
    return BasePolicyIdentity(
        requested=requested,
        source_type="huggingface_hub" if requested == BASE_POLICY_REPO_ID else "local",
        resolved_revision=resolved_revision,
        files_sha256=_hash_identity_files(resolved_path),
    )


def package_versions() -> dict[str, str | None]:
    names = ("accelerate", "lerobot", "torch", "transformers")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _legacy_repository_path(path: Path) -> Path | None:
    """Map a missing absolute path from another checkout onto this repository."""

    for index in range(len(path.parts) - 1, -1, -1):
        if path.parts[index] not in _REPOSITORY_PATH_ANCHORS:
            continue
        candidate = (REPO_ROOT / Path(*path.parts[index:])).resolve()
        if candidate.exists():
            return candidate
    return None


def resolve_repository_path(path: str | Path) -> Path:
    """Resolve repository-relative paths independently of the caller's cwd."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return (REPO_ROOT / candidate).resolve()
    resolved = candidate.resolve()
    if resolved.exists():
        return resolved
    return _legacy_repository_path(resolved) or resolved


def portable_path(path: str | Path) -> str:
    resolved = resolve_repository_path(path)
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_run_config(
    *,
    arguments: FreshArguments,
    split: BackgroundSplit,
    dataset_audit: DatasetAudit,
    base_identity: BasePolicyIdentity,
    train_config: Any,
) -> dict[str, Any]:
    resolved_train_config = train_config.to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "smolvla_libero_b0",
        "result_label": "Formal" if arguments.mode == "formal" else "Smoke",
        "created_at": utc_now(),
        "inputs": {
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_root": portable_path(arguments.dataset_root),
            "split_file": portable_path(split.source),
            "split_sha256": split.sha256,
            "overwrite": arguments.overwrite,
            "base_policy": dataclasses.asdict(base_identity),
        },
        "dataset": {
            "tasks": len(split.tasks),
            "train_episodes": len(split.train_episodes),
            "train_frames": split.train_frames,
            "validation_episodes": len(split.validation_episodes),
            "validation_frames": split.validation_frames,
            "fps": dataset_audit.fps,
            "features": dataset_audit.features,
            "sampler": "lerobot_native_global_frame_shuffle",
            "normalization_scope": {
                "visual": "imagenet",
                "state_action": "full_dataset_metadata",
            },
        },
        "training": {
            "mode": arguments.mode,
            "world_size": arguments.num_processes,
            "per_gpu_batch_size": arguments.batch_size,
            "global_batch_size": arguments.num_processes * arguments.batch_size,
            "mixed_precision": arguments.mixed_precision,
            "steps": arguments.steps,
            "save_freq": arguments.save_freq,
            "seed": arguments.train_seed,
            "num_workers_per_process": arguments.num_workers,
            "optimizer": resolved_train_config["optimizer"],
            "scheduler": resolved_train_config["scheduler"],
            "use_policy_training_preset": train_config.use_policy_training_preset,
            "output_dir": portable_path(arguments.output_dir),
        },
        "resolved_train_config": resolved_train_config,
        "versions": package_versions(),
    }


def checkpoint_run_dir(checkpoint_dir: str | Path) -> Path:
    checkpoint = resolve_repository_path(checkpoint_dir)
    if checkpoint.parent.name != "checkpoints" or not _STEP_NAME.fullmatch(checkpoint.name):
        raise ValueError("--resume-from must point to <run>/checkpoints/<six-digit-step>")
    return checkpoint.parent.parent


def _complete_checkpoint(path: Path) -> bool:
    policy_dir = path / "pretrained_model"
    state_dir = path / "training_state"
    return all((policy_dir / name).is_file() for name in _REQUIRED_POLICY_FILES) and all(
        (state_dir / name).is_file() for name in _REQUIRED_TRAINING_STATE_FILES
    )


def latest_complete_checkpoint(run_dir: str | Path) -> Path:
    checkpoints_root = Path(run_dir) / "checkpoints"
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {checkpoints_root}")
    complete = sorted(
        (
            child
            for child in checkpoints_root.iterdir()
            if child.is_dir() and _STEP_NAME.fullmatch(child.name) and _complete_checkpoint(child)
        ),
        key=lambda path: int(path.name),
    )
    if not complete:
        raise FileNotFoundError(f"No complete checkpoints found in {checkpoints_root}")
    return complete[-1]


def inspect_resume_launch(resume_from: str | Path) -> tuple[int, str]:
    run_dir = checkpoint_run_dir(resume_from)
    run_config = load_json_object(run_dir / "run_config.json")
    if run_config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported B0 run_config schema")
    if run_config.get("experiment") != "smolvla_libero_b0":
        raise ValueError("Checkpoint is not a SmolVLA LIBERO B0 run")
    training = run_config.get("training")
    if not isinstance(training, dict):
        raise TypeError("run_config.json lacks training settings")
    world_size = training.get("world_size")
    mixed_precision = training.get("mixed_precision")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("Saved world_size is invalid")
    if mixed_precision != FORMAL_MIXED_PRECISION:
        raise ValueError("Saved mixed precision is not bf16")
    return world_size, mixed_precision


def _read_training_step(training_state_dir: Path) -> int:
    payload = load_json_object(training_state_dir / "training_step.json")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("training_step.json contains an invalid step")
    return step


def _metrics_history(path: Path) -> tuple[int, int | None]:
    max_attempt = -1
    tail_step: int | None = None
    if not path.is_file():
        raise FileNotFoundError(f"Resume metrics history is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Invalid JSONL record at line {line_number}")
            attempt_id = payload.get("attempt_id")
            if isinstance(attempt_id, int) and not isinstance(attempt_id, bool):
                max_attempt = max(max_attempt, attempt_id)
            step = payload.get("step")
            if isinstance(step, int) and not isinstance(step, bool):
                tail_step = step
    if max_attempt < 0:
        raise ValueError("Resume metrics history contains no attempt_id")
    return max_attempt + 1, tail_step


def _resolve_recorded_path(value: str, *, run_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return resolve_repository_path(path)
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (run_dir / path).resolve()


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise ValueError(f"Resume contract mismatch for {field}: {actual!r} != {expected!r}")


def validate_base_policy_identity(recorded: Any) -> None:
    if not isinstance(recorded, dict):
        raise TypeError("B0 run_config base policy identity is invalid")
    requested = recorded.get("requested")
    source_type = recorded.get("source_type")
    resolved_revision = recorded.get("resolved_revision")
    if not isinstance(requested, str) or source_type not in {
        "local",
        "huggingface_hub",
    }:
        raise ValueError("B0 run_config base policy reference is invalid")

    if source_type == "local" and resolved_revision is not None:
        raise ValueError("Local B0 base policy must not have a Hub revision")
    if source_type == "huggingface_hub" and (
        not isinstance(resolved_revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", resolved_revision)
    ):
        raise ValueError("Hub B0 base policy lacks an immutable resolved revision")
    files_sha256 = recorded.get("files_sha256")
    if not isinstance(files_sha256, dict) or not files_sha256:
        raise ValueError("B0 run_config base policy hashes are missing")
    for filename, digest in files_sha256.items():
        if not isinstance(filename, str) or not filename:
            raise ValueError("B0 run_config contains an invalid base policy filename")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"B0 run_config contains an invalid SHA-256 for {filename}")
    if "config.json" not in files_sha256:
        raise ValueError("B0 run_config base policy config hash is missing")
    if not any(filename.startswith("model") for filename in files_sha256):
        raise ValueError("B0 run_config base policy weight hash is missing")


def validate_resume(
    arguments: ResumeArguments,
) -> ResumeContext:
    checkpoint_dir = arguments.resume_from
    run_dir = checkpoint_run_dir(checkpoint_dir)
    if not _complete_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Resume checkpoint is incomplete: {checkpoint_dir}")
    latest = latest_complete_checkpoint(run_dir)
    if checkpoint_dir != latest:
        raise ValueError(f"Resume requires latest complete checkpoint: {latest}")

    run_config = load_json_object(run_dir / "run_config.json")
    if run_config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported B0 run_config schema")
    if run_config.get("experiment") != "smolvla_libero_b0":
        raise ValueError("Checkpoint is not a SmolVLA LIBERO B0 run")
    training = run_config.get("training")
    inputs = run_config.get("inputs")
    if not isinstance(training, dict) or not isinstance(inputs, dict):
        raise TypeError("B0 run_config lacks training or inputs")
    _require_equal(
        arguments.num_processes,
        training.get("world_size"),
        field="world_size",
    )
    _require_equal(
        arguments.mixed_precision,
        training.get("mixed_precision"),
        field="mixed_precision",
    )
    world_size = training.get("world_size")
    per_gpu_batch = training.get("per_gpu_batch_size")
    if not isinstance(world_size, int) or not isinstance(per_gpu_batch, int):
        raise TypeError("B0 run_config batch contract is invalid")
    _require_equal(
        training.get("global_batch_size"),
        world_size * per_gpu_batch,
        field="global_batch_size",
    )
    mode = training.get("mode")
    if mode not in {"smoke", "formal"}:
        raise ValueError("B0 run_config mode is invalid")
    _require_equal(training.get("seed"), FORMAL_TRAIN_SEED, field="train_seed")
    if mode == "formal":
        formal_contract = {
            "world_size": FORMAL_WORLD_SIZE,
            "per_gpu_batch_size": FORMAL_BATCH_SIZE,
            "global_batch_size": FORMAL_GLOBAL_BATCH_SIZE,
            "mixed_precision": FORMAL_MIXED_PRECISION,
            "steps": FORMAL_STEPS,
            "save_freq": FORMAL_SAVE_FREQ,
        }
        for field, expected in formal_contract.items():
            _require_equal(training.get(field), expected, field=field)
    validate_base_policy_identity(inputs.get("base_policy"))

    split_path_value = inputs.get("split_file")
    dataset_root_value = inputs.get("dataset_root")
    if not isinstance(split_path_value, str) or not isinstance(dataset_root_value, str):
        raise TypeError("B0 run_config input paths are invalid")
    split_path = _resolve_recorded_path(split_path_value, run_dir=run_dir)
    dataset_root = _resolve_recorded_path(dataset_root_value, run_dir=run_dir)
    split = load_background_split(split_path)
    _require_equal(split.sha256, inputs.get("split_sha256"), field="split_sha256")
    dataset_audit = validate_dataset(dataset_root, split)

    config_path = checkpoint_dir / "pretrained_model" / "train_config.json"
    training_state_dir = checkpoint_dir / "training_state"
    saved_config = load_json_object(config_path)
    saved_step = _read_training_step(training_state_dir)
    _require_equal(saved_step, int(checkpoint_dir.name), field="checkpoint_step")
    total_steps = training.get("steps")
    if not isinstance(total_steps, int) or saved_step >= total_steps:
        raise ValueError("Resume checkpoint step must be smaller than configured total steps")
    output_dir_from_config = saved_config.get("output_dir")
    if not isinstance(output_dir_from_config, str):
        raise TypeError("Saved train config output directory is invalid")
    _require_equal(
        _resolve_recorded_path(output_dir_from_config, run_dir=run_dir),
        run_dir,
        field="output_dir",
    )
    dataset = saved_config.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError("Saved train config lacks dataset")
    _require_equal(
        tuple(sorted(dataset.get("episodes", ()))),
        split.train_episodes,
        field="dataset.episodes",
    )
    _require_equal(
        saved_config.get("batch_size"), training.get("per_gpu_batch_size"), field="batch"
    )
    _require_equal(saved_config.get("steps"), total_steps, field="steps")
    _require_equal(saved_config.get("save_freq"), training.get("save_freq"), field="save_freq")
    _require_equal(saved_config.get("seed"), training.get("seed"), field="seed")
    _require_equal(
        saved_config.get("use_policy_training_preset"),
        True,
        field="use_policy_training_preset",
    )
    _require_equal(saved_config.get("optimizer"), training.get("optimizer"), field="optimizer")
    _require_equal(saved_config.get("scheduler"), training.get("scheduler"), field="scheduler")
    _require_equal(
        saved_config.get("rename_map"),
        CAMERA_RENAME_MAP,
        field="rename_map",
    )
    _require_equal(dataset.get("repo_id"), DATASET_REPO_ID, field="dataset.repo_id")
    _require_equal(
        dataset.get("use_imagenet_stats"),
        True,
        field="dataset.use_imagenet_stats",
    )
    dataset_root_from_config = dataset.get("root")
    if not isinstance(dataset_root_from_config, str):
        raise TypeError("Saved train config dataset root is invalid")
    _require_equal(
        _resolve_recorded_path(dataset_root_from_config, run_dir=run_dir),
        dataset_root,
        field="dataset.root",
    )

    from lerobot.configs.policies import PreTrainedConfig

    checkpoint_policy = PreTrainedConfig.from_pretrained(checkpoint_dir / "pretrained_model")
    validate_policy_preset(checkpoint_policy)

    attempt_id, tail_step = _metrics_history(run_dir / "train_metrics.jsonl")
    return ResumeContext(
        checkpoint_dir=checkpoint_dir,
        run_dir=run_dir,
        config_path=config_path,
        training_state_dir=training_state_dir,
        saved_step=saved_step,
        attempt_id=attempt_id,
        previous_metrics_tail_step=tail_step,
        run_config=run_config,
        split=split,
        dataset_audit=dataset_audit,
    )


@contextmanager
def instrument_update_policy(
    lerobot_train_module: Any,
    sink: MetricsSink | None,
) -> Iterator[None]:
    if sink is None:
        yield
        return
    original = lerobot_train_module.update_policy

    def observed_update_policy(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        tracker = result[0]
        values = tracker.to_dict(use_avg=False)
        sink.write(
            {
                "record_type": "train_step",
                "timestamp": utc_now(),
                "wall_time_s": time.perf_counter() - sink.started_monotonic,
                "step": int(tracker.steps) + 1,
                "samples": int(tracker.samples)
                + int(tracker._batch_size) * int(tracker.accelerator.num_processes),
                "epochs": (
                    int(tracker.samples)
                    + int(tracker._batch_size) * int(tracker.accelerator.num_processes)
                )
                / int(tracker._num_frames),
                "loss": float(values["loss"]),
                "grad_norm": float(values["grad_norm"]),
                "lr": float(values["lr"]),
                "dataloading_s": float(values["dataloading_s"]),
                "update_s": float(values["update_s"]),
            }
        )
        return result

    lerobot_train_module.update_policy = observed_update_policy
    try:
        yield
    finally:
        lerobot_train_module.update_policy = original


def runtime_path(output_dir: Path, *, attempt_id: int, rank: int) -> Path:
    filename = f"runtime_attempt_{attempt_id:03d}_rank_{rank}.json"
    if output_dir.is_dir():
        return output_dir / filename
    return output_dir.parent / f".{output_dir.name}.{filename}"


def runtime_payload(
    *,
    attempt_id: int,
    rank: int,
    device: str,
    started_at: str,
    elapsed_s: float,
    status: str,
    peak_allocated_bytes: int | None,
    peak_reserved_bytes: int | None,
    error: BaseException | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "rank": rank,
        "device": device,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_s": elapsed_s,
        "status": status,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return payload


def metrics_tail(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    last = payload
    return last


__all__ = [
    "BASE_POLICY_REPO_ID",
    "CAMERA_RENAME_MAP",
    "FORMAL_MIXED_PRECISION",
    "LOCAL_BASE_POLICY_PATH",
    "LOCAL_BASE_POLICY_REFERENCE",
    "LOCAL_VLM_ASSETS_REFERENCE",
    "OFFLINE_VLM_ASSETS_DIRNAME",
    "REQUIRED_OFFLINE_VLM_ASSET_FILES",
    "BackgroundSplit",
    "BasePolicyIdentity",
    "DatasetAudit",
    "FreshArguments",
    "LaunchInspection",
    "MetricsSink",
    "ResumeArguments",
    "ResumeContext",
    "build_run_config",
    "build_train_config",
    "checkpoint_run_dir",
    "configure_smolvla_policy",
    "identify_base_policy",
    "inspect_resume_launch",
    "instrument_update_policy",
    "latest_complete_checkpoint",
    "load_background_split",
    "load_json_object",
    "metrics_tail",
    "normalize_base_policy_reference",
    "parse_arguments",
    "portable_path",
    "resolve_base_policy",
    "resolve_repository_path",
    "runtime_path",
    "runtime_payload",
    "utc_now",
    "validate_base_policy_identity",
    "validate_dataset",
    "validate_fresh_contract",
    "validate_local_base_policy",
    "validate_local_vlm_assets",
    "validate_policy_preset",
    "validate_resume",
    "validate_split_against_catalog",
    "write_json_atomic",
]
