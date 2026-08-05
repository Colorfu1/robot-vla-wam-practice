"""Contracts for SmolVLA LIBERO C1 default non-PEFT adaptation."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .b0_training import (
    CAMERA_RENAME_MAP,
    DATASET_REPO_ID,
    EXPECTED_ACTION_DIM,
    EXPECTED_STATE_DIM,
    FORMAL_LEARNING_RATE,
    FORMAL_LOG_FREQ,
    FORMAL_MIXED_PRECISION,
    FORMAL_SAVE_FREQ,
    FORMAL_STEPS,
    FORMAL_TRAIN_SEED,
    OFFLINE_VLM_ASSETS_DIRNAME,
    BasePolicyIdentity,
    DatasetAudit,
    _complete_checkpoint,
    _metrics_history,
    _read_training_step,
    _require_equal,
    _resolve_recorded_path,
    checkpoint_run_dir,
    identify_base_policy,
    latest_complete_checkpoint,
    load_json_object,
    normalize_base_policy_reference,
    package_versions,
    portable_path,
    resolve_repository_path,
    validate_base_policy_identity,
    validate_local_vlm_assets,
    validate_policy_preset,
)
from .data_preparation import sha256_file
from .target_training import (
    SAMPLING_STRATEGY,
    TargetSplit,
    load_target_split,
    validate_target_dataset,
)

EXPERIMENT_NAME = "smolvla_libero_c1_default_non_peft"
RESULT_NAME = "C1 Default Non-PEFT Adaptation"
DEFAULT_B0_SELECTED_REFERENCE = (
    "outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/"
    "checkpoints/selected/pretrained_model"
)
DEFAULT_DATASET_ROOT = "datasets/lerobot/lerobot/libero"
DEFAULT_SPLIT_FILE = "configs/experiments/smolvla_libero/target_split.json"
DEFAULT_OUTPUT_DIR = "outputs/smolvla_libero/pilot/c1_default_non_peft"
DEFAULT_SMOKE_OUTPUT_DIR = "outputs/smolvla_libero/debug/c1_default_non_peft_smoke"
DEFAULT_WORLD_SIZE = 8
DEFAULT_BATCH_SIZE = 1
FORMAL_GLOBAL_BATCH_SIZE = DEFAULT_WORLD_SIZE * DEFAULT_BATCH_SIZE

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
    "--experiment-tag",
}


class _UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class C1FreshArguments:
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
    experiment_tag: str | None = None


@dataclass(frozen=True)
class C1ResumeArguments:
    resume_from: Path
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class LaunchInspection:
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class C1ResumeContext:
    checkpoint_dir: Path
    run_dir: Path
    config_path: Path
    training_state_dir: Path
    saved_step: int
    attempt_id: int
    previous_metrics_tail_step: int | None
    run_config: dict[str, Any]
    split: TargetSplit
    dataset_audit: DatasetAudit


@dataclass(frozen=True)
class C1BasePolicy:
    root: Path
    vlm_assets: Path
    selection: dict[str, Any] | None


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
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> C1FreshArguments | C1ResumeArguments | LaunchInspection:
    raw = tuple(sys.argv[1:] if argv is None else argv)
    parser = _UsageParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal", "diagnostic"))
    parser.add_argument("--base-policy", default=DEFAULT_B0_SELECTED_REFERENCE)
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
        "--experiment-tag",
        help=(
            "Short label required by --mode=diagnostic; written into run_config "
            "so diagnostic runs can never be mistaken for formal results"
        ),
    )
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
        return C1ResumeArguments(
            resume_from=resolve_repository_path(namespace.resume_from),
            num_processes=namespace.num_processes,
            mixed_precision=namespace.mixed_precision,
        )

    formal = namespace.mode == "formal"
    if namespace.mode is None:
        parser.error("--mode is required for a fresh run")
    if namespace.mode == "diagnostic" and namespace.experiment_tag is None:
        parser.error("diagnostic mode requires --experiment-tag")
    if namespace.mode != "diagnostic" and namespace.experiment_tag is not None:
        parser.error("--experiment-tag is only valid in diagnostic mode")
    if namespace.dataset_root is None:
        parser.error("--dataset-root is required for a fresh run")
    if namespace.split_file is None:
        parser.error("--split-file is required for a fresh run")
    if namespace.output_dir is None:
        parser.error("--output-dir is required for a fresh run")
    output_dir = namespace.output_dir or Path(
        DEFAULT_OUTPUT_DIR if formal else DEFAULT_SMOKE_OUTPUT_DIR
    )
    arguments = C1FreshArguments(
        mode=namespace.mode,
        base_policy=normalize_base_policy_reference(namespace.base_policy),
        dataset_root=resolve_repository_path(namespace.dataset_root),
        split_file=resolve_repository_path(namespace.split_file),
        num_processes=namespace.num_processes
        if namespace.num_processes is not None
        else (DEFAULT_WORLD_SIZE if formal else 1),
        batch_size=namespace.batch_size or DEFAULT_BATCH_SIZE,
        steps=namespace.steps if namespace.steps is not None else (FORMAL_STEPS if formal else 2),
        save_freq=namespace.save_freq
        if namespace.save_freq is not None
        else (FORMAL_SAVE_FREQ if formal else 2),
        learning_rate=namespace.learning_rate or FORMAL_LEARNING_RATE,
        train_seed=namespace.train_seed if namespace.train_seed is not None else FORMAL_TRAIN_SEED,
        num_workers=namespace.num_workers,
        mixed_precision=namespace.mixed_precision or FORMAL_MIXED_PRECISION,
        output_dir=resolve_repository_path(output_dir),
        overwrite=namespace.overwrite,
        experiment_tag=namespace.experiment_tag,
    )
    validate_fresh_contract(arguments, check_gpu_capacity=not namespace.inspect_launch)
    if namespace.inspect_launch:
        return LaunchInspection(arguments.num_processes, arguments.mixed_precision)
    return arguments


def validate_fresh_contract(
    arguments: C1FreshArguments,
    *,
    check_gpu_capacity: bool = True,
) -> None:
    if arguments.output_dir.exists() and not arguments.overwrite:
        raise FileExistsError(f"Output directory already exists: {arguments.output_dir}")
    if arguments.save_freq > arguments.steps:
        raise ValueError("save frequency cannot exceed total steps")
    if arguments.mode != "diagnostic" and arguments.learning_rate != FORMAL_LEARNING_RATE:
        raise ValueError(f"C1 requires learning_rate={FORMAL_LEARNING_RATE}")
    if arguments.train_seed != FORMAL_TRAIN_SEED:
        raise ValueError(f"C1 requires train_seed={FORMAL_TRAIN_SEED}")
    if check_gpu_capacity and arguments.num_processes > torch.cuda.device_count():
        raise ValueError(
            f"Requested --num-processes={arguments.num_processes} but only "
            f"{torch.cuda.device_count()} CUDA GPU(s) are visible; select GPUs with "
            "CUDA_VISIBLE_DEVICES"
        )
    if arguments.mode == "diagnostic":
        if arguments.train_seed != FORMAL_TRAIN_SEED:
            raise ValueError(f"Diagnostic C1 requires train_seed={FORMAL_TRAIN_SEED}")
        if arguments.mixed_precision != FORMAL_MIXED_PRECISION:
            raise ValueError(f"Diagnostic C1 requires mixed_precision={FORMAL_MIXED_PRECISION}")
        if arguments.save_freq != FORMAL_SAVE_FREQ:
            raise ValueError(f"Diagnostic C1 requires save_freq={FORMAL_SAVE_FREQ}")
        if arguments.batch_size * arguments.num_processes != FORMAL_GLOBAL_BATCH_SIZE:
            raise ValueError(
                "Diagnostic C1 requires global batch size "
                f"{FORMAL_GLOBAL_BATCH_SIZE} (legal layouts: 8x1, 4x2, 2x4, 1x8); "
                f"got {arguments.num_processes}x{arguments.batch_size}"
            )
        return
    if arguments.mode != "formal":
        return
    if arguments.batch_size * arguments.num_processes != FORMAL_GLOBAL_BATCH_SIZE:
        raise ValueError(
            "Formal C1 requires global batch size "
            f"{FORMAL_GLOBAL_BATCH_SIZE} (legal layouts: 8x1, 4x2, 2x4, 1x8); "
            f"got {arguments.num_processes}x{arguments.batch_size}"
        )
    expected = {
        "steps": FORMAL_STEPS,
        "save_freq": FORMAL_SAVE_FREQ,
        "learning_rate": FORMAL_LEARNING_RATE,
        "train_seed": FORMAL_TRAIN_SEED,
        "mixed_precision": FORMAL_MIXED_PRECISION,
    }
    for field, value in expected.items():
        actual = getattr(arguments, field)
        if actual != value:
            raise ValueError(f"Formal C1 requires {field}={value!r}, got {actual!r}")


def _resolve_vlm_assets(policy: Any) -> Path:
    reference = getattr(policy, "vlm_model_name", None)
    if not isinstance(reference, (str, Path)):
        raise TypeError("Selected B0 config has no local VLM asset reference")
    candidate = resolve_repository_path(reference)
    if not candidate.is_dir():
        fallback = resolve_repository_path(
            Path("policies/smolvla_or_pi0/smolvla_base") / OFFLINE_VLM_ASSETS_DIRNAME
        )
        if fallback.is_dir():
            candidate = fallback
    return validate_local_vlm_assets(candidate)


def _validate_selection(root: Path) -> dict[str, Any]:
    selection_path = root.parent / "selected_checkpoint.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Formal C1 requires the selected B0 checkpoint copy; missing {selection_path}"
        )
    selection = load_json_object(selection_path)
    if selection.get("status") != "complete":
        raise ValueError("Selected B0 checkpoint status is not complete")
    if selection.get("selection_set") != "background_validation":
        raise ValueError("C1 base was not selected on the B0 background validation set")
    step = selection.get("selected_checkpoint_step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("Selected B0 checkpoint step is invalid")
    selected_reference = selection.get("selected_checkpoint")
    if not isinstance(selected_reference, str):
        raise TypeError("Selected B0 checkpoint reference is invalid")
    if resolve_repository_path(selected_reference) != root.parent:
        raise ValueError("Selected B0 checkpoint provenance does not match its directory")
    return selection


def resolve_c1_base_policy(reference: str, *, require_selection: bool) -> C1BasePolicy:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    root = resolve_repository_path(reference)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Selected B0 pretrained_model is missing: {root}; publish it first with "
            "scripts/experiments/smolvla_libero/publish_selected_b0.sh. Do not point "
            "--base-policy at checkpoints/last or an arbitrary step"
        )
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    )
    missing = [filename for filename in required if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Selected B0 pretrained_model is incomplete; missing {', '.join(missing)}. "
            "Publish the trusted selection with "
            "scripts/experiments/smolvla_libero/publish_selected_b0.sh instead of "
            "pointing --base-policy at an arbitrary step"
        )
    policy = PreTrainedConfig.from_pretrained(root)
    if not isinstance(policy, SmolVLAConfig):
        raise TypeError(f"C1 base must be SmolVLA, got {type(policy).__name__}")
    validate_policy_preset(policy)
    train_config = load_json_object(root / "train_config.json")
    if train_config.get("job_name") != "smolvla_libero_b0_domain_adaptation":
        raise ValueError("C1 base checkpoint is not a B0 domain-adaptation checkpoint")
    selection = _validate_selection(root) if require_selection else None
    return C1BasePolicy(root=root, vlm_assets=_resolve_vlm_assets(policy), selection=selection)


def configure_c1_policy(
    pretrained_path: str | Path,
    *,
    device: str,
    learning_rate: float,
):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    root = resolve_repository_path(pretrained_path)
    policy = PreTrainedConfig.from_pretrained(root)
    if not isinstance(policy, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(policy).__name__}")
    policy.pretrained_path = Path(portable_path(root))
    policy.vlm_model_name = portable_path(_resolve_vlm_assets(policy))
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
    policy.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(EXPECTED_STATE_DIM,)),
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
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(EXPECTED_ACTION_DIM,))
    }
    validate_c1_policy_preset(policy, expected_learning_rate=learning_rate)
    return policy


def validate_c1_policy_preset(
    policy: Any,
    *,
    expected_learning_rate: float | None = None,
) -> None:
    validate_policy_preset(policy, expected_learning_rate=expected_learning_rate)
    if policy.use_peft is not False:
        raise ValueError("C1 must not use PEFT")
    if policy.train_expert_only is not True or policy.freeze_vision_encoder is not True:
        raise ValueError("C1 must preserve SmolVLA's default expert-only freeze boundary")


def build_train_config(
    arguments: C1FreshArguments,
    split: TargetSplit,
    policy: Any,
):
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=DATASET_REPO_ID,
            root=portable_path(arguments.dataset_root),
            episodes=list(split.train_episodes),
            use_imagenet_stats=True,
        ),
        policy=policy,
        output_dir=Path(portable_path(arguments.output_dir)),
        job_name="smolvla_libero_c1_default_non_peft",
        seed=arguments.train_seed,
        num_workers=arguments.num_workers,
        batch_size=arguments.batch_size,
        steps=arguments.steps,
        eval_freq=0,
        log_freq=min(arguments.steps, FORMAL_LOG_FREQ),
        save_checkpoint=True,
        save_freq=arguments.save_freq,
        use_policy_training_preset=True,
        optimizer=policy.get_optimizer_preset(),
        scheduler=policy.get_scheduler_preset(),
        wandb=WandBConfig(enable=False),
        rename_map=dict(CAMERA_RENAME_MAP),
    )


def build_run_config(
    *,
    arguments: C1FreshArguments,
    split: TargetSplit,
    dataset_audit: DatasetAudit,
    base_identity: BasePolicyIdentity,
    base_selection: dict[str, Any] | None,
    train_config: Any,
) -> dict[str, Any]:
    resolved_train_config = train_config.to_dict()
    selection_summary = None
    if base_selection is not None:
        selection_summary = {
            "selected_checkpoint_step": base_selection["selected_checkpoint_step"],
            "selection_set": base_selection["selection_set"],
            "policy_seed": base_selection.get("policy_seed"),
            "provenance_sha256": sha256_file(
                resolve_repository_path(arguments.base_policy).parent / "selected_checkpoint.json"
            ),
        }
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "experiment_name": RESULT_NAME,
        "result_label": (
            "Formal"
            if arguments.mode == "formal"
            else ("Diagnostic" if arguments.mode == "diagnostic" else "Smoke")
        ),
        "inputs": {
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_root": portable_path(arguments.dataset_root),
            "split_file": portable_path(split.source),
            "split_sha256": split.sha256,
            "overwrite": arguments.overwrite,
            "base_policy": dataclasses.asdict(base_identity),
            "base_selection": selection_summary,
        },
        "dataset": {
            "task_ids": list(split.task_ids),
            "combined_task_indices": list(split.combined_task_indices),
            "train_episodes": len(split.train_episodes),
            "train_frames": split.train_frames,
            "validation_episodes": len(split.validation_episodes),
            "validation_frames": split.validation_frames,
            "fps": dataset_audit.fps,
            "features": dataset_audit.features,
            "sampler": SAMPLING_STRATEGY,
            "normalization_scope": {
                "visual": "imagenet",
                "state_action": "full_dataset_metadata",
            },
        },
        "training": {
            "mode": arguments.mode,
            "experiment_tag": getattr(arguments, "experiment_tag", None),
            "world_size": arguments.num_processes,
            "per_gpu_batch_size": arguments.batch_size,
            "global_batch_size": arguments.num_processes * arguments.batch_size,
            "data_parallel_layout": f"{arguments.num_processes}x{arguments.batch_size}",
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


def inspect_resume_launch(resume_from: str | Path) -> tuple[int, str]:
    run_dir = checkpoint_run_dir(resume_from)
    run_config = load_json_object(run_dir / "run_config.json")
    if run_config.get("schema_version") != 1:
        raise ValueError("Unsupported C1 run_config schema")
    if run_config.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Checkpoint is not a SmolVLA LIBERO C1 run")
    training = run_config.get("training")
    if not isinstance(training, dict):
        raise TypeError("C1 run_config lacks training settings")
    world_size = training.get("world_size")
    mixed_precision = training.get("mixed_precision")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("Saved C1 world_size is invalid")
    if mixed_precision != FORMAL_MIXED_PRECISION:
        raise ValueError("Saved C1 mixed precision is not bf16")
    return world_size, mixed_precision


def validate_resume_batch_contract(
    *,
    requested_world_size: int,
    training: Mapping[str, Any],
) -> None:
    """Enforce the recorded batch contract without hard-coding the world size."""

    world_size = training.get("world_size")
    per_gpu_batch = training.get("per_gpu_batch_size")
    if not isinstance(world_size, int) or not isinstance(per_gpu_batch, int):
        raise TypeError("C1 run_config batch contract is invalid")
    _require_equal(
        training.get("global_batch_size"),
        world_size * per_gpu_batch,
        field="global_batch_size",
    )
    _require_equal(requested_world_size, world_size, field="world_size")


def validate_resume_formal_contract(training: Mapping[str, Any]) -> None:
    """Check frozen formal settings while allowing any legal data-parallel layout."""

    _require_equal(
        training.get("global_batch_size"),
        FORMAL_GLOBAL_BATCH_SIZE,
        field="global_batch_size",
    )
    for field, expected in {
        "mixed_precision": FORMAL_MIXED_PRECISION,
        "steps": FORMAL_STEPS,
        "save_freq": FORMAL_SAVE_FREQ,
    }.items():
        _require_equal(training.get(field), expected, field=field)


def validate_resume(arguments: C1ResumeArguments) -> C1ResumeContext:
    checkpoint_dir = arguments.resume_from
    run_dir = checkpoint_run_dir(checkpoint_dir)
    if not _complete_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Resume checkpoint is incomplete: {checkpoint_dir}")
    latest = latest_complete_checkpoint(run_dir)
    if checkpoint_dir != latest:
        raise ValueError(f"Resume requires latest complete checkpoint: {latest}")
    run_config = load_json_object(run_dir / "run_config.json")
    if run_config.get("schema_version") != 1:
        raise ValueError("Unsupported C1 run_config schema")
    if run_config.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Checkpoint is not a SmolVLA LIBERO C1 run")
    training = run_config.get("training")
    inputs = run_config.get("inputs")
    dataset_record = run_config.get("dataset")
    if not all(isinstance(value, dict) for value in (training, inputs, dataset_record)):
        raise TypeError("C1 run_config lacks training, inputs, or dataset settings")
    validate_resume_batch_contract(
        requested_world_size=arguments.num_processes,
        training=training,
    )
    _require_equal(
        arguments.mixed_precision,
        training.get("mixed_precision"),
        field="mixed_precision",
    )
    _require_equal(training.get("seed"), FORMAL_TRAIN_SEED, field="train_seed")
    mode = training.get("mode")
    if mode not in {"smoke", "formal"}:
        raise ValueError("C1 run_config mode is invalid")
    if mode == "formal":
        validate_resume_formal_contract(training)
    validate_base_policy_identity(inputs.get("base_policy"))
    _require_equal(dataset_record.get("sampler"), SAMPLING_STRATEGY, field="sampler")

    split_value = inputs.get("split_file")
    dataset_root_value = inputs.get("dataset_root")
    if not isinstance(split_value, str) or not isinstance(dataset_root_value, str):
        raise TypeError("C1 run_config input paths are invalid")
    split_path = _resolve_recorded_path(split_value, run_dir=run_dir)
    dataset_root = _resolve_recorded_path(dataset_root_value, run_dir=run_dir)
    split = load_target_split(split_path)
    _require_equal(split.sha256, inputs.get("split_sha256"), field="split_sha256")
    dataset_audit = validate_target_dataset(dataset_root, split)

    config_path = checkpoint_dir / "pretrained_model" / "train_config.json"
    training_state_dir = checkpoint_dir / "training_state"
    saved_config = load_json_object(config_path)
    saved_step = _read_training_step(training_state_dir)
    _require_equal(saved_step, int(checkpoint_dir.name), field="checkpoint_step")
    total_steps = training.get("steps")
    if not isinstance(total_steps, int) or saved_step >= total_steps:
        raise ValueError("Resume checkpoint step must be smaller than configured total steps")
    output_value = saved_config.get("output_dir")
    if not isinstance(output_value, str):
        raise TypeError("Saved C1 output directory is invalid")
    _require_equal(
        _resolve_recorded_path(output_value, run_dir=run_dir),
        run_dir,
        field="output_dir",
    )
    dataset = saved_config.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError("Saved C1 train config lacks dataset")
    _require_equal(
        tuple(sorted(dataset.get("episodes", ()))),
        split.train_episodes,
        field="dataset.episodes",
    )
    fixed_fields = {
        "batch_size": training.get("per_gpu_batch_size"),
        "steps": total_steps,
        "save_freq": training.get("save_freq"),
        "seed": training.get("seed"),
        "use_policy_training_preset": True,
        "optimizer": training.get("optimizer"),
        "scheduler": training.get("scheduler"),
        "rename_map": CAMERA_RENAME_MAP,
    }
    for field, expected in fixed_fields.items():
        _require_equal(saved_config.get(field), expected, field=field)
    _require_equal(dataset.get("repo_id"), DATASET_REPO_ID, field="dataset.repo_id")
    _require_equal(
        dataset.get("use_imagenet_stats"),
        True,
        field="dataset.use_imagenet_stats",
    )
    dataset_root_from_config = dataset.get("root")
    if not isinstance(dataset_root_from_config, str):
        raise TypeError("Saved C1 dataset root is invalid")
    _require_equal(
        _resolve_recorded_path(dataset_root_from_config, run_dir=run_dir),
        dataset_root,
        field="dataset.root",
    )

    from lerobot.configs.policies import PreTrainedConfig

    checkpoint_policy = PreTrainedConfig.from_pretrained(checkpoint_dir / "pretrained_model")
    validate_c1_policy_preset(checkpoint_policy)
    attempt_id, tail_step = _metrics_history(run_dir / "train_metrics.jsonl")
    return C1ResumeContext(
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


def base_policy_identity(arguments: C1FreshArguments, base: C1BasePolicy) -> BasePolicyIdentity:
    return identify_base_policy(arguments.base_policy, base.root, None)
