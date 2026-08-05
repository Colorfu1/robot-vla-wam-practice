"""Contracts for SmolVLA LIBERO C2 canonical LoRA adaptation."""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .b0_training import (
    CAMERA_RENAME_MAP,
    DATASET_REPO_ID,
    FORMAL_LEARNING_RATE,
    FORMAL_MIXED_PRECISION,
    FORMAL_SAVE_FREQ,
    FORMAL_STEPS,
    FORMAL_TRAIN_SEED,
    BasePolicyIdentity,
    DatasetAudit,
    _metrics_history,
    _read_training_step,
    _require_equal,
    _resolve_recorded_path,
    checkpoint_run_dir,
    load_json_object,
    normalize_base_policy_reference,
    resolve_repository_path,
    validate_base_policy_identity,
)
from .c1_training import (
    DEFAULT_B0_SELECTED_REFERENCE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_FILE,
    DEFAULT_WORLD_SIZE,
    FORMAL_GLOBAL_BATCH_SIZE,
    C1BasePolicy,
    base_policy_identity,
    configure_c1_policy,
    resolve_c1_base_policy,
    validate_c1_policy_preset,
    validate_resume_batch_contract,
    validate_resume_formal_contract,
)
from .c1_training import build_run_config as build_c1_run_config
from .c1_training import build_train_config as build_c1_train_config
from .target_training import (
    SAMPLING_STRATEGY,
    TargetSplit,
    load_target_split,
    validate_target_dataset,
)

EXPERIMENT_NAME = "smolvla_libero_c2_canonical_lora"
RESULT_NAME = "C2 Canonical LoRA"
DEFAULT_OUTPUT_DIR = "outputs/smolvla_libero/pilot/c2_canonical_lora"
DEFAULT_SMOKE_OUTPUT_DIR = "outputs/smolvla_libero/debug/c2_canonical_lora_smoke"
FORMAL_LORA_RANK = 32
SMOLVLA_LORA_TARGET = (
    r"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|"
    r"model\.(state_proj|action_in_proj|action_out_proj|"
    r"action_time_mlp_in|action_time_mlp_out))"
)

_FRESH_ONLY_OPTIONS = {
    "--mode",
    "--base-policy",
    "--dataset-root",
    "--split-file",
    "--batch-size",
    "--steps",
    "--save-freq",
    "--learning-rate",
    "--lora-rank",
    "--train-seed",
    "--num-workers",
    "--output-dir",
    "--overwrite",
}
_REQUIRED_POLICY_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "config.json",
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
_STEP_NAME = re.compile(r"^[0-9]{6}$")


class _UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class C2FreshArguments:
    mode: str
    base_policy: str
    dataset_root: Path
    split_file: Path
    num_processes: int
    batch_size: int
    steps: int
    save_freq: int
    learning_rate: float
    lora_rank: int
    train_seed: int
    num_workers: int
    mixed_precision: str
    output_dir: Path
    overwrite: bool = False


@dataclass(frozen=True)
class C2ResumeArguments:
    resume_from: Path
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class LaunchInspection:
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class C2ResumeContext:
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
class PeftInspection:
    total_parameters: int
    trainable_parameters: int
    trainable_parameter_names: tuple[str, ...]
    effective_config: dict[str, Any]


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
) -> C2FreshArguments | C2ResumeArguments | LaunchInspection:
    raw = tuple(sys.argv[1:] if argv is None else argv)
    parser = _UsageParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--base-policy", default=DEFAULT_B0_SELECTED_REFERENCE)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-file", type=Path, default=Path(DEFAULT_SPLIT_FILE))
    parser.add_argument("--num-processes", type=_positive_integer)
    parser.add_argument("--batch-size", type=_positive_integer)
    parser.add_argument("--steps", type=_positive_integer)
    parser.add_argument("--save-freq", type=_positive_integer)
    parser.add_argument("--learning-rate", type=_positive_float)
    parser.add_argument("--lora-rank", type=_positive_integer)
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
        return C2ResumeArguments(
            resume_from=resolve_repository_path(namespace.resume_from),
            num_processes=namespace.num_processes,
            mixed_precision=namespace.mixed_precision,
        )

    formal = namespace.mode == "formal"
    output_dir = namespace.output_dir or Path(
        DEFAULT_OUTPUT_DIR if formal else DEFAULT_SMOKE_OUTPUT_DIR
    )
    arguments = C2FreshArguments(
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
        lora_rank=namespace.lora_rank or FORMAL_LORA_RANK,
        train_seed=namespace.train_seed if namespace.train_seed is not None else FORMAL_TRAIN_SEED,
        num_workers=namespace.num_workers,
        mixed_precision=namespace.mixed_precision or FORMAL_MIXED_PRECISION,
        output_dir=resolve_repository_path(output_dir),
        overwrite=namespace.overwrite,
    )
    validate_fresh_contract(arguments, check_gpu_capacity=not namespace.inspect_launch)
    if namespace.inspect_launch:
        return LaunchInspection(arguments.num_processes, arguments.mixed_precision)
    return arguments


def validate_fresh_contract(
    arguments: C2FreshArguments,
    *,
    check_gpu_capacity: bool = True,
) -> None:
    if arguments.output_dir.exists() and not arguments.overwrite:
        raise FileExistsError(f"Output directory already exists: {arguments.output_dir}")
    if arguments.save_freq > arguments.steps:
        raise ValueError("save frequency cannot exceed total steps")
    fixed = {
        "learning_rate": FORMAL_LEARNING_RATE,
        "lora_rank": FORMAL_LORA_RANK,
        "train_seed": FORMAL_TRAIN_SEED,
        "mixed_precision": FORMAL_MIXED_PRECISION,
    }
    for field, expected in fixed.items():
        actual = getattr(arguments, field)
        if actual != expected:
            raise ValueError(f"C2 requires {field}={expected!r}, got {actual!r}")
    if check_gpu_capacity and arguments.num_processes > torch.cuda.device_count():
        raise ValueError(
            f"Requested --num-processes={arguments.num_processes} but only "
            f"{torch.cuda.device_count()} CUDA GPU(s) are visible; select GPUs with "
            "CUDA_VISIBLE_DEVICES"
        )
    if arguments.mode != "formal":
        return
    if arguments.batch_size * arguments.num_processes != FORMAL_GLOBAL_BATCH_SIZE:
        raise ValueError(
            "Formal C2 requires global batch size "
            f"{FORMAL_GLOBAL_BATCH_SIZE} (legal layouts: 8x1, 4x2, 2x4, 1x8); "
            f"got {arguments.num_processes}x{arguments.batch_size}"
        )
    formal = {
        "steps": FORMAL_STEPS,
        "save_freq": FORMAL_SAVE_FREQ,
    }
    for field, expected in formal.items():
        actual = getattr(arguments, field)
        if actual != expected:
            raise ValueError(f"Formal C2 requires {field}={expected!r}, got {actual!r}")


def requested_peft_config(rank: int):
    from lerobot.configs.default import PeftConfig

    return PeftConfig(method_type="LORA", r=rank)


def build_train_config(
    arguments: C2FreshArguments,
    split: TargetSplit,
    policy: Any,
):
    config = build_c1_train_config(arguments, split, policy)
    config.job_name = "smolvla_libero_c2_canonical_lora"
    config.peft = requested_peft_config(arguments.lora_rank)
    return config


def inspect_peft_policy(
    pretrained_path: str | Path,
    *,
    learning_rate: float,
    rank: int,
) -> PeftInspection:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    root = resolve_repository_path(pretrained_path)
    policy_config = configure_c1_policy(
        root,
        device="cpu",
        learning_rate=learning_rate,
    )
    policy = SmolVLAPolicy.from_pretrained(
        root,
        config=policy_config,
        local_files_only=True,
        strict=True,
    )
    load_vlm_weights = policy.config.load_vlm_weights
    policy.config.load_vlm_weights = True
    try:
        wrapped = policy.wrap_with_peft(
            peft_cli_overrides=dataclasses.asdict(requested_peft_config(rank))
        )
    finally:
        policy.config.load_vlm_weights = load_vlm_weights
    adapter = wrapped.peft_config[wrapped.active_adapter]
    effective = adapter.to_dict()
    peft_type = effective.get("peft_type")
    if getattr(peft_type, "value", peft_type) != "LORA":
        raise ValueError("C2 effective PEFT method is not LoRA")
    if int(effective.get("r", -1)) != rank:
        raise ValueError("C2 effective LoRA rank mismatch")
    if effective.get("target_modules") != SMOLVLA_LORA_TARGET:
        raise ValueError("C2 effective LoRA target modules mismatch")
    if effective.get("modules_to_save") not in ([], None):
        raise ValueError("C2 must not fully train modules outside LoRA")
    trainable_names = tuple(
        name for name, parameter in wrapped.named_parameters() if parameter.requires_grad
    )
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise ValueError("C2 found unexpected non-LoRA trainable parameters")
    total = sum(parameter.numel() for parameter in wrapped.parameters())
    trainable = sum(
        parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad
    )
    del wrapped
    return PeftInspection(
        total_parameters=total,
        trainable_parameters=trainable,
        trainable_parameter_names=trainable_names,
        effective_config=effective,
    )


def build_run_config(
    *,
    arguments: C2FreshArguments,
    split: TargetSplit,
    dataset_audit: DatasetAudit,
    base_identity: BasePolicyIdentity,
    base_selection: dict[str, Any] | None,
    train_config: Any,
    peft_inspection: PeftInspection,
) -> dict[str, Any]:
    payload = build_c1_run_config(
        arguments=arguments,
        split=split,
        dataset_audit=dataset_audit,
        base_identity=base_identity,
        base_selection=base_selection,
        train_config=train_config,
    )
    payload["experiment"] = EXPERIMENT_NAME
    payload["experiment_name"] = RESULT_NAME
    payload["training"]["lora_rank"] = arguments.lora_rank
    payload["peft"] = {
        "requested": dataclasses.asdict(train_config.peft),
        "effective": peft_inspection.effective_config,
        "trainable_parameters": peft_inspection.trainable_parameters,
        "total_parameters": peft_inspection.total_parameters,
        "trainable_parameter_names": list(peft_inspection.trainable_parameter_names),
    }
    payload["resolved_train_config"] = train_config.to_dict()
    return payload


def resolve_c2_base_policy(reference: str, *, require_selection: bool) -> C1BasePolicy:
    return resolve_c1_base_policy(reference, require_selection=require_selection)


def c2_base_policy_identity(
    arguments: C2FreshArguments,
    base: C1BasePolicy,
) -> BasePolicyIdentity:
    return base_policy_identity(arguments, base)


def _complete_checkpoint(path: Path) -> bool:
    policy_dir = path / "pretrained_model"
    state_dir = path / "training_state"
    return all((policy_dir / name).is_file() for name in _REQUIRED_POLICY_FILES) and all(
        (state_dir / name).is_file() for name in _REQUIRED_TRAINING_STATE_FILES
    )


def latest_complete_checkpoint(run_dir: str | Path) -> Path:
    checkpoints_root = resolve_repository_path(run_dir) / "checkpoints"
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
        raise FileNotFoundError(f"No complete C2 checkpoints found in {checkpoints_root}")
    return complete[-1]


def inspect_resume_launch(resume_from: str | Path) -> tuple[int, str]:
    run_dir = checkpoint_run_dir(resume_from)
    run_config = load_json_object(run_dir / "run_config.json")
    if run_config.get("schema_version") != 1:
        raise ValueError("Unsupported C2 run_config schema")
    if run_config.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Checkpoint is not a SmolVLA LIBERO C2 run")
    training = run_config.get("training")
    if not isinstance(training, dict):
        raise TypeError("C2 run_config lacks training settings")
    world_size = training.get("world_size")
    mixed_precision = training.get("mixed_precision")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("Saved C2 world_size is invalid")
    if mixed_precision != FORMAL_MIXED_PRECISION:
        raise ValueError("Saved C2 mixed precision is not bf16")
    return world_size, mixed_precision


def _validate_adapter(policy_dir: Path, *, rank: int) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from peft import PeftConfig

    policy = PreTrainedConfig.from_pretrained(policy_dir)
    if policy.use_peft is not True:
        raise ValueError("C2 checkpoint policy is not marked as PEFT")
    policy.use_peft = False
    validate_c1_policy_preset(policy)
    adapter = PeftConfig.from_pretrained(policy_dir)
    if str(adapter.peft_type) not in {"PeftType.LORA", "LORA"}:
        raise ValueError("C2 checkpoint adapter is not LoRA")
    if int(getattr(adapter, "r", -1)) != rank:
        raise ValueError("C2 checkpoint LoRA rank mismatch")
    if adapter.target_modules != SMOLVLA_LORA_TARGET:
        raise ValueError("C2 checkpoint target modules mismatch")
    if getattr(adapter, "modules_to_save", None) not in ([], None):
        raise ValueError("C2 checkpoint unexpectedly stores full-training modules")


def validate_resume(arguments: C2ResumeArguments) -> C2ResumeContext:
    checkpoint_dir = arguments.resume_from
    run_dir = checkpoint_run_dir(checkpoint_dir)
    if not _complete_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Resume checkpoint is incomplete: {checkpoint_dir}")
    latest = latest_complete_checkpoint(run_dir)
    if checkpoint_dir != latest:
        raise ValueError(f"Resume requires latest complete checkpoint: {latest}")
    run_config = load_json_object(run_dir / "run_config.json")
    if run_config.get("schema_version") != 1:
        raise ValueError("Unsupported C2 run_config schema")
    if run_config.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Checkpoint is not a SmolVLA LIBERO C2 run")
    training = run_config.get("training")
    inputs = run_config.get("inputs")
    dataset_record = run_config.get("dataset")
    peft_record = run_config.get("peft")
    if not all(
        isinstance(value, dict) for value in (training, inputs, dataset_record, peft_record)
    ):
        raise TypeError("C2 run_config lacks required records")
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
    _require_equal(training.get("lora_rank"), FORMAL_LORA_RANK, field="lora_rank")
    mode = training.get("mode")
    if mode not in {"smoke", "formal"}:
        raise ValueError("C2 run_config mode is invalid")
    if mode == "formal":
        validate_resume_formal_contract(training)
    validate_base_policy_identity(inputs.get("base_policy"))
    _require_equal(dataset_record.get("sampler"), SAMPLING_STRATEGY, field="sampler")

    split_value = inputs.get("split_file")
    dataset_root_value = inputs.get("dataset_root")
    if not isinstance(split_value, str) or not isinstance(dataset_root_value, str):
        raise TypeError("C2 run_config input paths are invalid")
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
        raise TypeError("Saved C2 output directory is invalid")
    _require_equal(
        _resolve_recorded_path(output_value, run_dir=run_dir),
        run_dir,
        field="output_dir",
    )
    dataset = saved_config.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError("Saved C2 train config lacks dataset")
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
        "peft": peft_record.get("requested"),
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
        raise TypeError("Saved C2 dataset root is invalid")
    _require_equal(
        _resolve_recorded_path(dataset_root_from_config, run_dir=run_dir),
        dataset_root,
        field="dataset.root",
    )
    _validate_adapter(checkpoint_dir / "pretrained_model", rank=FORMAL_LORA_RANK)
    attempt_id, tail_step = _metrics_history(run_dir / "train_metrics.jsonl")
    return C2ResumeContext(
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
