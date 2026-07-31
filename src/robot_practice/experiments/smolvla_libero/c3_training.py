"""Contracts for SmolVLA LIBERO C3 LoRA plus paraphrase adaptation."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    portable_path,
    resolve_repository_path,
    validate_base_policy_identity,
)
from .c1_training import (
    DEFAULT_B0_SELECTED_REFERENCE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_FILE,
    FORMAL_BATCH_SIZE,
    FORMAL_GLOBAL_BATCH_SIZE,
    FORMAL_WORLD_SIZE,
    C1BasePolicy,
    configure_c1_policy,
)
from .c2_training import (
    FORMAL_LORA_RANK,
    PeftInspection,
    _complete_checkpoint,
    _validate_adapter,
    c2_base_policy_identity,
    inspect_peft_policy,
    resolve_c2_base_policy,
)
from .c2_training import build_run_config as build_c2_run_config
from .c2_training import build_train_config as build_c2_train_config
from .paraphrase_training import (
    CANONICAL_PROBABILITY,
    INSTRUCTION_SAMPLING_CONTRACT,
    PARAPHRASE_PROBABILITY,
    ParaphraseTrainingSpec,
    load_paraphrase_training_spec,
    load_rank_instruction_baseline,
)
from .target_training import (
    SAMPLING_STRATEGY,
    TargetSplit,
    load_target_split,
    validate_target_dataset,
)

EXPERIMENT_NAME = "smolvla_libero_c3_paraphrase_lora"
RESULT_NAME = "C3 LoRA + Paraphrase"
DEFAULT_PARAPHRASES_FILE = "configs/experiments/smolvla_libero/paraphrases.json"
DEFAULT_OUTPUT_DIR = "outputs/smolvla_libero/pilot/c3_paraphrase_lora"
DEFAULT_SMOKE_OUTPUT_DIR = "outputs/smolvla_libero/debug/c3_paraphrase_lora_smoke"

_FRESH_ONLY_OPTIONS = {
    "--mode",
    "--base-policy",
    "--dataset-root",
    "--split-file",
    "--paraphrases",
    "--canonical-probability",
    "--batch-size",
    "--steps",
    "--save-freq",
    "--learning-rate",
    "--lora-rank",
    "--train-seed",
    "--num-workers",
    "--output-dir",
}
_STEP_NAME = re.compile(r"^[0-9]{6}$")


class _UsageParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


@dataclass(frozen=True)
class C3FreshArguments:
    mode: str
    base_policy: str
    dataset_root: Path
    split_file: Path
    paraphrases: Path
    canonical_probability: float
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


@dataclass(frozen=True)
class C3ResumeArguments:
    resume_from: Path
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class LaunchInspection:
    num_processes: int
    mixed_precision: str


@dataclass(frozen=True)
class C3ResumeContext:
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
    paraphrase_spec: ParaphraseTrainingSpec


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


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("must be strictly between zero and one")
    return parsed


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> C3FreshArguments | C3ResumeArguments | LaunchInspection:
    raw = tuple(sys.argv[1:] if argv is None else argv)
    parser = _UsageParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--base-policy", default=DEFAULT_B0_SELECTED_REFERENCE)
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-file", type=Path, default=Path(DEFAULT_SPLIT_FILE))
    parser.add_argument(
        "--paraphrases",
        type=Path,
        default=Path(DEFAULT_PARAPHRASES_FILE),
    )
    parser.add_argument("--canonical-probability", type=_probability)
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
        return C3ResumeArguments(
            resume_from=resolve_repository_path(namespace.resume_from),
            num_processes=namespace.num_processes,
            mixed_precision=namespace.mixed_precision,
        )

    formal = namespace.mode == "formal"
    output_dir = namespace.output_dir or Path(
        DEFAULT_OUTPUT_DIR if formal else DEFAULT_SMOKE_OUTPUT_DIR
    )
    arguments = C3FreshArguments(
        mode=namespace.mode,
        base_policy=normalize_base_policy_reference(namespace.base_policy),
        dataset_root=resolve_repository_path(namespace.dataset_root),
        split_file=resolve_repository_path(namespace.split_file),
        paraphrases=resolve_repository_path(namespace.paraphrases),
        canonical_probability=(
            namespace.canonical_probability
            if namespace.canonical_probability is not None
            else CANONICAL_PROBABILITY
        ),
        num_processes=namespace.num_processes
        if namespace.num_processes is not None
        else (FORMAL_WORLD_SIZE if formal else 1),
        batch_size=namespace.batch_size or FORMAL_BATCH_SIZE,
        steps=namespace.steps if namespace.steps is not None else (FORMAL_STEPS if formal else 2),
        save_freq=namespace.save_freq
        if namespace.save_freq is not None
        else (FORMAL_SAVE_FREQ if formal else 2),
        learning_rate=namespace.learning_rate or FORMAL_LEARNING_RATE,
        lora_rank=namespace.lora_rank or FORMAL_LORA_RANK,
        train_seed=namespace.train_seed
        if namespace.train_seed is not None
        else FORMAL_TRAIN_SEED,
        num_workers=namespace.num_workers,
        mixed_precision=namespace.mixed_precision or FORMAL_MIXED_PRECISION,
        output_dir=resolve_repository_path(output_dir),
    )
    validate_fresh_contract(arguments)
    if namespace.inspect_launch:
        return LaunchInspection(arguments.num_processes, arguments.mixed_precision)
    return arguments


def validate_fresh_contract(arguments: C3FreshArguments) -> None:
    if arguments.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {arguments.output_dir}")
    if arguments.save_freq > arguments.steps:
        raise ValueError("save frequency cannot exceed total steps")
    fixed = {
        "batch_size": FORMAL_BATCH_SIZE,
        "learning_rate": FORMAL_LEARNING_RATE,
        "lora_rank": FORMAL_LORA_RANK,
        "train_seed": FORMAL_TRAIN_SEED,
        "mixed_precision": FORMAL_MIXED_PRECISION,
        "canonical_probability": CANONICAL_PROBABILITY,
    }
    for field, expected in fixed.items():
        actual = getattr(arguments, field)
        if actual != expected:
            raise ValueError(f"C3 requires {field}={expected!r}, got {actual!r}")
    if arguments.mode != "formal":
        return
    formal = {
        "num_processes": FORMAL_WORLD_SIZE,
        "steps": FORMAL_STEPS,
        "save_freq": FORMAL_SAVE_FREQ,
    }
    for field, expected in formal.items():
        actual = getattr(arguments, field)
        if actual != expected:
            raise ValueError(f"Formal C3 requires {field}={expected!r}, got {actual!r}")
    if arguments.batch_size * arguments.num_processes != FORMAL_GLOBAL_BATCH_SIZE:
        raise ValueError(f"Formal C3 requires global batch size {FORMAL_GLOBAL_BATCH_SIZE}")


def build_train_config(
    arguments: C3FreshArguments,
    split: TargetSplit,
    policy: Any,
):
    config = build_c2_train_config(arguments, split, policy)
    config.job_name = "smolvla_libero_c3_paraphrase_lora"
    return config


def build_run_config(
    *,
    arguments: C3FreshArguments,
    split: TargetSplit,
    dataset_audit: DatasetAudit,
    base_identity: BasePolicyIdentity,
    base_selection: dict[str, Any] | None,
    train_config: Any,
    peft_inspection: PeftInspection,
    paraphrase_spec: ParaphraseTrainingSpec,
) -> dict[str, Any]:
    payload = build_c2_run_config(
        arguments=arguments,
        split=split,
        dataset_audit=dataset_audit,
        base_identity=base_identity,
        base_selection=base_selection,
        train_config=train_config,
        peft_inspection=peft_inspection,
    )
    payload["experiment"] = EXPERIMENT_NAME
    payload["experiment_name"] = RESULT_NAME
    payload["inputs"]["paraphrases_file"] = portable_path(paraphrase_spec.source)
    payload["inputs"]["paraphrases_sha256"] = paraphrase_spec.sha256
    payload["training"]["canonical_probability"] = arguments.canonical_probability
    payload["training"]["paraphrase_probability"] = PARAPHRASE_PROBABILITY
    payload["instruction_augmentation"] = {
        "sampling_contract": INSTRUCTION_SAMPLING_CONTRACT,
        "sampling_stage": "after_dataloader_before_tokenizer",
        "canonical_probability": paraphrase_spec.canonical_probability,
        "paraphrase_probability": PARAPHRASE_PROBABILITY,
        "seed": arguments.train_seed,
        "held_out_text_available_to_selector": False,
        "tasks": {
            str(task_id): {
                "canonical": task.canonical,
                "train_paraphrase_count": len(task.train_paraphrases),
                "held_out_paraphrase_count": len(task.held_out_sha256),
                "held_out_sha256": list(task.held_out_sha256),
                "semantic_review_status": paraphrase_spec.audit["tasks"][str(task_id)][
                    "semantic_review_status"
                ],
            }
            for task_id, task in paraphrase_spec.tasks.items()
        },
    }
    payload["resolved_train_config"] = train_config.to_dict()
    return payload


def resolve_c3_base_policy(reference: str, *, require_selection: bool) -> C1BasePolicy:
    return resolve_c2_base_policy(reference, require_selection=require_selection)


def c3_base_policy_identity(
    arguments: C3FreshArguments,
    base: C1BasePolicy,
) -> BasePolicyIdentity:
    return c2_base_policy_identity(arguments, base)


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
        raise FileNotFoundError(f"No complete C3 checkpoints found in {checkpoints_root}")
    return complete[-1]


def inspect_resume_launch(resume_from: str | Path) -> tuple[int, str]:
    run_dir = checkpoint_run_dir(resume_from)
    run_config = load_json_object(run_dir / "run_config.json")
    if run_config.get("schema_version") != 1:
        raise ValueError("Unsupported C3 run_config schema")
    if run_config.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Checkpoint is not a SmolVLA LIBERO C3 run")
    training = run_config.get("training")
    if not isinstance(training, dict):
        raise TypeError("C3 run_config lacks training settings")
    world_size = training.get("world_size")
    mixed_precision = training.get("mixed_precision")
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("Saved C3 world_size is invalid")
    if mixed_precision != FORMAL_MIXED_PRECISION:
        raise ValueError("Saved C3 mixed precision is not bf16")
    return world_size, mixed_precision


def validate_resume(arguments: C3ResumeArguments) -> C3ResumeContext:
    checkpoint_dir = arguments.resume_from
    run_dir = checkpoint_run_dir(checkpoint_dir)
    if not _complete_checkpoint(checkpoint_dir):
        raise FileNotFoundError(f"Resume checkpoint is incomplete: {checkpoint_dir}")
    latest = latest_complete_checkpoint(run_dir)
    if checkpoint_dir != latest:
        raise ValueError(f"Resume requires latest complete checkpoint: {latest}")
    run_config = load_json_object(run_dir / "run_config.json")
    if run_config.get("schema_version") != 1:
        raise ValueError("Unsupported C3 run_config schema")
    if run_config.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("Checkpoint is not a SmolVLA LIBERO C3 run")
    training = run_config.get("training")
    inputs = run_config.get("inputs")
    dataset_record = run_config.get("dataset")
    peft_record = run_config.get("peft")
    instruction_record = run_config.get("instruction_augmentation")
    if not all(
        isinstance(value, dict)
        for value in (training, inputs, dataset_record, peft_record, instruction_record)
    ):
        raise TypeError("C3 run_config lacks required records")
    _require_equal(arguments.num_processes, training.get("world_size"), field="world_size")
    _require_equal(
        arguments.mixed_precision,
        training.get("mixed_precision"),
        field="mixed_precision",
    )
    world_size = training.get("world_size")
    per_gpu_batch = training.get("per_gpu_batch_size")
    if not isinstance(world_size, int) or not isinstance(per_gpu_batch, int):
        raise TypeError("C3 run_config batch contract is invalid")
    _require_equal(
        training.get("global_batch_size"),
        world_size * per_gpu_batch,
        field="global_batch_size",
    )
    _require_equal(training.get("seed"), FORMAL_TRAIN_SEED, field="train_seed")
    _require_equal(training.get("lora_rank"), FORMAL_LORA_RANK, field="lora_rank")
    _require_equal(
        training.get("canonical_probability"),
        CANONICAL_PROBABILITY,
        field="canonical_probability",
    )
    _require_equal(
        training.get("paraphrase_probability"),
        PARAPHRASE_PROBABILITY,
        field="paraphrase_probability",
    )
    _require_equal(
        instruction_record.get("sampling_contract"),
        INSTRUCTION_SAMPLING_CONTRACT,
        field="instruction_sampling_contract",
    )
    _require_equal(
        instruction_record.get("held_out_text_available_to_selector"),
        False,
        field="held_out_text_available_to_selector",
    )
    mode = training.get("mode")
    if mode not in {"smoke", "formal"}:
        raise ValueError("C3 run_config mode is invalid")
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
    _require_equal(dataset_record.get("sampler"), SAMPLING_STRATEGY, field="sampler")

    split_value = inputs.get("split_file")
    dataset_root_value = inputs.get("dataset_root")
    paraphrases_value = inputs.get("paraphrases_file")
    if not all(isinstance(value, str) for value in (split_value, dataset_root_value, paraphrases_value)):
        raise TypeError("C3 run_config input paths are invalid")
    split_path = _resolve_recorded_path(split_value, run_dir=run_dir)
    dataset_root = _resolve_recorded_path(dataset_root_value, run_dir=run_dir)
    paraphrases_path = _resolve_recorded_path(paraphrases_value, run_dir=run_dir)
    split = load_target_split(split_path)
    _require_equal(split.sha256, inputs.get("split_sha256"), field="split_sha256")
    paraphrase_spec = load_paraphrase_training_spec(paraphrases_path, split)
    _require_equal(
        paraphrase_spec.sha256,
        inputs.get("paraphrases_sha256"),
        field="paraphrases_sha256",
    )
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
        raise TypeError("Saved C3 output directory is invalid")
    _require_equal(
        _resolve_recorded_path(output_value, run_dir=run_dir),
        run_dir,
        field="output_dir",
    )
    _require_equal(
        saved_config.get("job_name"),
        "smolvla_libero_c3_paraphrase_lora",
        field="job_name",
    )
    dataset = saved_config.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError("Saved C3 train config lacks dataset")
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
        raise TypeError("Saved C3 dataset root is invalid")
    _require_equal(
        _resolve_recorded_path(dataset_root_from_config, run_dir=run_dir),
        dataset_root,
        field="dataset.root",
    )
    _validate_adapter(checkpoint_dir / "pretrained_model", rank=FORMAL_LORA_RANK)
    for rank in range(world_size):
        load_rank_instruction_baseline(
            run_dir=run_dir,
            spec=paraphrase_spec,
            split=split,
            seed=FORMAL_TRAIN_SEED,
            step=saved_step,
            rank=rank,
            world_size=world_size,
            batch_size=per_gpu_batch,
        )
    attempt_id, tail_step = _metrics_history(run_dir / "train_metrics.jsonl")
    return C3ResumeContext(
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
        paraphrase_spec=paraphrase_spec,
    )


__all__ = [
    "DEFAULT_B0_SELECTED_REFERENCE",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PARAPHRASES_FILE",
    "DEFAULT_SPLIT_FILE",
    "EXPERIMENT_NAME",
    "FORMAL_GLOBAL_BATCH_SIZE",
    "FORMAL_LORA_RANK",
    "FORMAL_WORLD_SIZE",
    "C3FreshArguments",
    "C3ResumeArguments",
    "C3ResumeContext",
    "LaunchInspection",
    "PeftInspection",
    "build_run_config",
    "build_train_config",
    "c3_base_policy_identity",
    "configure_c1_policy",
    "inspect_peft_policy",
    "load_paraphrase_training_spec",
    "parse_arguments",
    "resolve_c3_base_policy",
    "validate_resume",
]
