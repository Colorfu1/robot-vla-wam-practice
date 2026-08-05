"""Deterministic instruction augmentation for SmolVLA LIBERO C3."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .b0_training import load_json_object, portable_path, resolve_repository_path, write_json_atomic
from .data_preparation import (
    HELD_OUT_PARAPHRASE_COUNT,
    TRAIN_PARAPHRASE_COUNT,
    verify_paraphrase_binding,
)
from .target_training import TargetSplit

INSTRUCTION_SAMPLING_CONTRACT = "blake2b_seed_step_rank_batch_index_v1"
CANONICAL_PROBABILITY = 0.25
PARAPHRASE_PROBABILITY = 0.75
# The draw is the first 8 digest bytes as a 64-bit unsigned integer, so comparing it against
# CANONICAL_PROBABILITY * 2**64 is the integer form of ``draw / 2**64 < CANONICAL_PROBABILITY``.
# The threshold is frozen by INSTRUCTION_SAMPLING_CONTRACT: to change the ratio, bump the
# contract version instead of editing CANONICAL_PROBABILITY.
CANONICAL_DRAW_THRESHOLD = 1 << 62
assert CANONICAL_DRAW_THRESHOLD == int(CANONICAL_PROBABILITY * (1 << 64)), (
    "CANONICAL_PROBABILITY drifted from the frozen v1 canonical draw threshold"
)
SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_PREFIX = "instruction_counts_step"
SUMMARY_FILENAME = "instruction_counts.json"


@dataclass(frozen=True)
class ParaphraseTask:
    task_id: int
    combined_task_index: int
    canonical: str
    train_paraphrases: tuple[str, ...]
    held_out_sha256: tuple[str, ...]


@dataclass(frozen=True)
class ParaphraseTrainingSpec:
    source: Path
    sha256: str
    canonical_probability: float
    tasks: Mapping[int, ParaphraseTask]
    audit: dict[str, Any]

    @property
    def tasks_by_combined_index(self) -> Mapping[int, ParaphraseTask]:
        return {task.combined_task_index: task for task in self.tasks.values()}


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_paraphrase_training_spec(
    path: str | Path,
    split: TargetSplit,
) -> ParaphraseTrainingSpec:
    source = resolve_repository_path(path)
    audit = verify_paraphrase_binding(
        target_split=split.source,
        paraphrases=source,
    )
    payload = load_json_object(source)
    probability = payload.get("canonical_probability")
    if probability != CANONICAL_PROBABILITY:
        raise ValueError(f"C3 canonical probability must be {CANONICAL_PROBABILITY}")
    tasks_payload = payload.get("tasks")
    if not isinstance(tasks_payload, dict):
        raise TypeError("C3 paraphrase tasks must be an object")

    tasks: dict[int, ParaphraseTask] = {}
    combined_indices: set[int] = set()
    for task_id in split.task_ids:
        record = tasks_payload.get(str(task_id))
        split_record = split.payload["tasks"][str(task_id)]
        if not isinstance(record, dict) or not isinstance(split_record, dict):
            raise TypeError(f"C3 task {task_id} paraphrase or split record is invalid")
        canonical = record.get("canonical")
        train = record.get("train_paraphrases")
        held_out = record.get("held_out_paraphrases")
        if canonical != split_record.get("canonical"):
            raise ValueError(f"C3 task {task_id} canonical instruction mismatch")
        if not isinstance(train, list) or len(train) != TRAIN_PARAPHRASE_COUNT:
            raise ValueError(f"C3 task {task_id} train paraphrase count mismatch")
        if not isinstance(held_out, list) or len(held_out) != HELD_OUT_PARAPHRASE_COUNT:
            raise ValueError(f"C3 task {task_id} held-out paraphrase count mismatch")
        if any(not isinstance(value, str) for value in [canonical, *train, *held_out]):
            raise TypeError(f"C3 task {task_id} instructions must be strings")
        combined_task_index = split_record.get("combined_task_index")
        if isinstance(combined_task_index, bool) or not isinstance(combined_task_index, int):
            raise TypeError(f"C3 task {task_id} combined task index is invalid")
        if combined_task_index in combined_indices:
            raise ValueError("C3 combined task indices must be unique")
        combined_indices.add(combined_task_index)
        tasks[task_id] = ParaphraseTask(
            task_id=task_id,
            combined_task_index=combined_task_index,
            canonical=canonical,
            train_paraphrases=tuple(train),
            held_out_sha256=tuple(_text_sha256(value) for value in held_out),
        )
    if set(tasks_payload) != {str(task_id) for task_id in split.task_ids}:
        raise ValueError("C3 paraphrase task IDs do not match the target split")
    return ParaphraseTrainingSpec(
        source=source,
        sha256=audit["source_sha256"],
        canonical_probability=float(probability),
        tasks=tasks,
        audit=audit,
    )


def _selection_digest(
    *,
    seed: int,
    step: int,
    rank: int,
    batch_index: int,
    dataset_index: int,
    task_id: int,
) -> bytes:
    identity = (
        f"{INSTRUCTION_SAMPLING_CONTRACT}|{seed}|{step}|{rank}|"
        f"{batch_index}|{dataset_index}|{task_id}"
    )
    return hashlib.blake2b(identity.encode("utf-8"), digest_size=16).digest()


def choose_instruction(
    task: ParaphraseTask,
    *,
    seed: int,
    step: int,
    rank: int,
    batch_index: int,
    dataset_index: int,
) -> tuple[str, str]:
    digest = _selection_digest(
        seed=seed,
        step=step,
        rank=rank,
        batch_index=batch_index,
        dataset_index=dataset_index,
        task_id=task.task_id,
    )
    canonical_draw = int.from_bytes(digest[:8], byteorder="big", signed=False)
    if canonical_draw < CANONICAL_DRAW_THRESHOLD:
        return task.canonical, "canonical"
    paraphrase_index = int.from_bytes(digest[8:], byteorder="big", signed=False) % len(
        task.train_paraphrases
    )
    return task.train_paraphrases[paraphrase_index], f"train_{paraphrase_index}"


def instruction_snapshot_path(run_dir: str | Path, *, step: int, rank: int) -> Path:
    return resolve_repository_path(run_dir) / (f"{SNAPSHOT_PREFIX}_{step:06d}_rank_{rank}.json")


def _batch_values(value: Any, *, field: str) -> list[Any]:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        result = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
    else:
        raise TypeError(f"C3 batch field {field} must be a tensor or sequence")
    return result


def _empty_counts(spec: ParaphraseTrainingSpec) -> dict[int, Counter[str]]:
    return {task_id: Counter() for task_id in spec.tasks}


def _snapshot_counts(
    payload: Mapping[str, Any],
    spec: ParaphraseTrainingSpec,
) -> dict[int, Counter[str]]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != {str(value) for value in spec.tasks}:
        raise ValueError("C3 instruction snapshot task IDs mismatch")
    counts = _empty_counts(spec)
    for task_id, task in spec.tasks.items():
        record = tasks[str(task_id)]
        if not isinstance(record, dict):
            raise TypeError(f"C3 instruction snapshot task {task_id} is invalid")
        canonical_count = record.get("canonical_count")
        train_counts = record.get("train_paraphrase_counts")
        total = record.get("total")
        if (
            isinstance(canonical_count, bool)
            or not isinstance(canonical_count, int)
            or canonical_count < 0
        ):
            raise ValueError(f"C3 task {task_id} canonical count is invalid")
        if not isinstance(train_counts, list) or len(train_counts) != len(task.train_paraphrases):
            raise ValueError(f"C3 task {task_id} train count vector is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in train_counts
        ):
            raise ValueError(f"C3 task {task_id} train counts are invalid")
        observed_total = canonical_count + sum(train_counts)
        if total != observed_total:
            raise ValueError(f"C3 task {task_id} instruction total mismatch")
        counts[task_id]["canonical"] = canonical_count
        for index, value in enumerate(train_counts):
            counts[task_id][f"train_{index}"] = value
    return counts


def validate_instruction_snapshot(
    path: str | Path,
    *,
    spec: ParaphraseTrainingSpec,
    split: TargetSplit,
    seed: int,
    step: int,
    rank: int,
    world_size: int,
    batch_size: int,
) -> dict[str, Any]:
    payload = load_json_object(path)
    expected = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "sampling_contract": INSTRUCTION_SAMPLING_CONTRACT,
        "paraphrases_sha256": spec.sha256,
        "split_sha256": split.sha256,
        "seed": seed,
        "rank": rank,
        "world_size": world_size,
        "batch_size": batch_size,
        "completed_through_step": step,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"C3 instruction snapshot {field} mismatch")
    counts = _snapshot_counts(payload, spec)
    total = sum(sum(task_counts.values()) for task_counts in counts.values())
    if total != step * batch_size:
        raise ValueError(
            f"C3 rank {rank} snapshot expected {step * batch_size} selections, got {total}"
        )
    if payload.get("total_selections") != total:
        raise ValueError("C3 instruction snapshot aggregate total mismatch")
    return payload


class InstructionSelectionTracker:
    """Apply and count deterministic C3 instructions one successful update at a time."""

    def __init__(
        self,
        *,
        spec: ParaphraseTrainingSpec,
        split: TargetSplit,
        seed: int,
        rank: int,
        world_size: int,
        batch_size: int,
        initial_step: int,
        total_steps: int,
        save_freq: int,
        output_dir: str | Path,
        attempt_id: int,
        baseline: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.split = split
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size = int(batch_size)
        self.initial_step = int(initial_step)
        self.total_steps = int(total_steps)
        self.save_freq = int(save_freq)
        self.output_dir = resolve_repository_path(output_dir)
        self.attempt_id = int(attempt_id)
        self.completed_step = self.initial_step
        self._tasks_by_combined_index = dict(spec.tasks_by_combined_index)
        self._counts = _empty_counts(spec)
        self._pending: list[tuple[int, str]] | None = None

        if baseline is not None:
            self._counts = _snapshot_counts(baseline, spec)
        baseline_total = sum(sum(values.values()) for values in self._counts.values())
        if baseline_total != self.initial_step * self.batch_size:
            raise ValueError("C3 instruction baseline does not match the resume step")

    def apply(self, batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        if self._pending is not None:
            raise RuntimeError("C3 received another batch before the previous update completed")
        task_indices = _batch_values(batch.get("task_index"), field="task_index")
        dataset_indices = _batch_values(batch.get("index"), field="index")
        tasks = _batch_values(batch.get("task"), field="task")
        if not (len(task_indices) == len(dataset_indices) == len(tasks) == self.batch_size):
            raise ValueError("C3 batch fields do not match the frozen per-GPU batch size")

        step = self.completed_step + 1
        selected_tasks: list[str] = []
        pending: list[tuple[int, str]] = []
        for batch_index, (combined_index, dataset_index, current_task) in enumerate(
            zip(task_indices, dataset_indices, tasks, strict=True)
        ):
            task = self._tasks_by_combined_index.get(int(combined_index))
            if task is None:
                raise ValueError(f"C3 encountered non-target task index {combined_index}")
            if current_task != task.canonical:
                raise ValueError(f"C3 dataset task text for task {task.task_id} is not canonical")
            instruction, instruction_id = choose_instruction(
                task,
                seed=self.seed,
                step=step,
                rank=self.rank,
                batch_index=batch_index,
                dataset_index=int(dataset_index),
            )
            selected_tasks.append(instruction)
            pending.append((task.task_id, instruction_id))
        batch["task"] = selected_tasks
        self._pending = pending
        return batch

    def complete_step(self) -> None:
        if self._pending is None:
            raise RuntimeError("C3 update completed without a selected instruction batch")
        for task_id, instruction_id in self._pending:
            self._counts[task_id][instruction_id] += 1
        self._pending = None
        self.completed_step += 1
        if self.completed_step % self.save_freq == 0 or self.completed_step == self.total_steps:
            self.write_snapshot()

    def snapshot_payload(self) -> dict[str, Any]:
        task_payload: dict[str, Any] = {}
        canonical_total = 0
        total = 0
        for task_id, task in self.spec.tasks.items():
            counts = self._counts[task_id]
            canonical_count = int(counts["canonical"])
            train_counts = [
                int(counts[f"train_{index}"]) for index in range(len(task.train_paraphrases))
            ]
            task_total = canonical_count + sum(train_counts)
            canonical_total += canonical_count
            total += task_total
            task_payload[str(task_id)] = {
                "canonical_count": canonical_count,
                "train_paraphrase_counts": train_counts,
                "total": task_total,
            }
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "sampling_contract": INSTRUCTION_SAMPLING_CONTRACT,
            "paraphrases_sha256": self.spec.sha256,
            "split_sha256": self.split.sha256,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "batch_size": self.batch_size,
            "attempt_id": self.attempt_id,
            "initial_step": self.initial_step,
            "completed_through_step": self.completed_step,
            "total_selections": total,
            "canonical_count": canonical_total,
            "canonical_ratio": canonical_total / total if total else None,
            "tasks": task_payload,
        }

    def write_snapshot(self) -> Path:
        return write_json_atomic(
            instruction_snapshot_path(
                self.output_dir,
                step=self.completed_step,
                rank=self.rank,
            ),
            self.snapshot_payload(),
        )

    def ensure_complete(self) -> None:
        if self._pending is not None:
            raise RuntimeError("C3 finished with an uncommitted instruction batch")
        if self.completed_step != self.total_steps:
            raise RuntimeError(
                f"C3 instruction tracker ended at {self.completed_step}, "
                f"expected {self.total_steps}"
            )


class _InstructionAugmentingPreprocessor:
    def __init__(self, preprocessor: Any, tracker: InstructionSelectionTracker) -> None:
        self._preprocessor = preprocessor
        self._tracker = tracker

    def __call__(self, batch: MutableMapping[str, Any]) -> Any:
        return self._preprocessor(self._tracker.apply(batch))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._preprocessor, name)


@contextmanager
def override_instruction_preprocessor(
    lerobot_train_module: Any,
    tracker: InstructionSelectionTracker,
) -> Iterator[None]:
    """Inject C3 task text immediately before the policy processor pipeline."""

    original = lerobot_train_module.make_pre_post_processors

    def make_instruction_processors(*args: Any, **kwargs: Any):
        preprocessor, postprocessor = original(*args, **kwargs)
        return _InstructionAugmentingPreprocessor(preprocessor, tracker), postprocessor

    lerobot_train_module.make_pre_post_processors = make_instruction_processors
    try:
        yield
    finally:
        lerobot_train_module.make_pre_post_processors = original


@contextmanager
def track_successful_instruction_updates(
    lerobot_train_module: Any,
    tracker: InstructionSelectionTracker,
) -> Iterator[None]:
    """Commit instruction counts only after a successful optimizer update."""

    original = lerobot_train_module.update_policy

    def tracked_update_policy(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        tracker.complete_step()
        return result

    lerobot_train_module.update_policy = tracked_update_policy
    try:
        yield
    finally:
        lerobot_train_module.update_policy = original


def load_rank_instruction_baseline(
    *,
    run_dir: str | Path,
    spec: ParaphraseTrainingSpec,
    split: TargetSplit,
    seed: int,
    step: int,
    rank: int,
    world_size: int,
    batch_size: int,
) -> dict[str, Any] | None:
    if step == 0:
        return None
    path = instruction_snapshot_path(run_dir, step=step, rank=rank)
    if not path.is_file():
        raise FileNotFoundError(f"C3 resume instruction snapshot is missing: {path}")
    return validate_instruction_snapshot(
        path,
        spec=spec,
        split=split,
        seed=seed,
        step=step,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
    )


def aggregate_instruction_counts(
    *,
    run_dir: str | Path,
    spec: ParaphraseTrainingSpec,
    split: TargetSplit,
    seed: int,
    step: int,
    world_size: int,
    batch_size: int,
) -> dict[str, Any]:
    aggregate = _empty_counts(spec)
    snapshots: list[str] = []
    for rank in range(world_size):
        path = instruction_snapshot_path(run_dir, step=step, rank=rank)
        payload = validate_instruction_snapshot(
            path,
            spec=spec,
            split=split,
            seed=seed,
            step=step,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
        )
        counts = _snapshot_counts(payload, spec)
        for task_id, task_counts in counts.items():
            aggregate[task_id].update(task_counts)
        snapshots.append(portable_path(path))

    canonical_total = 0
    total = 0
    covered = 0
    possible = 0
    task_payload: dict[str, Any] = {}
    for task_id, task in spec.tasks.items():
        counts = aggregate[task_id]
        canonical_count = int(counts["canonical"])
        train_counts = [
            int(counts[f"train_{index}"]) for index in range(len(task.train_paraphrases))
        ]
        task_total = canonical_count + sum(train_counts)
        canonical_total += canonical_count
        total += task_total
        covered += sum(value > 0 for value in train_counts)
        possible += len(train_counts)
        task_payload[str(task_id)] = {
            "canonical": task.canonical,
            "canonical_count": canonical_count,
            "train_paraphrases": [
                {"instruction": instruction, "count": train_counts[index]}
                for index, instruction in enumerate(task.train_paraphrases)
            ],
            "train_paraphrase_coverage": sum(value > 0 for value in train_counts)
            / len(train_counts),
            "total": task_total,
        }
    expected_total = step * world_size * batch_size
    if total != expected_total:
        raise ValueError(f"C3 expected {expected_total} aggregate instructions, got {total}")
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "complete",
        "result_label": "Offline",
        "sampling_contract": INSTRUCTION_SAMPLING_CONTRACT,
        "paraphrases_file": portable_path(spec.source),
        "paraphrases_sha256": spec.sha256,
        "split_sha256": split.sha256,
        "seed": seed,
        "completed_through_step": step,
        "world_size": world_size,
        "batch_size": batch_size,
        "total_selections": total,
        "canonical_count": canonical_total,
        "canonical_ratio": canonical_total / total,
        "target_canonical_probability": spec.canonical_probability,
        "train_paraphrase_coverage": covered / possible,
        "rank_snapshots": snapshots,
        "tasks": task_payload,
    }
    write_json_atomic(resolve_repository_path(run_dir) / SUMMARY_FILENAME, payload)
    return payload
