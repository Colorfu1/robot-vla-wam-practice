"""Write independently loadable PushT development task Recordings."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import rerun as rr
import rerun.dataframe as rdf

from .config import load_json_object
from .rerun_visualization import (
    DEVELOPMENT_ROLLOUT_ROOT,
    build_development_rollout_blueprint,
    log_rollout_to_stream,
)
from .rollout import RolloutResult

DEVELOPMENT_EXPERIMENT_ID = "development_checkpoint_comparison"
DEVELOPMENT_ENVIRONMENT_SEEDS = tuple(range(1000, 1020))
MODEL_NAMES = ("bc_chunk", "diffusion")
CATEGORY_ORDER = (
    "bc_pass_diffusion_fail",
    "diffusion_pass_bc_fail",
    "both_fail",
    "both_pass",
)
MODEL_APPLICATION_IDS = {
    "bc_chunk": "robot_practice_pusht_bc_chunk",
    "diffusion": "robot_practice_pusht_diffusion",
}
MODEL_DISPLAY_NAMES = {
    "bc_chunk": "BCChunk",
    "diffusion": "Diffusion",
}
TASK_COMPONENTS = {
    "/task/model": ("rerun.archetypes.TextDocument", "rerun.components.Text"),
    "/task/environment_seed": ("rerun.archetypes.Scalars", "rerun.components.Scalar"),
    "/task/policy_seed": ("rerun.archetypes.Scalars", "rerun.components.Scalar"),
    "/task/success": ("rerun.archetypes.Scalars", "rerun.components.Scalar"),
    "/task/max_reward": ("rerun.archetypes.Scalars", "rerun.components.Scalar"),
    "/task/checkpoint_step": ("rerun.archetypes.Scalars", "rerun.components.Scalar"),
    "/task/checkpoint_id": ("rerun.archetypes.TextDocument", "rerun.components.Text"),
}
REQUIRED_ROLLOUT_LEAVES = (
    "complete_agent_path",
    "complete_block_path",
    "complete_action_targets",
    "simulator",
    "current_agent",
    "current_block",
    "current_action_target",
    "reward",
    "success",
    "rollout_index",
    "metadata",
)


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _require_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


@dataclass(frozen=True)
class TaskRecordingMetadata:
    model: str
    environment_seed: int
    policy_seed: int
    success: bool
    max_reward: float
    checkpoint_step: int
    checkpoint_id: str

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"Unsupported model: {self.model!r}")
        _require_int(self.environment_seed, field="environment_seed", minimum=0)
        _require_int(self.policy_seed, field="policy_seed", minimum=0)
        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean")
        _require_finite_number(self.max_reward, field="max_reward")
        _require_int(self.checkpoint_step, field="checkpoint_step", minimum=0)
        _require_sha256(self.checkpoint_id, field="checkpoint_id")

    @property
    def recording_id(self) -> str:
        return f"env_{self.environment_seed:04d}"

    @property
    def recording_name(self) -> str:
        outcome = "success" if self.success else "failure"
        return f"Task env={self.environment_seed} | {MODEL_DISPLAY_NAMES[self.model]} | {outcome}"

    @property
    def recording_filename(self) -> str:
        return f"{self.recording_id}.rrd"

    @property
    def recording_relative_path(self) -> str:
        return f"recordings/{self.model}/{self.recording_filename}"

    @property
    def rollout_root(self) -> str:
        return DEVELOPMENT_ROLLOUT_ROOT


@dataclass(frozen=True)
class TaskRecordingArtifact:
    metadata: TaskRecordingMetadata
    path: Path


def task_metadata_from_result(
    result: RolloutResult,
    *,
    checkpoint_step: int,
    checkpoint_id: str,
) -> TaskRecordingMetadata:
    if result.condition_id != DEVELOPMENT_EXPERIMENT_ID:
        raise ValueError(
            f"Development task recordings require condition_id {DEVELOPMENT_EXPERIMENT_ID!r}"
        )
    return TaskRecordingMetadata(
        model=result.model,
        environment_seed=result.environment_seed,
        policy_seed=result.policy_seed,
        success=result.success,
        max_reward=result.max_reward,
        checkpoint_step=checkpoint_step,
        checkpoint_id=checkpoint_id,
    )


def _expected_category(*, bc_success: bool, diffusion_success: bool) -> str:
    if bc_success and not diffusion_success:
        return "bc_pass_diffusion_fail"
    if diffusion_success and not bc_success:
        return "diffusion_pass_bc_fail"
    if not bc_success and not diffusion_success:
        return "both_fail"
    return "both_pass"


def load_comparison_task_metadata(
    path: str | Path,
    *,
    expected_environment_seeds: Sequence[int] = DEVELOPMENT_ENVIRONMENT_SEEDS,
) -> dict[str, tuple[TaskRecordingMetadata, ...]]:
    """Validate ``comparison.json`` and return canonical per-model task metadata."""

    payload = load_json_object(path)
    expected_seeds = tuple(expected_environment_seeds)
    if payload.get("schema_version") != 1:
        raise ValueError("comparison.schema_version must be 1")
    if payload.get("experiment") != DEVELOPMENT_EXPERIMENT_ID:
        raise ValueError("comparison.json has an unexpected experiment identifier")
    raw_environment_seeds = payload.get("environment_seeds")
    if not isinstance(raw_environment_seeds, list):
        raise TypeError("comparison.environment_seeds must be a list")
    environment_seeds = tuple(
        _require_int(seed, field="comparison.environment_seeds[]", minimum=0)
        for seed in raw_environment_seeds
    )
    if environment_seeds != expected_seeds:
        raise ValueError("comparison.environment_seeds must exactly match the development seed set")
    policy_seed = _require_int(
        payload.get("policy_seed"),
        field="comparison.policy_seed",
        minimum=0,
    )
    if policy_seed != 0:
        raise ValueError("Development comparison task recordings require policy_seed=0")
    if payload.get("num_rollouts_per_policy") != len(expected_seeds):
        raise ValueError("comparison.num_rollouts_per_policy is inconsistent")
    if payload.get("num_rollouts") != len(expected_seeds) * len(MODEL_NAMES):
        raise ValueError("comparison.num_rollouts is inconsistent")

    summaries = _require_mapping(payload.get("summaries"), field="comparison.summaries")
    checkpoint_by_model: dict[str, tuple[int, str]] = {}
    for model in MODEL_NAMES:
        summary = _require_mapping(summaries.get(model), field=f"summaries.{model}")
        checkpoint_by_model[model] = (
            _require_int(
                summary.get("checkpoint_step"),
                field=f"summaries.{model}.checkpoint_step",
                minimum=0,
            ),
            _require_sha256(
                summary.get("checkpoint_id"),
                field=f"summaries.{model}.checkpoint_id",
            ),
        )

    raw_categories = _require_mapping(
        payload.get("categories"),
        field="comparison.categories",
    )
    if set(raw_categories) != set(CATEGORY_ORDER):
        raise ValueError("comparison.categories must contain exactly the four categories")
    category_by_seed: dict[int, str] = {}
    for category in CATEGORY_ORDER:
        values = raw_categories[category]
        if not isinstance(values, list):
            raise TypeError(f"comparison.categories.{category} must be a list")
        seeds = tuple(
            _require_int(seed, field=f"comparison.categories.{category}[]", minimum=0)
            for seed in values
        )
        expected_order = tuple(seed for seed in expected_seeds if seed in set(seeds))
        if seeds != expected_order:
            raise ValueError(f"comparison.categories.{category} must use development order")
        for seed in seeds:
            if seed not in expected_seeds:
                raise ValueError(f"Unexpected seed {seed} in category {category}")
            if seed in category_by_seed:
                raise ValueError(f"Seed {seed} appears in more than one category")
            category_by_seed[seed] = category
    if set(category_by_seed) != set(expected_seeds):
        raise ValueError("The four comparison categories must cover every development seed")

    raw_counts = _require_mapping(
        payload.get("category_counts"),
        field="comparison.category_counts",
    )
    if set(raw_counts) != set(CATEGORY_ORDER):
        raise ValueError("comparison.category_counts must contain exactly four entries")
    for category in CATEGORY_ORDER:
        observed_count = _require_int(
            raw_counts[category],
            field=f"comparison.category_counts.{category}",
            minimum=0,
        )
        if observed_count != len(raw_categories[category]):
            raise ValueError(f"comparison.category_counts.{category} is inconsistent")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise TypeError("comparison.cases must be a list")
    cases_by_seed: dict[int, Mapping[str, Any]] = {}
    for raw_case in raw_cases:
        case = _require_mapping(raw_case, field="comparison.cases[]")
        seed = _require_int(
            case.get("environment_seed"),
            field="comparison.cases[].environment_seed",
            minimum=0,
        )
        if seed in cases_by_seed:
            raise ValueError(f"comparison.cases contains duplicate seed {seed}")
        cases_by_seed[seed] = case
    if tuple(cases_by_seed) != expected_seeds:
        raise ValueError("comparison.cases must contain every development seed in order")

    metadata_by_model: dict[str, list[TaskRecordingMetadata]] = {model: [] for model in MODEL_NAMES}
    for seed in expected_seeds:
        case = cases_by_seed[seed]
        results: dict[str, Mapping[str, Any]] = {}
        for model in MODEL_NAMES:
            result = _require_mapping(case.get(model), field=f"case[{seed}].{model}")
            success = result.get("success")
            if not isinstance(success, bool):
                raise TypeError(f"case[{seed}].{model}.success must be a boolean")
            max_reward = _require_finite_number(
                result.get("max_reward"),
                field=f"case[{seed}].{model}.max_reward",
            )
            checkpoint_step, checkpoint_id = checkpoint_by_model[model]
            metadata = TaskRecordingMetadata(
                model=model,
                environment_seed=seed,
                policy_seed=policy_seed,
                success=success,
                max_reward=max_reward,
                checkpoint_step=checkpoint_step,
                checkpoint_id=checkpoint_id,
            )
            if result.get("rerun_entity_path") != metadata.rollout_root:
                raise ValueError(f"case[{seed}].{model}.rerun_entity_path does not match its task")
            if result.get("rerun_recording_path") != metadata.recording_relative_path:
                raise ValueError(
                    f"case[{seed}].{model}.rerun_recording_path does not match its task"
                )
            metadata_by_model[model].append(metadata)
            results[model] = result
        expected_category = _expected_category(
            bc_success=bool(results["bc_chunk"]["success"]),
            diffusion_success=bool(results["diffusion"]["success"]),
        )
        if case.get("category") != expected_category:
            raise ValueError(f"case[{seed}].category is inconsistent with success values")
        if category_by_seed[seed] != expected_category:
            raise ValueError(f"Seed {seed} is registered under the wrong category")

    return {model: tuple(metadata_by_model[model]) for model in MODEL_NAMES}


def _log_task_context(
    recording: rr.RecordingStream,
    metadata: TaskRecordingMetadata,
) -> None:
    recording.log("/task/model", rr.TextDocument(metadata.model), static=True)
    recording.log(
        "/task/environment_seed",
        rr.Scalars(metadata.environment_seed),
        static=True,
    )
    recording.log("/task/policy_seed", rr.Scalars(metadata.policy_seed), static=True)
    recording.log("/task/success", rr.Scalars(int(metadata.success)), static=True)
    recording.log("/task/max_reward", rr.Scalars(metadata.max_reward), static=True)
    recording.log(
        "/task/checkpoint_step",
        rr.Scalars(metadata.checkpoint_step),
        static=True,
    )
    recording.log(
        "/task/checkpoint_id",
        rr.TextDocument(metadata.checkpoint_id),
        static=True,
    )


def _finish_stream(
    recording: rr.RecordingStream,
    *,
    suppress_errors: bool = False,
) -> None:
    errors: list[BaseException] = []
    try:
        recording.flush()
    except Exception as error:  # noqa: BLE001 - close every sink before reporting
        errors.append(error)
    try:
        recording.disconnect()
    except Exception as error:  # noqa: BLE001 - close every sink before reporting
        errors.append(error)
    if errors and not suppress_errors:
        raise errors[0]


def _run_rerun_cli(arguments: Sequence[str], *, operation: str) -> None:
    rerun_cli = shutil.which("rerun")
    if rerun_cli is None:
        raise RuntimeError("The rerun CLI is required to verify development Recordings")
    completed = subprocess.run(
        [rerun_cli, "rrd", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Rerun {operation} failed: {detail}")


def _write_task_recording(
    *,
    output_path: Path,
    metadata: TaskRecordingMetadata,
    result: RolloutResult,
    rollout_index: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Task Recording already exists: {output_path}")
    recording = rr.RecordingStream(
        MODEL_APPLICATION_IDS[metadata.model],
        recording_id=metadata.recording_id,
    )
    try:
        recording.save(output_path)
        recording.send_recording_name(metadata.recording_name)
        _log_task_context(recording, metadata)
        rr.send_blueprint(
            build_development_rollout_blueprint(),
            make_active=False,
            make_default=True,
            recording=recording,
        )
        next_frame = log_rollout_to_stream(
            recording,
            result,
            global_frame_start=0,
            rollout_index=rollout_index,
            entity_root=metadata.rollout_root,
        )
        if next_frame != result.agent_path.shape[0]:
            raise RuntimeError("Task-local Rerun timeline length is inconsistent")
    except BaseException:
        _finish_stream(recording, suppress_errors=True)
        raise
    _finish_stream(recording)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Task Recording was not written: {output_path}")
    return output_path


def write_rollout_task_recording(
    *,
    output_path: str | Path,
    result: RolloutResult,
    metadata: TaskRecordingMetadata,
    rollout_index: int,
) -> Path:
    """Write one directly generated rollout as one task-local Recording."""

    expected = task_metadata_from_result(
        result,
        checkpoint_step=metadata.checkpoint_step,
        checkpoint_id=metadata.checkpoint_id,
    )
    if expected != metadata:
        raise ValueError("Rollout data and task Recording metadata disagree")
    return _write_task_recording(
        output_path=Path(output_path),
        metadata=metadata,
        result=result,
        rollout_index=rollout_index,
    )


def _single_nested_value(
    table: Mapping[str, Any],
    column: str,
    *,
    recording_id: str,
) -> Any:
    values = table.get(column)
    if values is None or len(values) != 1 or values[0] is None or len(values[0]) != 1:
        raise RuntimeError(f"Recording {recording_id} has no exact value for {column}")
    return values[0][0]


def _verify_task_context(recording: Any, metadata: TaskRecordingMetadata) -> None:
    descriptors_by_path: dict[str, list[Any]] = {}
    for descriptor in recording.schema().component_columns():
        descriptors_by_path.setdefault(str(descriptor.entity_path), []).append(descriptor)
    for entity_path, (archetype, component_type) in TASK_COMPONENTS.items():
        descriptors = descriptors_by_path.get(entity_path, [])
        if len(descriptors) != 1:
            raise RuntimeError(
                f"Recording {metadata.recording_id} must contain exactly one {entity_path}"
            )
        descriptor = descriptors[0]
        if (
            str(descriptor.archetype) != archetype
            or str(descriptor.component_type) != component_type
            or not descriptor.is_static
        ):
            raise RuntimeError(
                f"Recording {metadata.recording_id} has the wrong schema for {entity_path}"
            )

    values = recording.view(index=None, contents="/task/**").select_static().read_all().to_pydict()
    observed = {
        "model": _single_nested_value(
            values,
            "/task/model:TextDocument:text",
            recording_id=metadata.recording_id,
        ),
        "environment_seed": _single_nested_value(
            values,
            "/task/environment_seed:Scalars:scalars",
            recording_id=metadata.recording_id,
        ),
        "policy_seed": _single_nested_value(
            values,
            "/task/policy_seed:Scalars:scalars",
            recording_id=metadata.recording_id,
        ),
        "success": _single_nested_value(
            values,
            "/task/success:Scalars:scalars",
            recording_id=metadata.recording_id,
        ),
        "max_reward": _single_nested_value(
            values,
            "/task/max_reward:Scalars:scalars",
            recording_id=metadata.recording_id,
        ),
        "checkpoint_step": _single_nested_value(
            values,
            "/task/checkpoint_step:Scalars:scalars",
            recording_id=metadata.recording_id,
        ),
        "checkpoint_id": _single_nested_value(
            values,
            "/task/checkpoint_id:TextDocument:text",
            recording_id=metadata.recording_id,
        ),
    }
    expected = {
        "model": metadata.model,
        "environment_seed": float(metadata.environment_seed),
        "policy_seed": float(metadata.policy_seed),
        "success": float(int(metadata.success)),
        "max_reward": float(metadata.max_reward),
        "checkpoint_step": float(metadata.checkpoint_step),
        "checkpoint_id": metadata.checkpoint_id,
    }
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if isinstance(expected_value, float):
            if not math.isclose(float(observed_value), expected_value, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"Recording {metadata.recording_id} task context disagrees on {key}"
                )
        elif observed_value != expected_value:
            raise RuntimeError(f"Recording {metadata.recording_id} task context disagrees on {key}")


def _recording_name(recording: Any) -> str:
    values = (
        recording.view(index=None, contents="/__properties/**")
        .select_static()
        .read_all()
        .to_pydict()
    )
    return str(
        _single_nested_value(
            values,
            "property:RecordingInfo:name",
            recording_id=str(recording.recording_id()),
        )
    )


def verify_task_recording(
    recording_path: str | Path,
    expected_metadata: TaskRecordingMetadata,
) -> str:
    """Verify one physical task RRD and return its Recording ID."""

    path = Path(recording_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Task Recording was not written: {path}")
    if path.name != expected_metadata.recording_filename:
        raise RuntimeError(
            f"Task Recording filename must be {expected_metadata.recording_filename}"
        )
    _run_rerun_cli(["verify", str(path)], operation="recording verification")
    archive = rdf.load_archive(path)
    if archive.num_recordings() != 1:
        raise RuntimeError(f"{path.name} must contain exactly one data Recording")
    recording = archive.all_recordings()[0]
    recording_id = str(recording.recording_id())
    if recording_id != expected_metadata.recording_id:
        raise RuntimeError(f"{path.name} has the wrong Recording ID")
    if str(recording.application_id()) != MODEL_APPLICATION_IDS[expected_metadata.model]:
        raise RuntimeError(f"Recording {recording_id} has the wrong application ID")
    if _recording_name(recording) != expected_metadata.recording_name:
        raise RuntimeError(f"Recording {recording_id} has the wrong selector name")

    component_paths = {
        str(descriptor.entity_path) for descriptor in recording.schema().component_columns()
    }
    unexpected_paths = {
        entity_path
        for entity_path in component_paths
        if entity_path != "/__properties"
        and entity_path not in TASK_COMPONENTS
        and not entity_path.startswith(f"{expected_metadata.rollout_root}/")
    }
    if unexpected_paths:
        raise RuntimeError(
            f"Recording {recording_id} contains unrelated entities: {sorted(unexpected_paths)}"
        )
    for leaf in REQUIRED_ROLLOUT_LEAVES:
        if f"{expected_metadata.rollout_root}/{leaf}" not in component_paths:
            raise RuntimeError(f"Recording {recording_id} is missing rollout {leaf}")
    simulator_descriptors = [
        descriptor
        for descriptor in recording.schema().component_columns()
        if str(descriptor.entity_path) == f"{expected_metadata.rollout_root}/simulator"
    ]
    if not any(
        str(descriptor.archetype) == "rerun.archetypes.EncodedImage"
        for descriptor in simulator_descriptors
    ):
        raise RuntimeError(f"Recording {recording_id} has no encoded simulator image")

    timeline = (
        recording.view(index="global_frame", contents=f"{expected_metadata.rollout_root}/**")
        .select()
        .read_all()
        .column("global_frame")
        .to_pylist()
    )
    if not timeline or timeline[0] != 0:
        raise RuntimeError(f"Recording {recording_id} timeline does not start at zero")
    if any(
        value is None or isinstance(value, bool) or not isinstance(value, int) for value in timeline
    ):
        raise RuntimeError(f"Recording {recording_id} has an invalid global_frame")
    if any(current <= previous for previous, current in pairwise(timeline)):
        raise RuntimeError(f"Recording {recording_id} timeline is not increasing")
    _verify_task_context(recording, expected_metadata)
    return recording_id


def _metadata_equivalent(
    observed: Sequence[TaskRecordingMetadata],
    expected: Sequence[TaskRecordingMetadata],
) -> bool:
    if len(observed) != len(expected):
        return False
    for left, right in zip(observed, expected, strict=True):
        if (
            left.model != right.model
            or left.environment_seed != right.environment_seed
            or left.policy_seed != right.policy_seed
            or left.success != right.success
            or left.checkpoint_step != right.checkpoint_step
            or left.checkpoint_id != right.checkpoint_id
            or not math.isclose(
                float(left.max_reward),
                float(right.max_reward),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False
    return True


def validate_paired_recording_directories(
    recordings_directory: str | Path,
    *,
    artifacts_by_model: Mapping[str, Sequence[TaskRecordingArtifact]],
    expected_by_model: Mapping[str, Sequence[TaskRecordingMetadata]],
) -> dict[str, tuple[Path, ...]]:
    """Validate the exact per-model file sets and their paired Recording IDs."""

    root = Path(recordings_directory)
    if set(artifacts_by_model) != set(MODEL_NAMES):
        raise ValueError("Task artifacts must contain both policy models")
    if set(expected_by_model) != set(MODEL_NAMES):
        raise ValueError("Expected metadata must contain both policy models")
    observed_model_directories = {path.name for path in root.iterdir() if path.is_dir()}
    if observed_model_directories != set(MODEL_NAMES):
        raise RuntimeError("Recording directories must contain exactly both policy models")

    paths_by_model: dict[str, tuple[Path, ...]] = {}
    recording_ids: dict[str, tuple[str, ...]] = {}
    for model in MODEL_NAMES:
        expected_metadata = tuple(expected_by_model[model])
        artifacts = tuple(artifacts_by_model[model])
        observed_metadata = tuple(artifact.metadata for artifact in artifacts)
        if not _metadata_equivalent(observed_metadata, expected_metadata):
            raise RuntimeError(f"{model} task artifacts disagree with comparison.json")
        expected_paths = tuple(
            root / model / metadata.recording_filename for metadata in expected_metadata
        )
        actual_paths = tuple(sorted((root / model).glob("*.rrd")))
        if actual_paths != expected_paths:
            raise RuntimeError(f"{model} Recording directory has an unexpected RRD file set")
        artifact_paths = tuple(artifact.path for artifact in artifacts)
        if artifact_paths != expected_paths:
            raise RuntimeError(f"{model} task artifacts do not match their public paths")
        recording_ids[model] = tuple(
            verify_task_recording(path, metadata)
            for path, metadata in zip(expected_paths, expected_metadata, strict=True)
        )
        paths_by_model[model] = expected_paths

    if recording_ids["bc_chunk"] != recording_ids["diffusion"]:
        raise RuntimeError("BCChunk and Diffusion directories expose different task IDs")
    return paths_by_model


class PairedTaskRecordings:
    """Write and validate one public RRD per model/environment-seed pair."""

    def __init__(self, recordings_directory: str | Path) -> None:
        self.recordings_directory = Path(recordings_directory)
        self.recordings_directory.mkdir(parents=True, exist_ok=False)
        for model in MODEL_NAMES:
            (self.recordings_directory / model).mkdir()
        self.artifacts: dict[str, list[TaskRecordingArtifact]] = {
            model: [] for model in MODEL_NAMES
        }
        self._keys: set[tuple[str, int]] = set()
        self._rollout_index = 0
        self._finalized = False

    def add(self, result: RolloutResult, metadata: TaskRecordingMetadata) -> Path:
        if self._finalized:
            raise RuntimeError("Cannot add a task Recording after final validation")
        key = (metadata.model, metadata.environment_seed)
        if key in self._keys:
            raise RuntimeError(f"Duplicate task Recording: {key}")
        path = self.recordings_directory / metadata.model / metadata.recording_filename
        write_rollout_task_recording(
            output_path=path,
            result=result,
            metadata=metadata,
            rollout_index=self._rollout_index,
        )
        verify_task_recording(path, metadata)
        self.artifacts[metadata.model].append(TaskRecordingArtifact(metadata=metadata, path=path))
        self._keys.add(key)
        self._rollout_index += 1
        return path

    def finalize(
        self,
        *,
        comparison_path: str | Path,
    ) -> dict[str, tuple[Path, ...]]:
        if self._finalized:
            raise RuntimeError("Task Recording directories were already finalized")
        expected = load_comparison_task_metadata(comparison_path)
        paths = validate_paired_recording_directories(
            self.recordings_directory,
            artifacts_by_model=self.artifacts,
            expected_by_model=expected,
        )
        self._finalized = True
        return paths
