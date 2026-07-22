"""Resolve, download, replay, and cache single-task LIBERO demonstrations."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import platform
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import transform_utils

from robot_practice.evaluation.common import (
    iso_timestamp,
    local_timestamp,
    package_version,
    practice_root,
    read_json,
    write_json_atomic,
)
from robot_practice.evaluation.libero_tasks import (
    UPSTREAM_DATASET_LICENSE,
    UPSTREAM_DATASET_REPO_ID,
    UPSTREAM_DATASET_URL,
    LiberoTaskIdentity,
    normalize_task_language,
)

LOGGER = logging.getLogger(__name__)

COMBINED_DATASET_REPO_ID = "lerobot/libero"
CACHE_SCHEMA_VERSION = 1
CONVERTER_VERSION = 1
RAW_SOURCE_SCHEMA_VERSION = 1
REPLAY_REPORT_SCHEMA_VERSION = 1
OUTPUT_FPS = 10
OUTPUT_IMAGE_SHAPE = (256, 256, 3)
STATE_DIM = 8
ACTION_DIM = 7
CAMERA_KEYS = ("observation.images.image", "observation.images.image2")
AGENTVIEW_IMAGE_CONVENTIONS = ("auto", "lerobot_180", "physical_opencv")
RESOLVED_AGENTVIEW_IMAGE_CONVENTIONS = AGENTVIEW_IMAGE_CONVENTIONS[1:]
GIB = 1024**3


class DatasetTaskResolutionError(ValueError):
    """A LeRobot dataset cannot be mapped unambiguously to the requested task."""


class IncompatibleDatasetCacheError(ValueError):
    """A generated cache cannot be reused by this converter version."""


class DatasetWriteError(RuntimeError):
    """A generated LeRobot episode could not be committed safely."""


class ReplayDemoError(RuntimeError):
    """One source demonstration failed while preserving replay progress."""

    def __init__(
        self,
        message: str,
        *,
        failed_action_index: int | None,
        last_completed_action_index: int | None,
        kept_steps: int,
        filtered_noops: int,
    ) -> None:
        super().__init__(message)
        self.failed_action_index = failed_action_index
        self.last_completed_action_index = last_completed_action_index
        self.kept_steps = kept_steps
        self.filtered_noops = filtered_noops


@dataclass(frozen=True)
class DatasetSelection:
    """One LeRobot root and the dataset-native episodes for a LIBERO task."""

    repo_id: str
    root: Path
    task_index: int
    episodes: tuple[int, ...]
    source: str
    episode_namespace: str
    source_manifest: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "root": self.root.resolve(),
            "source_manifest": (
                None if self.source_manifest is None else self.source_manifest.resolve()
            ),
        }


def resolve_agentview_image_convention(
    selection: DatasetSelection,
    requested: str,
    *,
    visualization_enabled: bool,
) -> str | None:
    """Resolve a physical-projection image convention without guessing provenance."""

    if requested not in AGENTVIEW_IMAGE_CONVENTIONS:
        raise ValueError(
            "--agentview-image-convention must be one of "
            f"{list(AGENTVIEW_IMAGE_CONVENTIONS)}, got {requested!r}"
        )
    if not visualization_enabled:
        return None
    if requested != "auto":
        return requested
    if selection.source_manifest is not None:
        manifest = read_json(selection.source_manifest)
        output_contract = manifest.get("output_contract", {})
        if isinstance(output_contract, dict):
            convention = output_contract.get("agentview_image_convention")
            if convention in RESOLVED_AGENTVIEW_IMAGE_CONVENTIONS:
                return str(convention)
    raise ValueError(
        "Cannot verify the stored agentview orientation from dataset provenance. "
        "Set --agentview-image-convention=lerobot_180 for the verified LeRobot/OpenVLA "
        "LIBERO convention, or =physical_opencv for images already matching the physical "
        "camera projection. This option changes visualization only."
    )


@dataclass(frozen=True)
class RawHdf5Summary:
    language: str
    demo_names: tuple[str, ...]
    total_frames: int
    image_shape: tuple[int, int, int]
    state_dim: int
    action_dim: int
    source_control_frequency_hz: float | None


@dataclass(frozen=True)
class RawDatasetSource:
    path: Path
    repository: str
    revision: str
    relative_path: str
    size_bytes: int
    lfs_sha256: str | None
    license: str
    attribution_url: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "path": self.path.resolve()}


def combined_dataset_root() -> Path:
    return practice_root() / "datasets" / "lerobot" / "lerobot" / "libero"


def raw_dataset_cache_root() -> Path:
    return practice_root() / "datasets" / "raw" / "libero"


def converted_dataset_cache_root() -> Path:
    return practice_root() / "datasets" / "lerobot" / "libero_tasks"


def task_cache_root(task: LiberoTaskIdentity, cache_root: Path | None = None) -> Path:
    base = converted_dataset_cache_root() if cache_root is None else cache_root
    return base / task.suite / task.cache_name


def generated_repo_id(task: LiberoTaskIdentity) -> str:
    return f"robot-practice/{task.suite}_{task.cache_name}"


def _read_dataset_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset is missing {info_path}")
    return read_json(info_path)


def _validate_dataset_contract(root: Path, info: dict[str, Any]) -> None:
    fps = float(info.get("fps", -1))
    if fps != OUTPUT_FPS:
        raise ValueError(f"LIBERO dataset at {root} must use {OUTPUT_FPS} Hz, got {fps}")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"LIBERO dataset at {root} has no feature mapping")
    expected_shapes = {
        "observation.state": [STATE_DIM],
        "action": [ACTION_DIM],
        CAMERA_KEYS[0]: list(OUTPUT_IMAGE_SHAPE),
        CAMERA_KEYS[1]: list(OUTPUT_IMAGE_SHAPE),
    }
    for key, shape in expected_shapes.items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise ValueError(f"LIBERO dataset at {root} is missing feature {key}")
        if list(feature.get("shape", [])) != shape:
            raise ValueError(
                f"LIBERO dataset feature {key} at {root} must have shape {shape}, "
                f"got {feature.get('shape')}"
            )
    for key in CAMERA_KEYS:
        if features[key].get("dtype") != "video":
            raise ValueError(f"LIBERO dataset camera {key} at {root} must use video dtype")


def _task_rows(root: Path) -> list[tuple[int, str]]:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset is missing {tasks_path}")
    table = pq.read_table(tasks_path)
    if "task_index" not in table.column_names:
        raise ValueError(f"LIBERO tasks metadata at {tasks_path} has no task_index column")
    string_columns = [
        field.name
        for field in table.schema
        if field.name != "task_index" and pa.types.is_string(field.type)
    ]
    if len(string_columns) != 1:
        raise ValueError(
            f"Expected one task-language column in {tasks_path}, got {string_columns}"
        )
    language_column = string_columns[0]
    return [
        (int(row["task_index"]), normalize_task_language(str(row[language_column])))
        for row in table.select(["task_index", language_column]).to_pylist()
    ]


def _episodes_by_task_index(root: Path) -> dict[int, dict[int, int]]:
    data_root = root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"LeRobot dataset is missing data directory {data_root}")
    table = pads.dataset(data_root, format="parquet").to_table(
        columns=["episode_index", "task_index"]
    )
    if table.num_rows == 0:
        raise ValueError(f"LeRobot dataset at {root} contains no frames")
    episode_tasks: dict[int, set[int]] = defaultdict(set)
    for row in table.to_pylist():
        episode_tasks[int(row["episode_index"])].add(int(row["task_index"]))
    mixed = {episode: values for episode, values in episode_tasks.items() if len(values) != 1}
    if mixed:
        sample = next(iter(mixed.items()))
        raise ValueError(
            f"LIBERO dataset episode {sample[0]} at {root} contains multiple task indices: "
            f"{sorted(sample[1])}"
        )
    result: dict[int, dict[int, int]] = defaultdict(dict)
    for episode, values in episode_tasks.items():
        result[next(iter(values))][episode] = 1
    return dict(result)


def inspect_lerobot_dataset(
    *,
    root: Path,
    repo_id: str,
    task: LiberoTaskIdentity,
    source: str,
    episode_namespace: str,
    source_manifest: Path | None = None,
) -> DatasetSelection:
    """Validate a local LeRobot root and map a language to one task index."""

    root = root.expanduser().resolve()
    info = _read_dataset_info(root)
    _validate_dataset_contract(root, info)
    matches = [index for index, language in _task_rows(root) if language == task.language]
    if len(matches) != 1:
        raise DatasetTaskResolutionError(
            f"Dataset {root} has {len(matches)} exact matches for {task.suite} "
            f"task {task.task_id} language {task.language!r}; expected exactly one"
        )
    task_index = matches[0]
    episodes = tuple(sorted(_episodes_by_task_index(root).get(task_index, {})))
    if not episodes:
        raise DatasetTaskResolutionError(
            f"Dataset {root} task_index={task_index} has no episodes for {task.language!r}"
        )
    return DatasetSelection(
        repo_id=repo_id,
        root=root,
        task_index=task_index,
        episodes=episodes,
        source=source,
        episode_namespace=episode_namespace,
        source_manifest=source_manifest,
    )


def select_episodes(
    selection: DatasetSelection,
    requested: tuple[int, ...] | None,
) -> DatasetSelection:
    if requested is None:
        return selection
    if not requested:
        raise ValueError("--episodes must contain at least one episode index")
    if len(set(requested)) != len(requested):
        raise ValueError("--episodes must not contain duplicate indices")
    if any(index < 0 for index in requested):
        raise ValueError("--episodes values must be non-negative")
    available = set(selection.episodes)
    missing = [index for index in requested if index not in available]
    if missing:
        raise ValueError(
            f"Episodes {missing} do not belong to the selected LIBERO task in "
            f"{selection.root}; available episodes: {list(selection.episodes)}"
        )
    return DatasetSelection(**{**asdict(selection), "episodes": requested})


def is_noop_action(action: np.ndarray, previous_kept_action: np.ndarray | None) -> bool:
    """Match the LIBERO/OpenVLA no-op filter without suppressing gripper changes."""

    array = np.asarray(action)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"LIBERO action must have shape ({ACTION_DIM},), got {array.shape}")
    arm_is_still = bool(np.linalg.norm(array[:-1]) < 1e-4)
    if previous_kept_action is None:
        return arm_is_still
    previous = np.asarray(previous_kept_action)
    if previous.shape != (ACTION_DIM,):
        raise ValueError(
            f"Previous LIBERO action must have shape ({ACTION_DIM},), got {previous.shape}"
        )
    return arm_is_still and bool(array[-1] == previous[-1])


def _natural_demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 2**31 - 1, name


def validate_raw_hdf5(path: Path, task: LiberoTaskIdentity) -> RawHdf5Summary:
    """Validate the upstream task HDF5 without loading its image payloads."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Raw LIBERO HDF5 does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if "data" not in handle or not isinstance(handle["data"], h5py.Group):
            raise ValueError(f"Raw LIBERO HDF5 {path} has no data group")
        data = handle["data"]
        try:
            problem_info = json.loads(data.attrs["problem_info"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Raw LIBERO HDF5 {path} has invalid problem_info") from exc
        raw_language = problem_info.get("language_instruction")
        if isinstance(raw_language, list):
            raw_language = "".join(str(item) for item in raw_language)
        language = normalize_task_language(str(raw_language or ""))
        if language != task.language:
            raise ValueError(
                f"Raw LIBERO HDF5 language mismatch: expected {task.language!r}, "
                f"got {language!r}"
            )

        demo_names = tuple(sorted(data.keys(), key=_natural_demo_key))
        if not demo_names:
            raise ValueError(f"Raw LIBERO HDF5 {path} contains no demonstrations")
        declared_demos = data.attrs.get("num_demos")
        if declared_demos is not None and int(declared_demos) != len(demo_names):
            raise ValueError(
                f"Raw LIBERO HDF5 declares {int(declared_demos)} demos but contains "
                f"{len(demo_names)}"
            )

        total_frames = 0
        image_shapes: set[tuple[int, int, int]] = set()
        for demo_name in demo_names:
            demo = data[demo_name]
            required = (
                "actions",
                "states",
                "robot_states",
                "obs/agentview_rgb",
                "obs/eye_in_hand_rgb",
                "obs/ee_states",
                "obs/gripper_states",
            )
            missing = [key for key in required if key not in demo]
            if missing:
                raise ValueError(f"Raw LIBERO demo {demo_name} is missing {missing}")
            actions = np.asarray(demo["actions"])
            states = np.asarray(demo["states"])
            robot_states = np.asarray(demo["robot_states"])
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"Raw LIBERO demo {demo_name} actions must have shape [T,{ACTION_DIM}], "
                    f"got {actions.shape}"
                )
            length = actions.shape[0]
            if length <= 0:
                raise ValueError(f"Raw LIBERO demo {demo_name} is empty")
            for key, array in (("states", states), ("robot_states", robot_states)):
                if array.ndim != 2 or array.shape[0] != length:
                    raise ValueError(
                        f"Raw LIBERO demo {demo_name} {key} length does not match actions"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(
                        f"Raw LIBERO demo {demo_name} {key} contains non-finite values"
                    )
            if not np.isfinite(actions).all():
                raise ValueError(f"Raw LIBERO demo {demo_name} actions contain non-finite values")
            expected_obs_shapes = {
                "obs/ee_states": (length, 6),
                "obs/gripper_states": (length, 2),
            }
            for key, expected in expected_obs_shapes.items():
                if demo[key].shape != expected:
                    raise ValueError(
                        f"Raw LIBERO demo {demo_name} {key} must have shape {expected}, "
                        f"got {demo[key].shape}"
                    )
            for key in ("obs/agentview_rgb", "obs/eye_in_hand_rgb"):
                shape = tuple(int(value) for value in demo[key].shape)
                if len(shape) != 4 or shape[0] != length or shape[-1] != 3:
                    raise ValueError(
                        f"Raw LIBERO demo {demo_name} {key} must have shape [T,H,W,3], "
                        f"got {shape}"
                    )
                image_shapes.add(shape[1:])
            declared_samples = demo.attrs.get("num_samples")
            if declared_samples is not None and int(declared_samples) != length:
                raise ValueError(
                    f"Raw LIBERO demo {demo_name} declares {int(declared_samples)} samples "
                    f"but contains {length}"
                )
            total_frames += length

        if len(image_shapes) != 1:
            raise ValueError(f"Raw LIBERO HDF5 has inconsistent image shapes: {image_shapes}")
        source_frequency = None
        try:
            env_args = json.loads(data.attrs["env_args"])
            source_frequency = float(env_args["env_kwargs"]["control_freq"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    return RawHdf5Summary(
        language=language,
        demo_names=demo_names,
        total_frames=total_frames,
        image_shape=next(iter(image_shapes)),
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        source_control_frequency_hz=source_frequency,
    )


def _raw_source_sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.source.json")


def _raw_source_from_payload(path: Path, payload: dict[str, Any]) -> RawDatasetSource:
    if payload.get("schema_version") != RAW_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported raw source sidecar schema at {_raw_source_sidecar(path)}")
    return RawDatasetSource(
        path=path,
        repository=str(payload["repository"]),
        revision=str(payload["revision"]),
        relative_path=str(payload["relative_path"]),
        size_bytes=int(payload["size_bytes"]),
        lfs_sha256=(
            None if payload.get("lfs_sha256") is None else str(payload["lfs_sha256"])
        ),
        license=str(payload["license"]),
        attribution_url=str(payload["attribution_url"]),
    )


def _lfs_value(lfs: Any, name: str) -> Any:
    if isinstance(lfs, dict):
        return lfs.get(name)
    return getattr(lfs, name, None)


def _hub_source_metadata(
    task: LiberoTaskIdentity,
    *,
    api: HfApi,
) -> RawDatasetSource:
    info = api.dataset_info(
        UPSTREAM_DATASET_REPO_ID,
        revision="main",
        files_metadata=True,
    )
    revision = str(info.sha)
    sibling = next(
        (
            item
            for item in info.siblings
            if str(item.rfilename) == task.source_relative_path
        ),
        None,
    )
    if sibling is None:
        raise FileNotFoundError(
            f"Upstream dataset {UPSTREAM_DATASET_REPO_ID}@{revision} has no file "
            f"{task.source_relative_path}"
        )
    lfs = getattr(sibling, "lfs", None)
    size = getattr(sibling, "size", None) or _lfs_value(lfs, "size")
    if size is None or int(size) <= 0:
        raise ValueError(f"Upstream file has no valid size metadata: {task.source_relative_path}")
    lfs_sha = _lfs_value(lfs, "sha256") or _lfs_value(lfs, "oid")
    return RawDatasetSource(
        path=Path(),
        repository=UPSTREAM_DATASET_REPO_ID,
        revision=revision,
        relative_path=task.source_relative_path,
        size_bytes=int(size),
        lfs_sha256=None if lfs_sha is None else str(lfs_sha),
        license=UPSTREAM_DATASET_LICENSE,
        attribution_url=UPSTREAM_DATASET_URL,
    )


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def download_raw_task(
    task: LiberoTaskIdentity,
    *,
    raw_root: Path | None = None,
    api: HfApi | None = None,
    downloader: Callable[..., str] = hf_hub_download,
    free_bytes: int | None = None,
) -> RawDatasetSource:
    """Download exactly one task HDF5, retaining provenance in a sidecar."""

    root = raw_dataset_cache_root() if raw_root is None else raw_root.expanduser().resolve()
    path = root / task.source_relative_path
    sidecar = _raw_source_sidecar(path)
    if path.is_file() and sidecar.is_file():
        source = _raw_source_from_payload(path, read_json(sidecar))
        if (
            source.repository == UPSTREAM_DATASET_REPO_ID
            and source.relative_path == task.source_relative_path
            and source.size_bytes == path.stat().st_size
        ):
            LOGGER.info("[Download] cache hit: %s", path)
            return source

    metadata = _hub_source_metadata(task, api=api or HfApi())
    missing_raw_bytes = (
        0
        if path.is_file() and path.stat().st_size == metadata.size_bytes
        else metadata.size_bytes
    )
    conversion_headroom = max(4 * metadata.size_bytes, 5 * GIB)
    required_bytes = missing_raw_bytes + conversion_headroom
    root.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    LOGGER.info("[Download] repository: %s", metadata.repository)
    LOGGER.info("[Download] revision: %s", metadata.revision)
    LOGGER.info("[Download] file: %s", metadata.relative_path)
    LOGGER.info("[Download] size: %s", _format_bytes(metadata.size_bytes))
    LOGGER.info("[Download] destination: %s", path)
    LOGGER.info("[Download] estimated required space: %s", _format_bytes(required_bytes))
    LOGGER.info("[Download] free disk space: %s", _format_bytes(available))
    LOGGER.info("[Download] Raw and converted data will remain cached.")
    if available < required_bytes:
        raise OSError(
            f"Insufficient disk space for {task.source_relative_path}: "
            f"need {_format_bytes(required_bytes)}, have {_format_bytes(available)}"
        )

    if missing_raw_bytes:
        downloaded = Path(
            downloader(
                repo_id=metadata.repository,
                filename=metadata.relative_path,
                repo_type="dataset",
                revision=metadata.revision,
                local_dir=root,
            )
        ).resolve()
        if downloaded != path.resolve() and not path.is_file():
            raise RuntimeError(
                f"Hugging Face download returned {downloaded}, expected stable path {path}"
            )
    if not path.is_file() or path.stat().st_size != metadata.size_bytes:
        actual = None if not path.exists() else path.stat().st_size
        raise IOError(
            f"Downloaded LIBERO file size mismatch at {path}: "
            f"expected {metadata.size_bytes}, got {actual}"
        )
    source = RawDatasetSource(**{**asdict(metadata), "path": path.resolve()})
    write_json_atomic(
        sidecar,
        {
            "schema_version": RAW_SOURCE_SCHEMA_VERSION,
            **source.to_dict(),
            "downloaded_at": iso_timestamp(),
        },
    )
    return source


def _cache_retry_command(task: LiberoTaskIdentity) -> str:
    return (
        "scripts/experiments/act/eval_offline.sh "
        "--policy-path=<pretrained_model> "
        f"--suite={task.suite} --task-id={task.task_id} --rebuild-dataset-cache"
    )


def validate_generated_cache(
    root: Path,
    task: LiberoTaskIdentity,
) -> DatasetSelection:
    root = root.expanduser().resolve()
    manifest_path = root / "source_manifest.json"
    if (root / ".incomplete").exists() or not manifest_path.is_file():
        raise IncompatibleDatasetCacheError(
            f"Generated cache at {root} is incomplete or lacks source_manifest.json. "
            f"Rebuild with: {_cache_retry_command(task)}"
        )
    manifest = read_json(manifest_path)
    schema_version = manifest.get("schema_version")
    converter_version = manifest.get("converter_version")
    if schema_version != CACHE_SCHEMA_VERSION or converter_version != CONVERTER_VERSION:
        raise IncompatibleDatasetCacheError(
            f"Generated cache at {root} uses schema={schema_version}, "
            f"converter={converter_version}; required schema={CACHE_SCHEMA_VERSION}, "
            f"converter={CONVERTER_VERSION}. Rebuild with: {_cache_retry_command(task)}"
        )
    if manifest.get("task") != task.to_dict():
        raise IncompatibleDatasetCacheError(
            f"Generated cache task identity does not match {task.suite} task {task.task_id}. "
            f"Rebuild with: {_cache_retry_command(task)}"
        )
    selection = inspect_lerobot_dataset(
        root=root,
        repo_id=generated_repo_id(task),
        task=task,
        source="generated_task_cache",
        episode_namespace="generated single-task indices",
        source_manifest=manifest_path,
    )
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise IncompatibleDatasetCacheError(
            f"Generated cache manifest at {manifest_path} has no output summary. "
            f"Rebuild with: {_cache_retry_command(task)}"
        )
    expected_episodes = int(output.get("episodes", -1))
    if expected_episodes != len(selection.episodes):
        raise IncompatibleDatasetCacheError(
            f"Generated cache manifest expects {expected_episodes} episodes, "
            f"dataset contains {len(selection.episodes)}. Rebuild with: "
            f"{_cache_retry_command(task)}"
        )
    info = _read_dataset_info(root)
    expected_frames = int(output.get("frames", -1))
    actual_frames = int(info.get("total_frames", -1))
    if expected_frames != actual_frames:
        raise IncompatibleDatasetCacheError(
            f"Generated cache manifest expects {expected_frames} frames, metadata contains "
            f"{actual_frames}. Rebuild with: {_cache_retry_command(task)}"
        )
    if not (root / "replay_report.json").is_file():
        raise IncompatibleDatasetCacheError(
            f"Generated cache at {root} has no replay_report.json. "
            f"Rebuild with: {_cache_retry_command(task)}"
        )
    try:
        _validate_video_sample(selection)
    except Exception as exc:
        raise IncompatibleDatasetCacheError(
            f"Generated cache at {root} cannot decode its first video frame. "
            f"Rebuild with: {_cache_retry_command(task)}"
        ) from exc
    return selection


def _log_dataset_selection(selection: DatasetSelection, *, all_episodes: bool) -> None:
    LOGGER.info("[Dataset] source: %s", selection.source)
    LOGGER.info("[Dataset] root: %s", selection.root)
    LOGGER.info("[Dataset] mapped task_index: %d", selection.task_index)
    LOGGER.info("[Dataset] episode namespace: %s", selection.episode_namespace)
    suffix = " (all demonstrations)" if all_episodes else ""
    LOGGER.info("[Dataset] selected episodes: %d%s", len(selection.episodes), suffix)


def resolve_offline_dataset(
    task: LiberoTaskIdentity,
    *,
    episodes: tuple[int, ...] | None,
    dataset_root: Path | None = None,
    rebuild_dataset_cache: bool = False,
    combined_root: Path | None = None,
    cache_root: Path | None = None,
    prepare: Callable[..., DatasetSelection] | None = None,
) -> DatasetSelection:
    """Resolve explicit, combined, or generated data in deterministic priority order."""

    if dataset_root is not None and rebuild_dataset_cache:
        raise ValueError(
            "--dataset-root cannot be combined with --rebuild-dataset-cache; "
            "explicit dataset roots are user-managed"
        )
    all_episodes = episodes is None
    if dataset_root is not None:
        selection = inspect_lerobot_dataset(
            root=dataset_root,
            repo_id="robot-practice/explicit_libero",
            task=task,
            source="explicit_dataset_root",
            episode_namespace="explicit-dataset native indices",
        )
        selected = select_episodes(selection, episodes)
        _log_dataset_selection(selected, all_episodes=all_episodes)
        return selected

    existing_root = combined_dataset_root() if combined_root is None else combined_root
    if (existing_root / "meta" / "info.json").is_file():
        try:
            selection = inspect_lerobot_dataset(
                root=existing_root,
                repo_id=COMBINED_DATASET_REPO_ID,
                task=task,
                source="existing_combined_dataset",
                episode_namespace="combined-dataset global indices",
            )
        except DatasetTaskResolutionError as exc:
            LOGGER.info("[Dataset] Existing combined dataset cannot satisfy task: %s", exc)
        else:
            if rebuild_dataset_cache:
                LOGGER.info(
                    "[Dataset] --rebuild-dataset-cache is not applicable because the "
                    "existing combined dataset satisfies this task"
                )
            selected = select_episodes(selection, episodes)
            _log_dataset_selection(selected, all_episodes=all_episodes)
            return selected

    generated_root = task_cache_root(task, cache_root)
    if generated_root.is_dir() and not rebuild_dataset_cache:
        selection = validate_generated_cache(generated_root, task)
        LOGGER.info("[Dataset] cache hit: %s", generated_root)
        selected = select_episodes(selection, episodes)
        _log_dataset_selection(selected, all_episodes=all_episodes)
        return selected

    if not generated_root.is_dir():
        LOGGER.info("[Dataset] No converted demonstrations found for this task.")
    prepare_function = prepare or prepare_task_dataset
    selection = prepare_function(
        task,
        cache_root=cache_root,
        rebuild=rebuild_dataset_cache,
    )
    selected = select_episodes(selection, episodes)
    _log_dataset_selection(selected, all_episodes=all_episodes)
    return selected


@contextmanager
def _task_cache_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOGGER.info("[Dataset] Waiting for task cache lock: %s", lock_path)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _replay_report_path(task: LiberoTaskIdentity, cache_root: Path) -> Path:
    stamp = f"{local_timestamp()}_{os.getpid()}"
    return (
        cache_root
        / "replay_reports"
        / task.suite
        / task.cache_name
        / stamp
        / "replay_report.json"
    )


def _remove_stale_temp_directories(task: LiberoTaskIdentity, suite_root: Path) -> None:
    prefix = f".{task.cache_name}.tmp-"
    for path in suite_root.glob(f"{prefix}*"):
        if path.is_dir():
            LOGGER.warning("[Dataset] Removing stale incomplete conversion: %s", path)
            shutil.rmtree(path)


def _make_replay_environment(task: LiberoTaskIdentity) -> OffScreenRenderEnv:
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl_path.is_file():
        raise FileNotFoundError(f"LIBERO BDDL file is missing: {bddl_path}")
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=OUTPUT_IMAGE_SHAPE[0],
        camera_widths=OUTPUT_IMAGE_SHAPE[1],
    )


def _dataset_features() -> dict[str, dict[str, Any]]:
    return {
        CAMERA_KEYS[0]: {
            "dtype": "video",
            "shape": OUTPUT_IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        CAMERA_KEYS[1]: {
            "dtype": "video",
            "shape": OUTPUT_IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["actions"],
        },
    }


def _processed_replay_frame(raw_observation: dict[str, Any], action: np.ndarray) -> dict[str, Any]:
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in raw_observation]
    if missing:
        raise KeyError(f"Replayed LIBERO observation is missing {missing}")
    eef_pos = np.asarray(raw_observation["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(raw_observation["robot0_eef_quat"])
    axis_angle = np.asarray(transform_utils.quat2axisangle(eef_quat), dtype=np.float32)
    gripper = np.asarray(raw_observation["robot0_gripper_qpos"], dtype=np.float32)
    state = np.concatenate((eef_pos, axis_angle, gripper)).astype(np.float32)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"Replayed LIBERO state must be finite shape ({STATE_DIM},), got {state}")

    images: list[np.ndarray] = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.asarray(raw_observation[key])
        if image.shape != OUTPUT_IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(
                f"Replayed camera {key} must be uint8 {OUTPUT_IMAGE_SHAPE}, "
                f"got dtype={image.dtype}, shape={image.shape}"
            )
        images.append(image[::-1, ::-1].copy())
    return {
        CAMERA_KEYS[0]: images[0],
        CAMERA_KEYS[1]: images[1],
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32),
    }


def _replay_one_demo(
    *,
    environment: OffScreenRenderEnv,
    demo: h5py.Group,
    dataset: LeRobotDataset,
    task: LiberoTaskIdentity,
) -> dict[str, Any]:
    actions = np.asarray(demo["actions"], dtype=np.float32)
    initial_state = np.asarray(demo["states"][0])
    previous_kept: np.ndarray | None = None
    filtered_noops = 0
    kept_steps = 0
    success = False
    failed_action_index: int | None = None
    last_completed_action_index: int | None = None
    try:
        environment.reset()
        observation = environment.set_init_state(initial_state)
        for robot in environment.robots:
            robot.controller.use_delta = True
        for _ in range(10):
            observation, _, _, _ = environment.step([0, 0, 0, 0, 0, 0, -1])

        for action_index, action in enumerate(actions):
            failed_action_index = action_index
            if is_noop_action(action, previous_kept):
                filtered_noops += 1
                continue
            frame = _processed_replay_frame(observation, action)
            frame["task"] = task.language
            dataset.add_frame(frame)
            kept_steps += 1
            previous_kept = action.copy()
            observation, _, _, _ = environment.step(action.tolist())
            last_completed_action_index = action_index
            if bool(environment.check_success()):
                success = True
                break
    except Exception as exc:
        dataset.clear_episode_buffer(delete_images=True)
        raise ReplayDemoError(
            str(exc),
            failed_action_index=failed_action_index,
            last_completed_action_index=last_completed_action_index,
            kept_steps=kept_steps,
            filtered_noops=filtered_noops,
        ) from exc

    if not success or kept_steps == 0:
        dataset.clear_episode_buffer(delete_images=True)
        return {
            "status": "failed",
            "reason": "task_not_successful" if kept_steps else "no_kept_actions",
            "failed_action_index": None,
            "last_completed_action_index": last_completed_action_index,
            "kept_steps": kept_steps,
            "filtered_noops": filtered_noops,
        }

    output_episode_index = dataset.meta.total_episodes
    try:
        dataset.save_episode(parallel_encoding=False)
    except Exception as exc:
        raise DatasetWriteError(
            f"Could not save generated episode {output_episode_index}"
        ) from exc
    return {
        "status": "success",
        "output_episode_index": output_episode_index,
        "last_completed_action_index": last_completed_action_index,
        "kept_steps": kept_steps,
        "filtered_noops": filtered_noops,
    }


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "lerobot": package_version("lerobot"),
        "libero": package_version("libero"),
        "robosuite": package_version("robosuite"),
        "mujoco": package_version("mujoco"),
    }


def _validate_video_sample(selection: DatasetSelection) -> None:
    dataset = LeRobotDataset(
        selection.repo_id,
        root=selection.root,
        episodes=[selection.episodes[0]],
        download_videos=False,
    )
    item = dataset[0]
    for key in CAMERA_KEYS:
        image = item[key]
        shape = tuple(int(value) for value in image.shape)
        if shape != (3, OUTPUT_IMAGE_SHAPE[0], OUTPUT_IMAGE_SHAPE[1]):
            raise ValueError(f"Generated cache camera {key} decoded to unexpected shape {shape}")


def _publish_cache(temporary: Path, final: Path) -> None:
    backup = final.with_name(f".{final.name}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if final.exists():
        final.rename(backup)
    try:
        temporary.rename(final)
    except BaseException:
        if backup.exists() and not final.exists():
            backup.rename(final)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def prepare_task_dataset(
    task: LiberoTaskIdentity,
    *,
    cache_root: Path | None = None,
    raw_root: Path | None = None,
    rebuild: bool = False,
    environment_factory: Callable[[LiberoTaskIdentity], OffScreenRenderEnv] = (
        _make_replay_environment
    ),
) -> DatasetSelection:
    """Prepare one generated LeRobot task cache under an advisory lock."""

    base = (
        converted_dataset_cache_root()
        if cache_root is None
        else cache_root.expanduser().resolve()
    )
    final = task_cache_root(task, base)
    suite_root = final.parent
    lock_path = suite_root / f".{task.cache_name}.lock"
    with _task_cache_lock(lock_path):
        if final.is_dir() and not rebuild:
            selection = validate_generated_cache(final, task)
            LOGGER.info("[Dataset] cache hit after lock: %s", final)
            return selection

        suite_root.mkdir(parents=True, exist_ok=True)
        _remove_stale_temp_directories(task, suite_root)
        source = download_raw_task(task, raw_root=raw_root)
        summary = validate_raw_hdf5(source.path, task)
        LOGGER.info("[Replay] output: %s", final)
        LOGGER.info(
            "[Replay] source camera shape: %dx%d; regenerated shape: %dx%d",
            summary.image_shape[0],
            summary.image_shape[1],
            OUTPUT_IMAGE_SHAPE[0],
            OUTPUT_IMAGE_SHAPE[1],
        )
        LOGGER.info(
            "[Replay] source control metadata: %s Hz; output dataset contract: %d Hz",
            summary.source_control_frequency_hz,
            OUTPUT_FPS,
        )
        LOGGER.info("[Replay] source demonstrations: %d", len(summary.demo_names))

        report_path = _replay_report_path(task, base)
        report: dict[str, Any] = {
            "schema_version": REPLAY_REPORT_SCHEMA_VERSION,
            "status": "incomplete",
            "started_at": iso_timestamp(),
            "task": task.to_dict(),
            "source": source.to_dict(),
            "demonstrations": [],
        }
        write_json_atomic(report_path, report)

        temporary = suite_root / f".{task.cache_name}.tmp-{os.getpid()}-{local_timestamp()}"
        dataset = LeRobotDataset.create(
            repo_id=generated_repo_id(task),
            root=temporary,
            robot_type="panda",
            fps=OUTPUT_FPS,
            features=_dataset_features(),
            use_videos=True,
            image_writer_threads=4,
            vcodec="libsvtav1",
        )
        (temporary / ".incomplete").write_text(
            "LIBERO replay conversion has not completed.\n", encoding="utf-8"
        )
        environment: OffScreenRenderEnv | None = None
        interrupted: BaseException | None = None
        try:
            environment = environment_factory(task)
            with h5py.File(source.path, "r") as handle:
                for demo_name in summary.demo_names:
                    try:
                        result = _replay_one_demo(
                            environment=environment,
                            demo=handle["data"][demo_name],
                            dataset=dataset,
                            task=task,
                        )
                    except DatasetWriteError:
                        raise
                    except ReplayDemoError as exc:
                        cause = exc.__cause__ or exc
                        LOGGER.warning("[Replay] %s failed: %s", demo_name, cause)
                        result = {
                            "status": "failed",
                            "reason": "exception",
                            "exception_type": type(cause).__name__,
                            "message": str(cause),
                            "failed_action_index": exc.failed_action_index,
                            "last_completed_action_index": (
                                exc.last_completed_action_index
                            ),
                            "kept_steps": exc.kept_steps,
                            "filtered_noops": exc.filtered_noops,
                        }
                    except Exception as exc:
                        LOGGER.warning("[Replay] %s failed: %s", demo_name, exc)
                        result = {
                            "status": "failed",
                            "reason": "exception",
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    report["demonstrations"].append({"source_demo": demo_name, **result})
                    write_json_atomic(report_path, report)
        except BaseException as exc:
            interrupted = exc
        finally:
            if environment is not None:
                try:
                    environment.close()
                except BaseException as exc:
                    if interrupted is None:
                        interrupted = exc
                    else:
                        LOGGER.exception("[Replay] Environment cleanup also failed")
            try:
                dataset.finalize()
            except BaseException as exc:
                if interrupted is None:
                    interrupted = exc
                else:
                    LOGGER.exception("[Replay] Dataset finalization also failed")

        successes = [
            item for item in report["demonstrations"] if item.get("status") == "success"
        ]
        failures = [
            item for item in report["demonstrations"] if item.get("status") != "success"
        ]
        filtered_noops = sum(
            int(item.get("filtered_noops", 0)) for item in report["demonstrations"]
        )
        generated_frames = sum(int(item.get("kept_steps", 0)) for item in successes)
        if interrupted is not None:
            report["status"] = "interrupted"
        elif successes:
            report["status"] = "complete"
        else:
            report["status"] = "failed"
        report["completed_at"] = iso_timestamp()
        report["summary"] = {
            "source_demonstrations": len(summary.demo_names),
            "successful": len(successes),
            "failed": len(failures),
            "filtered_noops": filtered_noops,
            "generated_frames": generated_frames,
        }
        write_json_atomic(report_path, report)
        LOGGER.info("[Replay] successful: %d", len(successes))
        LOGGER.info("[Replay] failed: %d", len(failures))
        LOGGER.info("[Replay] filtered no-op steps: %d", filtered_noops)
        LOGGER.info("[Replay] generated frames: %d", generated_frames)
        LOGGER.info("[Replay] report: %s", report_path)
        if interrupted is not None:
            raise interrupted
        if not successes:
            raise RuntimeError(
                f"All {len(summary.demo_names)} LIBERO demonstrations failed replay; "
                f"raw data remains at {source.path}; report: {report_path}"
            )

        shutil.copy2(report_path, temporary / "replay_report.json")
        source_manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "converter_version": CONVERTER_VERSION,
            "created_at": iso_timestamp(),
            "task": task.to_dict(),
            "source": source.to_dict(),
            "raw_contract": {
                "image_shape": summary.image_shape,
                "state_dim": summary.state_dim,
                "action_dim": summary.action_dim,
                "control_frequency_hz": summary.source_control_frequency_hz,
            },
            "output_contract": {
                "fps": OUTPUT_FPS,
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "camera_keys": CAMERA_KEYS,
                "image_shape": OUTPUT_IMAGE_SHAPE,
                "image_rotation_degrees": 180,
                "agentview_image_convention": "lerobot_180",
                "timeline_note": (
                    "No-op-filtered source transitions are stored under the existing "
                    "10 Hz LeRobot LIBERO contract and are not asserted to be a physically "
                    "uniform source timeline."
                ),
            },
            "runtime": _runtime_versions(),
            "replay_report": report_path.resolve(),
            "output": {
                "episodes": len(successes),
                "frames": generated_frames,
                "failed_episodes": len(failures),
                "filtered_noops": filtered_noops,
                "episode_map": [
                    {
                        "source_demo": item["source_demo"],
                        "output_episode_index": item["output_episode_index"],
                    }
                    for item in successes
                ],
            },
        }
        write_json_atomic(temporary / "source_manifest.json", source_manifest)
        prepublication = inspect_lerobot_dataset(
            root=temporary,
            repo_id=generated_repo_id(task),
            task=task,
            source="generated_task_cache",
            episode_namespace="generated single-task indices",
            source_manifest=temporary / "source_manifest.json",
        )
        _validate_video_sample(prepublication)
        (temporary / ".incomplete").unlink()
        _publish_cache(temporary, final)
        selection = validate_generated_cache(final, task)
        LOGGER.info("[Dataset] cache ready: %s", final)
        return selection
