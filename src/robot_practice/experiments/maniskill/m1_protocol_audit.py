"""M1 ManiSkill data, controller and protocol audit.

This module is the machine-readable implementation of
``plans/2026-08-11-M1三任务数据与控制协议预检.md``.  It deliberately separates
metadata-only phases from phases that construct ManiSkill.  The latter are only
invoked by the remote runner; importing this module never creates an environment,
starts a replay, or trains a policy.
"""

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import practice_root

from .state_identity import (
    invert_unique_fingerprints,
    start_state_fingerprint,
    state_fingerprint,
)
from .success_predicates import quaternion_canonicalize
from .rollout_contract import (
    M1W_DATA_PROVENANCE,
    OFFICIAL_DATA_PROVENANCE,
    require_rollout_backend,
)

KNOWN_TASKS = ("StackCube-v1", "PegInsertionSide-v1", "PlugCharger-v1")
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/experiments/maniskill/m1_protocol_audit.json")
DEFAULT_OUTPUT = Path("outputs/maniskill/m1_protocol_audit")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON payload contains NaN or Infinity")
    return value


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON without losing recoverability on a cross-device rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        try:
            temporary.replace(path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(temporary, path)
            temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(practice_root()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = _rooted(path)
    payload = _read_json(config_path)
    payload["_config_path"] = config_path.as_posix()
    return payload


def _configured_rollout_contract(config: Mapping[str, Any]) -> dict[str, str]:
    """Resolve the backend contract declared by an M1 protocol config."""

    protocol = _require_mapping(config.get("protocol"), "protocol")
    is_wam_protocol = str(config.get("protocol_id")) == "m1_wam_protocol_audit"
    expected_provenance = M1W_DATA_PROVENANCE if is_wam_protocol else OFFICIAL_DATA_PROVENANCE
    declared_provenance = config.get("data_provenance")
    if declared_provenance is None:
        raise ValueError(
            "M1 config must declare data_provenance: official or m1w"
        )
    contract = require_rollout_backend(
        str(declared_provenance), str(protocol.get("backend", ""))
    )
    if contract["data_provenance"] != expected_provenance:
        raise ValueError(
            f"M1 config protocol_id={config.get('protocol_id')!r} must use "
            f"data_provenance={expected_provenance!r}, got {contract['data_provenance']!r}"
        )
    return contract


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def configured_tasks(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the task set owned by a protocol config, preserving file order."""

    tasks = _require_mapping(config.get("tasks"), "tasks")
    names = tuple(str(name) for name in tasks)
    if not names:
        raise ValueError("tasks must contain at least one task")
    unknown = tuple(name for name in names if name not in KNOWN_TASKS)
    if unknown:
        raise ValueError(f"Unsupported task(s): {unknown}; known tasks are {KNOWN_TASKS}")
    return names


def selected_tasks(config: Mapping[str, Any], task: str) -> tuple[str, ...]:
    """Resolve a CLI task selector against the tasks in the loaded config."""

    tasks = configured_tasks(config)
    if task == "all":
        return tasks
    if task not in tasks:
        raise ValueError(f"Task {task!r} is not enabled by this config; expected one of {tasks}")
    return (task,)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all frozen, pre-result protocol choices."""

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported M1 config schema: {config.get('schema_version')!r}")
    archive = _require_mapping(config.get("archive"), "archive")
    if archive.get("repo_id") != "haosulab/ManiSkill_Demonstrations":
        raise ValueError("M1 archive repo_id must be the official demonstration dataset")
    revision = archive.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("M1 archive revision must be a full immutable git revision")
    files = _require_mapping(archive.get("files"), "archive.files")
    tasks = _require_mapping(config.get("tasks"), "tasks")
    task_names = configured_tasks(config)
    protocol = _require_mapping(config.get("protocol"), "protocol")
    required_protocol = {
        "source_control_mode": "pd_joint_pos",
        "source_obs_mode": "none",
        "source_type": "motionplanning",
        "replay_reward_mode": "sparse",
        "replay_use_first_env_state": True,
        "replay_allow_failure": True,
        "replay_max_retry": 0,
    }
    is_wam_protocol = str(config.get("protocol_id")) == "m1_wam_protocol_audit"
    rollout_contract = _configured_rollout_contract(config)
    if is_wam_protocol:
        required_protocol.update(
            {
                "data_route": "frozen_gpu_carrier_live_rgb_replay_7d",
                "native_source_backend": "physx_cuda",
                "native_source_control_mode": "pd_ee_delta_pose",
                "native_source_action_dim": 7,
                "replay_obs_mode": "rgb",
            }
        )
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"protocol.{key} must be {expected!r}, got {protocol.get(key)!r}")
    backend = rollout_contract["actual_rollout_backend"]
    if backend not in {"physx_cpu", "physx_cuda"}:
        raise ValueError(f"protocol.backend must be physx_cpu or physx_cuda, got {backend!r}")
    for key in ("replay_backend", "data_backend", "rollout_backend"):
        configured = str(protocol.get(key, backend))
        if configured != backend:
            raise ValueError(f"protocol.{key} must match protocol.backend ({backend!r})")
    if backend == "physx_cuda":
        for key in ("replay_num_envs", "data_num_envs", "state_bank_num_envs"):
            if int(protocol.get(key, 0)) <= 0:
                raise ValueError(f"protocol.{key} must be positive for GPU protocol")
    if is_wam_protocol:
        for key in ("baseline_replay_h5", "repair_replay_h5", "frozen_layout_manifest"):
            value = protocol.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"protocol.{key} must be a non-empty path")
        if int(protocol.get("required_formal_episodes", 0)) <= 0:
            raise ValueError("protocol.required_formal_episodes must be positive")
    if set(files) != set(task_names):
        raise ValueError(
            f"archive.files must match configured tasks {task_names}, got {tuple(files)}"
        )
    for task in task_names:
        archive_item = _require_mapping(files.get(task), f"archive.files.{task}")
        digest = archive_item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"archive.files.{task}.sha256 must be a SHA-256 hex digest")
        item = _require_mapping(tasks[task], f"tasks.{task}")
        required_task_keys = (
            "source_path",
            "target_control_mode",
            "action_dim",
            "native_horizon",
        )
        if is_wam_protocol:
            required_task_keys = required_task_keys + ("native_source_dir",)
        for key in required_task_keys:
            if key not in item:
                raise ValueError(f"tasks.{task} is missing {key}")
        if item["target_control_mode"] not in {"pd_ee_delta_pos", "pd_ee_delta_pose"}:
            raise ValueError(f"Unsupported target controller for {task}")
        if task == "PlugCharger-v1":
            if item.get("formal_horizon") is not None:
                raise ValueError("PlugCharger formal_horizon must remain null before replay")
            if not isinstance(item.get("formal_horizon_rule"), Mapping):
                raise ValueError("PlugCharger must define formal_horizon_rule")
        elif not isinstance(item.get("formal_horizon"), int) or item["formal_horizon"] <= 0:
            raise ValueError(f"{task}.formal_horizon must be a positive integer")
    schemas = _require_mapping(config.get("observation_schemas"), "observation_schemas")
    visual = _require_mapping(schemas.get("visual_student_v1"), "visual_student_v1")
    if visual.get("state_dim") != 18:
        raise ValueError("visual_student_v1 state_dim must be 18")
    return {
        "schema_version": int(config["schema_version"]),
        "protocol_id": str(config["protocol_id"]),
        "tasks": list(task_names),
        "archive_revision": revision,
        "backend": protocol["backend"],
        **rollout_contract,
    }


def task_config(config: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    if task not in configured_tasks(config):
        raise ValueError(f"Unknown M1 task {task!r}; expected one of {configured_tasks(config)}")
    return _require_mapping(_require_mapping(config["tasks"], "tasks")[task], f"tasks.{task}")


def output_root(config: Mapping[str, Any], override: str | Path | None = None) -> Path:
    if override is not None:
        return _rooted(override)
    return _rooted(DEFAULT_OUTPUT)


def task_root(config: Mapping[str, Any], task: str, override: str | Path | None = None) -> Path:
    return output_root(config, override) / task


def source_h5(config: Mapping[str, Any], task: str) -> Path:
    archive = _require_mapping(config["archive"], "archive")
    source_root = _rooted(str(archive["source_root"]))
    return source_root / str(task_config(config, task)["source_path"])


def native_source_h5(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    override_output: str | Path | None = None,
) -> Path:
    """Return the formal GPU-native 7D source, never the official CPU source."""

    source_dir = task_root(config, task, override_output) / str(
        task_config(config, task)["native_source_dir"]
    )
    return source_dir / ("full" if full else "smoke") / "trajectory.h5"


def _native_source_input(config: Mapping[str, Any]) -> Path:
    return _rooted(str(config["protocol"]["native_source_input_h5"]))


def _metadata_path(h5_path: Path) -> Path:
    return h5_path.with_suffix(".json")


def _prepare_replay_metadata(
    source_json: Path,
    destination_json: Path,
    *,
    reward_mode: str,
    obs_mode: str | None = None,
) -> dict[str, Any]:
    """Prepare replay-only metadata without mutating the source metadata."""

    metadata = _read_json(source_json)
    env_info = _require_mapping(metadata.get("env_info"), "env_info")
    env_kwargs = _require_mapping(env_info.get("env_kwargs"), "env_info.env_kwargs")
    original_reward_mode = env_kwargs.get("reward_mode")
    replay_metadata = copy.deepcopy(metadata)
    replay_env_info = _require_mapping(replay_metadata["env_info"], "env_info")
    replay_env_kwargs = _require_mapping(
        replay_env_info["env_kwargs"], "env_info.env_kwargs"
    )
    replay_env_kwargs["reward_mode"] = reward_mode
    if obs_mode is not None:
        replay_env_kwargs["obs_mode"] = obs_mode
    _write_json(destination_json, replay_metadata)
    return {
        "source_reward_mode": original_reward_mode,
        "replay_reward_mode": reward_mode,
        "source_obs_mode": env_kwargs.get("obs_mode"),
        "replay_obs_mode": replay_env_kwargs.get("obs_mode"),
        "source_json": source_json.as_posix(),
        "replay_json": destination_json.as_posix(),
    }


def episode_id(name: str) -> int:
    if not name.startswith("traj_"):
        raise ValueError(f"Unexpected trajectory group name {name!r}")
    return int(name.split("_", 1)[1])


def metadata_records(metadata: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("trajectory metadata must contain an episodes list")
    records: dict[int, Mapping[str, Any]] = {}
    for item in episodes:
        record = _require_mapping(item, "episodes[]")
        value = record.get("episode_id")
        if not isinstance(value, int):
            raise TypeError(f"episode_id must be an integer, got {value!r}")
        if value in records:
            raise ValueError(f"duplicate episode_id {value}")
        records[value] = record
    return records


def _success_series(episode: h5py.Group, record: Mapping[str, Any]) -> np.ndarray:
    if "success" in episode:
        values = np.asarray(episode["success"], dtype=bool).reshape(-1)
    else:
        values = np.full(int(episode["actions"].shape[0]), bool(record.get("success", False)))
    action_count = int(episode["actions"].shape[0])
    if values.size not in {action_count, action_count + 1}:
        raise ValueError(
            f"{episode.name} success length {values.size} is incompatible with {action_count} actions"
        )
    return values


def first_success_step(success: Sequence[bool]) -> int | None:
    indices = np.flatnonzero(np.asarray(success, dtype=bool))
    return None if indices.size == 0 else int(indices[0]) + 1


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("mean", "p01", "p10", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(array)),
        "p01": float(np.percentile(array, 1)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _action_stats(actions: np.ndarray, action_dim: int) -> dict[str, Any]:
    names = (
        ("translation_x", "translation_y", "translation_z", "gripper")
        if action_dim == 4
        else tuple(f"action_{index}" for index in range(action_dim))
    )
    finite = bool(np.isfinite(actions).all())
    absolute = np.abs(actions.astype(np.float64, copy=False))
    channels: dict[str, Any] = {}
    for index, name in enumerate(names):
        values = actions[:, index] if actions.size else np.asarray([], dtype=np.float64)
        channels[name] = {
            **_quantiles(values),
            "min": None if values.size == 0 else float(np.min(values)),
            "std": None if values.size == 0 else float(np.std(values)),
            "out_of_bounds_count": int(np.sum(np.abs(values) > 1.00001)),
        }
    return {
        "action_dim": action_dim,
        "channel_order": list(names),
        "finite": finite,
        "max_abs": None if absolute.size == 0 else float(np.max(absolute)),
        "out_of_bounds_count": int(np.sum(absolute > 1.00001)),
        "channels": channels,
    }


def chunk_slicing_audit(
    episode_lengths: Sequence[int], *, n_obs_steps: int, horizon: int, n_action_steps: int
) -> dict[str, Any]:
    """Count valid windows and endpoint padding without crossing episodes."""

    if min(episode_lengths, default=0) < 0:
        raise ValueError("episode lengths must be non-negative")
    valid_windows = 0
    padded_windows = 0
    padded_actions = 0
    for length in episode_lengths:
        for start in range(int(length)):
            valid_windows += 1
            obs_end = start + n_obs_steps
            action_end = start + n_action_steps
            if obs_end > length or start + horizon > length:
                padded_windows += 1
            if action_end > length:
                padded_actions += max(0, action_end - length)
    return {
        "n_obs_steps": n_obs_steps,
        "horizon": horizon,
        "n_action_steps": n_action_steps,
        "episode_count": len(episode_lengths),
        "valid_windows": valid_windows,
        "episode_end_padded_windows": padded_windows,
        "episode_end_padded_actions": padded_actions,
        "cross_episode_windows": 0,
        "mask_policy": "pad_only_within_episode_and_mark_mask_false",
    }


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Archive member escapes destination: {member.filename}")
        archive.extractall(destination)


def download_archives(
    config: Mapping[str, Any], *, override_output: str | Path | None = None
) -> dict[str, Any]:
    """Download fixed-revision archives and publish source trees without overwrite."""

    from huggingface_hub import hf_hub_download

    archive = _require_mapping(config["archive"], "archive")
    download_dir = _rooted(str(archive["download_dir"]))
    source_root = _rooted(str(archive["source_root"]))
    download_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    files = _require_mapping(archive["files"], "archive.files")
    for task in configured_tasks(config):
        item = _require_mapping(files[task], f"archive.files.{task}")
        target_zip = download_dir / str(item["archive_name"])
        expected_sha = str(item["sha256"])
        if target_zip.is_file():
            actual_sha = sha256_file(target_zip)
            if actual_sha != expected_sha:
                raise RuntimeError(f"Existing archive hash mismatch for {task}: {actual_sha}")
            downloaded_from = "existing"
        else:
            with tempfile.TemporaryDirectory(prefix=f"m1_{task}_", dir=download_dir) as temp_dir:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=str(archive["repo_id"]),
                        repo_type=str(archive["repo_type"]),
                        filename=str(item["filename"]),
                        revision=str(archive["revision"]),
                        local_dir=temp_dir,
                    )
                )
                actual_sha = sha256_file(downloaded)
                if actual_sha != expected_sha:
                    raise RuntimeError(f"Downloaded archive hash mismatch for {task}: {actual_sha}")
                shutil.copy2(downloaded, target_zip)
            downloaded_from = "huggingface_hub"
        task_dir = source_root / task
        expected_h5 = task_dir / "motionplanning" / "trajectory.h5"
        expected_json = _metadata_path(expected_h5)
        if not expected_h5.is_file() or not expected_json.is_file():
            with tempfile.TemporaryDirectory(
                prefix=f"m1_extract_{task}_", dir=source_root
            ) as temp_dir:
                extracted = Path(temp_dir) / "extract"
                _safe_extract(target_zip, extracted)
                h5_candidates = list(extracted.rglob("motionplanning/trajectory.h5"))
                if len(h5_candidates) != 1:
                    raise RuntimeError(
                        f"Expected one motionplanning trajectory in {task}, got {h5_candidates}"
                    )
                extracted_root = h5_candidates[0].parents[1]
                if task_dir.exists():
                    raise FileExistsError(
                        f"Partial source directory exists; refusing overwrite: {task_dir}"
                    )
                shutil.copytree(extracted_root, task_dir, symlinks=False)
        records[task] = {
            "archive": target_zip.as_posix(),
            "sha256": sha256_file(target_zip),
            "downloaded_from": downloaded_from,
            "source_h5": expected_h5.as_posix(),
            "source_json": expected_json.as_posix(),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "revision": archive["revision"],
        "repo_id": archive["repo_id"],
        "tasks": records,
    }
    _write_json(output_root(config, override_output) / "download_manifest.json", result)
    return result


def audit_source(
    config: Mapping[str, Any], task: str, *, override_output: str | Path | None = None
) -> dict[str, Any]:
    path = source_h5(config, task)
    metadata_path = _metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Source H5/JSON pair is missing: {path}")
    metadata = _read_json(metadata_path)
    records = metadata_records(metadata)
    env_info = _require_mapping(metadata.get("env_info"), "env_info")
    env_kwargs = _require_mapping(env_info.get("env_kwargs"), "env_info.env_kwargs")
    raw_source_type = metadata.get("source_type", metadata.get("source"))
    normalized_source_type = (
        None if raw_source_type is None else str(raw_source_type).split("|", 1)[0].strip().lower()
    )
    source_contract = {
        "env_id": env_info.get("env_id"),
        "source_type": normalized_source_type,
        "source_type_raw": raw_source_type,
        "source_desc": metadata.get("source_desc"),
        "control_mode": env_kwargs.get("control_mode"),
        "obs_mode": env_kwargs.get("obs_mode"),
        "sim_backend": env_kwargs.get("sim_backend"),
    }
    if source_contract["env_id"] != task:
        raise ValueError(f"Source env_id is not {task}: {source_contract['env_id']!r}")
    if source_contract["control_mode"] != "pd_joint_pos" or source_contract["obs_mode"] != "none":
        raise ValueError(
            f"Source controller/observation contract is wrong for {task}: {source_contract}"
        )
    episode_rows: list[dict[str, Any]] = []
    group_ids: list[int] = []
    action_lengths: list[int] = []
    success_count = 0
    with h5py.File(path, "r") as handle:
        for name in sorted(handle.keys(), key=episode_id):
            group = handle[name]
            current_id = episode_id(name)
            group_ids.append(current_id)
            if "actions" not in group:
                raise ValueError(f"{name} has no actions dataset")
            length = int(group["actions"].shape[0])
            action_lengths.append(length)
            record = records.get(current_id, {})
            success = _success_series(group, record)
            row = {
                "episode_id": current_id,
                "num_action_steps": length,
                "metadata_success": bool(record.get("success", False)),
                "success_once": bool(np.any(success)),
                "success_at_end": bool(success[-1]) if success.size else False,
                "first_success_step": first_success_step(success),
                "source_seed": record.get(
                    "episode_seed", record.get("reset_kwargs", {}).get("seed")
                ),
            }
            episode_rows.append(row)
            success_count += int(row["metadata_success"])
    expected_count = int(config["protocol"]["required_source_episodes"])
    ids_match = group_ids == sorted(records)
    archive_manifest_path = output_root(config, override_output) / "download_manifest.json"
    archive_manifest = (
        _read_json(archive_manifest_path) if archive_manifest_path.is_file() else None
    )
    archive_item = None if archive_manifest is None else archive_manifest.get("tasks", {}).get(task)
    archive_path = (
        None
        if not isinstance(archive_item, Mapping)
        else Path(str(archive_item.get("archive", "")))
    )
    archive_gate = bool(
        isinstance(archive_item, Mapping)
        and archive_path is not None
        and archive_path.is_file()
        and archive_item.get("sha256")
        == _require_mapping(config["archive"]["files"], "archive.files")[task]["sha256"]
        and sha256_file(archive_path) == archive_item.get("sha256")
        and archive_manifest.get("revision") == config["archive"]["revision"]
    )
    gates = {
        "archive_identity": {
            "manifest": archive_manifest_path.as_posix(),
            "passed": archive_gate,
        },
        "episode_count": {
            "observed": len(records),
            "expected": expected_count,
            "passed": len(records) == expected_count,
        },
        "h5_group_count": {
            "observed": len(group_ids),
            "expected": expected_count,
            "passed": len(group_ids) == expected_count,
        },
        "h5_json_ids": {"observed": ids_match, "passed": ids_match},
        "metadata_success": {
            "observed": success_count,
            "expected": expected_count,
            "passed": success_count == expected_count,
        },
        "source_contract": {
            "observed": source_contract,
            "passed": source_contract["source_type"] == "motionplanning"
            and source_contract["control_mode"] == "pd_joint_pos"
            and source_contract["obs_mode"] == "none",
        },
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "source_h5": path.as_posix(),
        "source_json": metadata_path.as_posix(),
        "source_sha256": sha256_file(path),
        "source_json_sha256": sha256_file(metadata_path),
        "source_contract": source_contract,
        "archive_manifest": archive_manifest,
        "summary": {
            "num_episodes": len(records),
            "total_frames": int(sum(action_lengths)),
            "episode_length": _quantiles(action_lengths),
            "first_success_step": _quantiles(
                [
                    float(row["first_success_step"])
                    for row in episode_rows
                    if row["first_success_step"] is not None
                ]
            ),
        },
        "episodes": episode_rows,
        "gates": gates,
        "gate_pass": all(bool(item["passed"]) for item in gates.values()),
    }
    root = task_root(config, task, override_output)
    _write_json(
        root / "source_manifest.json",
        {
            "task": task,
            "h5_sha256": result["source_sha256"],
            "json_sha256": result["source_json_sha256"],
        },
    )
    _write_json(root / "source_audit.json", result)
    return result


def _observation_schema(observation: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
            return
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        result[path] = {"shape": list(array.shape), "dtype": str(array.dtype)}

    walk(observation, "")
    return result


def environment_audit(
    config: Mapping[str, Any], task: str, *, override_output: str | Path | None = None
) -> dict[str, Any]:
    """Record the actual configured-backend environment contract."""

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    item = task_config(config, task)
    protocol = _require_mapping(config["protocol"], "protocol")
    rollout_contract = _configured_rollout_contract(config)
    backend = rollout_contract["actual_rollout_backend"]
    kwargs = {
        "obs_mode": "rgb",
        "control_mode": item["target_control_mode"],
        "sim_backend": backend,
        "reward_mode": "sparse",
        "reconfiguration_freq": 1,
        "render_mode": "rgb_array",
    }
    env = gym.make(task, **kwargs)
    state_env = gym.make(
        task,
        # ``state`` is a flattened tensor in ManiSkill3 and loses the qpos/qvel
        # field boundaries needed for the 18D student-state audit.  Keep the
        # structured form for contract inspection; this does not change the
        # visual replay or training observation contract.
        obs_mode="state_dict",
        control_mode=item["target_control_mode"],
        sim_backend=backend,
        reward_mode="sparse",
        reconfiguration_freq=1,
    )
    try:
        observation, reset_info = env.reset(seed=10000)
        state_observation, _ = state_env.reset(seed=10000)
        action_space = env.action_space
        action_low = np.asarray(action_space.low, dtype=np.float64)
        action_high = np.asarray(action_space.high, dtype=np.float64)
        unwrapped = env.unwrapped
        reward_modes = list(getattr(unwrapped, "SUPPORTED_REWARD_MODES", []))
        observation_schema = _observation_schema(observation)
        state_observation_schema = _observation_schema(state_observation)
        state_dict = unwrapped.get_state_dict()
        state_structure = _observation_schema(state_dict)
        camera_schema = {
            key: value
            for key, value in observation_schema.items()
            if ".rgb" in key or key.endswith("rgb")
        }
        required_cameras = _require_mapping(
            _require_mapping(config["observation_schemas"], "observation_schemas")[
                "visual_student_v1"
            ],
            "visual_student_v1",
        )["cameras"]
        camera_gate = all(
            any(
                name in key
                and value["shape"][-3:] == expected["shape"]
                and value["dtype"] == expected["dtype"]
                for key, value in camera_schema.items()
            )
            for name, expected in _require_mapping(
                required_cameras, "visual_student_v1.cameras"
            ).items()
        )
        state_fields = _require_mapping(config["observation_schemas"], "observation_schemas")
        expected_state_schema_name = str(item["observation_schema"])
        expected_state_dim = int(
            _require_mapping(state_fields[expected_state_schema_name], expected_state_schema_name)[
                "state_dim"
            ]
        )
        state_leaves = {
            key: value
            for key, value in state_observation_schema.items()
            if "qpos" in key or "qvel" in key or "extra" in key
        }
        state_dim_candidates = [
            int(value["shape"][-1])
            for key, value in state_leaves.items()
            if value["shape"] and key.endswith(("qpos", "qvel"))
        ]
        if expected_state_schema_name == "visual_student_v1":
            observed_state_dim = sum(state_dim_candidates)
        else:
            observed_state_dim = sum(
                int(value["shape"][-1]) for key, value in state_leaves.items() if value["shape"]
            )
        state_gate = observed_state_dim == expected_state_dim
        result = {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "env_id": getattr(unwrapped, "__class__", type(unwrapped)).__name__,
            "env_kwargs": kwargs,
            "control_freq_hz": getattr(unwrapped, "control_freq", None),
            "sim_backend": str(
                getattr(getattr(unwrapped, "backend", None), "sim_backend", backend)
            ),
            **rollout_contract,
            "action_space": {
                "shape": list(action_space.shape),
                "low": action_low.tolist(),
                "high": action_high.tolist(),
                "finite": bool(np.isfinite(action_low).all() and np.isfinite(action_high).all()),
            },
            "observation_schema": observation_schema,
            "state_observation_schema": state_observation_schema,
            "state_dict_structure": state_structure,
            "reset_info_keys": sorted(str(key) for key in reset_info),
            "reward_modes": reward_modes,
            "termination_contract": {
                "formal_backend": protocol["backend"],
                "ignore_terminations": True,
                "record_metrics": True,
                "partial_reset": False,
            },
            "gates": {
                "action_dim": {
                    "observed": int(np.prod(action_space.shape)),
                    "expected": int(item["action_dim"]),
                    "passed": int(np.prod(action_space.shape)) == int(item["action_dim"]),
                },
                "control_frequency": {
                    "observed": getattr(unwrapped, "control_freq", None),
                    "expected": 20,
                    "passed": getattr(unwrapped, "control_freq", None) == 20,
                },
                "camera_schema": {"observed": camera_schema, "passed": camera_gate},
                "state_schema": {
                    "observed_dim": observed_state_dim,
                    "expected_dim": expected_state_dim,
                    "passed": state_gate,
                },
                "sparse_reward_supported": {
                    "observed": reward_modes,
                    "passed": "sparse" in reward_modes or task == "PlugCharger-v1",
                },
            },
        }
    finally:
        env.close()
        state_env.close()
    result["gate_pass"] = all(bool(item["passed"]) for item in result["gates"].values())
    root = task_root(config, task, override_output)
    _write_json(root / "environment_contract.json", result)
    _write_json(root / "observation_schema.json", result["observation_schema"])
    return result


def _copy_pair(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"Refusing to overwrite different file: {destination}")
        return
    shutil.copy2(source, destination)


def replay_task(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    num_workers: int | None = None,
    num_envs: int | None = None,
    override_output: str | Path | None = None,
    source_override: str | Path | None = None,
    phase_name: str | None = None,
    output_subdir: str | None = None,
    success_threshold: float | None = None,
) -> dict[str, Any]:
    """Run one replay in the configured backend in an isolated attempt.

    ``source_override`` is used by the formal M1-W route for the offline CPU
    joint-to-EE artifact.  The default remains the raw joint replay diagnostic
    so older callers do not silently change meaning.
    """

    item = task_config(config, task)
    root = task_root(config, task, override_output)
    phase_name = phase_name or ("replay_full" if full else "replay_smoke")
    phase_root = root / phase_name
    marker = phase_root / "phase_manifest.json"
    if marker.is_file():
        completed = _read_json(marker)
        expected = 10 if not full else int(config["protocol"]["required_source_episodes"])
        required_cached_success = float(
            completed.get(
                "required_success_rate",
                config["protocol"].get("replay_survival_threshold", 1.0),
            )
        )
        if full:
            if (
                completed.get("full") is not True
                or completed.get("expected_replays") != expected
                or completed.get("replay_count") != expected
                or (
                    required_cached_success > 0
                    and completed.get("successful_replays") != expected
                )
            ):
                raise RuntimeError(
                    f"Existing {phase_name} manifest is not a hard-gate success; "
                    f"move {phase_root} aside and rerun"
                )
        elif completed.get("replay_count") != expected or (
            required_cached_success > 0 and completed.get("successful_replays") != expected
        ):
            raise RuntimeError(
                f"Existing {phase_name} manifest is not a {expected}/{expected} success; "
                f"move {phase_root} aside and rerun"
            )
        return completed
    if phase_root.exists() and any(phase_root.iterdir()):
        raise FileExistsError(f"Incomplete {phase_name} exists; inspect or remove it: {phase_root}")
    phase_root.mkdir(parents=True, exist_ok=True)
    protocol = _require_mapping(config["protocol"], "protocol")
    rollout_contract = _configured_rollout_contract(config)
    backend = rollout_contract["actual_rollout_backend"]
    workers = int(num_workers or protocol["replay_num_workers"])
    replay_envs = int(num_envs or protocol.get("replay_num_envs", workers))
    source = _rooted(source_override) if source_override is not None else source_h5(config, task)
    source_json = _metadata_path(source)
    if not source.is_file() or not source_json.is_file():
        raise FileNotFoundError(source)
    attempt = phase_root / f".attempt-{os.getpid()}"
    input_dir = attempt / "input"
    input_dir.mkdir(parents=True)
    input_h5 = input_dir / "trajectory.h5"
    input_json = input_dir / "trajectory.json"
    shutil.copy2(source, input_h5)
    replay_contract = _prepare_replay_metadata(
        source_json,
        input_json,
        reward_mode=str(config["protocol"]["replay_reward_mode"]),
        obs_mode=str(config["protocol"]["replay_obs_mode"]),
    )
    log_path = phase_root / "replay.log"
    command = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(input_h5),
        "--use-first-env-state",
        "--obs-mode",
        str(config["protocol"]["replay_obs_mode"]),
        "--sim-backend",
        backend,
        "--reward-mode",
        str(config["protocol"]["replay_reward_mode"]),
        "--save-traj",
        "--record-rewards",
        "--allow-failure",
        "--max-retry",
        "0",
        "--num-envs",
        str(replay_envs if backend == "physx_cuda" else workers),
    ]
    if backend == "physx_cpu":
        command.extend(["--target-control-mode", str(item["target_control_mode"])])
    if not full:
        command.extend(["--count", "10"])
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=practice_root(), stdout=log, stderr=subprocess.STDOUT, check=True
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Replay failed for {task}; inspect {log_path} for the child-process traceback"
        ) from exc
    candidates = [
        path
        for path in input_dir.glob("*.h5")
        if path.name != input_h5.name and not path.name.endswith(".tmp")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one replay H5 for {task}, found {candidates}; see {log_path}")
    replay_parent = phase_root / output_subdir if output_subdir else (
        phase_root / "source_joint" if backend == "physx_cuda" else phase_root
    )
    replay_parent.mkdir(parents=True, exist_ok=True)
    replay_h5 = replay_parent / "replay.h5"
    replay_json = replay_parent / "replay.json"
    shutil.copy2(candidates[0], replay_h5)
    candidate_json = _metadata_path(candidates[0])
    if not candidate_json.is_file():
        raise RuntimeError(f"Replay metadata is missing beside {candidates[0]}")
    shutil.copy2(candidate_json, replay_json)
    replay_output_metadata = _read_json(replay_json)
    replay_records = metadata_records(replay_output_metadata)
    with h5py.File(replay_h5, "r") as handle:
        replay_episode_ids = [episode_id(name) for name in handle]
        successful_replays = sum(
            bool(np.any(_success_series(handle[name], replay_records[episode_id(name)])))
            for name in handle
        )
    replay_count = len(replay_episode_ids)
    expected_replays = 10 if not full else len(metadata_records(_read_json(input_json)))
    success_rate = successful_replays / max(1, expected_replays)
    required_success_rate = (
        float(success_threshold)
        if success_threshold is not None
        else (1.0 if not full else float(config["protocol"].get("replay_survival_threshold", 1.0)))
    )
    gate_pass = replay_count == expected_replays and success_rate >= required_success_rate
    if not gate_pass:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "task": task,
            "phase": phase_name,
            "full": full,
            "expected_replays": expected_replays,
            "replay_count": replay_count,
            "successful_replays": successful_replays,
            "failed_replays": expected_replays - successful_replays,
            "success_rate": success_rate,
            "required_success_rate": required_success_rate,
            "replay_h5": replay_h5.as_posix(),
            "replay_json": replay_json.as_posix(),
            "log": log_path.as_posix(),
            "command": command,
        }
        _write_json(phase_root / "phase_failure.json", failure)
        shutil.rmtree(attempt)
        raise RuntimeError(
            f"{phase_name} gate failed for {task}: "
            f"{successful_replays}/{expected_replays} successful; "
            f"inspect {phase_root / 'phase_failure.json'} and {log_path}"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "phase": phase_name,
        "full": full,
        "num_workers": workers,
        "num_envs": replay_envs if backend == "physx_cuda" else workers,
        "backend": backend,
        **rollout_contract,
        "source_only": backend == "physx_cuda" and source_override is None,
        "source_override": None if source_override is None else source.as_posix(),
        "command": command,
        "source_h5": source.as_posix(),
        "replay_h5": replay_h5.as_posix(),
        "replay_json": replay_json.as_posix(),
        "replay_h5_sha256": sha256_file(replay_h5),
        "replay_json_sha256": sha256_file(replay_json),
        "expected_replays": expected_replays,
        "replay_count": replay_count,
        "successful_replays": successful_replays,
        "success_rate": success_rate,
        "required_success_rate": required_success_rate,
        "success_gate": "hard" if required_success_rate > 0 else "episode_count_only",
        "replay_metadata": replay_contract,
    }
    _write_json(marker, result)
    shutil.rmtree(attempt)
    return result


def offline_convert_ee(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    num_envs: int | None = None,
    override_output: str | Path | None = None,
) -> dict[str, Any]:
    """Convert the official joint demos to native EE demos on CPU.

    ManiSkill3 does not support cross-controller conversion in its GPU replay
    path.  This artifact is intentionally an intermediate converter output;
    all canonical replay and later data repair happen on GPU.
    """

    protocol = _require_mapping(config["protocol"], "protocol")
    if protocol.get("backend") != "physx_cuda":
        raise ValueError("M1-W offline EE conversion requires the active GPU protocol")
    task_item = task_config(config, task)
    if task_item.get("target_control_mode") != "pd_ee_delta_pose" or int(
        task_item.get("action_dim", -1)
    ) != 7:
        raise ValueError("GPU StackCube data generation requires a 7D pd_ee_delta_pose target")
    root = task_root(config, task, override_output)
    phase_name = "offline_convert_full" if full else "offline_convert_smoke"
    phase_root = root / phase_name
    manifest_path = phase_root / "conversion_manifest.json"
    expected = int(protocol["required_source_episodes"]) if full else 10
    if manifest_path.is_file() and (phase_root / "replay.h5").is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("gate_pass") and manifest.get("episode_count") == expected:
            return manifest
        raise RuntimeError(f"Existing offline conversion is not a hard-gate success: {manifest_path}")
    if phase_root.exists() and any(phase_root.iterdir()):
        raise FileExistsError(f"Incomplete offline conversion exists: {phase_root}")
    phase_root.mkdir(parents=True, exist_ok=True)
    source = source_h5(config, task)
    source_json = _metadata_path(source)
    input_dir = phase_root / "input"
    input_dir.mkdir(parents=True)
    input_h5 = input_dir / "trajectory.h5"
    input_json = input_dir / "trajectory.json"
    shutil.copy2(source, input_h5)
    _prepare_replay_metadata(
        source_json,
        input_json,
        reward_mode=str(protocol["replay_reward_mode"]),
        obs_mode="rgb",
    )
    log_path = phase_root / "conversion.log"
    command = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(input_h5),
        "--use-first-env-state",
        "--target-control-mode",
        str(task_item["target_control_mode"]),
        "--obs-mode",
        "rgb",
        "--sim-backend",
        "physx_cpu",
        "--reward-mode",
        str(protocol["replay_reward_mode"]),
        "--save-traj",
        "--record-rewards",
        "--allow-failure",
        "--max-retry",
        "0",
        "--num-envs",
        "1",
    ]
    if not full:
        command.extend(["--count", "10"])
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=practice_root(), stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Offline EE conversion failed for {task}; inspect {log_path}") from exc
    candidates = [path for path in input_dir.glob("*.h5") if path.name != input_h5.name]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one offline EE H5, found {candidates}; see {log_path}")
    output_h5 = phase_root / "replay.h5"
    output_json = phase_root / "replay.json"
    shutil.copy2(candidates[0], output_h5)
    shutil.copy2(_metadata_path(candidates[0]), output_json)
    metadata = _read_json(output_json)
    records = metadata_records(metadata)
    converted_fingerprints: dict[int, str] = {}
    with h5py.File(output_h5, "r") as handle:
        episode_count = len(handle)
        end_success = sum(
            bool(np.asarray(handle[name]["success"], dtype=bool).reshape(-1)[-1])
            for name in handle
        )
        action_dims = sorted({int(handle[name]["actions"].shape[1]) for name in handle})
        for name in handle:
            converted_fingerprints[episode_id(name)] = start_state_fingerprint(handle[name])
    converted_by_fingerprint = invert_unique_fingerprints(
        converted_fingerprints, label="offline native EE conversion"
    )
    identity_gate = episode_count == expected and len(converted_by_fingerprint) == expected
    for record in records.values():
        converted_id = int(record["episode_id"])
        fingerprint = converted_fingerprints[converted_id]
        record["offline_episode_id"] = converted_id
        record["start_state_fingerprint"] = fingerprint
        record.pop("episode_seed", None)
        reset_kwargs = record.get("reset_kwargs")
        if isinstance(reset_kwargs, dict):
            reset_kwargs.pop("seed", None)
    metadata["episodes"] = list(records.values())
    metadata["identity"] = {
        "key": "start_state_fingerprint",
        "source_frame": "offline EE env_states[0]",
        "tolerance": 1e-4,
        "seed_is_not_identity": True,
        "raw_joint_source_crosscheck": "not required; CPU converter may normalize initial actor poses",
    }
    _write_json(output_json, metadata)
    gate_pass = episode_count == expected and end_success == expected and action_dims == [7]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "phase": phase_name,
        "route": "offline_cpu_joint_to_native_ee",
        "source_h5": source.as_posix(),
        "replay_h5": output_h5.as_posix(),
        "replay_json": output_json.as_posix(),
        "conversion_backend": "physx_cpu",
        "canonical_downstream_backend": "physx_cuda",
        "target_control_mode": task_item["target_control_mode"],
        "episode_count": episode_count,
        "expected_episodes": expected,
        "successful_at_end": end_success,
        "action_dims": action_dims,
        "identity": {
            "key": "start_state_fingerprint from env_states frame 0",
            "tolerance": 1e-4,
            "offline_unique_count": len(converted_by_fingerprint),
            "expected_count": expected,
            "seed_is_not_identity": True,
            "gate_pass": identity_gate,
        },
        "command": command,
        "log": log_path.as_posix(),
        "gate_pass": gate_pass,
    }
    _write_json(manifest_path, manifest)
    gate_pass = gate_pass and identity_gate
    manifest["gate_pass"] = gate_pass
    if not gate_pass:
        raise RuntimeError(
            f"Offline EE conversion gate failed: {end_success}/{expected} end-success, "
            f"action_dims={action_dims}; inspect {manifest_path}"
        )
    shutil.rmtree(input_dir)
    return manifest


def prepare_gpu_native_source(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    override_output: str | Path | None = None,
) -> dict[str, Any]:
    """Snapshot the existing GPU-native 7D action source.

    The official motion-planning H5 remains provenance-only.  This phase does
    not invoke the CPU converter or recreate a controller conversion.  It
    copies the already generated GPU ``pd_ee_delta_pose`` action artifact and
    creates a deterministic 10-episode smoke subset when requested.
    """

    protocol = _require_mapping(config["protocol"], "protocol")
    if protocol.get("data_route") != "gpu_native_live_rgb_replay_with_failed_ee_repair_7d":
        raise ValueError(
            "M1-W data preparation requires "
            "protocol.data_route=gpu_native_live_rgb_replay_with_failed_ee_repair_7d"
        )
    input_h5 = _native_source_input(config)
    input_json = _metadata_path(input_h5)
    if not input_h5.is_file() or not input_json.is_file():
        raise FileNotFoundError(
            "GPU-native StackCube input artifact is missing: "
            f"{input_h5} (+ JSON metadata)"
        )
    phase_root = task_root(config, task, override_output) / str(
        task_config(config, task)["native_source_dir"]
    ) / ("full" if full else "smoke")
    output_h5 = phase_root / "trajectory.h5"
    output_json = output_h5.with_suffix(".json")
    manifest_path = phase_root / "source_manifest.json"
    expected = int(protocol["required_source_episodes"]) if full else 10
    if manifest_path.is_file() and output_h5.is_file() and output_json.is_file():
        manifest = _read_json(manifest_path)
        if (
            manifest.get("gate_pass")
            and manifest.get("episode_count") == expected
            and manifest.get("route") == "gpu_native_action_source_7d"
        ):
            return manifest
        raise RuntimeError(f"Existing GPU-native source is not a hard-gate success: {manifest_path}")
    if phase_root.exists() and any(phase_root.iterdir()):
        raise FileExistsError(f"Incomplete GPU-native source exists: {phase_root}")
    phase_root.mkdir(parents=True, exist_ok=True)

    input_metadata = _read_json(input_json)
    input_records = metadata_records(input_metadata)
    input_ids = sorted(input_records)
    if len(input_ids) < expected:
        raise RuntimeError(
            f"GPU-native input has {len(input_ids)} episodes; expected at least {expected}"
        )
    selected_ids = input_ids if full else input_ids[:expected]
    with h5py.File(input_h5, "r") as source_file, h5py.File(output_h5, "w") as destination:
        for source_id in selected_ids:
            source_file.copy(f"traj_{source_id}", destination, name=f"traj_{source_id}")

    output_metadata = copy.deepcopy(input_metadata)
    output_metadata["episodes"] = []
    for source_id in selected_ids:
        record = copy.deepcopy(input_records[source_id])
        record["episode_id"] = source_id
        record["native_source_episode_id"] = source_id
        record.pop("episode_seed", None)
        reset_kwargs = record.get("reset_kwargs")
        if isinstance(reset_kwargs, dict):
            reset_kwargs.pop("seed", None)
        output_metadata["episodes"].append(record)
    output_metadata["m1_wam_data_route"] = {
        "route": "gpu_native_action_source_7d",
        "input_artifact": _relative(input_h5),
        "backend": "physx_cuda",
        "control_mode": "pd_ee_delta_pose",
        "action_dim": 7,
        "identity": "env_states frame 0 fingerprint",
        "seed_is_not_identity": True,
    }
    _write_json(output_json, output_metadata)

    fingerprints: dict[int, str] = {}
    action_dims: set[int] = set()
    finite_actions = True
    with h5py.File(output_h5, "r") as handle:
        for source_id in selected_ids:
            group = handle[f"traj_{source_id}"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise RuntimeError(f"traj_{source_id} actions must be rank-2, got {actions.shape}")
            action_dims.add(int(actions.shape[1]))
            finite_actions = finite_actions and bool(np.isfinite(actions).all())
            fingerprints[source_id] = start_state_fingerprint(group, tolerance=1e-4)
    invert_unique_fingerprints(fingerprints, label="GPU-native source")
    env_kwargs = _require_mapping(input_metadata.get("env_info"), "env_info").get("env_kwargs")
    env_kwargs = _require_mapping(env_kwargs, "env_info.env_kwargs")
    route_gate = (
        str(env_kwargs.get("sim_backend")) == "physx_cuda"
        and str(env_kwargs.get("control_mode")) == "pd_ee_delta_pose"
        and action_dims == {7}
        and finite_actions
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "phase": "gpu_source_full" if full else "gpu_source_smoke",
        "route": "gpu_native_action_source_7d",
        "input_artifact": _relative(input_h5),
        "input_artifact_sha256": sha256_file(input_h5),
        "replay_h5": _relative(output_h5),
        "replay_json": _relative(output_json),
        "backend": "physx_cuda",
        "control_mode": "pd_ee_delta_pose",
        "episode_count": len(selected_ids),
        "expected_episodes": expected,
        "action_dims": sorted(action_dims),
        "finite_actions": finite_actions,
        "unique_start_state_fingerprints": len(fingerprints),
        "identity": {
            "key": "start_state_fingerprint from env_states frame 0",
            "tolerance": 1e-4,
            "seed_is_not_identity": True,
        },
        "gate_pass": len(selected_ids) == expected and len(fingerprints) == expected and route_gate,
    }
    _write_json(manifest_path, manifest)
    if not manifest["gate_pass"]:
        raise RuntimeError(f"GPU-native source gate failed; inspect {manifest_path}")
    return manifest


def materialize_gpu_data(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    num_workers: int | None = None,
    num_envs: int | None = None,
    override_output: str | Path | None = None,
) -> dict[str, Any]:
    raise RuntimeError(
        "State materialization is retired from the M1-W formal route. "
        "Use baseline_gpu_replay -> repair_gpu_ee_failures -> "
        "merge_gpu_ee_data -> final_live_rgb_replay."
    )


def baseline_gpu_replay(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    num_envs: int | None = None,
    override_output: str | Path | None = None,
) -> dict[str, Any]:
    """Replay native 7D actions and record only live state/action signals.

    This is stage 1 of the WAM route.  It intentionally uses ``obs_mode=none``:
    RGB is not materialized from the recorded states, and it is not needed to
    decide which source episodes require EE repair.  The output still contains
    the action stream, success stream, and live ``env_states`` produced by the
    actual ``env.step`` calls.
    """

    if str(config.get("protocol_id")) != "m1_wam_protocol_audit":
        raise ValueError("baseline_gpu_replay is only defined for the M1-W protocol")
    source = native_source_h5(config, task, full=full, override_output=override_output)
    if not source.is_file():
        raise FileNotFoundError(f"GPU-native source preparation is required first: {source}")
    root = task_root(config, task, override_output)
    output_dir = root / ("gpu_ee_replay_full" if full else "gpu_ee_replay_smoke") / "baseline"
    from .gpu_ee_replay import replay

    expected = int(config["protocol"]["required_source_episodes"]) if full else 10
    return replay(
        source_h5=source,
        output_dir=output_dir,
        num_envs=int(num_envs or config["protocol"].get("data_num_envs", 100)),
        expected_episodes=expected,
        obs_mode="none",
        source_type="m1_w_gpu_baseline_live_replay",
        source_desc=(
            "Stage-1 live GPU env.step replay of native pd_ee_delta_pose actions; "
            "records action, success, and env_states without RGB"
        ),
        required_success_rate=float(config["protocol"].get("baseline_replay_success_threshold", 0.0)),
    )


def repair_gpu_ee_failures(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    num_envs: int | None = None,
    override_output: str | Path | None = None,
) -> dict[str, Any]:
    """Repair only failed GPU EE episodes using the recorded TCP reference."""

    root = task_root(config, task, override_output)
    phase_name = "gpu_ee_repair_full" if full else "gpu_ee_repair_smoke"
    phase_root = root / phase_name
    manifest_path = phase_root / "repair_manifest.json"
    if manifest_path.is_file():
        cached = _read_json(manifest_path)
        if not cached.get("gate_pass"):
            raise RuntimeError(f"Existing EE repair is not a hard-gate success: {manifest_path}")
        return cached
    replay_root = root / ("gpu_ee_replay_full" if full else "gpu_ee_replay_smoke")
    baseline = replay_root / "baseline" / "replay.h5"
    offline = native_source_h5(config, task, full=full, override_output=override_output)
    if not baseline.is_file() or not offline.is_file():
        raise FileNotFoundError("GPU-native source and GPU baseline replay are required before repair")
    from .gpu_ee_repair import repair

    return repair(
        offline_ee_h5=offline,
        gpu_replay_h5=baseline,
        output_dir=phase_root,
        task_id=task,
        num_envs=int(num_envs or _require_mapping(config["protocol"], "protocol").get("data_num_envs", 100)),
    )


def merge_gpu_ee_data(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    override_output: str | Path | None = None,
) -> dict[str, Any]:
    """Merge baseline and repaired groups by unique start-state fingerprint."""

    root = task_root(config, task, override_output)
    phase_name = "merge_data_full" if full else "merge_data_smoke"
    phase_root = root / phase_name
    manifest_path = phase_root / "data_manifest.json"
    if manifest_path.is_file():
        cached = _read_json(manifest_path)
        if not cached.get("gate_pass"):
            raise RuntimeError(f"Existing merged data is not a hard-gate success: {manifest_path}")
        return cached
    replay_root = root / ("gpu_ee_replay_full" if full else "gpu_ee_replay_smoke")
    repair_root = root / ("gpu_ee_repair_full" if full else "gpu_ee_repair_smoke")
    baseline_h5 = replay_root / "baseline" / "replay.h5"
    repair_h5 = repair_root / "repair.h5"
    offline_h5 = native_source_h5(config, task, full=full, override_output=override_output)
    if not baseline_h5.is_file() or not offline_h5.is_file():
        raise FileNotFoundError("GPU-native source and GPU baseline artifacts are required before merge")
    baseline_json = _read_json(baseline_h5.with_suffix(".json"))
    baseline_records = metadata_records(baseline_json)
    source_fingerprints: dict[int, str] = {}
    with h5py.File(offline_h5, "r") as offline_file:
        for name in offline_file:
            source_fingerprints[episode_id(name)] = start_state_fingerprint(
                offline_file[name]
            )
    source_by_fingerprint = invert_unique_fingerprints(
        source_fingerprints, label="GPU-native source"
    )
    repair_json = _read_json(repair_h5.with_suffix(".json")) if repair_h5.is_file() else {"episodes": []}
    repair_records_by_group = {
        int(record["episode_id"]): record for record in repair_json.get("episodes", [])
    }
    repair_fingerprints = {
        group_id: str(record["repair_start_state_fingerprint"])
        for group_id, record in repair_records_by_group.items()
        if record.get("repair_start_state_fingerprint")
    }
    repair_by_fingerprint = invert_unique_fingerprints(
        repair_fingerprints, label="GPU repaired EE trajectories"
    )
    expected = int(config["protocol"]["required_source_episodes"]) if full else 10
    if len(baseline_records) != expected:
        raise RuntimeError(f"Baseline GPU EE replay has {len(baseline_records)} episodes; expected {expected}")
    failed_ids = []
    baseline_fingerprints: dict[int, str] = {}
    with h5py.File(baseline_h5, "r") as baseline_file:
        for name in baseline_file:
            baseline_fingerprints[episode_id(name)] = start_state_fingerprint(baseline_file[name])
            success = np.asarray(baseline_file[name]["success"], dtype=bool).reshape(-1)
            if not success.size or not bool(success[-1]):
                failed_ids.append(episode_id(name))
    if failed_ids and not repair_h5.is_file():
        raise FileNotFoundError(f"GPU repair output is required for {len(failed_ids)} failures: {repair_h5}")
    with h5py.File(baseline_h5, "r") as baseline_file:
        baseline_fingerprint_map = {
            target_id: baseline_fingerprints[target_id] for target_id in baseline_records
        }
        invert_unique_fingerprints(
            baseline_fingerprint_map, label="GPU baseline replay"
        )
    if set(baseline_fingerprints.values()) != set(source_fingerprints.values()):
        raise RuntimeError(
            "GPU baseline replay does not cover exactly the GPU-native source identity set"
        )
    with h5py.File(baseline_h5, "r") as baseline_file:
        failed_source_ids = {
            target_id: source_by_fingerprint.get(baseline_fingerprints[target_id])
            for target_id in failed_ids
        }
    missing_repair = []
    for target_id in failed_ids:
        source_id = failed_source_ids[target_id]
        repair_group_id = repair_by_fingerprint.get(baseline_fingerprints[target_id])
        if source_id is None or repair_group_id is None:
            missing_repair.append({"target_episode_id": target_id, "source_episode_id": source_id})
    if missing_repair:
        raise RuntimeError(f"Missing repaired trajectories for failed GPU episodes: {missing_repair[:5]}")
    phase_root.mkdir(parents=True, exist_ok=True)
    final_h5 = phase_root / "replay.h5"
    final_json = final_h5.with_suffix(".json")
    if final_h5.exists() or final_json.exists():
        raise FileExistsError(f"Incomplete merged data exists at {phase_root}")
    with h5py.File(baseline_h5, "r") as baseline_file, h5py.File(final_h5, "w") as destination:
        repair_file = h5py.File(repair_h5, "r") if repair_h5.is_file() else None
        try:
            for target_id in sorted(baseline_records):
                baseline_fingerprint = baseline_fingerprints[target_id]
                source_id = source_by_fingerprint.get(baseline_fingerprint)
                if source_id is None:
                    raise KeyError(
                        f"GPU baseline episode {target_id} has no GPU-native source identity"
                    )
                repair_group_id = repair_by_fingerprint.get(baseline_fingerprint)
                if repair_group_id is not None and repair_file is not None:
                    repair_record = repair_records_by_group[repair_group_id]
                    if (
                        repair_record.get("source_start_state_fingerprint")
                        != source_fingerprints[source_id]
                        or repair_record.get("baseline_start_state_fingerprint")
                        != baseline_fingerprint
                        or repair_record.get("repair_start_state_fingerprint")
                        != baseline_fingerprint
                    ):
                        raise RuntimeError(
                            f"Repair identity metadata mismatch for baseline episode {target_id}"
                        )
                    repair_file.copy(
                        f"traj_{repair_group_id}",
                        destination,
                        name=f"traj_{target_id}",
                    )
                    destination[f"traj_{target_id}"].attrs["repaired_from_source_episode_id"] = source_id
                    destination[f"traj_{target_id}"].attrs[
                        "start_state_fingerprint"
                    ] = baseline_fingerprint
                else:
                    baseline_file.copy(
                        f"traj_{target_id}", destination, name=f"traj_{target_id}"
                    )
                    destination[f"traj_{target_id}"].attrs[
                        "start_state_fingerprint"
                    ] = baseline_fingerprint
        finally:
            if repair_file is not None:
                repair_file.close()
    merged_metadata = copy.deepcopy(baseline_json)
    merged_metadata["env_info"]["env_kwargs"]["sim_backend"] = "physx_cuda"
    # RGB is intentionally absent from this intermediate artifact.  It is
    # captured only by final_live_rgb_replay after the action merge.
    merged_metadata["env_info"]["env_kwargs"]["obs_mode"] = "none"
    merged_metadata["m1_wam_data_route"] = {
        "gpu_native_source": offline_h5.as_posix(),
        "gpu_baseline": baseline_h5.as_posix(),
        "gpu_failed_ee_repair": None if not repair_h5.is_file() else repair_h5.as_posix(),
        "merge_rule": "successful baseline groups plus successful failed-only repair groups matched only by start-state fingerprint",
        "identity": "env_states frame 0 fingerprint; seed and trajectory group id are not identity keys",
        "identity_tolerance": 1e-4,
    }
    _write_json(final_json, merged_metadata)
    final_success = 0
    action_dims: set[int] = set()
    final_episode_count = 0
    final_fingerprints: dict[int, str] = {}
    with h5py.File(final_h5, "r") as handle:
        final_episode_count = len(handle)
        for name, group in handle.items():
            final_id = episode_id(name)
            final_fingerprints[final_id] = start_state_fingerprint(group)
            final_success += int(bool(np.asarray(group["success"], dtype=bool).reshape(-1)[-1]))
            action_dims.add(int(group["actions"].shape[1]))
    invert_unique_fingerprints(final_fingerprints, label="merged GPU data")
    if set(final_fingerprints.values()) != set(baseline_fingerprints.values()):
        raise RuntimeError("Merged GPU data does not preserve the baseline start-state identity set")
    repaired_count = len(repair_by_fingerprint)
    gate_pass = final_episode_count == expected and final_success == expected and action_dims == {7}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "phase": phase_name,
        "route": "gpu_native_7d_baseline_plus_failed_only_ee_repair",
        "replay_h5": final_h5.as_posix(),
        "replay_json": final_json.as_posix(),
        "episode_count": final_episode_count,
        "successful_at_end": final_success,
        "failed_baseline_episodes": len(failed_ids),
        "repaired_episodes": repaired_count,
        "rgb_captured": False,
        "final_live_replay_required": True,
        "identity": {
            "key": "start_state_fingerprint from env_states frame 0",
            "tolerance": 1e-4,
            "unique_episode_count": len(final_fingerprints),
            "seed_is_not_identity": True,
            "gate_pass": True,
        },
        "action_dims": sorted(action_dims),
        "gate_pass": gate_pass,
    }
    _write_json(manifest_path, manifest)
    if not gate_pass:
        raise RuntimeError(f"Merged GPU data gate failed: {final_success}/{expected}, dims={action_dims}")
    return manifest


def final_live_rgb_replay(
    config: Mapping[str, Any],
    task: str,
    *,
    full: bool,
    num_envs: int | None = None,
    override_output: str | Path | None = None,
) -> dict[str, Any]:
    """Capture the frozen baseline/repair carrier layout with live RGB.

    This is the only stage whose observations are eligible for D_core.  Every
    RGB frame is returned by the same GPU environment step that produced the
    corresponding action transition; no frame is reconstructed from an
    ``env_states`` snapshot afterwards.
    """

    if not full:
        raise ValueError("The frozen 987-episode layout is a full-only formal capture")
    root = task_root(config, task, override_output)
    output_dir = root / ("final_live_rgb_full" if full else "final_live_rgb_smoke")
    from .capture_gpu_ee_live_rgb import capture, freeze_layout

    protocol = _require_mapping(config["protocol"], "protocol")
    baseline = _rooted(str(protocol["baseline_replay_h5"]))
    repair = _rooted(str(protocol["repair_replay_h5"]))
    layout = _rooted(str(protocol["frozen_layout_manifest"]))
    freeze_layout(
        baseline_h5=baseline,
        repair_h5=repair,
        output=layout,
        num_envs=int(num_envs or protocol.get("data_num_envs", 100)),
    )
    return capture(
        baseline_h5=baseline,
        repair_h5=repair,
        layout_path=layout,
        output_dir=output_dir,
    )


def _replay_path(config: Mapping[str, Any], task: str) -> Path:
    path = task_root(config, task) / "final_live_rgb_full" / "replay.h5"
    if not path.is_file():
        raise FileNotFoundError(f"Full replay is required before trajectory audit: {path}")
    return path


def audit_trajectory(
    config: Mapping[str, Any], task: str, *, override_output: str | Path | None = None
) -> dict[str, Any]:
    item = task_config(config, task)
    root = task_root(config, task, override_output)
    replay = root / "final_live_rgb_full" / "replay.h5"
    replay_json = _metadata_path(replay)
    if (
        not replay.is_file()
        and str(config.get("protocol_id")) != "m1_wam_protocol_audit"
    ):
        replay = root / "replay_full" / "replay.h5"
        replay_json = _metadata_path(replay)
    if not replay.is_file() or not replay_json.is_file():
        raise FileNotFoundError(f"Full replay H5/JSON pair is missing: {replay}")
    metadata = _read_json(replay_json)
    records = metadata_records(metadata)
    rows: list[dict[str, Any]] = []
    action_blocks: list[np.ndarray] = []
    lengths: list[int] = []
    reward_matches = 0
    reward_values = 0
    observation_schema: dict[str, Any] = {}
    with h5py.File(replay, "r") as handle:
        for name in sorted(handle.keys(), key=episode_id):
            group = handle[name]
            current_id = episode_id(name)
            if current_id not in records:
                raise ValueError(f"Replay metadata has no record for {name}")
            actions = np.asarray(group["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] != int(item["action_dim"]):
                raise ValueError(
                    f"{name} action shape is {actions.shape}; expected [T,{item['action_dim']}]"
                )
            success = _success_series(group, records[current_id])
            first = first_success_step(success)
            length = int(actions.shape[0])
            lengths.append(length)
            action_blocks.append(actions)
            native_horizon = int(item["native_horizon"])
            success_by_native = first is not None and first <= native_horizon
            formal = item.get("formal_horizon")
            success_by_formal = (
                None if formal is None else first is not None and first <= int(formal)
            )
            if "rewards" in group:
                reward = np.asarray(group["rewards"], dtype=np.float64).reshape(-1)
                comparable = min(reward.size, success.size)
                reward_matches += int(
                    np.sum(
                        np.isclose(
                            reward[:comparable], success[:comparable].astype(float), atol=1e-6
                        )
                    )
                )
                reward_values += comparable
            if "obs" in group:
                observation_schema.update(_h5_schema(group["obs"], prefix="obs"))
            rows.append(
                {
                    "episode_id": current_id,
                    "source_episode_id": int(
                        records[current_id].get("source_episode_id", current_id)
                    ),
                    "start_state_fingerprint": records[current_id].get(
                        "source_start_state_fingerprint"
                    ),
                    "num_action_steps": length,
                    "success_once": bool(np.any(success)),
                    "success_at_end": bool(success[-1]) if success.size else False,
                    "first_success_step": first,
                    "success_by_native_horizon": bool(success_by_native),
                    "success_by_formal_horizon": success_by_formal,
                    "raw_frame_count": length,
                    "trim_action_count": None
                    if first is None
                    else min(length, first + int(config["protocol"]["trim_success_tail_frames"])),
                    "success_series_length": int(success.size),
                    "has_env_states": "env_states" in group,
                }
            )
    if not action_blocks:
        raise ValueError(f"Replay contains no trajectory groups: {replay}")
    actions = np.concatenate(action_blocks, axis=0)
    action_stats = _action_stats(actions, int(item["action_dim"]))
    is_wam_protocol = str(config.get("protocol_id")) == "m1_wam_protocol_audit"
    successful = [row for row in rows if row["success_once"]]
    formal_values = [
        row["first_success_step"] for row in successful if row["first_success_step"] is not None
    ]
    formal_horizon = item.get("formal_horizon")
    if formal_horizon is not None:
        # Horizon coverage answers whether the live-replay successes lie inside
        # the formal horizon; replay survival is a separate hard gate.
        coverage_denominator = len(successful) if is_wam_protocol else len(rows)
        formal_coverage = sum(value <= int(formal_horizon) for value in formal_values) / max(
            1, coverage_denominator
        )
    else:
        formal_coverage = None
    pre_success_frames = sum(
        0 if row["first_success_step"] is None else int(row["first_success_step"]) - 1
        for row in rows
    )
    total_frames = int(sum(lengths))
    timing = {
        "native_horizon": int(item["native_horizon"]),
        "formal_horizon": formal_horizon,
        "num_episodes": len(rows),
        "success_once_count": len(successful),
        "success_at_end_count": sum(bool(row["success_at_end"]) for row in rows),
        "success_by_native_horizon_count": sum(
            bool(row["success_by_native_horizon"]) for row in rows
        ),
        "success_by_formal_horizon_count": None
        if formal_horizon is None
        else sum(bool(row["success_by_formal_horizon"]) for row in rows),
        "success_by_native_horizon_rate": sum(
            bool(row["success_by_native_horizon"]) for row in rows
        )
        / max(1, len(rows)),
        "success_by_formal_horizon_rate": formal_coverage,
        "first_success_step": _quantiles([float(value) for value in formal_values]),
        "episode_length": _quantiles(lengths),
        "pre_success_frame_share": pre_success_frames / max(1, total_frames),
    }
    chunks = {
        f"{spec['n_obs_steps']}/{spec['horizon']}/{spec['n_action_steps']}": chunk_slicing_audit(
            lengths,
            n_obs_steps=int(spec["n_obs_steps"]),
            horizon=int(spec["horizon"]),
            n_action_steps=int(spec["n_action_steps"]),
        )
        for spec in config["protocol"]["chunk_contracts"]
    }
    replay_success_count = len(successful)
    required_episodes = int(
        config["protocol"].get(
            "required_formal_episodes", config["protocol"]["required_source_episodes"]
        )
    )
    replay_survival = replay_success_count / required_episodes
    gates = {
        "replay_survival": {
            "observed": replay_survival,
            "threshold": config["protocol"]["replay_survival_threshold"],
            "replay_episode_count": len(rows),
            "successful_replay_count": replay_success_count,
            "definition": "final live replay episodes with at least one success",
            "passed": replay_survival >= float(config["protocol"]["replay_survival_threshold"]),
        },
        "action_contract": {
            "observed": action_stats,
            "passed": action_stats["finite"] and int(action_stats["out_of_bounds_count"]) == 0,
        },
        "formal_horizon_coverage": {
            "observed": formal_coverage,
            "threshold": config["protocol"]["formal_horizon_coverage_threshold"],
            "denominator": (
                "successful final live-replay episodes"
                if is_wam_protocol
                else "all replay episodes"
            ),
            "passed": formal_coverage is None
            or formal_coverage >= float(config["protocol"]["formal_horizon_coverage_threshold"]),
        },
        "post_success_trim": {
            "observed": all(
                row["trim_action_count"] is None
                or row["trim_action_count"] <= row["first_success_step"] + 5
                for row in rows
            ),
            "passed": True,
        },
        "reward_success_alignment": {
            "observed": None if reward_values == 0 else reward_matches / reward_values,
            "passed": reward_values > 0 and reward_matches == reward_values,
        },
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "replay_h5": replay.as_posix(),
        "replay_sha256": sha256_file(replay),
        "episodes": rows,
        "timing": timing,
        "actions": action_stats,
        "observations": {"h5_schema": observation_schema},
        "reward_termination": {
            "reward_values": reward_values,
            "reward_success_matches": reward_matches,
            "alignment_rate": None if reward_values == 0 else reward_matches / reward_values,
            "termination_semantics": {"ignore_terminations": True, "record_metrics": True},
        },
        "chunks": chunks,
        "gates": gates,
        "gate_pass": all(bool(gate["passed"]) for gate in gates.values()),
    }
    _write_json(root / "success_timing.json", timing)
    _write_json(root / "action_channel_stats.json", action_stats)
    _write_json(root / "reward_termination_audit.json", result["reward_termination"])
    _write_json(root / "chunk_slicing_audit.json", chunks)
    with (root / "episode_audit.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(root / "trajectory_audit.json", result)
    return result


def _h5_schema(group: h5py.Group, *, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def visitor(name: str, value: h5py.Dataset) -> None:
        if isinstance(value, h5py.Dataset):
            result[f"{prefix}/{name}"] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    group.visititems(visitor)
    return result


def _action_signature(group: h5py.Group) -> str:
    actions = np.asarray(group["actions"], dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()


def plug_horizon(
    mean_demo_length: float, *, multiplier: float = 2.0, round_to: int = 100, minimum: int = 200
) -> int:
    raw = float(mean_demo_length) * float(multiplier)
    rounded = int(round_to) * math.floor(raw / int(round_to) + 0.5)
    return max(int(minimum), int(rounded))


def freeze_plug_horizon(
    config: Mapping[str, Any], *, override_output: str | Path | None = None
) -> dict[str, Any]:
    task = "PlugCharger-v1"
    root = task_root(config, task, override_output)
    audit = _read_json(root / "trajectory_audit.json")
    rows = audit["episodes"]
    gates = _require_mapping(audit.get("gates"), "trajectory_audit.gates")
    survival_gate = _require_mapping(gates.get("replay_survival"), "gates.replay_survival")
    if not bool(survival_gate.get("passed")):
        observed = survival_gate.get("observed")
        threshold = survival_gate.get(
            "threshold", config["protocol"]["replay_survival_threshold"]
        )
        raise RuntimeError(
            "Plug horizon requires replay survival gate to pass: "
            f"{float(observed):.4f} < {float(threshold):.4f}"
        )
    successful_rows = [row for row in rows if bool(row["success_once"])]
    lengths = [float(row["num_action_steps"]) for row in successful_rows]
    if not lengths:
        raise RuntimeError("Plug horizon cannot be frozen without successful replay episodes")
    rule = task_config(config, task)["formal_horizon_rule"]
    horizon = plug_horizon(
        float(np.mean(lengths)),
        multiplier=float(rule["mean_length_multiplier"]),
        round_to=int(rule["round_to"]),
        minimum=int(rule["minimum"]),
    )
    covered = sum(
        row["first_success_step"] is not None and row["first_success_step"] <= horizon
        for row in successful_rows
    )
    coverage = covered / max(1, len(successful_rows))
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "mean_demo_length": float(np.mean(lengths)),
        "raw_horizon": float(np.mean(lengths) * float(rule["mean_length_multiplier"])),
        "formal_horizon": horizon,
        "rule": dict(rule),
        "replay_episode_count": len(rows),
        "successful_replay_count": len(successful_rows),
        "success_by_formal_horizon_count": covered,
        "success_by_formal_horizon_rate": coverage,
        "threshold": config["protocol"]["formal_horizon_coverage_threshold"],
        "gate_pass": coverage >= float(config["protocol"]["formal_horizon_coverage_threshold"]),
    }
    if not result["gate_pass"]:
        raise RuntimeError(f"Plug formal horizon coverage gate failed: {coverage:.4f} < 0.99")
    _write_json(root / "horizon_decision.json", result)
    return result


def _stratified_episode_order(rows: Sequence[Mapping[str, Any]], task: str) -> list[dict[str, Any]]:
    usable = [dict(row) for row in rows if row.get("first_success_step") is not None]
    usable.sort(key=lambda row: (int(row["first_success_step"]), int(row["episode_id"])))
    strata: dict[int, list[dict[str, Any]]] = {index: [] for index in range(10)}
    for rank, row in enumerate(usable):
        decile = min(9, (rank * 10) // max(1, len(usable)))
        row["first_success_decile"] = decile
        digest = hashlib.sha256(f"{task}:{row['episode_id']}".encode()).hexdigest()
        row["split_sort_key"] = digest
        strata[decile].append(row)
    for values in strata.values():
        values.sort(key=lambda row: row["split_sort_key"])
    result: list[dict[str, Any]] = []
    for index in itertools.count():
        added = False
        for decile in range(10):
            if index < len(strata[decile]):
                result.append(strata[decile][index])
                added = True
        if not added:
            break
    return result


def _episode_group(handle: h5py.File, row: Mapping[str, Any]) -> h5py.Group:
    return handle[f"traj_{int(row['episode_id'])}"]


def freeze_data(
    config: Mapping[str, Any], task: str, *, override_output: str | Path | None = None
) -> dict[str, Any]:
    root = task_root(config, task, override_output)
    audit = _read_json(root / "trajectory_audit.json")
    if not audit.get("gate_pass"):
        raise RuntimeError(f"Cannot freeze {task}: trajectory audit gates failed")
    audit_rows = [dict(row) for row in audit["episodes"]]
    predicate_contract = None
    if task == "PegInsertionSide-v1":
        predicate_path = root / "success_predicate" / "predicate_contract.json"
        predicate_contract = _read_json(predicate_path)
        if predicate_contract.get("frozen_env_id") == "RobotPracticePegInsertionSideStrict-v1":
            labels_path = root / "success_predicate" / "strict_labels.json"
            if not labels_path.is_file():
                raise FileNotFoundError(
                    f"Strict Peg labels are required before D_core freeze: {labels_path}"
                )
            strict_labels = _read_json(labels_path)
            for row in audit_rows:
                labels = strict_labels.get(str(row["episode_id"]))
                if labels is None:
                    raise ValueError(
                        f"Missing strict labels for replay episode {row['episode_id']}"
                    )
                first = first_success_step(labels)
                row["success_once"] = bool(any(labels))
                row["success_at_end"] = bool(labels[-1]) if labels else False
                row["first_success_step"] = first
                row["success_by_native_horizon"] = first is not None and first <= int(
                    task_config(config, task)["native_horizon"]
                )
                row["success_by_formal_horizon"] = first is not None and first <= int(
                    task_config(config, task)["formal_horizon"]
                )
                row["trim_action_count"] = (
                    None
                    if first is None
                    else min(
                        int(row["num_action_steps"]),
                        first + int(config["protocol"]["trim_success_tail_frames"]),
                    )
                )
    rows = _stratified_episode_order(audit_rows, task)
    split_sizes = config["splits"]["offline"]
    if len(rows) < int(split_sizes["test"]) + int(split_sizes["development"]):
        raise RuntimeError(f"Not enough successful episodes for offline split: {len(rows)}")
    test_rows = rows[: int(split_sizes["test"])]
    dev_rows = rows[
        int(split_sizes["test"]) : int(split_sizes["test"]) + int(split_sizes["development"])
    ]
    train_rows = rows[int(split_sizes["test"]) + int(split_sizes["development"]) :]
    source_replay = root / "final_live_rgb_full" / "replay.h5"
    source_json = _metadata_path(source_replay)
    dcore = root / "d_core"
    dcore.mkdir(parents=True, exist_ok=True)
    dcore_h5 = dcore / "trajectory.h5"
    dcore_json = dcore / "trajectory.json"
    if dcore_h5.exists() or dcore_json.exists():
        raise FileExistsError(f"D_core already exists; refusing overwrite: {dcore}")
    selected = test_rows + dev_rows + train_rows
    with h5py.File(source_replay, "r") as source, h5py.File(dcore_h5, "w") as destination:
        for row in selected:
            source_group = _episode_group(source, row)
            target_group = destination.create_group(f"traj_{int(row['episode_id'])}")
            trim_count = int(row["trim_action_count"] or row["num_action_steps"])

            def copy_limited(node: h5py.Group, parent: h5py.Group, limit: int) -> None:
                for child_name, child in node.items():
                    if isinstance(child, h5py.Group):
                        child_parent = parent.create_group(child_name)
                        copy_limited(child, child_parent, limit)
                    else:
                        parent.create_dataset(child_name, data=child[:limit])

            for key, value in source_group.items():
                if key != "actions" and key != "success":
                    if key in {"obs", "env_states"} and isinstance(value, h5py.Group):
                        # obs[t] and env_states[t] describe the reset state
                        # followed by each transition.  Keep one terminal
                        # frame when actions are trimmed.
                        copy_limited(value, target_group.create_group(key), trim_count + 1)
                    elif isinstance(value, h5py.Dataset) and value.ndim >= 1:
                        target_group.create_dataset(key, data=value[:trim_count])
                    else:
                        source_group.copy(value, target_group, name=key)
            target_group.create_dataset(
                "actions", data=np.asarray(source_group["actions"][:trim_count])
            )
            if "success" in source_group:
                target_group.create_dataset(
                    "success", data=np.asarray(source_group["success"][:trim_count])
                )
            target_group.attrs["source_episode_id"] = int(row["source_episode_id"])
    metadata = _read_json(source_json)
    records = metadata_records(metadata)
    ordered_records = []
    for row in selected:
        record = dict(records[int(row["episode_id"])])
        record["source_episode_id"] = int(row["source_episode_id"])
        record["first_success_decile"] = int(row["first_success_decile"])
        record["d_core_action_count"] = int(row["trim_action_count"] or row["num_action_steps"])
        record.pop("episode_seed", None)
        reset_kwargs = record.get("reset_kwargs")
        if isinstance(reset_kwargs, dict):
            reset_kwargs.pop("seed", None)
        ordered_records.append(record)
    metadata["episodes"] = ordered_records
    metadata["m1_d_core"] = {
        "task": task,
        "trim": "min(T, first_success+5)",
        "episode_count": len(selected),
    }
    dcore_json.write_text(
        json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "algorithm": "first_success_decile_then_sha256_task_source_episode_id",
        "counts": {"train": len(train_rows), "development": len(dev_rows), "test": len(test_rows)},
        "train": [int(row["source_episode_id"]) for row in train_rows],
        "development": [int(row["source_episode_id"]) for row in dev_rows],
        "test": [int(row["source_episode_id"]) for row in test_rows],
    }
    _write_json(dcore / "split_manifest.json", split_manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "d_core_h5": dcore_h5.as_posix(),
        "d_core_json": dcore_json.as_posix(),
        "split_manifest": split_manifest,
        "success_predicate": predicate_contract,
        "episode_count": len(selected),
        "gate_pass": len(test_rows) == 100
        and len(dev_rows) == 100
        and len(selected) >= int(config["protocol"]["d_core_min_episodes"]),
    }
    _write_json(root / "d_core_manifest.json", result)
    if not result["gate_pass"]:
        raise RuntimeError(f"D_core gate failed for {task}")
    return result


def _flatten_fingerprint(value: Any, *, path: str = "") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_flatten_fingerprint(child, path=f"{path}.{key}" if path else str(key)))
        return result
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        return result
    array = array.astype(np.float64, copy=False).reshape(-1)
    lowered = path.lower()
    if array.size >= 7 and ("pose" in lowered or lowered.endswith("orientation")):
        # Pose raw layout is [x,y,z,w,x,y,z]. Canonicalize only the quaternion.
        pose = array.copy()
        pose[3:7] = quaternion_canonicalize(pose[3:7])
        array = pose
    result[path] = array
    return result


def _fingerprint_from_observation(observation: Any) -> dict[str, Any]:
    components = _flatten_fingerprint(observation)
    return {
        key: value.tolist() for key, value in sorted(components.items()) if "rgb" not in key.lower()
    }


def _fingerprint_equal(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float) -> bool:
    if not left or not right or set(left) != set(right):
        return False
    return all(
        np.asarray(left[key], dtype=np.float64).shape
        == np.asarray(right[key], dtype=np.float64).shape
        and np.allclose(left[key], right[key], atol=tolerance, rtol=0.0)
        for key in left
    )


def _frozen_peg_env_id(root: Path) -> str:
    contract = root / "success_predicate" / "predicate_contract.json"
    if contract.is_file():
        payload = _read_json(contract)
        return str(payload.get("frozen_env_id", "PegInsertionSide-v1"))
    return "PegInsertionSide-v1"


def state_banks(
    config: Mapping[str, Any], task: str, *, override_output: str | Path | None = None
) -> dict[str, Any]:
    """Generate reset states in the configured backend and compare fingerprints."""

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    item = task_config(config, task)
    root = task_root(config, task, override_output)
    env_id = _frozen_peg_env_id(root) if task == "PegInsertionSide-v1" else task
    counts = config["splits"]["state_banks"]
    bank_root = root / "state_banks"
    if bank_root.exists() and (bank_root / "state_bank_manifest.json").is_file():
        return _read_json(bank_root / "state_bank_manifest.json")
    if bank_root.exists() and any(bank_root.iterdir()):
        raise FileExistsError(f"Incomplete state bank exists; refusing overwrite: {bank_root}")
    bank_root.mkdir(parents=True, exist_ok=True)
    protocol = _require_mapping(config["protocol"], "protocol")
    rollout_contract = _configured_rollout_contract(config)
    backend = rollout_contract["actual_rollout_backend"]
    state_bank_envs = int(protocol.get("state_bank_num_envs", 1)) if backend == "physx_cuda" else 1
    state_obs_mode = "state_dict" if backend == "physx_cuda" else "state"
    env = gym.make(
        env_id,
        obs_mode=state_obs_mode,
        control_mode=item["target_control_mode"],
        sim_backend=backend,
        reward_mode="sparse",
        reconfiguration_freq=1,
        num_envs=state_bank_envs,
    )
    try:
        sets: dict[str, list[dict[str, Any]]] = {}
        for set_name, spec in counts.items():
            count = int(spec["count"])
            start = int(spec["seed_start"])
            rows: list[dict[str, Any]] = []
            with (bank_root / f"{set_name}.jsonl").open("w", encoding="utf-8") as stream:
                for batch_start in range(0, count, state_bank_envs):
                    batch_seeds = [start + index for index in range(batch_start, min(count, batch_start + state_bank_envs))]
                    if backend == "physx_cuda":
                        import torch

                        observation, _ = env.reset(
                            seed=torch.tensor(batch_seeds, device=env.unwrapped.device)
                        )
                    else:
                        observation, _ = env.reset(seed=batch_seeds[0])
                    state = env.unwrapped.get_state_dict()
                    from mani_skill.trajectory import utils as trajectory_utils

                    if backend == "physx_cuda":
                        state_rows = trajectory_utils.dict_to_list_of_dicts(state)
                        observation_rows = trajectory_utils.dict_to_list_of_dicts(observation)
                    else:
                        state_rows = [state]
                        observation_rows = [observation]
                    for local_index, _seed in enumerate(batch_seeds):
                        state_row = state_rows[local_index]
                        row = {
                            "index": batch_start + local_index,
                            "state": _jsonable(state_row),
                            "start_state_fingerprint": state_fingerprint(
                                state_row, tolerance=float(protocol["state_match_tolerance"])
                            ),
                            "observation_fingerprint": _fingerprint_from_observation(
                                observation_rows[local_index]
                            ),
                        }
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
            sets[set_name] = rows
    finally:
        env.close()
    tolerance = float(config["protocol"]["state_match_tolerance"])
    pairs: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(sets, 2):
        duplicates: list[dict[str, Any]] = []
        for left in sets[left_name]:
            for right in sets[right_name]:
                if left["start_state_fingerprint"] == right["start_state_fingerprint"]:
                    duplicates.append(
                        {"left_index": left["index"], "right_index": right["index"]}
                    )
        pairs.append(
            {
                "left": left_name,
                "right": right_name,
                "num_duplicates": len(duplicates),
                "duplicates": duplicates[:20],
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "env_id": env_id,
        "backend": backend,
        **rollout_contract,
        "num_envs": state_bank_envs,
        "reconfiguration_freq": 1,
        "sets": {name: len(rows) for name, rows in sets.items()},
        "state_match_tolerance": tolerance,
        "identity": {
            "key": "start_state_fingerprint from reset state",
            "seed_is_not_identity": True,
        },
        "cross_split_pairs": pairs,
        "within_split_unique": {
            name: len({row["start_state_fingerprint"] for row in rows}) == len(rows)
            for name, rows in sets.items()
        },
        "nonempty_fingerprint_rate": sum(
            bool(row["start_state_fingerprint"]) for rows in sets.values() for row in rows
        )
        / max(1, sum(len(rows) for rows in sets.values())),
        "gate_pass": all(pair["num_duplicates"] == 0 for pair in pairs)
        and all(result for result in {
            name: len({row["start_state_fingerprint"] for row in rows}) == len(rows)
            for name, rows in sets.items()
        }.values())
        and all(bool(row["start_state_fingerprint"]) for rows in sets.values() for row in rows),
    }
    _write_json(bank_root / "state_bank_manifest.json", result)
    _write_json(root / "split_leakage_report.json", result)
    if not result["gate_pass"]:
        raise RuntimeError(f"State-bank leakage detected for {task}")
    return result


def finalize(
    config: Mapping[str, Any], *, override_output: str | Path | None = None
) -> dict[str, Any]:
    root = output_root(config, override_output)
    task_results: dict[str, Any] = {}
    all_passed = True
    for task in configured_tasks(config):
        task_root_path = root / task
        required = {
            "source": task_root_path / "source_audit.json",
            "environment": task_root_path / "environment_contract.json",
            "trajectory": task_root_path / "trajectory_audit.json",
            "d_core": task_root_path / "d_core_manifest.json",
            "state_banks": task_root_path / "state_banks" / "state_bank_manifest.json",
        }
        if config["protocol"]["backend"] == "physx_cuda":
            required["data_generation"] = (
                task_root_path / "final_live_rgb_full" / "replay_manifest.json"
            )
        missing = [name for name, path in required.items() if not path.is_file()]
        extra_paths: dict[str, Path] = {}
        if task == "PlugCharger-v1":
            extra_paths["horizon"] = task_root_path / "horizon_decision.json"
        if task == "PegInsertionSide-v1":
            extra_paths["predicate"] = (
                task_root_path / "success_predicate" / "predicate_contract.json"
            )
        missing.extend(name for name, path in extra_paths.items() if not path.is_file())
        records: dict[str, Any] = {"missing": missing}
        for name, path in required.items():
            if path.is_file():
                records[name] = _read_json(path)
        for name, path in extra_paths.items():
            if path.is_file():
                records[name] = _read_json(path)
        gate_names = list(required) + list(extra_paths)
        records["gate_pass"] = not missing and all(
            bool(records[name].get("gate_pass", False)) for name in gate_names if name in records
        )
        task_results[task] = records
        all_passed = all_passed and bool(records["gate_pass"])
    comparison = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": config["protocol_id"],
        "tasks": {
            task: {
                "source_episodes": data.get("source", {}).get("summary", {}).get("num_episodes"),
                "replay_survival": data.get("trajectory", {})
                .get("gates", {})
                .get("replay_survival"),
                "d_core_episodes": data.get("d_core", {}).get("episode_count"),
                "canonical_env_id": data.get("predicate", {}).get("frozen_env_id", task),
                "gate_pass": data.get("gate_pass", False),
            }
            for task, data in task_results.items()
        },
        "gate_pass": all_passed,
    }
    if not all_passed:
        _write_json(root / "comparison.json", comparison)
        raise RuntimeError(
            "M1 finalize gates are not all satisfied; frozen_protocol.json was not written"
        )
    frozen = copy.deepcopy(config)
    frozen.pop("_config_path", None)
    frozen["frozen_protocol"] = {
        "created_from": "m1_protocol_audit",
        "tasks": {},
        "backend": config["protocol"]["backend"],
        "control_frequency_hz": 20,
        "formal_evaluation": {
            "ignore_terminations": True,
            "record_metrics": True,
            "partial_reset": False,
        },
    }
    for task in configured_tasks(config):
        frozen["frozen_protocol"]["tasks"][task] = {
            "canonical_env_id": task_results[task].get("predicate", {}).get("frozen_env_id", task),
            "formal_horizon": task_config(config, task).get("formal_horizon")
            if task != "PlugCharger-v1"
            else _read_json(root / task / "horizon_decision.json")["formal_horizon"],
        }
    _write_json(root / "comparison.json", comparison)
    _write_json(root / "frozen_protocol.json", frozen)
    return {"comparison": comparison, "frozen_protocol": frozen}


def validate_frozen(
    protocol_path: str | Path, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    path = _rooted(protocol_path)
    payload = _read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Frozen protocol schema mismatch")
    frozen = _require_mapping(payload.get("frozen_protocol"), "frozen_protocol")
    if frozen.get("backend") not in {"physx_cpu", "physx_cuda"}:
        raise ValueError("Frozen protocol backend must be physx_cpu or physx_cuda")
    if config is not None and frozen.get("backend") != config["protocol"]["backend"]:
        raise ValueError("Frozen protocol backend does not match the configured protocol")
    frozen_tasks = _require_mapping(frozen.get("tasks"), "frozen_protocol.tasks")
    expected_tasks = configured_tasks(config) if config is not None else tuple(frozen_tasks)
    if tuple(frozen_tasks) != expected_tasks:
        raise ValueError(
            f"Frozen protocol tasks {tuple(frozen_tasks)} do not match configured tasks "
            f"{expected_tasks}"
        )
    for task in expected_tasks:
        item = _require_mapping(
            frozen_tasks.get(task), task
        )
        if not isinstance(item.get("formal_horizon"), int) or item["formal_horizon"] <= 0:
            raise ValueError(f"Frozen protocol has invalid formal horizon for {task}")
    return {"protocol": path.as_posix(), "gate_pass": True, "tasks": list(expected_tasks)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    frozen = subparsers.add_parser("validate-frozen")
    frozen.add_argument("--protocol", type=Path, required=True)
    frozen.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    for name in ("download", "finalize"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--output-root", type=Path, default=None)
    for name in (
        "source-audit",
        "environment-audit",
        "trajectory-audit",
        "freeze-data",
        "state-banks",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--task", type=str, required=True)
        command.add_argument("--output-root", type=Path, default=None)
    replay = subparsers.add_parser("replay-smoke")
    replay.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    replay.add_argument("--task", type=str, required=True)
    replay.add_argument("--num-workers", type=int, default=None)
    replay.add_argument("--num-envs", type=int, default=None)
    replay.add_argument("--output-root", type=Path, default=None)
    replay = subparsers.add_parser("replay-full")
    replay.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    replay.add_argument("--task", type=str, required=True)
    replay.add_argument("--num-workers", type=int, default=None)
    replay.add_argument("--num-envs", type=int, default=None)
    replay.add_argument("--output-root", type=Path, default=None)
    for name in ("gpu-source-smoke", "gpu-source-full"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--task", type=str, required=True)
        command.add_argument("--output-root", type=Path, default=None)
    for name in (
        "gpu-baseline-smoke",
        "gpu-baseline-full",
        "gpu-repair-smoke",
        "gpu-repair-full",
        "gpu-merge-smoke",
        "gpu-merge-full",
        "gpu-final-smoke",
        "gpu-final-full",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--task", type=str, required=True)
        command.add_argument("--num-envs", type=int, default=None)
        command.add_argument("--output-root", type=Path, default=None)
    peg = subparsers.add_parser("peg-predicate-audit")
    peg.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    peg.add_argument("--output-root", type=Path, default=None)
    horizon = subparsers.add_parser("freeze-horizon")
    horizon.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    horizon.add_argument("--task", type=str, required=True)
    horizon.add_argument("--output-root", type=Path, default=None)
    return parser


def _run_command(args: argparse.Namespace) -> Any:
    if args.command == "validate-config":
        config = load_config(args.config)
        result = validate_config(config)
    elif args.command == "validate-frozen":
        config = load_config(args.config)
        validate_config(config)
        result = validate_frozen(args.protocol, config)
    else:
        config = load_config(args.config)
        validate_config(config)
        override = args.output_root
        task = getattr(args, "task", None)
        selected = selected_tasks(config, task) if task is not None else ()
        if args.command == "download":
            result = download_archives(config, override_output=override)
        elif args.command == "finalize":
            result = finalize(config, override_output=override)
        elif args.command == "peg-predicate-audit":
            if "PegInsertionSide-v1" not in configured_tasks(config):
                raise ValueError("peg-predicate-audit requires PegInsertionSide-v1 in the config")
            from .m1_peg_predicate_audit import audit_peg_predicate

            result = audit_peg_predicate(config, override_output=override)
        elif args.command == "freeze-horizon":
            if task != "PlugCharger-v1":
                raise ValueError("freeze-horizon only supports PlugCharger-v1")
            result = freeze_plug_horizon(config, override_output=override)
        elif args.command == "replay-smoke":
            if str(config.get("protocol_id")) == "m1_wam_protocol_audit":
                result = [
                    baseline_gpu_replay(
                        config,
                        current,
                        full=False,
                        num_workers=args.num_workers,
                        num_envs=args.num_envs,
                        override_output=override,
                    )
                    for current in selected
                ]
            else:
                result = [
                    replay_task(
                        config,
                        current,
                        full=False,
                        num_workers=args.num_workers,
                        num_envs=args.num_envs,
                        override_output=override,
                    )
                    for current in selected
                ]
        elif args.command == "replay-full":
            if task == "all":
                raise ValueError("replay-full requires one configured task")
            if str(config.get("protocol_id")) == "m1_wam_protocol_audit":
                result = baseline_gpu_replay(
                    config,
                    task,
                    full=True,
                    num_workers=args.num_workers,
                    num_envs=args.num_envs,
                    override_output=override,
                )
            else:
                result = replay_task(
                    config,
                    task,
                    full=True,
                    num_workers=args.num_workers,
                    num_envs=args.num_envs,
                    override_output=override,
                )
        elif args.command in {
            "gpu-baseline-smoke",
            "gpu-baseline-full",
            "gpu-repair-smoke",
            "gpu-repair-full",
            "gpu-merge-smoke",
            "gpu-merge-full",
            "gpu-final-smoke",
            "gpu-final-full",
        }:
            if task == "all":
                raise ValueError(f"{args.command} requires one configured task")
            full = args.command.endswith("-full")
            if args.command.startswith("gpu-baseline"):
                result = baseline_gpu_replay(
                    config,
                    task,
                    full=full,
                    num_envs=args.num_envs,
                    override_output=override,
                )
            elif args.command.startswith("gpu-repair"):
                result = repair_gpu_ee_failures(
                    config,
                    task,
                    full=full,
                    num_envs=args.num_envs,
                    override_output=override,
                )
            elif args.command.startswith("gpu-merge"):
                result = merge_gpu_ee_data(
                    config,
                    task,
                    full=full,
                    override_output=override,
                )
            else:
                result = final_live_rgb_replay(
                    config,
                    task,
                    full=full,
                    num_envs=args.num_envs,
                    override_output=override,
                )
        elif args.command == "gpu-source-smoke":
            result = [
                prepare_gpu_native_source(
                    config,
                    current,
                    full=False,
                    override_output=override,
                )
                for current in selected
            ]
        elif args.command == "gpu-source-full":
            if task == "all":
                raise ValueError("gpu-source-full requires one configured task")
            result = prepare_gpu_native_source(
                config,
                task,
                full=True,
                override_output=override,
            )
        elif args.command == "source-audit":
            result = [
                audit_source(config, current, override_output=override)
                for current in selected
            ]
        elif args.command == "environment-audit":
            result = [
                environment_audit(config, current, override_output=override)
                for current in selected
            ]
        elif args.command == "trajectory-audit":
            result = [
                audit_trajectory(config, current, override_output=override)
                for current in selected
            ]
        elif args.command == "freeze-data":
            result = [
                freeze_data(config, current, override_output=override)
                for current in selected
            ]
        elif args.command == "state-banks":
            result = [
                state_banks(config, current, override_output=override)
                for current in selected
            ]
        else:
            raise AssertionError(f"Unhandled command {args.command}")
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    _run_command(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
