"""Audit ManiSkill PickCube demonstrations without replaying actions.

The raw-data part only reads HDF5 files.  The optional initial-state part
performs reset-only calls to ManiSkill on the remote evaluation machine and
compares the resulting states with the first ``env_states`` frame in the
training demonstrations.  It never steps an environment or trains a policy.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

from .split import (
    DEFAULT_SIM_BACKEND,
    DEFAULT_STATE_MATCH_TOLERANCE,
    TASK_ID,
    load_seed_sets,
    reset_batches_for_set,
)
from .rollout_contract import require_rollout_backend

AUDIT_SCHEMA_VERSION = 1
PANDA_QPOS_SLICE = slice(13, 22)
FINGERPRINT_COMPONENTS = (
    "robot_qpos",
    "cube_position",
    "cube_orientation",
    "goal_position",
)
ACTION_NAMES = {
    4: ("translation_x", "translation_y", "translation_z", "gripper"),
    7: (
        "translation_x",
        "translation_y",
        "translation_z",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "gripper",
    ),
    8: tuple(f"action_{index}" for index in range(8)),
}


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return value


def _metadata_path(h5_path: Path) -> Path:
    metadata_path = h5_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata sidecar does not exist: {metadata_path}")
    return metadata_path


def _episode_id(name: str) -> int:
    if not name.startswith("traj_"):
        raise ValueError(f"Unexpected trajectory group {name!r}")
    return int(name.removeprefix("traj_"))


def _metadata_records(metadata: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Trajectory metadata episodes must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for record in episodes:
        if not isinstance(record, Mapping):
            continue
        result[int(record["episode_id"])] = record
    return result


def _episode_seed(record: Mapping[str, Any] | None) -> int | None:
    if record is None:
        return None
    value = record.get("episode_seed")
    if value is None:
        reset_kwargs = record.get("reset_kwargs")
        if isinstance(reset_kwargs, Mapping):
            value = reset_kwargs.get("seed")
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else None
    return None if value is None else int(value)


def _series(episode: h5py.Group, path: Sequence[str]) -> np.ndarray | None:
    node: Any = episode
    for component in path:
        if not isinstance(node, h5py.Group) or component not in node:
            return None
        node = node[component]
    if not isinstance(node, h5py.Dataset):
        return None
    return np.asarray(node)


def _first_success(success: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(success, dtype=bool).reshape(-1))
    return None if indices.size == 0 else int(indices[0])


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        return {key: None for key in ("mean", "std", "p10", "p50", "p90", "p95", "p99")}
    return {
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "p10": float(np.percentile(flattened, 10)),
        "p50": float(np.percentile(flattened, 50)),
        "p90": float(np.percentile(flattened, 90)),
        "p95": float(np.percentile(flattened, 95)),
        "p99": float(np.percentile(flattened, 99)),
    }


def _action_measures(actions: np.ndarray) -> dict[str, np.ndarray | None]:
    if actions.ndim != 2:
        raise ValueError(f"Actions must have shape [T,A], got {actions.shape}")
    if actions.shape[1] not in ACTION_NAMES:
        raise ValueError(f"Unsupported PickCube action dimension: {actions.shape[1]}")
    absolute = np.abs(actions.astype(np.float64, copy=False))
    translation_norm = np.linalg.norm(actions[:, :3], axis=1)
    rotation_norm = np.linalg.norm(actions[:, 3:6], axis=1) if actions.shape[1] >= 7 else None
    gripper = absolute[:, -1]
    return {
        "translation_norm": translation_norm,
        "rotation_norm": rotation_norm,
        "gripper_abs": gripper,
    }


def _action_summary(actions: np.ndarray) -> dict[str, Any]:
    names = ACTION_NAMES[int(actions.shape[1])]
    absolute = np.abs(actions.astype(np.float64, copy=False))
    return {
        "action_dim": int(actions.shape[1]),
        "channel_order": list(names),
        "channels": {name: _quantiles(actions[:, index]) for index, name in enumerate(names)},
        "translation_norm": _quantiles(np.linalg.norm(actions[:, :3], axis=1)),
        "rotation_norm": (
            None if actions.shape[1] < 7 else _quantiles(np.linalg.norm(actions[:, 3:6], axis=1))
        ),
        "gripper_abs": _quantiles(absolute[:, -1]),
        "num_action_steps": int(actions.shape[0]),
    }


def _state_fingerprint_from_env_state(
    episode: h5py.Group, *, index: int = 0
) -> dict[str, list[float]]:
    cube = _series(episode, ("env_states", "actors", "cube"))
    goal = _series(episode, ("env_states", "actors", "goal_site"))
    panda = _series(episode, ("env_states", "articulations", "panda"))
    if cube is None or goal is None or panda is None:
        raise KeyError(
            "PickCube env_states must contain actors/cube, actors/goal_site, "
            "and articulations/panda"
        )
    cube_state = np.asarray(cube[index], dtype=np.float64).reshape(-1)
    goal_state = np.asarray(goal[index], dtype=np.float64).reshape(-1)
    panda_state = np.asarray(panda[index], dtype=np.float64).reshape(-1)
    if cube_state.size < 7 or goal_state.size < 3 or panda_state.size < 22:
        raise ValueError("PickCube env_states have unexpected dimensions")
    values = {
        "robot_qpos": panda_state[PANDA_QPOS_SLICE],
        "cube_position": cube_state[:3],
        "cube_orientation": cube_state[3:7],
        "goal_position": goal_state[:3],
    }
    return {key: [float(value) for value in array] for key, array in values.items()}


def _state_fingerprint_from_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Extract reset-only state fields from ManiSkill ``obs_mode=state_dict``."""

    extra = observation["extra"]
    agent = observation["agent"]

    def to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float64)

    qpos = to_numpy(agent["qpos"])
    cube_pose = to_numpy(extra["obj_pose"])
    goal = to_numpy(extra["goal_pos"])
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if cube_pose.ndim == 1:
        cube_pose = cube_pose[None, :]
    if goal.ndim == 1:
        goal = goal[None, :]
    if qpos.shape[1:] != (9,) or cube_pose.shape[1:] != (7,) or goal.shape[1:] != (3,):
        raise ValueError(
            "ManiSkill state observation must contain qpos=(9,), obj_pose=(7,), goal_pos=(3,)"
        )
    return {
        "robot_qpos": qpos,
        "cube_position": cube_pose[:, :3],
        "cube_orientation": cube_pose[:, 3:7],
        "goal_position": goal,
    }


def _as_finite_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("State fingerprint contains non-finite values")
    return array


def _fingerprint_row(
    *,
    episode_id: int,
    seed: int | None,
    values: Mapping[str, Any],
    source: str,
    state_set: str | None = None,
    batch_index: int | None = None,
    row_index: int | None = None,
) -> dict[str, Any]:
    arrays = {key: _as_finite_array(values[key]) for key in FINGERPRINT_COMPONENTS}
    fingerprint = np.concatenate([arrays[key] for key in FINGERPRINT_COMPONENTS])
    return {
        "episode_id": int(episode_id),
        "seed": None if seed is None else int(seed),
        "source": source,
        "state_set": state_set,
        "batch_index": batch_index,
        "row_index": row_index,
        **{key: arrays[key].tolist() for key in FINGERPRINT_COMPONENTS},
        "fingerprint": fingerprint.tolist(),
    }


def collect_training_fingerprints(
    source_h5: Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read first-frame fingerprints from a source trajectory HDF5."""

    source_h5 = Path(source_h5)
    metadata = _read_json(_metadata_path(source_h5)) if metadata is None else metadata
    records = _metadata_records(metadata)
    rows: list[dict[str, Any]] = []
    with h5py.File(source_h5, "r") as handle:
        for name in sorted(handle, key=_episode_id):
            episode_id = _episode_id(name)
            rows.append(
                _fingerprint_row(
                    episode_id=episode_id,
                    seed=_episode_seed(records.get(episode_id)),
                    values=_state_fingerprint_from_env_state(handle[name]),
                    source="training_demo",
                )
            )
    if not rows:
        raise ValueError(f"No trajectory groups found in {source_h5}")
    return rows


def compare_fingerprints(
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = DEFAULT_STATE_MATCH_TOLERANCE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare reset states with training states and report exact duplicates."""

    if not training_rows or not evaluation_rows:
        raise ValueError("Both training and evaluation fingerprint sets must be non-empty")
    if tolerance <= 0:
        raise ValueError("Fingerprint tolerance must be positive")
    train_arrays = {
        key: np.stack([_as_finite_array(row[key]) for row in training_rows])
        for key in FINGERPRINT_COMPONENTS
    }
    details: list[dict[str, Any]] = []
    for eval_row in evaluation_rows:
        component_distances = {
            key: np.linalg.norm(train_arrays[key] - _as_finite_array(eval_row[key]), axis=1)
            for key in FINGERPRINT_COMPONENTS
        }
        exact_mask = np.logical_and.reduce(
            [values <= tolerance for values in component_distances.values()]
        )
        flat_train = np.concatenate([train_arrays[key] for key in FINGERPRINT_COMPONENTS], axis=1)
        flat_eval = np.concatenate(
            [_as_finite_array(eval_row[key])[None, :] for key in FINGERPRINT_COMPONENTS], axis=1
        )
        flat_distances = np.linalg.norm(flat_train - flat_eval, axis=1)
        nearest_index = int(np.argmin(flat_distances))
        details.append(
            {
                "evaluation_episode_id": int(eval_row["episode_id"]),
                "state_set": eval_row.get("state_set"),
                "batch_index": eval_row.get("batch_index"),
                "row_index": eval_row.get("row_index"),
                "evaluation_seed_label": eval_row.get("seed"),
                "exact_duplicate": bool(exact_mask.any()),
                "exact_duplicate_training_episode_ids": [
                    int(training_rows[index]["episode_id"]) for index in np.flatnonzero(exact_mask)
                ],
                "nearest_training_episode_id": int(training_rows[nearest_index]["episode_id"]),
                "nearest_distance": float(flat_distances[nearest_index]),
                "nearest_component_distances": {
                    key: float(values[nearest_index]) for key, values in component_distances.items()
                },
            }
        )

    def summary(values: Sequence[float]) -> dict[str, float | None]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {key: None for key in ("p10", "p50", "p90", "p95", "max")}
        return {
            "p10": float(np.percentile(array, 10)),
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "method": "initial_state_fingerprint",
        "tolerance": tolerance,
        "num_training_states": len(training_rows),
        "num_evaluation_states": len(evaluation_rows),
        "num_exact_duplicates": sum(bool(row["exact_duplicate"]) for row in details),
        "exact_duplicate_evaluation_episode_ids": [
            row["evaluation_episode_id"] for row in details if row["exact_duplicate"]
        ],
        "nearest_distance": summary([row["nearest_distance"] for row in details]),
        "nearest_component_distances": {
            key: summary([row["nearest_component_distances"][key] for row in details])
            for key in FINGERPRINT_COMPONENTS
        },
        "status": "no_exact_duplicate"
        if not any(row["exact_duplicate"] for row in details)
        else "duplicate_present",
    }
    return details, report


def _write_fingerprints(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_backend(name: str) -> str:
    if name == "physx_cuda":
        return "gpu"
    if name == "physx_cpu":
        return "cpu"
    return name


def collect_reset_fingerprints(
    *,
    task: str,
    control_mode: str,
    split: Mapping[str, Any],
    state_set: str,
    sim_backend: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Reset one frozen batch and return its initial state fingerprints."""

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    batches = reset_batches_for_set(split, state_set, limit=limit)
    rows: list[dict[str, Any]] = []
    episode_offset = 0
    for batch_index, seed_batch in enumerate(batches):
        environment = gym.make(
            task,
            num_envs=len(seed_batch),
            # ``state`` is a flattened [B,42] Tensor in ManiSkill 3.0.1;
            # state_dict preserves agent/extra fields needed for the fingerprint.
            obs_mode="state_dict",
            control_mode=control_mode,
            sim_backend=_runtime_backend(sim_backend),
        )
        try:
            observation, _ = environment.reset(seed=[int(seed) for seed in seed_batch])
            fields = _state_fingerprint_from_observation(observation)
            for row_index, seed in enumerate(seed_batch):
                values = {key: fields[key][row_index] for key in FINGERPRINT_COMPONENTS}
                rows.append(
                    _fingerprint_row(
                        episode_id=episode_offset + row_index,
                        seed=int(seed),
                        values=values,
                        source="evaluation_reset",
                        state_set=state_set,
                        batch_index=batch_index,
                        row_index=row_index,
                    )
                )
        finally:
            environment.close()
        episode_offset += len(seed_batch)
    return rows


def audit_raw_data(
    source_h5: Path, *, task: str = TASK_ID
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the per-episode audit and the success-tail alignment report."""

    metadata = _read_json(_metadata_path(source_h5))
    records = _metadata_records(metadata)
    episodes: list[dict[str, Any]] = []
    all_actions: list[np.ndarray] = []
    pre_actions: list[np.ndarray] = []
    post_actions: list[np.ndarray] = []
    aligned: dict[int, list[dict[str, float]]] = {}
    total_tail_frames = 0
    total_frames = 0
    persistent_count = 0

    with h5py.File(source_h5, "r") as handle:
        for name in sorted(handle, key=_episode_id):
            episode_id = _episode_id(name)
            episode = handle[name]
            actions = np.asarray(episode["actions"], dtype=np.float64)
            if actions.ndim != 2 or not np.isfinite(actions).all():
                raise ValueError(f"Invalid actions in {name}: {actions.shape}")
            success_values = _series(episode, ("success",))
            if success_values is None:
                success_values = np.asarray(
                    [records.get(episode_id, {}).get("success", False)] * len(actions)
                )
            success = np.asarray(success_values, dtype=bool).reshape(-1)
            if success.size != actions.shape[0]:
                raise ValueError(f"Success/action lengths disagree in {name}")
            first = _first_success(success)
            last_indices = np.flatnonzero(success)
            last = None if last_indices.size == 0 else int(last_indices[-1])
            persists = bool(first is not None and np.all(success[first:]))
            if persists:
                persistent_count += 1
            total_frames += len(actions)
            tail_count = 0 if first is None else len(actions) - first
            total_tail_frames += tail_count
            measures = _action_measures(actions)
            all_actions.append(actions)
            if first is not None:
                pre_actions.append(actions[:first])
                post_actions.append(actions[first:])
                cube = _series(episode, ("env_states", "actors", "cube"))
                cube_position = None if cube is None else np.asarray(cube)[:, :3]
                tcp = _series(episode, ("obs", "extra", "tcp_pose"))
                tcp_position = None if tcp is None else np.asarray(tcp)[:, :3]
                for tau in range(len(actions) - first):
                    index = first + tau
                    row = {
                        "translation_norm": float(measures["translation_norm"][index]),
                        "rotation_norm": (
                            None
                            if measures["rotation_norm"] is None
                            else float(measures["rotation_norm"][index])
                        ),
                        "gripper_abs": float(measures["gripper_abs"][index]),
                        "cube_displacement": (
                            None
                            if cube_position is None
                            else float(
                                np.linalg.norm(cube_position[index, :3] - cube_position[first, :3])
                            )
                        ),
                        "tcp_displacement": (
                            None
                            if tcp_position is None
                            else float(
                                np.linalg.norm(tcp_position[index, :3] - tcp_position[first, :3])
                            )
                        ),
                        "success": float(success[index]),
                    }
                    aligned.setdefault(tau, []).append(row)

            episodes.append(
                {
                    "episode_id": episode_id,
                    "seed": _episode_seed(records.get(episode_id)),
                    "num_action_steps": len(actions),
                    "first_success_step": None if first is None else first + 1,
                    "last_success_step": None if last is None else last + 1,
                    "success_frame_count": int(np.sum(success)),
                    "success_tail_frame_count": int(tail_count),
                    "success_persists_to_end": persists,
                    "success_at_end": bool(success[-1]) if len(success) else False,
                    "action_summary": _action_summary(actions),
                    "pre_success_action_summary": (
                        None if first is None else _action_summary(actions[:first])
                    ),
                    "post_success_action_summary": (
                        None if first is None else _action_summary(actions[first:])
                    ),
                }
            )

    if not all_actions:
        raise ValueError(f"No trajectory groups found in {source_h5}")
    actions = np.concatenate(all_actions, axis=0)
    non_empty_pre = [value for value in pre_actions if value.size]
    non_empty_post = [value for value in post_actions if value.size]
    pre = (
        np.concatenate(non_empty_pre, axis=0)
        if non_empty_pre
        else np.empty((0, actions.shape[1]), dtype=np.float64)
    )
    post = (
        np.concatenate(non_empty_post, axis=0)
        if non_empty_post
        else np.empty((0, actions.shape[1]), dtype=np.float64)
    )

    def curve_summary(values: Sequence[Mapping[str, float]]) -> dict[str, float | None]:
        result: dict[str, float | None] = {"count": float(len(values))}
        for key in (
            "translation_norm",
            "rotation_norm",
            "gripper_abs",
            "cube_displacement",
            "tcp_displacement",
            "success",
        ):
            numeric = [float(value[key]) for value in values if value.get(key) is not None]
            result[key] = None if not numeric else float(np.mean(numeric))
        return result

    curve = [{"tau": tau, **curve_summary(aligned[tau])} for tau in sorted(aligned)]
    summary = {
        "num_episodes": len(episodes),
        "num_success_episodes": sum(row["first_success_step"] is not None for row in episodes),
        "num_episodes_success_persists_to_end": persistent_count,
        "total_frames": total_frames,
        "success_tail_frames": total_tail_frames,
        "success_tail_frame_share": total_tail_frames / total_frames if total_frames else None,
        "first_success_step": _quantiles(
            np.asarray(
                [
                    row["first_success_step"]
                    for row in episodes
                    if row["first_success_step"] is not None
                ],
                dtype=np.float64,
            )
        ),
        "success_tail_frame_count": _quantiles(
            np.asarray([row["success_tail_frame_count"] for row in episodes], dtype=np.float64)
        ),
        "action_summary": _action_summary(actions),
        "pre_success_action_summary": _action_summary(pre) if pre.size else None,
        "post_success_action_summary": _action_summary(post) if post.size else None,
    }
    raw = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "task": task,
        "source_h5": _repo_relative(source_h5),
        "metadata": _repo_relative(_metadata_path(source_h5)),
        "summary": summary,
        "episodes": episodes,
    }
    tail = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": raw["created_at"],
        "task": task,
        "source_h5": _repo_relative(source_h5),
        "summary": summary,
        "aligned_by_tau": curve,
        "displacement_note": (
            "TCP displacement is null when the source HDF5 has no recorded obs.extra.tcp_pose; "
            "the available env_states contain robot qpos but not TCP pose."
        ),
    }
    return raw, tail


def run_audit(
    *,
    source_h5: Path,
    output_dir: Path,
    split_file: Path | None = None,
    state_sets: Sequence[str] = (),
    task: str = TASK_ID,
    control_mode: str = "pd_ee_delta_pose",
    sim_backend: str | None = None,
    tolerance: float | None = None,
    include_raw: bool = True,
) -> dict[str, Any]:
    source_h5 = source_h5.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    if include_raw:
        raw, tail = audit_raw_data(source_h5, task=task)
        raw_path = output_dir / "raw_data_audit.json"
        tail_path = output_dir / "success_tail_stats.json"
        write_json_atomic(raw_path, raw)
        write_json_atomic(tail_path, tail)
        result.update(
            {
                "raw_data_audit": _repo_relative(raw_path),
                "success_tail_stats": _repo_relative(tail_path),
            }
        )
    if split_file is None or not state_sets:
        return result

    split = load_seed_sets(split_file, task=task)
    provenance = split.get("data_provenance")
    if provenance is None:
        raise ValueError(
            "split file must declare data_provenance for reset fingerprint audits; "
            "official requires CPU and m1w requires GPU"
        )
    requested_backend = sim_backend or str(split.get("sim_backend", DEFAULT_SIM_BACKEND))
    rollout_contract = require_rollout_backend(provenance, requested_backend)
    backend = rollout_contract["actual_rollout_backend"]
    match_tolerance = (
        float(tolerance)
        if tolerance is not None
        else float(split.get("state_match_tolerance", DEFAULT_STATE_MATCH_TOLERANCE))
    )
    training_rows = collect_training_fingerprints(source_h5)
    evaluation_rows: list[dict[str, Any]] = []
    for state_set in state_sets:
        evaluation_rows.extend(
            collect_reset_fingerprints(
                task=task,
                control_mode=control_mode,
                split=split,
                state_set=state_set,
                sim_backend=backend,
            )
        )
    details, leakage = compare_fingerprints(
        training_rows,
        evaluation_rows,
        tolerance=match_tolerance,
    )
    all_rows = training_rows + evaluation_rows
    fingerprint_path = output_dir / "initial_state_fingerprints.parquet"
    _write_fingerprints(fingerprint_path, all_rows)
    leakage.update(
        {
            "created_at": iso_timestamp(),
            "task": task,
            "training_h5": _repo_relative(source_h5),
            "split_file": _repo_relative(split_file),
            "sim_backend": backend,
            **rollout_contract,
            "reset_control_mode": control_mode,
            "evaluation_state_sets": list(state_sets),
            "fingerprints_parquet": _repo_relative(fingerprint_path),
            "details": details,
        }
    )
    leakage_path = output_dir / "split_leakage_report.json"
    write_json_atomic(leakage_path, leakage)
    result.update(
        {
            "initial_state_fingerprints": _repo_relative(fingerprint_path),
            "split_leakage_report": _repo_relative(leakage_path),
            "split_leakage_status": leakage["status"],
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/maniskill/m0_probe"),
    )
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument(
        "--state-set",
        dest="state_sets",
        action="append",
        choices=("development", "test"),
        default=[],
        help="Reset and fingerprint one frozen set; repeat for both sets.",
    )
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip the CPU-only raw audit; use when running reset-only fingerprint audit separately.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_audit(
        source_h5=arguments.source_h5,
        output_dir=arguments.output_dir,
        split_file=arguments.split_file,
        state_sets=arguments.state_sets,
        task=arguments.task,
        control_mode=arguments.control_mode,
        sim_backend=arguments.sim_backend,
        tolerance=arguments.tolerance,
        include_raw=not arguments.skip_raw,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
