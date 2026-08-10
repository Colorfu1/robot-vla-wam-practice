"""Convert ManiSkill RecordEpisode HDF5 files into LeRobotDataset roots.

The converter deliberately consumes the observations written by ManiSkill's
``RecordEpisode`` wrapper.  It does not replay or otherwise step an
environment; replay is an explicit remote prerequisite handled by ManiSkill's
own ``replay_trajectory`` command.
"""

from __future__ import annotations

import argparse
import errno
import importlib.metadata
import json
import logging
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

LOGGER = logging.getLogger(__name__)

TASK_ID = "PickCube-v1"
IMAGE_SHAPE = (128, 128, 3)
STATE_SHAPE = (21,)
DEFAULT_FPS = 20
CONVERTER_SCHEMA_VERSION = 1
CAMERA_KEY = "observation.images.base_camera"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_CHANNEL_NAMES = (
    "translation_x",
    "translation_y",
    "translation_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "gripper",
)
ACTION_CHANNEL_NAMES_BY_DIM = {
    4: ACTION_CHANNEL_NAMES[:3] + ("gripper",),
    7: ACTION_CHANNEL_NAMES,
    8: tuple(f"action_{index}" for index in range(8)),
}
STATE_FIELDS = (
    "agent.qpos[0:9]",
    "agent.qvel[0:9]",
    "extra.goal_pos[0:3]",
)
EXCLUDED_STATE_FIELDS = (
    "extra.is_grasped",
    "extra.obj_pose",
    "extra.tcp_to_obj_pos",
    "extra.obj_to_goal_pos",
    "extra.tcp_pose",
)


def dataset_features(action_dim: int) -> dict[str, dict[str, Any]]:
    """Return the fixed M0 feature contract for one action dimension."""

    if action_dim not in ACTION_CHANNEL_NAMES_BY_DIM:
        raise ValueError(f"PickCube-v1 M0 action dimension must be 4, 7 or 8, got {action_dim}")
    return {
        CAMERA_KEY: {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": STATE_SHAPE,
            "names": ["state"],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(path)


def _publish_directory(temporary: Path, output_dir: Path) -> None:
    """Publish a completed dataset, falling back to a cross-device copy."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory: {output_dir}")
    try:
        temporary.rename(output_dir)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        LOGGER.warning(
            "Atomic dataset rename crossed filesystems; copying %s to %s",
            temporary,
            output_dir,
        )

    try:
        shutil.copytree(temporary, output_dir, symlinks=False)
    except BaseException:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    shutil.rmtree(temporary)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata_path(replay_h5: Path) -> Path:
    path = replay_h5.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"ManiSkill replay metadata is missing: {path}")
    return path


def _environment_info(metadata: Mapping[str, Any], *, task: str) -> dict[str, Any]:
    env_info = metadata.get("env_info")
    if not isinstance(env_info, Mapping):
        raise TypeError("Replay metadata has no env_info object")
    env_id = env_info.get("env_id")
    if env_id != task:
        raise ValueError(f"Replay task must be {task!r}, got {env_id!r}")
    env_kwargs = env_info.get("env_kwargs")
    if not isinstance(env_kwargs, Mapping):
        raise TypeError("Replay metadata env_info has no env_kwargs object")
    return dict(env_info)


def _control_mode(metadata: Mapping[str, Any]) -> str:
    env_info = metadata.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {}) if isinstance(env_info, Mapping) else {}
    value = env_kwargs.get("control_mode") if isinstance(env_kwargs, Mapping) else None
    if isinstance(value, str) and value:
        return value
    episodes = metadata.get("episodes", [])
    if isinstance(episodes, list):
        for episode in episodes:
            if isinstance(episode, Mapping) and isinstance(episode.get("control_mode"), str):
                return str(episode["control_mode"])
    raise ValueError("Could not determine replay control mode")


def _episode_seed(record: Mapping[str, Any]) -> int | None:
    seed = record.get("episode_seed")
    if seed is None:
        reset_kwargs = record.get("reset_kwargs")
        if isinstance(reset_kwargs, Mapping):
            seed = reset_kwargs.get("seed")
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(f"Expected one seed per CPU replay episode, got {seed!r}")
        seed = seed[0]
    return None if seed is None else int(seed)


def _episode_records(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Replay metadata episodes must be a list")
    records: dict[int, dict[str, Any]] = {}
    for record in episodes:
        if not isinstance(record, Mapping):
            raise TypeError("Each replay episode metadata record must be an object")
        episode_id = int(record["episode_id"])
        if episode_id in records:
            raise ValueError(f"Duplicate replay episode_id={episode_id}")
        records[episode_id] = dict(record)
    return records


def _episode_id_from_name(name: str) -> int:
    if not name.startswith("traj_"):
        raise ValueError(f"Unexpected ManiSkill HDF5 group {name!r}; expected traj_<id>")
    try:
        return int(name.removeprefix("traj_"))
    except ValueError as exc:
        raise ValueError(f"Unexpected ManiSkill HDF5 group {name!r}") from exc


def _h5_value_at(node: h5py.Group, path: tuple[str, ...], index: int) -> np.ndarray:
    current: Any = node
    for component in path:
        if not isinstance(current, h5py.Group) or component not in current:
            joined = "/".join(path)
            raise KeyError(f"Replay observation is missing obs/{joined}")
        current = current[component]
    if not isinstance(current, h5py.Dataset):
        raise TypeError(f"Replay observation path {'/'.join(path)} is not a dataset")
    return np.asarray(current[index])


def state_from_observation(
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_pos: np.ndarray,
) -> np.ndarray:
    """Build the non-privileged 21D state used by both M0 datasets."""

    qpos_array = np.asarray(qpos, dtype=np.float32).reshape(-1)
    qvel_array = np.asarray(qvel, dtype=np.float32).reshape(-1)
    goal_array = np.asarray(goal_pos, dtype=np.float32).reshape(-1)
    if qpos_array.shape != (9,):
        raise ValueError(f"agent.qpos must have shape (9,), got {qpos_array.shape}")
    if qvel_array.shape != (9,):
        raise ValueError(f"agent.qvel must have shape (9,), got {qvel_array.shape}")
    if goal_array.shape != (3,):
        raise ValueError(f"extra.goal_pos must have shape (3,), got {goal_array.shape}")
    state = np.concatenate((qpos_array, qvel_array, goal_array)).astype(np.float32)
    if state.shape != STATE_SHAPE or not np.isfinite(state).all():
        raise ValueError(f"State must be finite with shape {STATE_SHAPE}, got {state.shape}")
    return state


def _frame_from_h5(
    episode: h5py.Group, index: int, *, action_dim: int, task: str
) -> dict[str, Any]:
    if "obs" not in episode or "actions" not in episode:
        raise KeyError("Replay episode must contain obs and actions")
    obs = episode["obs"]
    image = _h5_value_at(obs, ("sensor_data", "base_camera", "rgb"), index)
    if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError(
            "base_camera RGB must be uint8 (128, 128, 3), "
            f"got dtype={image.dtype}, shape={image.shape}"
        )
    action = np.asarray(episode["actions"][index], dtype=np.float32).reshape(-1)
    if action.shape != (action_dim,) or not np.isfinite(action).all():
        raise ValueError(
            f"Replay action must be finite with shape ({action_dim},), got {action.shape}"
        )
    state = state_from_observation(
        _h5_value_at(obs, ("agent", "qpos"), index),
        _h5_value_at(obs, ("agent", "qvel"), index),
        _h5_value_at(obs, ("extra", "goal_pos"), index),
    )
    return {
        CAMERA_KEY: image,
        STATE_KEY: state,
        ACTION_KEY: action,
        "task": task,
    }


def iter_episode_frames(
    episode: h5py.Group,
    *,
    action_dim: int,
    task: str = TASK_ID,
    frame_limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one LeRobot frame per transition, using obs[t] with action[t]."""

    if "actions" not in episode or "obs" not in episode:
        raise KeyError("Replay episode must contain actions and obs")
    num_actions = int(episode["actions"].shape[0])
    camera_dataset = episode["obs"]["sensor_data"]["base_camera"]["rgb"]
    if int(camera_dataset.shape[0]) != num_actions + 1:
        raise ValueError(
            "Replay observation timeline must contain one reset frame plus one frame per action: "
            f"obs={camera_dataset.shape[0]}, actions={num_actions}"
        )
    if frame_limit is None:
        frame_limit = num_actions
    if frame_limit <= 0 or frame_limit > num_actions:
        raise ValueError(f"frame_limit must be in [1,{num_actions}], got {frame_limit}")
    for index in range(frame_limit):
        yield _frame_from_h5(episode, index, action_dim=action_dim, task=task)


def episode_success(metadata_record: Mapping[str, Any], episode: h5py.Group) -> bool:
    """Read the replay success label written by ManiSkill, never infer it."""

    if "success" in metadata_record:
        return bool(metadata_record["success"])
    if "success" not in episode:
        raise ValueError(
            "Replay episode has no native success label; rerun replay with a task that exposes "
            'info["success"]'
        )
    values = np.asarray(episode["success"], dtype=bool).reshape(-1)
    if values.size == 0:
        raise ValueError("Replay episode success dataset is empty")
    return bool(values[-1])


def success_series(metadata_record: Mapping[str, Any], episode: h5py.Group) -> np.ndarray:
    """Return the native per-action success signal when it is available."""

    if "success" in episode:
        values = np.asarray(episode["success"], dtype=bool).reshape(-1)
        if values.size:
            return values
    actions = np.asarray(episode["actions"])
    return np.full(actions.shape[0], episode_success(metadata_record, episode), dtype=bool)


def trim_frame_count(
    metadata_record: Mapping[str, Any],
    episode: h5py.Group,
    *,
    success_tail_frames: int,
) -> int:
    """Keep the first-success step plus a fixed number of stabilizing frames."""

    if success_tail_frames < 0:
        raise ValueError("success_tail_frames must be non-negative")
    num_actions = int(episode["actions"].shape[0])
    success = success_series(metadata_record, episode)
    first = np.flatnonzero(success)
    if first.size == 0:
        return num_actions
    # ``first`` is zero-based while the plan's step number is one-based:
    # total frames = first_success_step + tail_frames.
    first_success_step = int(first[0]) + 1
    return min(num_actions, first_success_step + success_tail_frames)


def _episode_summary(
    *,
    episode_id: int,
    metadata_record: Mapping[str, Any],
    episode: h5py.Group,
    success: bool,
    frame_count: int,
) -> dict[str, Any]:
    return {
        "source_episode_id": episode_id,
        "seed": _episode_seed(metadata_record),
        "success": success,
        "frame_count": frame_count,
        "elapsed_steps": int(metadata_record.get("elapsed_steps", frame_count)),
        "control_mode": metadata_record.get("control_mode"),
        "has_native_success_dataset": "success" in episode,
    }


def _replay_summary(
    *,
    replay_h5: Path,
    metadata: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
    source_episode_count: int | None,
) -> dict[str, Any]:
    successful_count = 0
    source_success_count = sum(bool(record.get("success", False)) for record in records.values())
    seeds: list[int] = []
    replayed_episode_ids: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    with h5py.File(replay_h5, "r") as handle:
        for episode_id, record in records.items():
            if f"traj_{episode_id}" not in handle:
                continue
            replayed_episode_ids.append(episode_id)
            episode = handle[f"traj_{episode_id}"]
            success = episode_success(record, episode)
            if success:
                successful_count += 1
            seed = _episode_seed(record)
            if seed is not None:
                seeds.append(seed)
            success_values = success_series(record, episode)
            diagnostics.append(
                {
                    "episode_id": episode_id,
                    "seed": seed,
                    "num_action_steps": int(episode["actions"].shape[0]),
                    "success": success,
                    "success_once": bool(np.any(success_values)),
                    "success_at_end": bool(success_values[-1]) if success_values.size else False,
                    "first_success_step": (
                        None
                        if not np.any(success_values)
                        else int(np.flatnonzero(success_values)[0]) + 1
                    ),
                    "has_is_grasped": "is_grasped" in episode,
                    "has_is_obj_placed": "is_obj_placed" in episode,
                    "has_is_robot_static": "is_robot_static" in episode,
                }
            )
    replayed_count = len(replayed_episode_ids)
    denominator = (
        min(int(source_episode_count), len(records))
        if source_episode_count is not None
        else len(records)
    )
    return {
        "replay_h5": _repo_relative(replay_h5),
        "source_total_episode_count": source_episode_count,
        "source_episode_count": source_episode_count,
        "requested_episode_count": len(records),
        "replayed_episode_count": replayed_count,
        "missing_episode_ids": sorted(set(records) - set(replayed_episode_ids)),
        "successful_replayed_episode_count": successful_count,
        "replay_survival_rate": (None if not denominator else replayed_count / denominator),
        "successful_replay_rate": (
            None if not source_success_count else successful_count / source_success_count
        ),
        "replay_ok": replayed_count,
        "replay_ok_denominator": denominator,
        "replay_ok_rate": None if not denominator else replayed_count / denominator,
        "replay_success": successful_count,
        "replay_success_denominator": source_success_count,
        "replay_success_rate": (
            None if not source_success_count else successful_count / source_success_count
        ),
        "replayed_seeds": sorted(set(seeds)),
        "control_mode": _control_mode(metadata),
        "episode_diagnostics": diagnostics,
    }


def action_channel_stats(
    *,
    replay_h5: Path,
    metadata: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize all seven EEF action channels in a replay file.

    This is intentionally an offline HDF5 pass.  It does not infer task
    success and does not step ManiSkill.  The output makes the pre-registered
    rotation-vs-translation gate auditable for the 7D EEF replay.
    """

    values: list[np.ndarray] = []
    replayed_episode_ids: list[int] = []
    with h5py.File(replay_h5, "r") as handle:
        for episode_id in sorted(records):
            episode_name = f"traj_{episode_id}"
            if episode_name not in handle:
                continue
            actions = np.asarray(handle[episode_name]["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[1] not in ACTION_CHANNEL_NAMES_BY_DIM:
                raise ValueError(
                    "PickCube replay actions must have shape [T,4], [T,7] or [T,8], "
                    f"got {actions.shape} in {episode_name}"
                )
            if actions.shape[0] == 0:
                continue
            if not np.isfinite(actions).all():
                raise ValueError(f"Replay actions contain non-finite values in {episode_name}")
            values.append(actions)
            replayed_episode_ids.append(episode_id)
    if not values:
        raise ValueError("Replay HDF5 contains no non-empty 7D EEF action episodes")

    actions = np.concatenate(values, axis=0)
    absolute = np.abs(actions)

    def channel_summary(index: int) -> dict[str, float]:
        return {
            "mean": float(np.mean(actions[:, index])),
            "std": float(np.std(actions[:, index])),
            "abs_p50": float(np.percentile(absolute[:, index], 50)),
            "abs_p90": float(np.percentile(absolute[:, index], 90)),
            "abs_p95": float(np.percentile(absolute[:, index], 95)),
            "abs_p99": float(np.percentile(absolute[:, index], 99)),
        }

    action_dim = int(actions.shape[1])
    names = ACTION_CHANNEL_NAMES_BY_DIM[action_dim]
    translation_abs_median = float(np.median(absolute[:, :3]))
    rotation_abs_median = None if action_dim < 7 else float(np.median(absolute[:, 3:6]))
    ratio = (
        None
        if rotation_abs_median is None or translation_abs_median == 0.0
        else rotation_abs_median / translation_abs_median
    )
    return {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "task": TASK_ID,
        "control_mode": _control_mode(metadata),
        "action_dim": action_dim,
        "channel_order": list(names),
        "replayed_episode_count": len(replayed_episode_ids),
        "replayed_episode_ids": replayed_episode_ids,
        "num_action_steps": int(actions.shape[0]),
        "channels": {name: channel_summary(index) for index, name in enumerate(names)},
        "translation_abs_median": translation_abs_median,
        "rotation_abs_median": rotation_abs_median,
        "rotation_to_translation_abs_median_ratio": ratio,
        "rotation_effective_threshold_ratio": 0.10,
        "rotation_channels_effective": (None if ratio is None else bool(ratio >= 0.10)),
    }


def _update_survival_report(
    path: Path,
    *,
    summary: dict[str, Any],
    dataset_summary: dict[str, Any] | None = None,
) -> None:
    payload = (
        _read_json(path)
        if path.is_file()
        else {
            "schema_version": CONVERTER_SCHEMA_VERSION,
            "task": TASK_ID,
            "replays": {},
        }
    )
    replays = payload.setdefault("replays", {})
    if not isinstance(replays, dict):
        raise TypeError(f"Survival report replays must be an object: {path}")
    control_mode = str(summary["control_mode"])
    merged = dict(summary)
    if dataset_summary is not None:
        merged["converted_dataset"] = dataset_summary
    replays[control_mode] = merged
    payload["updated_at"] = iso_timestamp()
    write_json_atomic(path, payload)


def _source_episode_count(source_h5: Path | None) -> int | None:
    if source_h5 is None:
        return None
    source_meta = _metadata_path(source_h5)
    payload = _read_json(source_meta)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError(f"Source metadata episodes must be a list: {source_meta}")
    return len(episodes)


def convert_replay(
    *,
    replay_h5: Path,
    output_dir: Path,
    repo_id: str,
    task: str = TASK_ID,
    fps: int = DEFAULT_FPS,
    include_failed: bool = False,
    max_episodes: int | None = None,
    success_tail_frames: int | None = None,
    source_h5: Path | None = None,
    survival_report: Path | None = None,
    action_stats_output: Path | None = None,
    episode_order: Sequence[int] | None = None,
    episode_id_map: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Convert one replay file without deleting an existing output directory."""

    replay_h5 = replay_h5.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not replay_h5.is_file():
        raise FileNotFoundError(f"Replay HDF5 does not exist: {replay_h5}")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing ManiSkill dataset directory: {output_dir}"
        )
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if success_tail_frames is not None and success_tail_frames < 0:
        raise ValueError("success_tail_frames must be non-negative")

    metadata_path = _metadata_path(replay_h5)
    metadata = _read_json(metadata_path)
    env_info = _environment_info(metadata, task=task)
    control_mode = _control_mode(metadata)
    records = _episode_records(metadata)
    source_count = _source_episode_count(source_h5)
    if max_episodes is not None:
        records = dict(sorted(records.items())[:max_episodes])
    if not records:
        raise ValueError("Replay metadata contains no episodes to convert")

    if episode_order is None:
        ordered_episode_ids = tuple(sorted(records))
    else:
        ordered_episode_ids = tuple(int(value) for value in episode_order)
        if len(set(ordered_episode_ids)) != len(ordered_episode_ids):
            raise ValueError("episode_order must not contain duplicate replay episode ids")
        if set(ordered_episode_ids) != set(records):
            raise ValueError("episode_order must contain exactly the replay metadata episode ids")
    canonical_episode_ids = {int(key): int(value) for key, value in (episode_id_map or {}).items()}
    unknown_mapped_ids = set(canonical_episode_ids) - set(records)
    if unknown_mapped_ids:
        raise ValueError(
            f"episode_id_map contains unknown replay episode ids: {sorted(unknown_mapped_ids)}"
        )

    survival_summary = _replay_summary(
        replay_h5=replay_h5,
        metadata=metadata,
        records=records,
        source_episode_count=source_count,
    )
    action_stats = None
    if action_stats_output is not None:
        action_stats = action_channel_stats(
            replay_h5=replay_h5,
            metadata=metadata,
            records=records,
        )
        if source_h5 is not None and source_h5.resolve() != replay_h5.resolve():
            source_metadata = _read_json(_metadata_path(source_h5))
            source_records = _episode_records(source_metadata)
            action_stats["source"] = action_channel_stats(
                replay_h5=source_h5,
                metadata=source_metadata,
                records=source_records,
            )
        write_json_atomic(action_stats_output, action_stats)
    if survival_report is not None:
        _update_survival_report(survival_report, summary=survival_summary)

    with h5py.File(replay_h5, "r") as handle:
        selected: list[tuple[int, dict[str, Any], h5py.Group, bool]] = []
        action_dims: set[int] = set()
        if episode_order is None:
            episode_names = tuple(sorted(handle, key=_episode_id_from_name))
        else:
            episode_names = tuple(f"traj_{episode_id}" for episode_id in ordered_episode_ids)
            missing_groups = [name for name in episode_names if name not in handle]
            if missing_groups:
                raise FileNotFoundError(
                    f"Replay HDF5 is missing ordered episode groups: {missing_groups[:5]}"
                )
        for episode_name in episode_names:
            episode_id = _episode_id_from_name(episode_name)
            if episode_id not in records:
                continue
            episode = handle[episode_name]
            if not isinstance(episode, h5py.Group):
                raise TypeError(f"Replay trajectory {episode_name} is not an HDF5 group")
            action_shape = tuple(int(value) for value in episode["actions"].shape[1:])
            if len(action_shape) != 1:
                raise ValueError(f"Replay actions must be a flat [T,A] array, got {action_shape}")
            action_dim = action_shape[0]
            action_dims.add(action_dim)
            success = episode_success(records[episode_id], episode)
            if include_failed or success:
                selected.append((episode_id, records[episode_id], episode, success))
        if not action_dims:
            raise ValueError("Replay HDF5 contains no trajectory actions")
        if len(action_dims) != 1:
            raise ValueError(
                f"Replay trajectories disagree on action dimension: {sorted(action_dims)}"
            )
        action_dim = next(iter(action_dims))
        dataset_contract = dataset_features(action_dim)
        if not selected:
            raise ValueError(
                "No successful replay episodes remain for conversion; use --include-failed only "
                "for an explicitly diagnostic dataset"
            )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_dir.parent / f".{output_dir.name}.incomplete-{os.getpid()}"
        if temporary.exists():
            raise FileExistsError(f"Refusing to reuse incomplete conversion directory: {temporary}")
        dataset = None
        finalized = False
        converted_episodes: list[dict[str, Any]] = []
        converted_frames = 0
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=temporary,
                robot_type="panda",
                fps=fps,
                features=dataset_contract,
                use_videos=True,
                image_writer_threads=4,
                vcodec="libsvtav1",
            )
            for episode_id, record, episode, success in selected:
                frame_count = 0
                frame_limit = (
                    None
                    if success_tail_frames is None
                    else trim_frame_count(
                        record,
                        episode,
                        success_tail_frames=success_tail_frames,
                    )
                )
                try:
                    for frame in iter_episode_frames(
                        episode,
                        action_dim=action_dim,
                        task=task,
                        frame_limit=frame_limit,
                    ):
                        dataset.add_frame(frame)
                        frame_count += 1
                    if frame_count == 0:
                        raise ValueError(f"Replay episode {episode_id} contains no actions")
                    dataset.save_episode(parallel_encoding=False)
                except BaseException:
                    clear_buffer = getattr(dataset, "clear_episode_buffer", None)
                    if clear_buffer is not None:
                        clear_buffer(delete_images=True)
                    raise
                converted_frames += frame_count
                converted_episodes.append(
                    {
                        **_episode_summary(
                            episode_id=canonical_episode_ids.get(episode_id, episode_id),
                            metadata_record=record,
                            episode=episode,
                            success=success,
                            frame_count=frame_count,
                        ),
                        "full_frame_count": int(episode["actions"].shape[0]),
                        "trimmed": success_tail_frames is not None,
                    }
                )
            dataset.finalize()
            finalized = True
            manifest = {
                "schema_version": CONVERTER_SCHEMA_VERSION,
                "created_at": iso_timestamp(),
                "task": task,
                "repo_id": repo_id,
                "source": {
                    "replay_h5": _repo_relative(replay_h5),
                    "metadata": _repo_relative(metadata_path),
                    "source_h5": None if source_h5 is None else _repo_relative(source_h5),
                    "env_id": env_info.get("env_id"),
                    "env_kwargs": {
                        key: value
                        for key, value in dict(env_info.get("env_kwargs", {})).items()
                        if key in {"obs_mode", "control_mode", "reward_mode", "sim_backend"}
                    },
                    "source_type": metadata.get("source_type"),
                },
                "replay": survival_summary,
                "action_channel_stats": (
                    None if action_stats_output is None else _repo_relative(action_stats_output)
                ),
                "conversion": {
                    "fps": fps,
                    "control_frequency_hz": fps,
                    "control_mode": control_mode,
                    "episode_filter": "all" if include_failed else "native_success_only",
                    "source_episode_limit": max_episodes,
                    "episode_order": [
                        int(canonical_episode_ids.get(episode_id, episode_id))
                        for episode_id, _, _, _ in selected
                    ],
                    "replay_episode_id_map": {
                        str(episode_id): int(canonical_episode_ids.get(episode_id, episode_id))
                        for episode_id in ordered_episode_ids
                    },
                    "success_tail_frames": success_tail_frames,
                    "timeline_policy": (
                        "full" if success_tail_frames is None else "first_success_plus_fixed_tail"
                    ),
                    "action_dim": action_dim,
                    "features": dataset_contract,
                    "state_fields": list(STATE_FIELDS),
                    "excluded_state_fields": list(EXCLUDED_STATE_FIELDS),
                    "observation_keys": [CAMERA_KEY, STATE_KEY],
                },
                "output": {
                    "episodes": len(converted_episodes),
                    "frames": converted_frames,
                    "successful_episodes": sum(
                        1 for episode in converted_episodes if episode["success"]
                    ),
                    "failed_episodes": sum(
                        1 for episode in converted_episodes if not episode["success"]
                    ),
                    "episode_map": converted_episodes,
                },
                "runtime": {
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "mani_skill": _version("mani_skill"),
                    "lerobot": _version("lerobot"),
                },
            }
            write_json_atomic(temporary / "conversion_manifest.json", manifest)
            _publish_directory(temporary, output_dir)
        except BaseException:
            if dataset is not None and not finalized:
                try:
                    dataset.finalize()
                except BaseException:
                    LOGGER.exception("Could not finalize incomplete conversion dataset")
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    result = {
        "output_dir": _repo_relative(output_dir),
        "control_mode": control_mode,
        "action_dim": action_dim,
        "episodes": len(converted_episodes),
        "frames": converted_frames,
        "replay_survival": survival_summary,
        "action_channel_stats": action_stats,
        "success_tail_frames": success_tail_frames,
    }
    if survival_report is not None:
        _update_survival_report(
            survival_report,
            summary=survival_summary,
            dataset_summary={
                "output_dir": _repo_relative(output_dir),
                "episodes": len(converted_episodes),
                "frames": converted_frames,
                "episode_filter": "all" if include_failed else "native_success_only",
            },
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include episodes whose native replay success label is false.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--success-tail-frames",
        type=int,
        default=None,
        help="Trim each successful episode at first success plus this many frames.",
    )
    parser.add_argument(
        "--source-h5",
        type=Path,
        default=None,
        help="Original official trajectory HDF5, used as the survival-rate denominator.",
    )
    parser.add_argument(
        "--survival-report",
        type=Path,
        default=None,
        help="Optional shared replay_survival.json to create or update.",
    )
    parser.add_argument(
        "--action-stats-output",
        type=Path,
        default=None,
        help="Optional JSON output for the 7D EEF action-channel statistics.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = convert_replay(
        replay_h5=arguments.replay_h5,
        output_dir=arguments.output_dir,
        repo_id=arguments.repo_id,
        task=arguments.task,
        fps=arguments.fps,
        include_failed=arguments.include_failed,
        max_episodes=arguments.max_episodes,
        success_tail_frames=arguments.success_tail_frames,
        source_h5=arguments.source_h5,
        survival_report=arguments.survival_report,
        action_stats_output=arguments.action_stats_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
