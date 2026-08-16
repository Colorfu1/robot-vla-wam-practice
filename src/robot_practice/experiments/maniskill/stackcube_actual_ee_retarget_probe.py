"""Retarget selected joint demos through an actual-EEF reference path.

This diagnostic is intentionally separate from ManiSkill's official replay
converter.  For each selected source episode it:

1. replays ``pd_joint_pos`` on a CPU source environment;
2. records the source simulator's actual TCP pose after every source step;
3. linearly resamples translation and SLERPs orientation into a dense
   reference path;
4. follows the reference with native ManiSkill3 ``pd_ee_delta_pose``;
5. saves the sent actions, reference TCP path, executed TCP path, tracking
   errors, and success flags.

The output is a diagnostic native-EE trajectory.  It is not promoted to a
training dataset automatically and it does not claim that source and target
simulators are dynamically identical.
"""

from __future__ import annotations

import argparse
import copy
import errno
import json
import math
import multiprocessing as mp
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import practice_root

TASK_IDS = ("StackCube-v1", "PlugCharger-v1")
TASK_ID = "StackCube-v1"
SOURCE_CONTROL_MODE = "pd_joint_pos"
TARGET_CONTROL_MODES = ("pd_ee_delta_pos", "pd_ee_delta_pose")
CONTROL_FREQUENCY_HZ = 20
SCHEMA_VERSION = 1
DEFAULT_COUNT = 1
DEFAULT_WORKERS = 1


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
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(practice_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _scalar_bool(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _as_vector(value: Any, *, name: str, size: int | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if size is not None and result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Infinity")
    return result


def _normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    result = _as_vector(quaternion, name="quaternion", size=4)
    norm = float(np.linalg.norm(result))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    result = result / norm
    return result if result[0] >= 0 else -result


def quaternion_slerp(left: Sequence[float], right: Sequence[float], fraction: float) -> np.ndarray:
    """Spherical interpolation for wxyz quaternions."""

    q0 = _normalize_quaternion(left)
    q1 = _normalize_quaternion(right)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + fraction * (q1 - q0))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return _normalize_quaternion(
        math.sin((1.0 - fraction) * theta) / sin_theta * q0
        + math.sin(fraction * theta) / sin_theta * q1
    )


def interpolate_pose_path(path: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a [T,7] pose path with linear position and quaternion SLERP."""

    poses = np.asarray(path, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] < 1:
        raise ValueError(f"Expected pose path [T,7], got {poses.shape}")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if poses.shape[0] == target_length:
        return poses.astype(np.float32)
    old_grid = np.linspace(0.0, 1.0, poses.shape[0])
    new_grid = np.linspace(0.0, 1.0, target_length)
    positions = np.column_stack(
        [np.interp(new_grid, old_grid, poses[:, axis]) for axis in range(3)]
    )
    quaternions = np.empty((target_length, 4), dtype=np.float64)
    for index, time in enumerate(new_grid):
        right = min(int(np.searchsorted(old_grid, time, side="right")), len(poses) - 1)
        left = max(0, right - 1)
        denominator = old_grid[right] - old_grid[left]
        fraction = 0.0 if denominator <= 0 else (time - old_grid[left]) / denominator
        quaternions[index] = quaternion_slerp(
            poses[left, 3:], poses[right, 3:], float(fraction)
        )
    return np.concatenate((positions, quaternions), axis=1).astype(np.float32)


def _metadata_records(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Source metadata must contain an episodes list")
    result: dict[int, dict[str, Any]] = {}
    for record in episodes:
        if not isinstance(record, Mapping):
            raise TypeError("Episode metadata must be objects")
        episode_id = int(record["episode_id"])
        result[episode_id] = dict(record)
    return result


def _reset_kwargs(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record.get("reset_kwargs", {})))
    seed = record.get("episode_seed", result.get("seed"))
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(f"Expected one CPU seed, got {seed!r}")
        seed = seed[0]
    if seed is None:
        raise ValueError(f"Episode {record.get('episode_id')} has no seed")
    result["seed"] = int(seed)
    return result


def _state_list(group: h5py.Group) -> list[dict[str, Any]]:
    from mani_skill.trajectory import utils as trajectory_utils

    return trajectory_utils.dict_to_list_of_dicts(group["env_states"])


def _pose_array(pose: Any) -> np.ndarray:
    position = _as_vector(pose.p[0], name="pose.position", size=3)
    quaternion = _as_vector(pose.q[0], name="pose.quaternion", size=4)
    return np.concatenate((position, quaternion)).astype(np.float32)


def _tcp_pose_at_base(env: Any, root_link: Any) -> np.ndarray:
    world_pose = env.unwrapped.agent.tcp.pose
    base_pose = root_link.pose.inv() * world_pose
    return _pose_array(base_pose)


def _replace_initial_recording(recorder: Any, state: Mapping[str, Any]) -> None:
    from mani_skill.utils import common

    if recorder._trajectory_buffer is None:
        return

    def replace(target: Any, replacement: Any) -> None:
        if isinstance(target, np.ndarray):
            target[-1, :] = replacement[-1, :]
        else:
            for key in target:
                replace(target[key], replacement[key])

    replace(recorder._trajectory_buffer.state, common.batch(state))
    replace(
        recorder._trajectory_buffer.observation,
        common.to_numpy(common.batch(recorder.base_env.get_obs())),
    )


def _position_action(delta: np.ndarray, arm: Any) -> tuple[np.ndarray, bool]:
    low = arm.action_space_low[:3].detach().cpu().numpy()
    high = arm.action_space_high[:3].detach().cpu().numpy()
    action = (delta - 0.5 * (high + low)) / (0.5 * (high - low))
    clipped = bool(np.any(np.abs(action) > 1.0))
    return np.clip(action, -1.0, 1.0).astype(np.float32), clipped


def _rotation_action(
    delta_euler_xyz: Sequence[float], arm: Any
) -> tuple[np.ndarray, bool]:
    delta = _as_vector(delta_euler_xyz, name="delta_euler_xyz", size=3)
    lower = np.broadcast_to(_as_vector(arm.config.rot_lower, name="rot_lower"), (3,))
    if np.any(np.isclose(lower, 0.0)):
        raise ValueError("PDEEPoseController rot_lower must not contain zero")
    action = delta / lower
    norm = float(np.linalg.norm(action))
    clipped = norm > 1.0
    if clipped:
        action = action / norm
    return action.astype(np.float32), clipped


def _target_action(
    controller: Any,
    source_action_dict: Mapping[str, Any],
    arm_action: np.ndarray,
    common: Any,
) -> Any:
    values: dict[str, Any] = {}
    for key in controller.controllers:
        if key == "arm":
            values[key] = arm_action
        elif key in source_action_dict:
            values[key] = source_action_dict[key]
        else:
            raise KeyError(f"Missing source action for target controller key {key!r}")
    device = controller.controllers["arm"].device
    return controller.from_action_dict(
        {key: common.to_tensor(value, device=device) for key, value in values.items()}
    )


def _worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils import common
    from mani_skill.utils.geometry import rotation_conversions
    from mani_skill.utils.wrappers import RecordEpisode

    source_h5 = Path(str(spec["source_h5"]))
    task_id = str(spec["task_id"])
    worker_dir = Path(str(spec["worker_dir"]))
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_manifest = worker_dir / "worker_manifest.json"
    if worker_manifest.is_file():
        previous = _read_json(worker_manifest)
        trajectory = Path(str(previous.get("trajectory", "")))
        if not trajectory.is_absolute():
            trajectory = practice_root() / trajectory
        if (
            trajectory.is_file()
            and previous.get("task_id") == spec["task_id"]
            and int(previous.get("substeps_per_source_step", -1))
            == int(spec["substeps_per_source_step"])
            and previous.get("target_control_mode") == spec["target_control_mode"]
        ):
            return previous

    with h5py.File(source_h5, "r") as source_file:
        metadata = _read_json(source_h5.with_suffix(".json"))
        source_kwargs = dict(metadata["env_info"]["env_kwargs"])
        source_kwargs.update(
            obs_mode="none",
            control_mode=SOURCE_CONTROL_MODE,
            sim_backend="physx_cpu",
            reward_mode="sparse",
            reconfiguration_freq=1,
        )
        target_control_mode = str(spec["target_control_mode"])
        if target_control_mode not in TARGET_CONTROL_MODES:
            raise ValueError(f"Unsupported target control mode: {target_control_mode}")
        target_kwargs = dict(source_kwargs)
        target_kwargs["control_mode"] = target_control_mode
        source_env = gym.make(task_id, **source_kwargs)
        target_base_env = gym.make(task_id, **target_kwargs)
        recorder = RecordEpisode(
            target_base_env,
            output_dir=str(worker_dir),
            trajectory_name="trajectory",
            save_trajectory=True,
            save_video=False,
            save_on_reset=False,
            record_reward=True,
            record_env_state=True,
            source_type="m1_actual_ee_retarget_probe",
            source_desc=f"Actual source TCP path followed by native {target_control_mode}",
        )
        rows: list[dict[str, Any]] = []
        try:
            for record in spec["records"]:
                episode_id = int(record["episode_id"])
                group = source_file[f"traj_{episode_id}"]
                actions = np.asarray(group["actions"], dtype=np.float32)
                states = _state_list(group)
                reset_kwargs = _reset_kwargs(record)
                source_env.reset(**reset_kwargs)
                recorder.reset(**reset_kwargs)
                source_env.unwrapped.set_state_dict(states[0])
                recorder.base_env.set_state_dict(states[0])
                _replace_initial_recording(recorder, states[0])

                source_controller = source_env.unwrapped.agent.controller
                target_controller = recorder.base_env.agent.controller
                target_arm = target_controller.controllers["arm"]
                expected_frame = (
                    "root_translation"
                    if target_control_mode == "pd_ee_delta_pos"
                    else "root_translation:root_aligned_body_rotation"
                )
                if target_arm.config.frame != expected_frame:
                    raise RuntimeError(
                        f"Unexpected {task_id} EE frame: {target_arm.config.frame}; "
                        f"expected {expected_frame}"
                    )

                substeps = int(spec["substeps_per_source_step"])
                if substeps <= 0:
                    raise ValueError("substeps_per_source_step must be positive")

                source_pose_path: list[np.ndarray] = [
                    _tcp_pose_at_base(source_env, target_arm.root_link)
                ]
                source_action_dicts: list[Mapping[str, Any]] = []
                for source_action in actions:
                    tensor_action = common.to_tensor(
                        source_action, device=source_env.unwrapped.device
                    )
                    action_dict = source_controller.to_action_dict(tensor_action)
                    source_action_dicts.append(
                        {
                            key: value.detach().cpu().clone()
                            if hasattr(value, "detach")
                            else copy.deepcopy(value)
                            for key, value in action_dict.items()
                        }
                    )
                    source_env.step(tensor_action)
                    source_pose_path.append(
                        _tcp_pose_at_base(source_env, target_arm.root_link)
                    )

                reference = interpolate_pose_path(
                    np.asarray(source_pose_path), len(actions) * substeps + 1
                )
                executed: list[np.ndarray] = [
                    _tcp_pose_at_base(recorder, target_arm.root_link)
                ]
                sent_actions: list[np.ndarray] = []
                tracking_errors: list[float] = []
                position_clips = 0
                rotation_clips = 0
                rotation_deltas: list[float] = []
                success_once = False
                first_success_step: int | None = None
                final_info: Mapping[str, Any] = {}
                for dense_step in range(1, len(reference)):
                    source_step = min((dense_step - 1) // substeps, len(actions) - 1)
                    source_action_dict = source_action_dicts[source_step]
                    target_pose = reference[dense_step]
                    current_pose = target_arm.ee_pose_at_base
                    current_position = _as_vector(
                        current_pose.p[0], name="current_tcp", size=3
                    )
                    target_position = _as_vector(target_pose[:3], name="reference_tcp", size=3)
                    delta = target_position - current_position
                    position_action, position_clipped = _position_action(delta, target_arm)
                    position_clips += int(position_clipped)
                    if target_control_mode == "pd_ee_delta_pose":
                        from mani_skill.utils.structs import Pose

                        desired_pose = Pose.create_from_pq(
                            common.to_tensor(
                                target_pose[:3][None], device=target_arm.device
                            ),
                            common.to_tensor(
                                target_pose[3:][None], device=target_arm.device
                            ),
                        )
                        delta_pose = desired_pose * current_pose.inv()
                        delta_euler = rotation_conversions.matrix_to_euler_angles(
                            rotation_conversions.quaternion_to_matrix(delta_pose.q), "XYZ"
                        )[0]
                        rotation_action, rotation_clipped = _rotation_action(
                            delta_euler, target_arm
                        )
                        rotation_deltas.append(
                            float(np.linalg.norm(delta_euler.detach().cpu().numpy()))
                        )
                        rotation_clips += int(rotation_clipped)
                        arm_action = np.concatenate((position_action, rotation_action))
                    else:
                        arm_action = position_action
                    action = _target_action(
                        target_controller, source_action_dict, arm_action, common
                    )
                    action_dim = 7 if target_control_mode == "pd_ee_delta_pose" else 4
                    sent_actions.append(
                        _as_vector(action, name="target_action", size=action_dim).astype(
                            np.float32
                        )
                    )
                    _, _, _, _, final_info = recorder.step(action)
                    actual = _tcp_pose_at_base(recorder, target_arm.root_link)
                    executed.append(actual)
                    tracking_errors.append(float(np.linalg.norm(target_position - actual[:3])))
                    if _scalar_bool(final_info.get("success")):
                        success_once = True
                        if first_success_step is None:
                            first_success_step = dense_step

                recorder.flush_trajectory(ignore_empty_transition=True)
                final_success = _scalar_bool(final_info.get("success"))
                diagnostic_group = worker_dir / f"episode_{episode_id:04d}.npz"
                np.savez_compressed(
                    diagnostic_group,
                    source_tcp_pose=np.asarray(source_pose_path, dtype=np.float32),
                    reference_tcp_pose=np.asarray(reference, dtype=np.float32),
                    executed_tcp_pose=np.asarray(executed, dtype=np.float32),
                    actions=np.asarray(sent_actions, dtype=np.float32),
                    tracking_error_m=np.asarray(tracking_errors, dtype=np.float32),
                )
                rows.append(
                    {
                        "episode_id": episode_id,
                        "source_seed": int(reset_kwargs["seed"]),
                        "source_success": bool(np.asarray(group["success"]).reshape(-1).any()),
                        "source_action_steps": len(actions),
                        "target_action_steps": len(sent_actions),
                        "substeps_per_source_step": substeps,
                        "success_once": success_once,
                        "success_at_end": final_success,
                        "first_success_step": first_success_step,
                        "position_clip_count": position_clips,
                        "rotation_clip_count": rotation_clips,
                        "max_rotation_delta_rad": max(rotation_deltas, default=0.0),
                        "mean_tracking_error_m": float(np.mean(tracking_errors))
                        if tracking_errors
                        else None,
                        "max_tracking_error_m": max(tracking_errors, default=0.0),
                        "diagnostic_npz": _relative(diagnostic_group),
                    }
                )
        finally:
            recorder.close()
            source_env.close()

    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(spec["task_id"]),
        "worker_id": int(spec["worker_id"]),
        "target_control_mode": str(spec["target_control_mode"]),
        "substeps_per_source_step": int(spec["substeps_per_source_step"]),
        "trajectory": _relative(worker_dir / "trajectory.h5"),
        "episodes": rows,
    }
    _write_json(worker_manifest, result)
    return result


def _worker_entry(spec: Mapping[str, Any]) -> dict[str, Any]:
    return _worker(spec)


def _chunks(records: list[dict[str, Any]], worker_count: int) -> list[list[dict[str, Any]]]:
    worker_count = min(worker_count, len(records))
    chunk_size = math.ceil(len(records) / worker_count)
    return [records[start : start + chunk_size] for start in range(0, len(records), chunk_size)]


def run_probe(
    *,
    task_id: str = TASK_ID,
    source_h5: str | Path,
    output_dir: str | Path,
    episode_ids: Sequence[int],
    target_control_mode: str = "pd_ee_delta_pose",
    substeps_per_source_step: int = 4,
    num_workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if task_id not in TASK_IDS:
        raise ValueError(f"task_id must be one of {TASK_IDS}")
    if not episode_ids:
        raise ValueError("episode_ids must not be empty")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if substeps_per_source_step <= 0:
        raise ValueError("substeps_per_source_step must be positive")
    if target_control_mode not in TARGET_CONTROL_MODES:
        raise ValueError(f"target_control_mode must be one of {TARGET_CONTROL_MODES}")
    source_h5 = Path(source_h5).expanduser().resolve()
    if not source_h5.is_file() or not source_h5.with_suffix(".json").is_file():
        raise FileNotFoundError(f"Source H5/JSON pair is missing: {source_h5}")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = output_dir / "probe_manifest.json"
    if final_manifest_path.is_file():
        return _read_json(final_manifest_path)

    metadata = _read_json(source_h5.with_suffix(".json"))
    records_by_id = _metadata_records(metadata)
    selected_ids = [int(value) for value in episode_ids]
    missing = [value for value in selected_ids if value not in records_by_id]
    if missing:
        raise KeyError(f"Unknown source episode ids: {missing}")
    records = [records_by_id[value] for value in selected_ids]
    workers_root = output_dir / "workers"
    workers_root.mkdir(parents=True, exist_ok=True)
    specs = []
    for worker_id, chunk in enumerate(_chunks(records, num_workers)):
        specs.append(
            {
                "source_h5": str(source_h5),
                "task_id": task_id,
                "worker_id": worker_id,
                "worker_dir": str(workers_root / f"worker_{worker_id:02d}"),
                "records": chunk,
                "target_control_mode": target_control_mode,
                "substeps_per_source_step": int(substeps_per_source_step),
            }
        )

    from tqdm import tqdm

    pending = []
    results = []
    for spec in specs:
        manifest = Path(spec["worker_dir"]) / "worker_manifest.json"
        if manifest.is_file():
            previous = _read_json(manifest)
            trajectory = Path(str(previous.get("trajectory", "")))
            if not trajectory.is_absolute():
                trajectory = practice_root() / trajectory
            if trajectory.is_file():
                results.append(previous)
                continue
        pending.append(spec)
    if pending:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(pending)) as pool:
            results.extend(
                tqdm(
                    pool.imap(_worker_entry, pending),
                    total=len(pending),
                    desc=f"{task_id} actual-EE retarget workers",
                    unit="worker",
                )
            )
    results.sort(key=lambda result: int(result["worker_id"]))
    if len(results) != len(specs):
        raise RuntimeError(f"Not all {task_id} probe workers completed")

    from mani_skill.trajectory.merge_trajectory import merge_trajectories

    final_h5 = output_dir / "replay.h5"
    if final_h5.exists() or final_h5.with_suffix(".json").exists():
        raise FileExistsError(f"Incomplete probe output exists at {final_h5}")
    trajectories = [
        Path(result["trajectory"])
        if Path(result["trajectory"]).is_absolute()
        else practice_root() / result["trajectory"]
        for result in results
    ]
    merge_trajectories(str(final_h5), [str(path) for path in trajectories])
    rows = [row for result in results for row in result["episodes"]]
    rows.sort(key=lambda row: selected_ids.index(int(row["episode_id"])))
    successes = sum(bool(row["success_at_end"]) for row in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "task": task_id,
        "source_h5": _relative(source_h5),
        "source_control_mode": SOURCE_CONTROL_MODE,
        "target_control_mode": target_control_mode,
        "backend": "physx_cpu",
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "selected_episode_ids": selected_ids,
        "reference_contract": {
            "source_pose": "actual source TCP pose after each source joint action",
            "translation": "linear interpolation on a dense reference grid",
            "rotation": "wxyz quaternion SLERP on a dense reference grid",
            "target_follower": (
                f"native {target_control_mode} with repeated source gripper targets"
            ),
            "action_dim": 7 if target_control_mode == "pd_ee_delta_pose" else 4,
            "substeps_per_source_step": int(substeps_per_source_step),
        },
        "replay_h5": _relative(final_h5),
        "replay_json": _relative(final_h5.with_suffix(".json")),
        "summary": {
            "episodes": len(rows),
            "success_at_end": {"num_success": successes, "rate": successes / len(rows)},
            "success_once": {
                "num_success": sum(bool(row["success_once"]) for row in rows),
                "rate": sum(bool(row["success_once"]) for row in rows) / len(rows),
            },
            "mean_tracking_error_m": float(
                np.mean([row["mean_tracking_error_m"] for row in rows])
            ),
            "gate": "diagnostic_only; no formal promotion gate",
        },
        "episodes": rows,
    }
    _write_json(output_dir / "episode_diagnostics.json", {"episodes": rows})
    _write_json(final_manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_IDS, default=TASK_ID)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", required=True, help="Comma-separated source episode ids")
    parser.add_argument(
        "--target-control-mode",
        choices=TARGET_CONTROL_MODES,
        default="pd_ee_delta_pose",
    )
    parser.add_argument(
        "--substeps-per-source-step",
        type=int,
        default=4,
        help="Dense reference points followed per source action; default 4",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episode_ids = [int(value) for value in args.episode_ids.split(",") if value.strip()]
    manifest = run_probe(
        task_id=args.task,
        source_h5=args.source_h5,
        output_dir=args.output_dir,
        episode_ids=episode_ids,
        target_control_mode=args.target_control_mode,
        substeps_per_source_step=args.substeps_per_source_step,
        num_workers=args.num_workers,
    )
    print(json.dumps(_jsonable(manifest["summary"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
