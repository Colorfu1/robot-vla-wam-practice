"""Diagnostic conversion probes for the ManiSkill PegInsertionSide demos.

The M1 replay gate exposed a conversion-specific failure mode: the source
demonstrations are joint-position trajectories, while the downstream contract
uses an end-effector controller.  This module intentionally does not modify
the installed ManiSkill converter.  It provides two isolated, remote-only
probes:

``fixed_delta``
    Convert source target-qpos to an EEF target and converge to it with a
    corrected, explicitly documented delta-rotation contract.

``absolute``
    Convert source target-qpos to an absolute EEF pose and send it to
    ``pd_ee_pose``.  There is no delta accumulation in this path.

Importing this module is side-effect free.  ManiSkill environments are only
constructed inside worker processes started by :func:`run_probe`; local users
can therefore run the pure math tests without starting replay or rollout.
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

TASK_ID = "PegInsertionSide-v1"
SOURCE_CONTROL_MODE = "pd_joint_pos"
FIXED_DELTA_MODE = "fixed_delta"
ABSOLUTE_MODE = "absolute"
MODES = (FIXED_DELTA_MODE, ABSOLUTE_MODE)
DEFAULT_SOURCE_H5 = Path(
    "datasets/raw/maniskill_demos/PegInsertionSide-v1/motionplanning/trajectory.h5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/maniskill/m1_protocol_audit/PegInsertionSide-v1/conversion_probe"
)
DEFAULT_POSITION_TOLERANCE = 0.002
DEFAULT_ROTATION_TOLERANCE = 0.01
DEFAULT_MAX_INNER_STEPS = 16
SCHEMA_VERSION = 1


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
    """Write JSON with a copy fallback for cross-device output roots."""

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


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(practice_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Invalid quaternion: {quaternion!r}")
    value = value / norm
    # q and -q represent the same rotation.  Pick the short-arc hemisphere.
    if value[0] < 0:
        value = -value
    return value


def quaternion_inverse(quaternion: Sequence[float]) -> np.ndarray:
    value = _normalize_quaternion(quaternion)
    return np.asarray([value[0], -value[1], -value[2], -value[3]], dtype=np.float64)


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    a = _normalize_quaternion(left)
    b = _normalize_quaternion(right)
    return np.asarray(
        [
            a[0] * b[0] - np.dot(a[1:], b[1:]),
            a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3],
            a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1],
        ],
        dtype=np.float64,
    )


def compact_axis_angle(quaternion: Sequence[float]) -> np.ndarray:
    """Return a short-arc axis-angle vector for a wxyz quaternion."""

    value = _normalize_quaternion(quaternion)
    vector = value[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(value[0]))
    return vector / vector_norm * angle


def pose_error(
    desired_position: Sequence[float],
    desired_quaternion: Sequence[float],
    current_position: Sequence[float],
    current_quaternion: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return position error, rotation-vector error, and their norms."""

    position_error = np.asarray(desired_position, dtype=np.float64).reshape(3) - np.asarray(
        current_position, dtype=np.float64
    ).reshape(3)
    delta_quaternion = quaternion_multiply(
        desired_quaternion, quaternion_inverse(current_quaternion)
    )
    rotation_error = compact_axis_angle(delta_quaternion)
    return (
        position_error,
        rotation_error,
        float(np.linalg.norm(position_error)),
        float(np.linalg.norm(rotation_error)),
    )


def fixed_delta_action(
    desired_position: Sequence[float],
    desired_quaternion: Sequence[float],
    current_position: Sequence[float],
    current_quaternion: Sequence[float],
    *,
    position_bound: float = 0.1,
    rotation_bound: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a corrected normalized ``pd_ee_delta_pose`` action.

    ManiSkill's Panda ``PDEEPoseController`` currently maps normalized
    rotation actions with ``physical_rotvec = action * rot_lower`` and the
    default ``rot_lower`` is negative.  Therefore the normalized action must
    contain ``-desired_rotvec / abs(rot_lower)`` to produce the desired
    physical rotation.  The desired relative quaternion is
    ``q_target * inverse(q_current)`` because the controller applies the delta
    on the left.  Keeping both parts here makes the convention explicit and
    prevents a future helper from silently reintroducing the inverse sign.
    """

    if position_bound <= 0 or rotation_bound <= 0:
        raise ValueError("position_bound and rotation_bound must be positive")
    position_error, rotation_error, position_norm, rotation_norm = pose_error(
        desired_position,
        desired_quaternion,
        current_position,
        current_quaternion,
    )
    position_action = position_error / float(position_bound)
    position_clipped = bool(np.any(np.abs(position_action) > 1.0))
    position_action = np.clip(position_action, -1.0, 1.0)

    # The negative sign is the controller's rot_lower sign, not an arbitrary
    # quaternion-order tweak.  It is part of the physical action contract.
    rotation_action = -rotation_error / float(rotation_bound)
    rotation_clipped = bool(np.linalg.norm(rotation_action) > 1.0)
    rotation_action_norm = float(np.linalg.norm(rotation_action))
    if rotation_action_norm > 1.0:
        rotation_action = rotation_action / rotation_action_norm

    action = np.concatenate((position_action, rotation_action)).astype(np.float32)
    return action, {
        "position_error_m": position_norm,
        "rotation_error_rad": rotation_norm,
        "position_error_vector_m": position_error,
        "rotation_error_vector_rad": rotation_error,
        "position_clipped": position_clipped,
        "rotation_clipped": rotation_clipped,
        "rotation_contract": "physical_rotvec=-normalized_action*abs(rot_lower)",
        "relative_quaternion_contract": "q_target*inverse(q_current)",
    }


def _scalar_bool(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _scalar_int(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return int(np.asarray(value).reshape(-1)[0])


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return value


def _metadata_records(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Source metadata must contain an episodes list")
    result: dict[int, dict[str, Any]] = {}
    for record in episodes:
        if not isinstance(record, Mapping):
            raise TypeError("Source episode metadata must be objects")
        episode_id = int(record["episode_id"])
        if episode_id in result:
            raise ValueError(f"Duplicate source episode id: {episode_id}")
        result[episode_id] = dict(record)
    return result


def _reset_kwargs(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("reset_kwargs", {})
    if not isinstance(value, Mapping):
        value = {}
    result = copy.deepcopy(dict(value))
    seed = record.get("episode_seed", result.get("seed"))
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(f"Expected one CPU seed, got {seed!r}")
        seed = seed[0]
    if seed is None:
        raise ValueError(f"Source episode {record.get('episode_id')} has no seed")
    result["seed"] = int(seed)
    return result


def _state_list(group: h5py.Group) -> list[dict[str, Any]]:
    from mani_skill.trajectory import utils as trajectory_utils

    if "env_states" not in group:
        raise RuntimeError(f"{group.name} has no env_states")
    return trajectory_utils.dict_to_list_of_dicts(group["env_states"])


def _source_success(group: h5py.Group, record: Mapping[str, Any]) -> bool:
    if "success" in group:
        values = np.asarray(group["success"])
        if values.size:
            return bool(values.reshape(-1)[-1])
    return bool(record.get("success", False))


def _replace_initial_recording(recorder: Any, state: Mapping[str, Any]) -> None:
    """Keep RecordEpisode's initial state/observation aligned after set_state_dict."""

    from mani_skill.utils import common

    if recorder._trajectory_buffer is None:
        return

    def recursive_replace(target: Any, replacement: Any) -> None:
        if isinstance(target, np.ndarray):
            target[-1, :] = replacement[-1, :]
        else:
            for key in target:
                recursive_replace(target[key], replacement[key])

    if recorder.record_env_state:
        recursive_replace(recorder._trajectory_buffer.state, common.batch(state))
    fixed_observation = recorder.base_env.get_obs()
    recursive_replace(
        recorder._trajectory_buffer.observation,
        common.to_numpy(common.batch(fixed_observation)),
    )


def _desired_world_pose(
    ori_controller: Any, ori_arm: Any, target_arm: Any, pin_model: Any
) -> Any:
    """FK the source controller's target qpos into a world-frame EEF pose."""

    from mani_skill.utils.structs import Pose

    full_qpos = ori_controller.articulation.get_qpos().clone()
    full_qpos[:, ori_arm.active_joint_indices] = ori_arm._target_qpos
    pin_model.compute_forward_kinematics(full_qpos.detach().cpu().numpy()[0])
    link_pose = pin_model.get_link_pose(target_arm.ee_link.index)
    return Pose.create(ori_controller.articulation.pose.sp * link_pose)


def _action_dict_for_target(
    target_controller: Any,
    source_action_dict: Mapping[str, Any],
    arm_action: np.ndarray,
) -> Any:
    from mani_skill.utils import common

    values: dict[str, Any] = {}
    for key in target_controller.controllers:
        if key == "arm":
            values[key] = arm_action
        elif key in source_action_dict:
            values[key] = source_action_dict[key]
        else:
            raise KeyError(f"Target controller key {key!r} is missing from source action")
    device = target_controller.controllers["arm"].device
    return target_controller.from_action_dict(
        {key: common.to_tensor(value, device=device) for key, value in values.items()}
    )


def _base_pose_components(
    target_arm: Any, desired_world_pose: Any
) -> tuple[np.ndarray, np.ndarray]:
    desired_base_pose = target_arm.root_link.pose.inv() * desired_world_pose
    position = desired_base_pose.p[0].detach().cpu().numpy().astype(np.float64)
    quaternion = desired_base_pose.q[0].detach().cpu().numpy().astype(np.float64)
    return position, quaternion


def _absolute_arm_action(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    from mani_skill.utils import common
    from mani_skill.utils.geometry import rotation_conversions

    quaternion_tensor = common.to_tensor(quaternion[None, :], device="cpu")
    matrix = rotation_conversions.quaternion_to_matrix(quaternion_tensor)
    euler = rotation_conversions.matrix_to_euler_angles(matrix, "XYZ")[0].detach().cpu().numpy()
    return np.concatenate((position, euler)).astype(np.float32)


def _worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one source episode subset in one CPU process."""

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils import common
    from mani_skill.utils.wrappers import RecordEpisode

    mode = str(spec["mode"])
    source_h5 = Path(str(spec["source_h5"]))
    worker_dir = Path(str(spec["worker_dir"]))
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_manifest = worker_dir / "worker_manifest.json"
    if worker_manifest.is_file():
        previous = _read_json(worker_manifest)
        previous_trajectory = Path(str(previous.get("trajectory", "")))
        if not previous_trajectory.is_absolute():
            previous_trajectory = practice_root() / previous_trajectory
        if previous_trajectory.is_file():
            return previous

    target_control_mode = "pd_ee_delta_pose" if mode == FIXED_DELTA_MODE else "pd_ee_pose"
    with h5py.File(source_h5, "r") as source_file:
        source_metadata = _read_json(source_h5.with_suffix(".json"))
        source_env_kwargs = dict(source_metadata["env_info"]["env_kwargs"])
        source_env_kwargs.update(
            obs_mode="none",
            control_mode=SOURCE_CONTROL_MODE,
            sim_backend="physx_cpu",
            reward_mode="sparse",
            reconfiguration_freq=1,
        )
        target_env_kwargs = dict(source_env_kwargs)
        target_env_kwargs["control_mode"] = target_control_mode
        source_env = gym.make(TASK_ID, **source_env_kwargs)
        target_base_env = gym.make(TASK_ID, **target_env_kwargs)
        recorder = RecordEpisode(
            target_base_env,
            output_dir=str(worker_dir),
            # RecordEpisode appends .h5 itself; passing a suffix here creates
            # the confusing trajectory.h5.h5 filename.
            trajectory_name="trajectory",
            save_trajectory=True,
            save_video=False,
            save_on_reset=False,
            record_reward=True,
            record_env_state=True,
            source_type=f"m1_peg_conversion_probe:{mode}",
            source_desc=(
                "Repo-local diagnostic conversion from official pd_joint_pos demos; "
                f"mode={mode}"
            ),
        )
        rows: list[dict[str, Any]] = []
        try:
            for record in spec["records"]:
                source_episode_id = int(record["episode_id"])
                source_group = source_file[f"traj_{source_episode_id}"]
                source_actions = np.asarray(source_group["actions"], dtype=np.float32)
                states = _state_list(source_group)
                if len(states) < 1:
                    raise RuntimeError(f"Source episode {source_episode_id} has no states")
                reset_kwargs = _reset_kwargs(record)
                source_env.reset(**reset_kwargs)
                recorder.reset(**reset_kwargs)
                source_env.unwrapped.set_state_dict(states[0])
                recorder.base_env.set_state_dict(states[0])
                _replace_initial_recording(recorder, states[0])

                # ManiSkill rebuilds controller instances during reset.  Do
                # not retain the instances created before the episode reset:
                # they keep the pre-reset qpos and would make FK appear frozen
                # even though source_env.step() advances the live controller.
                source_controller = source_env.unwrapped.agent.controller
                target_controller = recorder.base_env.agent.controller
                source_arm = source_controller.controllers["arm"]
                target_arm = target_controller.controllers["arm"]
                pin_model = source_controller.articulation.create_pinocchio_model()

                source_action_count = 0
                target_action_count = 0
                success_once = False
                first_success_target_step: int | None = None
                final_info: Mapping[str, Any] = {}
                inner_steps_by_source_step: list[int] = []
                position_errors: list[float] = []
                rotation_errors: list[float] = []
                position_clips = 0
                rotation_clips = 0
                max_inner_reached = 0

                for source_action in source_actions:
                    source_action = np.asarray(source_action, dtype=np.float32).reshape(-1)
                    source_action_dict = source_controller.to_action_dict(source_action)
                    source_env.step(
                        common.to_tensor(source_action, device=source_env.unwrapped.device)
                    )
                    desired_world_pose = _desired_world_pose(
                        source_controller, source_arm, target_arm, pin_model
                    )
                    desired_position, desired_quaternion = _base_pose_components(
                        target_arm, desired_world_pose
                    )
                    source_action_count += 1
                    inner_steps = 0
                    converged = False
                    inner_limit = 1 if mode == ABSOLUTE_MODE else int(spec["max_inner_steps"])
                    for _ in range(inner_limit):
                        current_pose = target_arm.ee_pose_at_base
                        current_position = current_pose.p[0].detach().cpu().numpy()
                        current_quaternion = current_pose.q[0].detach().cpu().numpy()
                        if mode == FIXED_DELTA_MODE:
                            arm_action, action_info = fixed_delta_action(
                                desired_position,
                                desired_quaternion,
                                current_position,
                                current_quaternion,
                                position_bound=float(spec["position_bound"]),
                                rotation_bound=float(spec["rotation_bound"]),
                            )
                            position_clips += int(action_info["position_clipped"])
                            rotation_clips += int(action_info["rotation_clipped"])
                        else:
                            arm_action = _absolute_arm_action(
                                desired_position, desired_quaternion
                            )
                            _, _, before_position_error, before_rotation_error = pose_error(
                                desired_position,
                                desired_quaternion,
                                current_position,
                                current_quaternion,
                            )
                            action_info = {
                                "position_error_m": before_position_error,
                                "rotation_error_rad": before_rotation_error,
                                "position_clipped": False,
                                "rotation_clipped": False,
                            }
                        action = _action_dict_for_target(
                            target_controller, source_action_dict, arm_action
                        )
                        _, _, _, _, final_info = recorder.step(action)
                        target_action_count += 1
                        inner_steps += 1
                        next_pose = target_arm.ee_pose_at_base
                        next_position = next_pose.p[0].detach().cpu().numpy()
                        next_quaternion = next_pose.q[0].detach().cpu().numpy()
                        _, _, after_position_error, after_rotation_error = pose_error(
                            desired_position,
                            desired_quaternion,
                            next_position,
                            next_quaternion,
                        )
                        position_errors.append(after_position_error)
                        rotation_errors.append(after_rotation_error)
                        if _scalar_bool(final_info.get("success")):
                            success_once = True
                            if first_success_target_step is None:
                                first_success_target_step = target_action_count
                        if mode == FIXED_DELTA_MODE and (
                            after_position_error <= float(spec["position_tolerance"])
                            and after_rotation_error <= float(spec["rotation_tolerance"])
                        ):
                            converged = True
                            break
                    inner_steps_by_source_step.append(inner_steps)
                    if mode == FIXED_DELTA_MODE and not converged:
                        max_inner_reached += 1

                recorder.flush_trajectory(ignore_empty_transition=True)
                final_success = _scalar_bool(final_info.get("success"))
                rows.append(
                    {
                        "source_episode_id": source_episode_id,
                        "source_seed": int(reset_kwargs["seed"]),
                        "source_success": _source_success(source_group, record),
                        "source_action_steps": int(source_action_count),
                        "target_action_steps": int(target_action_count),
                        "success_once": success_once,
                        "success_at_end": final_success,
                        "first_success_target_step": first_success_target_step,
                        "inner_steps_by_source_step": inner_steps_by_source_step,
                        "position_clip_count": int(position_clips),
                        "rotation_clip_count": int(rotation_clips),
                        "max_inner_reached_count": int(max_inner_reached),
                        "max_position_error_m": max(position_errors, default=0.0),
                        "max_rotation_error_rad": max(rotation_errors, default=0.0),
                        "final_position_error_m": position_errors[-1]
                        if position_errors
                        else None,
                        "final_rotation_error_rad": rotation_errors[-1]
                        if rotation_errors
                        else None,
                    }
                )
        finally:
            recorder.close()
            source_env.close()

    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "worker_id": int(spec["worker_id"]),
        "trajectory": _relative(worker_dir / "trajectory.h5"),
        "episodes": rows,
    }
    _write_json(worker_manifest, result)
    return result


def _worker_entry(spec: Mapping[str, Any]) -> dict[str, Any]:
    return _worker(spec)


def _episode_chunks(records: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    if count <= 0:
        raise ValueError("count must be positive")
    worker_count = min(count, len(records))
    return [
        records[start : min(start + math.ceil(count / worker_count), count)]
        for start in range(0, count, math.ceil(count / worker_count))
    ]


def _summary(rows: Sequence[Mapping[str, Any]], *, mode: str, count: int) -> dict[str, Any]:
    final_successes = sum(bool(row["success_at_end"]) for row in rows)
    once_successes = sum(bool(row["success_once"]) for row in rows)
    return {
        "episodes": len(rows),
        "requested_episodes": count,
        "success_at_end": {"num_success": final_successes, "rate": final_successes / count},
        "success_once": {"num_success": once_successes, "rate": once_successes / count},
        "source_successes": sum(bool(row["source_success"]) for row in rows),
        "target_action_steps": sum(int(row["target_action_steps"]) for row in rows),
        "source_action_steps": sum(int(row["source_action_steps"]) for row in rows),
        "rotation_clip_events": sum(int(row["rotation_clip_count"]) for row in rows),
        "position_clip_events": sum(int(row["position_clip_count"]) for row in rows),
        "mode": mode,
        "gate": "success_at_end == requested_episodes",
        "gate_pass": len(rows) == count and final_successes == count,
    }


def run_probe(
    *,
    mode: str,
    source_h5: str | Path = DEFAULT_SOURCE_H5,
    output_dir: str | Path | None = None,
    count: int = 10,
    num_workers: int = 1,
    max_inner_steps: int = DEFAULT_MAX_INNER_STEPS,
    position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
    rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE,
    position_bound: float = 0.1,
    rotation_bound: float = 0.1,
) -> dict[str, Any]:
    """Run or resume one isolated B/C probe.

    This function creates ManiSkill environments in spawned worker processes and
    must only be called on the remote ManiSkill machine.  It is intentionally
    not used by the local unit tests.
    """

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    source_h5 = Path(source_h5)
    if not source_h5.is_file() or not source_h5.with_suffix(".json").is_file():
        raise FileNotFoundError(f"Source H5/JSON pair is missing: {source_h5}")
    metadata = _read_json(source_h5.with_suffix(".json"))
    records_by_id = _metadata_records(metadata)
    ordered_records = [records_by_id[key] for key in sorted(records_by_id)]
    count = min(int(count), len(ordered_records))
    if count <= 0:
        raise ValueError("No source episodes available")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if max_inner_steps <= 0:
        raise ValueError("max_inner_steps must be positive")
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = output_dir / "probe_manifest.json"
    if final_manifest_path.is_file():
        return _read_json(final_manifest_path)

    records = [dict(record) for record in ordered_records[:count]]
    chunks = _episode_chunks(records, min(int(num_workers), count))
    workers_root = output_dir / "workers"
    workers_root.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "mode": mode,
            "source_h5": str(source_h5),
            "worker_id": worker_id,
            "worker_dir": str(workers_root / f"worker_{worker_id:02d}"),
            "records": chunk,
            "max_inner_steps": int(max_inner_steps),
            "position_tolerance": float(position_tolerance),
            "rotation_tolerance": float(rotation_tolerance),
            "position_bound": float(position_bound),
            "rotation_bound": float(rotation_bound),
        }
        for worker_id, chunk in enumerate(chunks)
    ]

    worker_results: list[dict[str, Any]] = []
    pending_specs = []
    for spec in specs:
        manifest = Path(spec["worker_dir"]) / "worker_manifest.json"
        if manifest.is_file():
            previous = _read_json(manifest)
            previous_trajectory = Path(str(previous.get("trajectory", "")))
            if not previous_trajectory.is_absolute():
                previous_trajectory = practice_root() / previous_trajectory
            if previous_trajectory.is_file():
                worker_results.append(previous)
                continue
        pending_specs.append(spec)
    if pending_specs:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(pending_specs)) as pool:
            from tqdm import tqdm

            worker_results.extend(
                tqdm(
                    pool.imap(_worker_entry, pending_specs),
                    total=len(pending_specs),
                    desc=f"Peg {mode} workers",
                    unit="worker",
                )
            )
    worker_results.sort(key=lambda result: int(result["worker_id"]))
    if len(worker_results) != len(specs):
        raise RuntimeError("Not all conversion probe workers completed")

    from mani_skill.trajectory.merge_trajectory import merge_trajectories

    final_h5 = output_dir / "replay.h5"
    if final_h5.exists() or final_h5.with_suffix(".json").exists():
        raise FileExistsError(
            f"Incomplete final probe output exists at {final_h5}; remove only that incomplete "
            "probe directory before retrying"
        )
    worker_paths = [
        Path(result["trajectory"])
        if Path(result["trajectory"]).is_absolute()
        else practice_root() / result["trajectory"]
        for result in worker_results
    ]
    merge_trajectories(str(final_h5), [str(path) for path in worker_paths])

    rows: list[dict[str, Any]] = []
    output_episode_id = 0
    for result in worker_results:
        for row in result["episodes"]:
            rows.append({**row, "output_episode_id": output_episode_id})
            output_episode_id += 1
    summary = _summary(rows, mode=mode, count=count)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "mode": mode,
        "source_h5": _relative(source_h5),
        "source_metadata": _relative(source_h5.with_suffix(".json")),
        "target_control_mode": (
            "pd_ee_delta_pose" if mode == FIXED_DELTA_MODE else "pd_ee_pose"
        ),
        "conversion_contract": {
            "source_control_mode": SOURCE_CONTROL_MODE,
            "backend": "physx_cpu",
            "control_frequency_hz": 20,
            "fixed_delta_rotation": (
                "q_target*inverse(q_current); physical_rotvec=-normalized_action*abs(rot_lower)"
            ),
            "fixed_delta_inner_loop": {
                "max_inner_steps": max_inner_steps,
                "position_tolerance_m": position_tolerance,
                "rotation_tolerance_rad": rotation_tolerance,
            },
            "absolute": "source controller target qpos -> Pinocchio FK -> pd_ee_pose",
        },
        "workers": len(worker_results),
        "replay_h5": _relative(final_h5),
        "replay_json": _relative(final_h5.with_suffix(".json")),
        "summary": summary,
        "episodes": rows,
    }
    _write_json(output_dir / "episode_diagnostics.json", {"episodes": rows})
    _write_json(final_manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_SOURCE_H5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-inner-steps", type=int, default=DEFAULT_MAX_INNER_STEPS)
    parser.add_argument("--position-tolerance", type=float, default=DEFAULT_POSITION_TOLERANCE)
    parser.add_argument("--rotation-tolerance", type=float, default=DEFAULT_ROTATION_TOLERANCE)
    parser.add_argument("--position-bound", type=float, default=0.1)
    parser.add_argument("--rotation-bound", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_probe(
        mode=args.mode,
        source_h5=args.source_h5,
        output_dir=args.output_dir,
        count=args.count,
        num_workers=args.num_workers,
        max_inner_steps=args.max_inner_steps,
        position_tolerance=args.position_tolerance,
        rotation_tolerance=args.rotation_tolerance,
        position_bound=args.position_bound,
        rotation_bound=args.rotation_bound,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=False, indent=2))
    if not result["summary"]["gate_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
