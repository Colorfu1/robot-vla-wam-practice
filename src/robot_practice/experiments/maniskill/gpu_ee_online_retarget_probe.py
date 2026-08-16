"""Probe online EEF retargeting for GPU replay of a CPU EE trajectory.

The input trajectory is a CPU replay in native ``pd_ee_delta_pose`` mode.  Its
recorded environment states are used only to recover the CPU reference TCP
path.  The target rollout is then executed on a batched GPU environment.  At
each step the target pose is compared with the *current GPU TCP pose* and a
fresh delta action is sent to the native GPU controller.

This is a diagnostic, not a data conversion path.  It answers whether the
GPU failures come primarily from open-loop EEF drift or from the GPU
controller/physics contract itself.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import practice_root, write_json_atomic

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pose"
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


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(practice_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return value


def _scalar_bool(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def _pose_array(pose: Any) -> np.ndarray:
    position = np.asarray(pose.p[0].detach().cpu().numpy(), dtype=np.float32).reshape(3)
    quaternion = np.asarray(pose.q[0].detach().cpu().numpy(), dtype=np.float32).reshape(4)
    return np.concatenate((position, quaternion))


def _state_list(group: h5py.Group) -> list[dict[str, Any]]:
    from mani_skill.trajectory import utils as trajectory_utils

    return trajectory_utils.dict_to_list_of_dicts(group["env_states"])


def _episode_ids_from_failed_metadata(path: Path) -> list[int]:
    episodes = _read_json(path).get("episodes")
    if not isinstance(episodes, list):
        raise TypeError(f"Missing episodes list in {path}")
    return [int(row["episode_id"]) for row in episodes if not bool(row.get("success", False))]


def _reference_paths(
    replay_h5: Path,
    episode_ids: Sequence[int],
    *,
    task_id: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Read CPU states and recover their TCP pose path without stepping physics."""

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    metadata = _read_json(replay_h5.with_suffix(".json"))
    env_kwargs = dict(metadata["env_info"]["env_kwargs"])
    env_kwargs.update(
        obs_mode="none",
        control_mode=CONTROL_MODE,
        sim_backend="physx_cpu",
        render_mode=None,
        reward_mode="sparse",
        num_envs=1,
    )
    cpu_env = gym.make(task_id, **env_kwargs)
    result: dict[int, dict[str, Any]] = {}
    try:
        cpu_env.reset(seed=0)
        with h5py.File(replay_h5, "r") as source_file:
            for episode_id in episode_ids:
                group = source_file[f"traj_{episode_id}"]
                actions = np.asarray(group["actions"], dtype=np.float32)
                states = _state_list(group)
                if len(states) != len(actions) + 1:
                    raise ValueError(
                        f"Episode {episode_id}: expected actions+1 states, "
                        f"got actions={len(actions)} states={len(states)}"
                    )
                poses = []
                for state in states:
                    cpu_env.unwrapped.set_state_dict(state)
                    arm = cpu_env.unwrapped.agent.controller.controllers["arm"]
                    poses.append(_pose_array(arm.ee_pose_at_base))
                result[episode_id] = {
                    "episode_id": episode_id,
                    "actions": actions,
                    "initial_state": states[0],
                    "reference_tcp_pose": np.asarray(poses, dtype=np.float32),
                    "source_success": bool(np.asarray(group["success"]).reshape(-1).any()),
                }
    finally:
        cpu_env.close()
    return result, metadata


def _batch_states(states: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from mani_skill.trajectory import utils as trajectory_utils

    return trajectory_utils.list_of_dicts_to_dict(list(states))


def _run_gpu_phase(
    references: dict[int, dict[str, Any]],
    metadata: dict[str, Any],
    selected_ids: list[int],
    task_id: str,
    num_envs: int,
    result_queue: Any,
) -> None:
    """Run GPU simulation in a fresh process before any CPU PhysX is created."""

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import torch
    from mani_skill.utils import common
    from mani_skill.utils.geometry import rotation_conversions
    from mani_skill.utils.structs import Pose

    env_kwargs = dict(metadata["env_info"]["env_kwargs"])
    env_kwargs.update(
        obs_mode="none",
        control_mode=CONTROL_MODE,
        sim_backend="physx_cuda",
        render_mode=None,
        reward_mode="sparse",
        num_envs=min(num_envs, len(selected_ids)),
    )
    episode_records = {
        int(record["episode_id"]): record for record in metadata["episodes"]
    }
    rows: list[dict[str, Any]] = []
    try:
        batch_size = int(env_kwargs["num_envs"])
        env = gym.make(task_id, **env_kwargs)
        try:
            arm = env.unwrapped.agent.controller.controllers["arm"]
            for start in range(0, len(selected_ids), batch_size):
                batch_ids = selected_ids[start : start + batch_size]
                batch = [references[episode_id] for episode_id in batch_ids]
                real_batch_size = len(batch)
                lengths = np.asarray([len(item["actions"]) for item in batch], dtype=np.int64)
                max_length = int(lengths.max())
                pad = batch_size - real_batch_size
                if pad:
                    batch_ids_padded = batch_ids + [batch_ids[-1]] * pad
                    batch_padded = batch + [batch[-1]] * pad
                    lengths_padded = np.concatenate((lengths, np.zeros(pad, dtype=np.int64)))
                else:
                    batch_ids_padded = batch_ids
                    batch_padded = batch
                    lengths_padded = lengths

                seeds = [int(episode_records[episode_id]["episode_seed"]) for episode_id in batch_ids]
                if pad:
                    seeds += [seeds[-1]] * pad
                env.reset(seed=torch.as_tensor(seeds, device=env.unwrapped.device))
                env.unwrapped.set_state_dict(
                    _batch_states([item["initial_state"] for item in batch_padded])
                )
                success_once = np.zeros(batch_size, dtype=bool)
                first_success = [None] * batch_size
                final_success = np.zeros(batch_size, dtype=bool)
                position_clips = np.zeros(batch_size, dtype=np.int64)
                rotation_clips = np.zeros(batch_size, dtype=np.int64)
                tracking_errors: list[list[float]] = [[] for _ in range(batch_size)]

                for step in range(max_length):
                    active = lengths_padded > step
                    target_pose = np.stack(
                        [
                            item["reference_tcp_pose"][
                                min(step + 1, len(item["reference_tcp_pose"]) - 1)
                            ]
                            for item in batch_padded
                        ],
                        axis=0,
                    )
                    source_gripper = np.asarray(
                        [
                            item["actions"][step, 6] if active[index] else 0.0
                            for index, item in enumerate(batch_padded)
                        ],
                        dtype=np.float32,
                    )
                    action, pos_clip, rot_clip = _online_delta_action(
                        arm.ee_pose_at_base,
                        target_pose,
                        source_gripper,
                        arm,
                        common=common,
                        rotation_conversions=rotation_conversions,
                        pose_cls=Pose,
                    )
                    action = torch.where(
                        torch.as_tensor(active, device=action.device).reshape(-1, 1),
                        action,
                        torch.zeros_like(action),
                    )
                    _, _, _, _, info = env.step(action)
                    success = info.get(
                        "success",
                        torch.zeros(batch_size, dtype=torch.bool, device=action.device),
                    )
                    success_np = np.asarray(success.detach().cpu().numpy(), dtype=bool).reshape(-1)
                    actual = arm.ee_pose_at_base
                    actual_position = actual.p.detach().cpu().numpy()
                    for index, is_active in enumerate(active):
                        if not is_active:
                            continue
                        if success_np[index]:
                            success_once[index] = True
                            if first_success[index] is None:
                                first_success[index] = step + 1
                        if step == lengths_padded[index] - 1:
                            final_success[index] = bool(success_np[index])
                        tracking_errors[index].append(
                            float(np.linalg.norm(actual_position[index] - target_pose[index, :3]))
                        )
                    position_clips += pos_clip.detach().cpu().numpy().astype(np.int64) * active
                    rotation_clips += rot_clip.detach().cpu().numpy().astype(np.int64) * active

                rows.extend(
                    {
                        "episode_id": episode_id,
                        "source_success": bool(batch[index]["source_success"]),
                        "source_action_steps": int(lengths[index]),
                        "success_once": bool(success_once[index]),
                        "success_at_end": bool(final_success[index]),
                        "first_success_step": first_success[index],
                        "position_clip_count": int(position_clips[index]),
                        "rotation_clip_count": int(rotation_clips[index]),
                        "mean_tracking_error_m": float(np.mean(tracking_errors[index]))
                        if tracking_errors[index]
                        else None,
                        "max_tracking_error_m": float(np.max(tracking_errors[index]))
                        if tracking_errors[index]
                        else None,
                    }
                    for index, episode_id in enumerate(batch_ids)
                )
        finally:
            env.close()
        result_queue.put({"ok": True, "rows": rows})
    except BaseException as exc:
        result_queue.put({"ok": False, "error": repr(exc)})


def _online_delta_action(
    current_pose: Any,
    target_pose_array: np.ndarray,
    source_gripper: Any,
    arm: Any,
    *,
    common: Any,
    rotation_conversions: Any,
    pose_cls: Any,
) -> tuple[Any, Any, Any]:
    """Build normalized GPU EE actions from current and target TCP poses."""

    import torch

    device = arm.device
    target = common.to_tensor(target_pose_array, device=device).to(dtype=torch.float32)
    desired_pose = pose_cls.create_from_pq(target[:, :3], target[:, 3:])
    delta_pose = desired_pose * current_pose.inv()
    delta_position = delta_pose.p
    delta_euler = rotation_conversions.matrix_to_euler_angles(
        rotation_conversions.quaternion_to_matrix(delta_pose.q), "XYZ"
    )

    low = arm.action_space_low[:3]
    high = arm.action_space_high[:3]
    position_action = (delta_position - 0.5 * (high + low)) / (0.5 * (high - low))
    position_clipped = torch.any(torch.abs(position_action) > 1.0, dim=1)
    position_action = torch.clamp(position_action, -1.0, 1.0)

    rot_lower = common.to_tensor(
        np.broadcast_to(float(arm.config.rot_lower), (3,)),
        device=device,
    ).to(dtype=torch.float32)
    rotation_action = delta_euler / rot_lower
    rotation_norm = torch.linalg.norm(rotation_action, dim=1)
    rotation_clipped = rotation_norm > 1.0
    safe_norm = torch.clamp(rotation_norm, min=1.0).unsqueeze(1)
    rotation_action = rotation_action / safe_norm

    gripper = common.to_tensor(source_gripper, device=device).to(dtype=torch.float32).reshape(-1, 1)
    return (
        torch.cat((position_action, rotation_action, gripper), dim=1),
        position_clipped,
        rotation_clipped,
    )


def _online_delta_pos_action(
    current_pose: Any,
    target_pose_array: np.ndarray,
    source_gripper: Any,
    arm: Any,
    *,
    common: Any,
    pose_cls: Any,
) -> tuple[Any, Any]:
    """Build a normalized native 4D ``pd_ee_delta_pos`` action.

    ``pd_ee_delta_pos`` has no orientation command.  The position target is
    still computed from the current TCP pose, so this helper is suitable for
    the failed-episode repair route without pretending that a 7D label was
    projected down to four coordinates.
    """

    import torch

    device = arm.device
    target = common.to_tensor(target_pose_array, device=device).to(dtype=torch.float32)
    desired_pose = pose_cls.create_from_pq(target[:, :3], target[:, 3:])
    delta_position = (desired_pose * current_pose.inv()).p
    low = arm.action_space_low[:3]
    high = arm.action_space_high[:3]
    position_action = (delta_position - 0.5 * (high + low)) / (0.5 * (high - low))
    position_clipped = torch.any(torch.abs(position_action) > 1.0, dim=1)
    position_action = torch.clamp(position_action, -1.0, 1.0)
    gripper = common.to_tensor(source_gripper, device=device).to(dtype=torch.float32).reshape(-1, 1)
    return torch.cat((position_action, gripper), dim=1), position_clipped


def run_probe(
    *,
    replay_h5: str | Path,
    failed_metadata: str | Path,
    output_dir: str | Path,
    task_id: str = TASK_ID,
    num_envs: int = 64,
    episode_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    if task_id != TASK_ID:
        raise ValueError(f"Only {TASK_ID} is supported")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    replay_h5 = Path(replay_h5).expanduser().resolve()
    failed_metadata = Path(failed_metadata).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not replay_h5.is_file() or not replay_h5.with_suffix(".json").is_file():
        raise FileNotFoundError(f"Replay H5/JSON pair is missing: {replay_h5}")
    if not failed_metadata.is_file():
        raise FileNotFoundError(f"Failed-episode metadata is missing: {failed_metadata}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = list(episode_ids) if episode_ids is not None else _episode_ids_from_failed_metadata(failed_metadata)
    selected_ids = [int(value) for value in selected_ids]
    if not selected_ids:
        raise ValueError("No failed episode ids selected")

    references, metadata = _reference_paths(replay_h5, selected_ids, task_id=task_id)
    missing = [episode_id for episode_id in selected_ids if episode_id not in references]
    if missing:
        raise KeyError(f"Missing episode ids in replay: {missing}")

    context = __import__("multiprocessing").get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_gpu_phase,
        args=(references, metadata, selected_ids, task_id, num_envs, result_queue),
    )
    process.start()
    child_result = result_queue.get()
    process.join()
    if process.exitcode != 0 or not child_result.get("ok"):
        raise RuntimeError(
            f"GPU online retarget phase failed (exitcode={process.exitcode}): "
            f"{child_result.get('error', 'unknown error')}"
        )
    rows = child_result["rows"]

    rows.sort(key=lambda row: selected_ids.index(int(row["episode_id"])))
    result = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "task": task_id,
        "control_mode": CONTROL_MODE,
        "source_replay_h5": _relative(replay_h5),
        "failed_metadata": _relative(failed_metadata),
        "target_backend": "physx_cuda",
        "num_envs": int(num_envs),
        "reference_contract": {
            "reference": "CPU replay env_states -> actual TCP pose",
            "online_target": "current GPU TCP pose -> next CPU TCP pose",
            "position": "root-frame delta, normalized and clipped by native controller bounds",
            "rotation": "root-aligned body rotation, XYZ Euler delta, normalized and norm-clipped",
            "gripper": "copied from source 7D action",
        },
        "summary": {
            "episodes": len(rows),
            "success_once": sum(bool(row["success_once"]) for row in rows),
            "success_at_end": sum(bool(row["success_at_end"]) for row in rows),
            "success_once_rate": sum(bool(row["success_once"]) for row in rows) / len(rows),
            "success_at_end_rate": sum(bool(row["success_at_end"]) for row in rows) / len(rows),
            "mean_tracking_error_m": float(
                np.mean([row["mean_tracking_error_m"] for row in rows])
            ),
            "gate": "diagnostic_only; no formal promotion gate",
        },
        "episodes": rows,
    }
    write_json_atomic(output_dir / "probe_manifest.json", _jsonable(result))
    write_json_atomic(output_dir / "episode_diagnostics.json", _jsonable({"episodes": rows}))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--failed-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-ids", help="Optional comma-separated ids; default: failed metadata")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episode_ids = None
    if args.episode_ids:
        episode_ids = [int(value) for value in args.episode_ids.split(",") if value.strip()]
    result = run_probe(
        replay_h5=args.replay_h5,
        failed_metadata=args.failed_metadata,
        output_dir=args.output_dir,
        task_id=args.task,
        num_envs=args.num_envs,
        episode_ids=episode_ids,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
