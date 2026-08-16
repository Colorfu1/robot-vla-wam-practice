"""Repair failed replays of an offline ManiSkill3 EE trajectory.

The route accepts a native 7D ``pd_ee_delta_pose`` or native 4D
``pd_ee_delta_pos`` artifact, including one produced by the official CPU
joint-to-EE converter. Episodes that fail when their actions
are replayed on GPU are re-recorded by following the TCP path recovered from
the same artifact's ``env_states``. The reference TCP path is computed with the
GPU ManiSkill controller/FK, while the replacement action is computed from the
current GPU TCP pose. No CPU PhysX environment is stepped during repair.
"""

from __future__ import annotations

import argparse
import errno
import json
import multiprocessing as mp
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .gpu_ee_online_retarget_probe import (
    _batch_states,
    _jsonable,
    _online_delta_action,
    _online_delta_pos_action,
    _read_json,
    _relative,
)
from .state_identity import invert_unique_fingerprints, start_state_fingerprint

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pose"
ACTION_DIM = 7
SCHEMA_VERSION = 1


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            temporary.replace(path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(temporary, path)
            temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata_records(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Replay metadata must contain an episodes list")
    return {int(row["episode_id"]): dict(row) for row in episodes}


def _state_list(group: h5py.Group) -> list[dict[str, Any]]:
    from mani_skill.trajectory import utils as trajectory_utils

    return trajectory_utils.dict_to_list_of_dicts(group["env_states"])


def _replace_initial_recording(recorder: Any, state: Mapping[str, Any]) -> None:
    """Make RecordEpisode's first state/observation match a direct state load."""

    from mani_skill.utils import common

    if recorder._trajectory_buffer is None:
        return

    def replace(target: Any, replacement: Any) -> None:
        if target is None or replacement is None:
            return
        if isinstance(target, Mapping):
            for key in target:
                if key in replacement:
                    replace(target[key], replacement[key])
            return
        if isinstance(target, np.ndarray) or hasattr(target, "shape"):
            target[-1, ...] = replacement[-1, ...]

    replace(recorder._trajectory_buffer.state, common.batch(state))
    replace(
        recorder._trajectory_buffer.observation,
        common.to_numpy(common.batch(recorder.base_env.get_obs())),
    )


def _project_visual_observation(replay_h5: Path) -> None:
    """Keep only visual and proprioceptive observations for the student."""

    with h5py.File(replay_h5, "r+") as handle:
        for group in handle.values():
            observation = group.get("obs")
            if observation is None:
                continue
            for forbidden in ("extra", "sensor_param"):
                if forbidden in observation:
                    del observation[forbidden]


def _remove_seed_identity_fields(record: dict[str, Any]) -> None:
    record.pop("episode_seed", None)
    reset_kwargs = record.get("reset_kwargs")
    if isinstance(reset_kwargs, dict):
        reset_kwargs.pop("seed", None)


def _reference_pose_worker(
    source_h5: str,
    episode_ids: Sequence[int],
    task_id: str,
    control_mode: str,
    result_queue: Any,
) -> None:
    """Recover recorded TCP poses with a GPU FK environment.

    This process only loads recorded states and evaluates the ManiSkill3
    controller's FK.  It never steps CPU PhysX and never converts joint
    actions.  Keeping it isolated also prevents a prior CPU diagnostic from
    contaminating the CUDA context used by the formal route.
    """

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import torch

        metadata = _read_json(Path(source_h5).with_suffix(".json"))
        env_kwargs = dict(metadata["env_info"]["env_kwargs"])
        env_kwargs.update(
            obs_mode="none",
            control_mode=control_mode,
            sim_backend="physx_cuda",
            render_mode=None,
            reward_mode="sparse",
            reconfiguration_freq=1,
            num_envs=len(episode_ids),
        )
        env = gym.make(task_id, **env_kwargs)
        try:
            states_by_id: dict[int, list[dict[str, Any]]] = {}
            with h5py.File(source_h5, "r") as source_file:
                for episode_id in episode_ids:
                    states_by_id[int(episode_id)] = _state_list(
                        source_file[f"traj_{int(episode_id)}"]
                    )
            lengths = {
                episode_id: len(states_by_id[episode_id]) for episode_id in episode_ids
            }
            max_length = max(lengths.values())
            device = env.unwrapped.device
            env.reset(
                seed=torch.arange(len(episode_ids), dtype=torch.int64, device=device)
            )
            poses: dict[int, list[np.ndarray]] = {
                int(episode_id): [] for episode_id in episode_ids
            }
            arm = env.unwrapped.agent.controller.controllers["arm"]
            for step in range(max_length):
                state_batch = [
                    states_by_id[int(episode_id)][
                        min(step, lengths[int(episode_id)] - 1)
                    ]
                    for episode_id in episode_ids
                ]
                env.unwrapped.set_state_dict(_batch_states(state_batch))
                env.unwrapped.scene._gpu_fetch_all()
                position = arm.ee_pose_at_base.p.detach().cpu().numpy()
                quaternion = arm.ee_pose_at_base.q.detach().cpu().numpy()
                pose_batch = np.concatenate((position, quaternion), axis=-1).astype(
                    np.float32
                )
                for index, episode_id in enumerate(episode_ids):
                    if step < lengths[int(episode_id)]:
                        poses[int(episode_id)].append(pose_batch[index])
            result_queue.put(
                {
                    "ok": True,
                    "poses": {
                        str(episode_id): np.asarray(values, dtype=np.float32)
                        for episode_id, values in poses.items()
                    },
                }
            )
        finally:
            env.close()
    except BaseException as exc:
        result_queue.put({"ok": False, "error": repr(exc)})


def _reference_paths_gpu(
    source_h5: Path,
    episode_ids: Sequence[int],
    *,
    task_id: str,
    num_envs: int,
    control_mode: str,
) -> dict[int, np.ndarray]:
    """Recover TCP paths from source states using GPU-only ManiSkill FK."""

    if not episode_ids:
        return {}
    context = mp.get_context("spawn")
    result: dict[int, np.ndarray] = {}
    for start in range(0, len(episode_ids), int(num_envs)):
        batch = [int(value) for value in episode_ids[start : start + int(num_envs)]]
        queue = context.Queue()
        process = context.Process(
            target=_reference_pose_worker,
            args=(str(source_h5), batch, task_id, control_mode, queue),
        )
        process.start()
        payload = queue.get()
        process.join()
        if process.exitcode != 0 or not payload.get("ok"):
            raise RuntimeError(
                f"GPU reference FK failed (exitcode={process.exitcode}): "
                f"{payload.get('error', 'unknown error')}"
            )
        result.update(
            {
                int(episode_id): np.asarray(poses, dtype=np.float32)
                for episode_id, poses in payload["poses"].items()
            }
        )
    return result


def _run_batch(
    references: Mapping[int, Mapping[str, Any]],
    source_metadata: Mapping[str, Any],
    baseline_records: Mapping[int, Mapping[str, Any]],
    baseline_fingerprints: Mapping[int, str],
    source_fingerprints: Mapping[int, str],
    target_source_pairs: Sequence[tuple[int, int]],
    output_dir: str,
    num_envs: int,
    control_mode: str,
    action_dim: int,
    result_queue: Any,
) -> None:
    """Run one replacement batch in a fresh process with a clean GPU context."""

    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import torch
        from mani_skill.utils import common
        from mani_skill.utils.wrappers import RecordEpisode

        pairs = [(int(target), int(source)) for target, source in target_source_pairs]
        if not pairs:
            raise ValueError("Replacement batch must not be empty")
        batch_size = len(pairs)
        if batch_size > int(num_envs):
            raise ValueError(f"Batch has {batch_size} episodes but num_envs={num_envs}")

        env_kwargs = dict(source_metadata["env_info"]["env_kwargs"])
        env_kwargs.update(
            obs_mode="rgb",
            control_mode=control_mode,
            sim_backend="physx_cuda",
            render_mode="rgb_array",
            reward_mode="sparse",
            reconfiguration_freq=1,
            num_envs=batch_size,
        )
        task_id = str(source_metadata["env_info"]["env_id"])
        target_base_env = gym.make(task_id, **env_kwargs)
        recorder = RecordEpisode(
            target_base_env,
            output_dir=output_dir,
            trajectory_name="repair",
            save_trajectory=True,
            save_video=False,
            save_on_reset=False,
            record_reward=True,
            record_env_state=True,
            source_type="m1_w_offline_ee_failed_repair_live",
            source_desc=(
                f"Offline native {action_dim}D EE reference path followed by native GPU "
                f"{control_mode} from the current GPU TCP state; records "
                "the repaired action, success, and live env_states"
            ),
        )
        arm = recorder.base_env.agent.controller.controllers["arm"]
        expected_frame = (
            "root_translation"
            if control_mode == "pd_ee_delta_pos"
            else "root_translation:root_aligned_body_rotation"
        )
        if arm.config.frame != expected_frame:
            raise RuntimeError(f"Unexpected StackCube GPU EE frame: {arm.config.frame}")

        target_ids = [target for target, _ in pairs]
        source_ids = [source for _, source in pairs]
        lengths = np.asarray(
            [len(references[source_id]["actions"]) for source_id in source_ids], dtype=np.int64
        )
        # Seeds only initialize the temporary GPU environments. Exact recorded
        # states are loaded immediately afterwards; no seed is an identity key.
        reset_seeds = list(range(batch_size))
        seed_tensor = torch.as_tensor(
            reset_seeds, dtype=torch.int64, device=recorder.base_env.unwrapped.device
        )
        recorder.reset(seed=seed_tensor)
        initial_state = _batch_states(
            [references[source_id]["initial_state"] for source_id in source_ids]
        )
        recorder.base_env.unwrapped.set_state_dict(initial_state)
        recorder.base_env.unwrapped.scene._gpu_fetch_all()
        _replace_initial_recording(recorder, initial_state)
        arm = recorder.base_env.agent.controller.controllers["arm"]

        success_once = np.zeros(batch_size, dtype=bool)
        first_success: list[int | None] = [None] * batch_size
        final_success = np.zeros(batch_size, dtype=bool)
        tracking_errors: list[list[float]] = [[] for _ in range(batch_size)]
        position_clips = np.zeros(batch_size, dtype=np.int64)
        rotation_clips = np.zeros(batch_size, dtype=np.int64)
        flushed = np.zeros(batch_size, dtype=bool)
        max_length = int(lengths.max())

        for step in range(max_length):
            active = lengths > step
            target_pose = np.stack(
                [
                    references[source_id]["reference_tcp_pose"][
                        min(step + 1, len(references[source_id]["reference_tcp_pose"]) - 1)
                    ]
                    for source_id in source_ids
                ],
                axis=0,
            )
            source_gripper = np.asarray(
                [
                    references[source_id]["actions"][step, action_dim - 1]
                    if active[index]
                    else 0.0
                    for index, source_id in enumerate(source_ids)
                ],
                dtype=np.float32,
            )
            pose_cls = __import__("mani_skill.utils.structs", fromlist=["Pose"]).Pose
            if control_mode == "pd_ee_delta_pos":
                action, pos_clip = _online_delta_pos_action(
                    arm.ee_pose_at_base,
                    target_pose,
                    source_gripper,
                    arm,
                    common=common,
                    pose_cls=pose_cls,
                )
                rot_clip = torch.zeros(batch_size, dtype=torch.bool, device=action.device)
            else:
                action, pos_clip, rot_clip = _online_delta_action(
                    arm.ee_pose_at_base,
                    target_pose,
                    source_gripper,
                    arm,
                    common=common,
                    rotation_conversions=__import__(
                        "mani_skill.utils.geometry", fromlist=["rotation_conversions"]
                    ).rotation_conversions,
                    pose_cls=pose_cls,
                )
            active_tensor = torch.as_tensor(active, device=action.device).reshape(-1, 1)
            action = torch.where(active_tensor, action, torch.zeros_like(action))
            _, _, _, _, info = recorder.step(action)
            success = info.get(
                "success", torch.zeros(batch_size, dtype=torch.bool, device=action.device)
            )
            success_np = np.asarray(common.to_numpy(success), dtype=bool).reshape(-1)
            actual_position = arm.ee_pose_at_base.p.detach().cpu().numpy()
            for index, is_active in enumerate(active):
                if not is_active:
                    continue
                if success_np[index]:
                    success_once[index] = True
                    if first_success[index] is None:
                        first_success[index] = step + 1
                if step == lengths[index] - 1:
                    final_success[index] = bool(success_np[index])
                tracking_errors[index].append(
                    float(np.linalg.norm(actual_position[index] - target_pose[index, :3]))
                )
            position_clips += np.asarray(common.to_numpy(pos_clip), dtype=np.int64) * active
            rotation_clips += np.asarray(common.to_numpy(rot_clip), dtype=np.int64) * active
            to_flush = (step >= lengths - 1) & ~flushed
            if np.any(to_flush):
                indices = np.flatnonzero(to_flush)
                flushed[indices] = True
                recorder.flush_trajectory(env_idxs_to_flush=indices)

        if not np.all(flushed):
            raise RuntimeError(f"Not all repair trajectories were flushed: {flushed.tolist()}")
        recorder.close()

        replay_h5 = Path(output_dir) / "repair.h5"
        replay_json = replay_h5.with_suffix(".json")
        if not replay_h5.is_file() or not replay_json.is_file():
            raise FileNotFoundError(f"RecordEpisode did not create {replay_h5}")
        target_metadata = _read_json(replay_json)
        records = target_metadata.get("episodes", [])
        if len(records) != batch_size:
            raise RuntimeError(f"Repair metadata count {len(records)} != batch size {batch_size}")
        baseline_by_fingerprint = invert_unique_fingerprints(
            {target_id: baseline_fingerprints[target_id] for target_id in target_ids},
            label="GPU baseline repair batch",
        )
        pair_by_target = {target_id: source_id for target_id, source_id in pairs}
        rows = []
        with h5py.File(replay_h5, "r") as repair_file:
            for record in records:
                repair_episode_id = int(record["episode_id"])
                repair_group = repair_file[f"traj_{repair_episode_id}"]
                repair_fingerprint = start_state_fingerprint(repair_group, tolerance=1e-4)
                target_id = baseline_by_fingerprint.get(repair_fingerprint)
                if target_id is None:
                    raise KeyError(
                        f"Repair trajectory {repair_episode_id} has no baseline start-state match: "
                        f"{repair_fingerprint}"
                    )
                source_id = pair_by_target[target_id]
                baseline_fingerprint = baseline_fingerprints[target_id]
                source_fingerprint = source_fingerprints[source_id]
                if not (
                    repair_fingerprint == baseline_fingerprint == source_fingerprint
                ):
                    raise RuntimeError(
                        "Repair identity mismatch: "
                        f"target={target_id}, source={source_id}, "
                        f"repair={repair_fingerprint}, baseline={baseline_fingerprint}, "
                        f"source={source_fingerprint}"
                    )
                batch_index = target_ids.index(target_id)
                success_series = np.asarray(
                    repair_group["success"], dtype=bool
                ).reshape(-1)
                record["source_episode_id"] = source_id
                record["target_baseline_episode_id"] = target_id
                record["source_start_state_fingerprint"] = source_fingerprint
                record["baseline_start_state_fingerprint"] = baseline_fingerprint
                record["repair_start_state_fingerprint"] = repair_fingerprint
                _remove_seed_identity_fields(record)
                rows.append(
                    {
                        "target_episode_id": repair_episode_id,
                        "source_episode_id": source_id,
                        "target_baseline_episode_id": target_id,
                        "source_start_state_fingerprint": source_fingerprint,
                        "baseline_start_state_fingerprint": baseline_fingerprint,
                        "repair_start_state_fingerprint": repair_fingerprint,
                        "source_action_steps": int(lengths[batch_index]),
                        "success_once": bool(np.any(success_series)),
                        "success_at_end": bool(success_series[-1]),
                        "first_success_step": (
                            int(np.flatnonzero(success_series)[0] + 1)
                            if np.any(success_series)
                            else None
                        ),
                        "position_clip_count": int(position_clips[batch_index]),
                        "rotation_clip_count": int(rotation_clips[batch_index]),
                        "mean_tracking_error_m": float(np.mean(tracking_errors[batch_index])),
                        "max_tracking_error_m": float(np.max(tracking_errors[batch_index])),
                    }
                )
        target_metadata["episodes"] = records
        _write_json(replay_json, target_metadata)
        result_queue.put({"ok": True, "rows": rows, "replay_h5": str(replay_h5)})
    except BaseException as exc:
        result_queue.put({"ok": False, "error": repr(exc)})


def _run_batch_isolated(
    references: Mapping[int, Mapping[str, Any]],
    source_metadata: Mapping[str, Any],
    baseline_records: Mapping[int, Mapping[str, Any]],
    baseline_fingerprints: Mapping[int, str],
    source_fingerprints: Mapping[int, str],
    target_source_pairs: Sequence[tuple[int, int]],
    output_dir: Path,
    num_envs: int,
    control_mode: str,
    action_dim: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_run_batch,
        args=(
            references,
            source_metadata,
            baseline_records,
            baseline_fingerprints,
            source_fingerprints,
            list(target_source_pairs),
            str(output_dir),
            int(num_envs),
            str(control_mode),
            int(action_dim),
            queue,
        ),
    )
    process.start()
    result = queue.get()
    process.join()
    if process.exitcode != 0 or not result.get("ok"):
        raise RuntimeError(
            f"GPU EE repair batch failed (exitcode={process.exitcode}): "
            f"{result.get('error', 'unknown error')}"
        )
    return result


def _failed_ids(baseline_h5: Path) -> list[int]:
    with h5py.File(baseline_h5, "r") as handle:
        result = []
        for name in handle:
            values = np.asarray(handle[name]["success"], dtype=bool).reshape(-1)
            if not values.size or not bool(values[-1]):
                result.append(int(name.split("_")[-1]))
    return sorted(result)


def repair(
    *,
    offline_ee_h5: str | Path,
    gpu_replay_h5: str | Path,
    output_dir: str | Path,
    task_id: str = TASK_ID,
    num_envs: int = 100,
    episode_ids: Sequence[int] | None = None,
    require_all_success: bool = True,
) -> dict[str, Any]:
    if task_id != TASK_ID:
        raise ValueError(f"Only {TASK_ID} is supported")
    offline_ee_h5 = Path(offline_ee_h5).expanduser().resolve()
    gpu_replay_h5 = Path(gpu_replay_h5).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (offline_ee_h5, gpu_replay_h5):
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise FileNotFoundError(f"H5/JSON pair is missing: {path}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "repair_manifest.json"
    if manifest_path.is_file():
        return _read_json(manifest_path)

    baseline_metadata = _read_json(gpu_replay_h5.with_suffix(".json"))
    baseline_records = _metadata_records(baseline_metadata)
    selected_targets = list(
        _failed_ids(gpu_replay_h5) if episode_ids is None else map(int, episode_ids)
    )
    missing = [episode_id for episode_id in selected_targets if episode_id not in baseline_records]
    if missing:
        raise KeyError(f"GPU replay has no metadata for failed episode ids: {missing}")
    source_metadata = _read_json(offline_ee_h5.with_suffix(".json"))
    source_env_kwargs = source_metadata.get("env_info", {}).get("env_kwargs", {})
    control_mode = str(source_env_kwargs.get("control_mode", CONTROL_MODE))
    action_dim = 4 if control_mode == "pd_ee_delta_pos" else 7
    if control_mode not in {"pd_ee_delta_pos", "pd_ee_delta_pose"}:
        raise ValueError(f"Unsupported native EE repair control mode: {control_mode!r}")
    if baseline_metadata.get("control_mode") not in {None, control_mode}:
        raise ValueError(
            "GPU replay and offline EE source use different control modes: "
            f"{baseline_metadata.get('control_mode')} vs {control_mode}"
        )
    source_records = _metadata_records(source_metadata)
    source_fingerprints = {}
    with h5py.File(offline_ee_h5, "r") as source_file:
        for name in source_file:
            source_id = int(name.split("_")[-1])
            source_fingerprints[source_id] = start_state_fingerprint(
                source_file[name], tolerance=1e-4
            )
    source_by_fingerprint = invert_unique_fingerprints(
        source_fingerprints, label="offline EE source"
    )
    with h5py.File(offline_ee_h5, "r") as source_file:
        for name, group in source_file.items():
            shape = tuple(group["actions"].shape)
            if len(shape) != 2 or shape[1] != action_dim:
                raise ValueError(
                    f"Offline EE source {name} must use [{action_dim}]D actions, got {shape}"
                )
    baseline_fingerprints: dict[int, str] = {}
    with h5py.File(gpu_replay_h5, "r") as baseline_file:
        source_by_target: dict[int, int] = {}
        for target_id in selected_targets:
            fingerprint = start_state_fingerprint(
                baseline_file[f"traj_{target_id}"], tolerance=1e-4
            )
            baseline_fingerprints[target_id] = fingerprint
            if fingerprint not in source_by_fingerprint:
                raise KeyError(
                    f"Offline EE source has no start-state match for GPU episode {target_id}: "
                    f"{fingerprint}"
                )
            source_by_target[target_id] = source_by_fingerprint[fingerprint]
    target_source_pairs: list[tuple[int, int]] = []
    for target_id in selected_targets:
        source_id = source_by_target[target_id]
        if source_id not in source_records:
            raise KeyError(f"Offline EE source has no matching episode id {source_id}")
        target_source_pairs.append((target_id, source_id))

    if not selected_targets:
        result = {
            "schema_version": SCHEMA_VERSION,
            "task": task_id,
            "offline_ee_source_h5": _relative(offline_ee_h5),
            "gpu_replay_h5": _relative(gpu_replay_h5),
            "selected_target_episode_ids": [],
            "repaired_episode_count": 0,
            "successful_at_end": 0,
            "gate_pass": True,
        }
        _write_json(manifest_path, result)
        return result

    source_ids = [source_id for _, source_id in target_source_pairs]
    reference_poses = _reference_paths_gpu(
        offline_ee_h5,
        source_ids,
        task_id=task_id,
        num_envs=min(int(num_envs), len(source_ids)),
        control_mode=control_mode,
    )
    references: dict[int, dict[str, Any]] = {}
    with h5py.File(offline_ee_h5, "r") as source_file:
        for source_id in source_ids:
            group = source_file[f"traj_{source_id}"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            states = _state_list(group)
            if len(states) != len(actions) + 1:
                raise ValueError(
                    f"Episode {source_id}: expected actions+1 states, "
                    f"got actions={len(actions)} states={len(states)}"
                )
            poses = reference_poses[source_id]
            if len(poses) != len(states):
                raise RuntimeError(
                    f"GPU reference FK length mismatch for episode {source_id}: "
                    f"{len(poses)} vs {len(states)}"
                )
            references[source_id] = {
                "episode_id": source_id,
                "actions": actions,
                "initial_state": states[0],
                "reference_tcp_pose": poses,
                "source_success": bool(np.asarray(group["success"]).reshape(-1).any()),
            }
    workers_root = output_dir / "batches"
    workers_root.mkdir(parents=True, exist_ok=True)
    batch_results = []
    for start in range(0, len(target_source_pairs), int(num_envs)):
        batch_pairs = target_source_pairs[start : start + int(num_envs)]
        batch_dir = workers_root / f"batch_{start:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_results.append(
            _run_batch_isolated(
                references,
                source_metadata,
                baseline_records,
                baseline_fingerprints,
                source_fingerprints,
                batch_pairs,
                batch_dir,
                len(batch_pairs),
                control_mode,
                action_dim,
            )
        )

    from mani_skill.trajectory.merge_trajectory import merge_trajectories

    final_h5 = output_dir / "repair.h5"
    final_json = final_h5.with_suffix(".json")
    trajectories = [Path(item["replay_h5"]) for item in batch_results]
    merge_trajectories(str(final_h5), [str(path) for path in trajectories])
    _project_visual_observation(final_h5)
    rows = [row for item in batch_results for row in item["rows"]]
    merged_metadata = _read_json(final_json)
    merged_records = merged_metadata.get("episodes", [])
    if len(merged_records) != len(rows):
        raise RuntimeError(f"Merged repair metadata count {len(merged_records)} != {len(rows)}")
    row_by_fingerprint = {
        row["repair_start_state_fingerprint"]: row for row in rows
    }
    used_fingerprints: set[str] = set()
    with h5py.File(final_h5, "r") as repair_file:
        for record in merged_records:
            repair_fingerprint = start_state_fingerprint(
                repair_file[f"traj_{int(record['episode_id'])}"], tolerance=1e-4
            )
            row = row_by_fingerprint.get(repair_fingerprint)
            if row is None:
                raise KeyError(
                    f"Merged repair group {record['episode_id']} has no identity match: "
                    f"{repair_fingerprint}"
                )
            if repair_fingerprint in used_fingerprints:
                raise ValueError(f"Duplicate merged repair start-state fingerprint: {repair_fingerprint}")
            used_fingerprints.add(repair_fingerprint)
            record["source_episode_id"] = int(row["source_episode_id"])
            record["target_baseline_episode_id"] = int(row["target_baseline_episode_id"])
            record["source_start_state_fingerprint"] = row["source_start_state_fingerprint"]
            record["baseline_start_state_fingerprint"] = row[
                "baseline_start_state_fingerprint"
            ]
            record["repair_start_state_fingerprint"] = repair_fingerprint
    if len(used_fingerprints) != len(merged_records):
        raise RuntimeError("Merged repair lost or duplicated start-state identities")
    merged_metadata["episodes"] = merged_records
    for record in merged_records:
        _remove_seed_identity_fields(record)
    _write_json(final_json, merged_metadata)

    successful_at_end = sum(bool(row["success_at_end"]) for row in rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task_id,
        "offline_ee_source_h5": _relative(offline_ee_h5),
        "gpu_replay_h5": _relative(gpu_replay_h5),
        "repair_h5": _relative(final_h5),
        "repair_json": _relative(final_json),
        "selected_target_episode_ids": selected_targets,
        "target_source_pairs": [
            {"target_episode_id": target, "source_episode_id": source}
            for target, source in target_source_pairs
        ],
        "repaired_episode_count": len(rows),
        "successful_once": sum(bool(row["success_once"]) for row in rows),
        "successful_at_end": successful_at_end,
        "success_at_end_rate": successful_at_end / len(rows),
        "reference_contract": {
            "reference_path": "offline EE source env_states -> actual TCP pose via GPU FK",
            "repair_action": "current GPU TCP pose -> next offline EE TCP pose",
            "gripper": f"copied from offline {action_dim}D EE action",
            "controller": f"native GPU {control_mode}",
            "action_dim": action_dim,
        },
        "observation_capture": {
            "obs_mode": "rgb",
            "rgb_saved_during_repair": True,
            "env_states_saved_during_env_step": True,
            "action_replay_used": True,
            "state_materialization_used": False,
        },
        "episodes": rows,
        "control_mode": control_mode,
        "action_dim": action_dim,
        "gate_pass": (successful_at_end == len(rows)) if require_all_success else True,
        "all_selected_targets_successful": successful_at_end == len(rows),
        "require_all_success": require_all_success,
    }
    _write_json(manifest_path, result)
    if require_all_success and not result["all_selected_targets_successful"]:
        raise RuntimeError(
            f"GPU EE repair gate failed: {successful_at_end}/{len(rows)} repaired episodes "
            f"succeeded at end; inspect {manifest_path}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-ee-h5", type=Path, required=True)
    parser.add_argument("--gpu-replay-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--episode-ids")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = None
    if args.episode_ids:
        selected = [int(value) for value in args.episode_ids.split(",") if value.strip()]
    result = repair(
        offline_ee_h5=args.offline_ee_h5,
        gpu_replay_h5=args.gpu_replay_h5,
        output_dir=args.output_dir,
        task_id=args.task,
        num_envs=args.num_envs,
        episode_ids=selected,
    )
    print(json.dumps(_jsonable({key: value for key, value in result.items() if key != "episodes"}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
