"""Capture the frozen StackCube GPU replay layout as a live RGB dataset.

The accepted data are the successful baseline trajectories plus successful EE
repairs. GPU PhysX replay is sensitive to vectorized batch composition, so
rejected trajectories are still executed as carriers in their original slots.
Carrier trajectories are never written to the formal dataset. The historical
default contract is 7D ``pd_ee_delta_pose``; the native M1-W route passes the
4D ``pd_ee_delta_pos`` contract explicitly.

This writer deliberately does not use ManiSkill ``RecordEpisode`` or its merge
utility.  Each stored transition consists of the exact action submitted to
``env.step`` and the observation, simulator state, reward and success returned
by that same vectorized execution.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .gpu_ee_online_retarget_probe import _batch_states, _jsonable, _read_json, _relative
from .validate_gpu_ee_candidates import Candidate, build_carrier_batches

SCHEMA_VERSION = 1
TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pose"
ACTION_DIM = 7
BACKEND = "physx_cuda"
CAMERAS = ("base_camera", "hand_camera")
EXPECTED_BASELINE_ACCEPTED = 805
EXPECTED_REPAIR_ACCEPTED = 182
EXPECTED_ACCEPTED = EXPECTED_BASELINE_ACCEPTED + EXPECTED_REPAIR_ACCEPTED


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    finally:
        temporary.unlink(missing_ok=True)


def _publish_directory(temporary: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    try:
        temporary.replace(destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copytree(temporary, destination, symlinks=False)
        shutil.rmtree(temporary)


def _refresh_target_fingerprints(output_dir: Path) -> dict[int, str]:
    """Record hashes of the states actually written by the live capture."""

    from .state_identity import start_state_fingerprint

    replay_h5 = output_dir / "replay.h5"
    replay_json = output_dir / "replay.json"
    if not replay_h5.is_file() or not replay_json.is_file():
        raise FileNotFoundError(f"Live RGB H5/JSON pair is incomplete: {output_dir}")
    fingerprints: dict[int, str] = {}
    with h5py.File(replay_h5, "r") as handle:
        for name, group in handle.items():
            episode_id = int(name.removeprefix("traj_"))
            fingerprints[episode_id] = start_state_fingerprint(group, tolerance=1e-4)
    if len(fingerprints) != len(set(fingerprints.values())):
        raise RuntimeError("Formal live RGB data contains duplicate target-state fingerprints")
    metadata = _read_json(replay_json)
    records = metadata.get("episodes")
    if not isinstance(records, list) or len(records) != len(fingerprints):
        raise RuntimeError("Formal live RGB metadata/H5 episode count mismatch")
    changed = False
    for record in records:
        episode_id = int(record["episode_id"])
        target = fingerprints[episode_id]
        if record.get("target_start_state_fingerprint") != target:
            record["target_start_state_fingerprint"] = target
            changed = True
    if changed:
        _write_json(replay_json, metadata)
    manifest_path = output_dir / "replay_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        manifest["identity"] = {
            "source_unique_start_state_fingerprints": len(
                {str(record["source_start_state_fingerprint"]) for record in records}
            ),
            "target_unique_start_state_fingerprints": len(set(fingerprints.values())),
            "source_target_exact_hash_matches": sum(
                str(record["source_start_state_fingerprint"])
                == str(record["target_start_state_fingerprint"])
                for record in records
            ),
            "runtime_state_load_check": "all leaves atol<=1e-4",
            "seed_is_not_identity": True,
        }
        _write_json(manifest_path, manifest)
    return fingerprints


def _state_list(group: h5py.Group) -> list[dict[str, Any]]:
    from mani_skill.trajectory import utils as trajectory_utils

    return trajectory_utils.dict_to_list_of_dicts(group["env_states"])


def _metadata_records(path: Path) -> dict[int, dict[str, Any]]:
    records = _read_json(path.with_suffix(".json")).get("episodes")
    if not isinstance(records, list):
        raise TypeError(f"Metadata must contain an episodes list: {path.with_suffix('.json')}")
    return {int(record["episode_id"]): dict(record) for record in records}


def _member(candidate: Candidate, action_steps: int) -> dict[str, Any]:
    return {
        "source_episode_id": int(candidate.source_episode_id),
        "route": candidate.route,
        "source_group_id": int(candidate.group_id),
        "start_state_fingerprint": candidate.start_state_fingerprint,
        "accepted": bool(candidate.accepted),
        "action_steps": int(action_steps),
    }


def freeze_layout(
    *,
    baseline_h5: str | Path,
    repair_h5: str | Path,
    output: str | Path,
    num_envs: int = 100,
    control_mode: str = CONTROL_MODE,
    action_dim: int = ACTION_DIM,
    expected_baseline_accepted: int | None = EXPECTED_BASELINE_ACCEPTED,
    expected_repair_accepted: int | None = EXPECTED_REPAIR_ACCEPTED,
    expected_accepted: int | None = EXPECTED_ACCEPTED,
) -> dict[str, Any]:
    """Write the exact carrier batches for a native EE action contract."""

    baseline_h5 = Path(baseline_h5).expanduser().resolve()
    repair_h5 = Path(repair_h5).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    for path in (baseline_h5, repair_h5):
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise FileNotFoundError(f"Missing H5/JSON pair: {path}")

    batches, summary = build_carrier_batches(
        baseline_h5,
        repair_h5,
        num_envs=int(num_envs),
        expected_baseline_accepted=expected_baseline_accepted,
        expected_repair_accepted=expected_repair_accepted,
        expected_accepted=expected_accepted,
    )
    handles = {
        "baseline": h5py.File(baseline_h5, "r"),
        "repair": h5py.File(repair_h5, "r"),
    }
    try:
        serialized_batches = []
        for batch_index, batch in enumerate(batches):
            members = []
            for candidate in batch:
                group = handles[candidate.route][f"traj_{candidate.group_id}"]
                if tuple(group["actions"].shape)[1] != int(action_dim):
                    raise RuntimeError(
                        f"{candidate.route}/traj_{candidate.group_id} action dim does not match "
                        f"layout contract {action_dim}"
                    )
                members.append(_member(candidate, int(group["actions"].shape[0])))
            serialized_batches.append(
                {
                    "batch_index": batch_index,
                    "route": batch[0].route,
                    "num_envs": len(batch),
                    "members": members,
                }
            )
    finally:
        for handle in handles.values():
            handle.close()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "backend": BACKEND,
        "control_mode": control_mode,
        "action_dim": int(action_dim),
        "identity": "env_states[0] start-state fingerprint; seed is not identity",
        "layout": summary,
        "inputs": {
            "baseline": {
                "h5": _relative(baseline_h5),
                "sha256": _sha256(baseline_h5),
            },
            "repair": {
                "h5": _relative(repair_h5),
                "sha256": _sha256(repair_h5),
            },
        },
        "accepted": {
            "baseline": summary["routes"]["baseline"]["accepted"],
            "repair": summary["routes"]["repair"]["accepted"],
            "total": summary["accepted_episode_count"],
        },
        "batches": serialized_batches,
        "gate_pass": (
            summary["total_episode_executions"] > 0
            and summary["accepted_episode_count"] > 0
        ),
    }
    if not payload["gate_pass"]:
        raise RuntimeError(f"Frozen layout gate failed: {summary}")
    _write_json(output, payload)
    return payload


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _split_rows(value: Any) -> list[Any]:
    from mani_skill.trajectory import utils as trajectory_utils

    return trajectory_utils.dict_to_list_of_dicts(value)


def _assert_state_close(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    source_episode_id: int,
    tolerance: float = 1e-4,
) -> None:
    """Check a loaded state numerically without quantization-boundary artifacts."""

    def visit(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or set(left) != set(right):
                raise RuntimeError(
                    f"Initial state schema changed for source {source_episode_id} at {path}"
                )
            for key in sorted(left):
                visit(left[key], right[key], f"{path}/{key}")
            return
        left_array = _numpy(left)
        right_array = _numpy(right)
        if left_array.shape != right_array.shape:
            raise RuntimeError(
                f"Initial state shape changed for source {source_episode_id} at {path}: "
                f"{left_array.shape} != {right_array.shape}"
            )
        if not np.allclose(left_array, right_array, rtol=0.0, atol=tolerance):
            difference = float(np.max(np.abs(left_array.astype(float) - right_array.astype(float))))
            raise RuntimeError(
                f"Initial state changed for source {source_episode_id} at {path}: "
                f"max_abs_diff={difference} > {tolerance}"
            )

    visit(expected, observed, "state")


def _create_state_tree(parent: h5py.Group, sample: Mapping[str, Any], frames: int) -> None:
    for key, value in sample.items():
        if isinstance(value, Mapping):
            _create_state_tree(parent.create_group(str(key)), value, frames)
            continue
        array = _numpy(value)
        parent.create_dataset(
            str(key),
            shape=(frames, *array.shape),
            dtype=array.dtype,
            chunks=(1, *array.shape) if array.shape else (1,),
        )


def _write_state_tree(parent: h5py.Group, value: Mapping[str, Any], frame: int) -> None:
    for key, child in value.items():
        if isinstance(child, Mapping):
            _write_state_tree(parent[str(key)], child, frame)
        else:
            parent[str(key)][frame] = _numpy(child)


def _create_episode_group(
    output: h5py.File,
    *,
    member: Mapping[str, Any],
    actions: np.ndarray,
    initial_state: Mapping[str, Any],
) -> h5py.Group:
    episode_id = int(member["source_episode_id"])
    group = output.create_group(f"traj_{episode_id}")
    steps = int(actions.shape[0])
    group.create_dataset("actions", data=actions, dtype="f4")
    group.create_dataset("rewards", shape=(steps,), dtype="f4")
    group.create_dataset("success", shape=(steps,), dtype="bool")
    group.create_dataset("terminated", shape=(steps,), dtype="bool")
    group.create_dataset("truncated", shape=(steps,), dtype="bool")
    obs = group.create_group("obs")
    agent = obs.create_group("agent")
    agent.create_dataset("qpos", shape=(steps + 1, 9), dtype="f4")
    agent.create_dataset("qvel", shape=(steps + 1, 9), dtype="f4")
    sensors = obs.create_group("sensor_data")
    for camera in CAMERAS:
        sensors.create_group(camera).create_dataset(
            "rgb",
            shape=(steps + 1, 128, 128, 3),
            dtype="u1",
            chunks=(1, 128, 128, 3),
            compression="lzf",
            shuffle=True,
        )
    _create_state_tree(group.create_group("env_states"), initial_state, steps + 1)
    group.attrs["source_episode_id"] = episode_id
    group.attrs["route"] = str(member["route"])
    group.attrs["source_group_id"] = int(member["source_group_id"])
    group.attrs["start_state_fingerprint"] = str(member["start_state_fingerprint"])
    return group


def _write_observation(group: h5py.Group, observation: Mapping[str, Any], slot: int, frame: int) -> None:
    qpos = _numpy(observation["agent"]["qpos"])[slot]
    qvel = _numpy(observation["agent"]["qvel"])[slot]
    if qpos.shape != (9,) or qvel.shape != (9,):
        raise RuntimeError(f"Unexpected qpos/qvel shape: {qpos.shape}/{qvel.shape}")
    group["obs/agent/qpos"][frame] = qpos
    group["obs/agent/qvel"][frame] = qvel
    for camera in CAMERAS:
        image = _numpy(observation["sensor_data"][camera]["rgb"])[slot]
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise RuntimeError(
                f"Unexpected {camera} RGB: dtype={image.dtype}, shape={image.shape}"
            )
        group[f"obs/sensor_data/{camera}/rgb"][frame] = image


def _capture_batch_worker(
    batch: Mapping[str, Any],
    inputs: Mapping[str, str],
    metadata_path: str,
    output_h5: str,
    control_mode: str,
    action_dim: int,
    result_queue: Any,
) -> None:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import torch
        from mani_skill.utils import common

        members = [dict(value) for value in batch["members"]]
        route = str(batch["route"])
        source_path = Path(inputs[route])
        metadata = _read_json(Path(metadata_path))
        env_kwargs = dict(metadata["env_info"]["env_kwargs"])
        env_kwargs.update(
            obs_mode="rgb",
            control_mode=control_mode,
            sim_backend=BACKEND,
            render_mode="rgb_array",
            reward_mode="sparse",
            reconfiguration_freq=1,
            num_envs=len(members),
        )
        env = gym.make(TASK_ID, **env_kwargs)
        try:
            actions: list[np.ndarray] = []
            initial_states: list[dict[str, Any]] = []
            with h5py.File(source_path, "r") as source:
                for member in members:
                    group = source[f"traj_{int(member['source_group_id'])}"]
                    action = np.asarray(group["actions"], dtype=np.float32)
                    states = _state_list(group)
                    if action.shape != (int(member["action_steps"]), action_dim):
                        raise RuntimeError(
                            f"Action contract changed for source {member['source_episode_id']}"
                        )
                    actions.append(action)
                    initial_states.append(states[0])

            device = env.unwrapped.device
            env.reset(seed=torch.arange(len(members), dtype=torch.int64, device=device))
            state_batch = _batch_states(initial_states)
            env.unwrapped.set_state_dict(state_batch)
            env.unwrapped.scene._gpu_fetch_all()
            observation = common.to_numpy(env.unwrapped.get_obs())
            current_states = _split_rows(common.to_numpy(env.unwrapped.get_state_dict()))
            for slot, member in enumerate(members):
                _assert_state_close(
                    initial_states[slot],
                    current_states[slot],
                    source_episode_id=int(member["source_episode_id"]),
                )

            lengths = np.asarray([len(value) for value in actions], dtype=np.int64)
            accepted_slots = [index for index, member in enumerate(members) if member["accepted"]]
            with h5py.File(output_h5, "a") as output:
                groups: dict[int, h5py.Group] = {}
                for slot in accepted_slots:
                    member = members[slot]
                    episode_id = int(member["source_episode_id"])
                    group = _create_episode_group(
                        output,
                        member=member,
                        actions=actions[slot],
                        initial_state=current_states[slot],
                    )
                    groups[episode_id] = group
                    _write_observation(group, observation, slot, 0)
                    _write_state_tree(group["env_states"], current_states[slot], 0)

                success_once = np.zeros(len(members), dtype=bool)
                first_success: list[int | None] = [None] * len(members)
                final_success = np.zeros(len(members), dtype=bool)
                for step in range(int(lengths.max())):
                    active = lengths > step
                    action_batch = np.zeros((len(members), action_dim), dtype=np.float32)
                    for slot, action in enumerate(actions):
                        if active[slot]:
                            action_batch[slot] = action[step]
                    next_observation, reward, terminated, truncated, info = env.step(
                        torch.as_tensor(action_batch, device=device)
                    )
                    next_observation = common.to_numpy(next_observation)
                    reward = _numpy(common.to_numpy(reward)).reshape(-1)
                    terminated = _numpy(common.to_numpy(terminated)).reshape(-1).astype(bool)
                    truncated = _numpy(common.to_numpy(truncated)).reshape(-1).astype(bool)
                    success = _numpy(
                        common.to_numpy(
                            info.get(
                                "success",
                                torch.zeros(len(members), dtype=torch.bool, device=device),
                            )
                        )
                    ).reshape(-1).astype(bool)
                    next_states = _split_rows(common.to_numpy(env.unwrapped.get_state_dict()))
                    for slot, member in enumerate(members):
                        if not active[slot]:
                            continue
                        if success[slot]:
                            success_once[slot] = True
                            if first_success[slot] is None:
                                first_success[slot] = step + 1
                        if step == lengths[slot] - 1:
                            final_success[slot] = bool(success[slot])
                        if not member["accepted"]:
                            continue
                        group = groups[int(member["source_episode_id"])]
                        group["rewards"][step] = reward[slot]
                        group["success"][step] = success[slot]
                        group["terminated"][step] = terminated[slot]
                        group["truncated"][step] = truncated[slot]
                        _write_observation(group, next_observation, slot, step + 1)
                        _write_state_tree(group["env_states"], next_states[slot], step + 1)
                output.flush()

            rows = []
            for slot, member in enumerate(members):
                rows.append(
                    {
                        **member,
                        "success_once": bool(success_once[slot]),
                        "success_at_end": bool(final_success[slot]),
                        "first_success_step": first_success[slot],
                    }
                )
            result_queue.put({"ok": True, "rows": rows})
        finally:
            env.close()
    except BaseException as exc:
        result_queue.put({"ok": False, "error": repr(exc)})


def _run_batch(
    batch: Mapping[str, Any],
    *,
    inputs: Mapping[str, str],
    metadata_path: Path,
    output_h5: Path,
    control_mode: str,
    action_dim: int,
) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_capture_batch_worker,
        args=(
            batch,
            inputs,
            str(metadata_path),
            str(output_h5),
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
            f"Live RGB batch {batch['batch_index']} failed (exitcode={process.exitcode}): "
            f"{result.get('error', 'unknown error')}"
        )
    return list(result["rows"])


def capture(
    *,
    baseline_h5: str | Path,
    repair_h5: str | Path,
    layout_path: str | Path,
    output_dir: str | Path,
    control_mode: str = CONTROL_MODE,
    action_dim: int = ACTION_DIM,
) -> dict[str, Any]:
    """Execute the frozen layout and publish the direct-RGB formal dataset."""

    baseline_h5 = Path(baseline_h5).expanduser().resolve()
    repair_h5 = Path(repair_h5).expanduser().resolve()
    layout_path = Path(layout_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    layout = _read_json(layout_path)
    if not layout.get("gate_pass"):
        raise RuntimeError(f"Frozen layout did not pass its gate: {layout_path}")
    layout_control_mode = layout.get("control_mode")
    layout_action_dim = layout.get("action_dim")
    if layout_control_mode is None and int(action_dim) == ACTION_DIM:
        layout_control_mode = CONTROL_MODE
    if layout_action_dim is None and int(action_dim) == ACTION_DIM:
        layout_action_dim = ACTION_DIM
    if layout_control_mode != control_mode or int(layout_action_dim or -1) != int(action_dim):
        raise RuntimeError(
            "Frozen layout action contract does not match capture request: "
            f"{layout_control_mode}/{layout_action_dim} vs "
            f"{control_mode}/{action_dim}"
        )
    expected_accepted = int(layout["accepted"]["total"])
    actual_inputs = {"baseline": baseline_h5, "repair": repair_h5}
    for route, path in actual_inputs.items():
        expected_hash = layout["inputs"][route]["sha256"]
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"{route} input changed after layout freeze: {path}")
    if output_dir.exists():
        manifest = output_dir / "replay_manifest.json"
        replay_h5 = output_dir / "replay.h5"
        if manifest.is_file() and replay_h5.is_file():
            cached = _read_json(manifest)
            if cached.get("gate_pass"):
                _refresh_target_fingerprints(output_dir)
                return cached
        raise FileExistsError(f"Incomplete or invalid formal dataset exists: {output_dir}")

    temporary = output_dir.parent / f".{output_dir.name}.incomplete-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Incomplete capture directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    replay_h5 = temporary / "replay.h5"
    inputs = {route: str(path) for route, path in actual_inputs.items()}
    metadata_path = baseline_h5.with_suffix(".json")
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    batch_summaries = []
    try:
        for batch in layout["batches"]:
            batch_rows = _run_batch(
                batch,
                inputs=inputs,
                metadata_path=metadata_path,
                output_h5=replay_h5,
                control_mode=control_mode,
                action_dim=int(action_dim),
            )
            rows.extend(batch_rows)
            accepted = [row for row in batch_rows if row["accepted"]]
            summary = {
                "batch_index": int(batch["batch_index"]),
                "route": str(batch["route"]),
                "episode_executions": len(batch_rows),
                "accepted_episode_count": len(accepted),
                "accepted_successful_at_end": sum(row["success_at_end"] for row in accepted),
            }
            batch_summaries.append(summary)
            print(
                f"[live-rgb] batch {summary['batch_index'] + 1}/{len(layout['batches'])} "
                f"{summary['route']}: {summary['accepted_successful_at_end']}/"
                f"{summary['accepted_episode_count']} accepted end-success; "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

        accepted_rows = sorted(
            (row for row in rows if row["accepted"]),
            key=lambda row: int(row["source_episode_id"]),
        )
        success_once = sum(row["success_once"] for row in accepted_rows)
        success_at_end = sum(row["success_at_end"] for row in accepted_rows)
        target_fingerprints: dict[int, str] = {}
        from .state_identity import start_state_fingerprint

        with h5py.File(replay_h5, "r") as output:
            group_ids = sorted(int(name.removeprefix("traj_")) for name in output)
            schema_ok = len(group_ids) == expected_accepted
            for row in accepted_rows:
                group = output[f"traj_{int(row['source_episode_id'])}"]
                target_fingerprints[int(row["source_episode_id"])] = start_state_fingerprint(
                    group, tolerance=1e-4
                )
                steps = int(row["action_steps"])
                schema_ok = schema_ok and group["actions"].shape == (steps, int(action_dim))
                schema_ok = schema_ok and group["obs/agent/qpos"].shape == (steps + 1, 9)
                schema_ok = schema_ok and group[
                    "obs/sensor_data/base_camera/rgb"
                ].shape == (steps + 1, 128, 128, 3)
        gate_pass = (
            len(accepted_rows) == expected_accepted
            and len({int(row["source_episode_id"]) for row in accepted_rows}) == expected_accepted
            and success_once == expected_accepted
            and success_at_end == expected_accepted
            and schema_ok
        )
        if not gate_pass:
            raise RuntimeError(
                f"Formal live RGB gate failed: accepted={len(accepted_rows)}, "
                f"once={success_once}, end={success_at_end}, schema={schema_ok}"
            )

        metadata = _read_json(metadata_path)
        metadata["env_info"]["env_kwargs"].update(
            obs_mode="rgb",
            control_mode=control_mode,
            sim_backend=BACKEND,
            render_mode="rgb_array",
            reward_mode="sparse",
            reconfiguration_freq=1,
            num_envs=int(layout["layout"]["num_envs"]),
        )
        metadata["source_type"] = f"m1_w_native_{action_dim}d_gpu_live_rgb"
        metadata["source_desc"] = (
            "Accepted StackCube baseline and repair actions executed in the frozen GPU "
            "carrier layout; RGB/action/env_state/success captured in the same env.step run"
        )
        metadata["episodes"] = [
            {
                "episode_id": int(row["source_episode_id"]),
                "source_episode_id": int(row["source_episode_id"]),
                "control_mode": control_mode,
                "elapsed_steps": int(row["action_steps"]),
                "success": True,
                "route": str(row["route"]),
                "source_group_id": int(row["source_group_id"]),
                "source_start_state_fingerprint": str(row["start_state_fingerprint"]),
                "target_start_state_fingerprint": target_fingerprints[
                    int(row["source_episode_id"])
                ],
                "first_success_step": row["first_success_step"],
                "reset_kwargs": {},
            }
            for row in accepted_rows
        ]
        metadata["m1_wam_data_route"] = {
            "route": f"frozen_gpu_carrier_live_rgb_replay_{action_dim}d",
            "layout_manifest": _relative(layout_path),
            "accepted_episodes": expected_accepted,
            "carrier_episode_executions": int(layout["layout"]["carrier_episode_executions"]),
            "rgb_capture": "directly from the same env.step execution",
            "actual_action_capture": True,
            "state_materialization": False,
            "record_episode_used": False,
            "identity": "env_states[0] start-state fingerprint; seed is not identity",
        }
        _write_json(temporary / "replay.json", metadata)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "task": TASK_ID,
            "route": f"frozen_gpu_carrier_live_rgb_replay_{action_dim}d",
            "backend": BACKEND,
            "control_mode": control_mode,
            "obs_mode": "rgb",
            "action_dim": int(action_dim),
            "layout_manifest": _relative(layout_path),
            "layout_manifest_sha256": _sha256(layout_path),
            "baseline_h5": _relative(baseline_h5),
            "repair_h5": _relative(repair_h5),
            "episode_count": len(accepted_rows),
            "expected_episodes": expected_accepted,
            "successful_once": success_once,
            "successful_at_end": success_at_end,
            "carrier_episode_executions": int(layout["layout"]["carrier_episode_executions"]),
            "total_episode_executions": int(layout["layout"]["total_episode_executions"]),
            "trajectory_writer": "direct_h5_per_source_episode",
            "record_episode_used": False,
            "state_materialization": False,
            "rgb_captured_during_env_step": True,
            "identity": {
                "source_unique_start_state_fingerprints": expected_accepted,
                "target_unique_start_state_fingerprints": len(set(target_fingerprints.values())),
                "source_target_exact_hash_matches": sum(
                    row["start_state_fingerprint"]
                    == target_fingerprints[int(row["source_episode_id"])]
                    for row in accepted_rows
                ),
                "runtime_state_load_check": "all leaves atol<=1e-4",
                "seed_is_not_identity": True,
            },
            "batches": batch_summaries,
            "excluded_source_episode_ids": sorted(
                int(row["source_episode_id"])
                for row in rows
                if row["route"] == "repair" and not row["accepted"]
            ),
            "elapsed_seconds": time.monotonic() - started,
            "gate_pass": gate_pass,
        }
        _write_json(temporary / "replay_manifest.json", manifest)
        _publish_directory(temporary, output_dir)
        return manifest
    except BaseException:
        failed = output_dir.parent / f".{output_dir.name}.failed-{os.getpid()}"
        if temporary.exists() and not failed.exists():
            temporary.replace(failed)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-h5", type=Path, required=True)
    parser.add_argument("--repair-h5", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument(
        "--control-mode",
        choices=("pd_ee_delta_pos", "pd_ee_delta_pose"),
        default=CONTROL_MODE,
    )
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--freeze-layout-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = freeze_layout(
        baseline_h5=args.baseline_h5,
        repair_h5=args.repair_h5,
        output=args.layout,
        num_envs=args.num_envs,
        control_mode=args.control_mode,
        action_dim=(
            int(args.action_dim)
            if args.action_dim is not None
            else (4 if args.control_mode == "pd_ee_delta_pos" else 7)
        ),
    )
    if args.freeze_layout_only:
        print(json.dumps(_jsonable(layout["layout"]), ensure_ascii=False, indent=2))
        return 0
    result = capture(
        baseline_h5=args.baseline_h5,
        repair_h5=args.repair_h5,
        layout_path=args.layout,
        output_dir=args.output_dir,
        control_mode=args.control_mode,
        action_dim=(
            int(args.action_dim)
            if args.action_dim is not None
            else (4 if args.control_mode == "pd_ee_delta_pos" else 7)
        ),
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
