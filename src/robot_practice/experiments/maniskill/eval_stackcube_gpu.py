"""GPU-only StackCube closed-loop rollout using exact state-bank states.

This entry point is intentionally separate from the M0 PickCube evaluator.
It never reconstructs an episode from a seed: each worker loads the exact
``state`` object stored in a state-bank JSONL row, verifies its
``start_state_fingerprint``, and then runs the visual policy and
``physx_cuda`` environment on the same GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

from .eval_closed_loop import (
    DiffusionPolicyDriver,
    _action_measures,
    _driver_action,
    _seed_policy,
    _to_numpy,
    _to_torch,
    latency_summary,
    load_diffusion_policy,
)
from .rollout_contract import M1W_DATA_PROVENANCE, require_rollout_backend
from .state_identity import state_fingerprint

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pose"
SIM_BACKEND = "physx_cuda"
OBS_MODE = "rgb"
STATE_KEY = "observation.state"
BASE_IMAGE_KEY = "observation.images.base_camera"
HAND_IMAGE_KEY = "observation.images.hand_camera"
STATE_DIM = 18
ACTION_DIM = 7
DEFAULT_FORMAL_HORIZON = 500
DEFAULT_N_ACTION_STEPS = 8
STAGE_KEYS = ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")
SCHEMA_VERSION = 1


def _portable(path: str | Path) -> str:
    value = Path(path).expanduser().resolve()
    try:
        return value.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(path)


def _version(package: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_stackcube_policy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Map ManiSkill RGB observations to StackCube's 18D/two-camera input."""

    import torch

    try:
        base_rgb = observation["sensor_data"]["base_camera"]["rgb"]
        hand_rgb = observation["sensor_data"]["hand_camera"]["rgb"]
        qpos = observation["agent"]["qpos"]
        qvel = observation["agent"]["qvel"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "StackCube obs_mode=rgb must contain base_camera/hand_camera RGB and agent qpos/qvel"
        ) from exc

    def image(value: Any, name: str) -> torch.Tensor:
        tensor = _to_torch(value)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tuple(tensor.shape[-1:]) != (3,):
            raise ValueError(f"{name} must be batched channel-last RGB, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.uint8:
            raise ValueError(f"{name} must be uint8, got {tensor.dtype}")
        return tensor.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32) / 255.0

    base = image(base_rgb, "base_camera.rgb")
    hand = image(hand_rgb, "hand_camera.rgb")
    qpos_tensor = _to_torch(qpos, dtype=torch.float32)
    qvel_tensor = _to_torch(qvel, dtype=torch.float32)
    if qpos_tensor.ndim == 1:
        qpos_tensor = qpos_tensor.unsqueeze(0)
    if qvel_tensor.ndim == 1:
        qvel_tensor = qvel_tensor.unsqueeze(0)
    if qpos_tensor.shape != qvel_tensor.shape or qpos_tensor.shape[0] != base.shape[0]:
        raise ValueError(
            "StackCube image/qpos/qvel batch dimensions disagree: "
            f"base={tuple(base.shape)}, hand={tuple(hand.shape)}, "
            f"qpos={tuple(qpos_tensor.shape)}, qvel={tuple(qvel_tensor.shape)}"
        )
    if qpos_tensor.shape[1:] != (9,) or qvel_tensor.shape[1:] != (9,):
        raise ValueError(
            "StackCube proprioception must be qpos/qvel=(9,), got "
            f"{tuple(qpos_tensor.shape[1:])}/{tuple(qvel_tensor.shape[1:])}"
        )
    state = torch.cat((qpos_tensor, qvel_tensor), dim=-1)
    if state.shape[1:] != (STATE_DIM,) or not bool(torch.isfinite(state).all()):
        raise ValueError(f"StackCube policy state must be finite [B,{STATE_DIM}], got {state.shape}")
    return {
        BASE_IMAGE_KEY: base,
        HAND_IMAGE_KEY: hand,
        STATE_KEY: state,
    }


def load_state_bank(path: Path, *, max_episodes: int | None = None) -> list[dict[str, Any]]:
    """Load and verify state-bank rows; the fingerprint is the only identity."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"State bank does not exist: {path}")
    rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_fingerprints: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"State-bank line {line_number} must be an object")
            index = int(row["index"])
            fingerprint = str(row["start_state_fingerprint"])
            state = row.get("state")
            if not isinstance(state, Mapping):
                raise TypeError(f"State-bank line {line_number} has no state object")
            if index in seen_indices:
                raise ValueError(f"Duplicate state-bank index {index}")
            if fingerprint in seen_fingerprints:
                raise ValueError(f"Duplicate state-bank fingerprint {fingerprint}")
            calculated = state_fingerprint(state, tolerance=1e-4)
            if calculated != fingerprint:
                raise ValueError(
                    f"State-bank fingerprint mismatch at line {line_number}: "
                    f"stored={fingerprint}, calculated={calculated}"
                )
            seen_indices.add(index)
            seen_fingerprints.add(fingerprint)
            rows.append(row)
    rows.sort(key=lambda item: int(item["index"]))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError(f"State bank is empty: {path}")
    return rows


def _batched_state(rows: Sequence[Mapping[str, Any]], *, device: str) -> dict[str, Any]:
    import torch

    first = rows[0].get("state")
    if not isinstance(first, Mapping):
        raise TypeError("State-bank row has no state mapping")
    result: dict[str, Any] = {}
    for group_name, group in first.items():
        if not isinstance(group, Mapping):
            raise TypeError(f"State-bank state group {group_name!r} must be a mapping")
        result[group_name] = {}
        for object_name in group:
            values = []
            for row in rows:
                state_group = row["state"].get(group_name)
                if not isinstance(state_group, Mapping) or object_name not in state_group:
                    raise ValueError(f"State-bank rows disagree on state key {group_name}/{object_name}")
                values.append(np.asarray(state_group[object_name], dtype=np.float32))
            result[group_name][object_name] = torch.as_tensor(
                np.stack(values, axis=0), dtype=torch.float32, device=device
            )
    return result


def _bool_signal(info: Mapping[str, Any], key: str, batch_size: int, *, required: bool) -> np.ndarray:
    if key not in info:
        if required:
            raise KeyError(f"ManiSkill info is missing native StackCube signal {key!r}")
        return np.zeros(batch_size, dtype=bool)
    values = _to_numpy(info[key]).astype(bool, copy=False).reshape(-1)
    if values.size == 1 and batch_size != 1:
        values = np.repeat(values, batch_size)
    if values.size != batch_size:
        raise ValueError(f"info[{key!r}] has {values.size} values, expected {batch_size}")
    return values.copy()


def _signals(info: Mapping[str, Any], batch_size: int) -> dict[str, np.ndarray]:
    result = {"success": _bool_signal(info, "success", batch_size, required=True)}
    result.update({key: _bool_signal(info, key, batch_size, required=True) for key in STAGE_KEYS})
    result["fail"] = _bool_signal(info, "fail", batch_size, required=False)
    return result


def _make_env(*, num_envs: int, max_steps: int) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    require_rollout_backend(M1W_DATA_PROVENANCE, SIM_BACKEND)
    return gym.make(
        TASK_ID,
        num_envs=num_envs,
        obs_mode=OBS_MODE,
        control_mode=CONTROL_MODE,
        sim_backend="gpu",
        max_episode_steps=max_steps,
    )


def _reset_exact_state(env: Any, rows: Sequence[Mapping[str, Any]], *, device: str) -> Mapping[str, Any]:
    # The seed here only satisfies Gymnasium's initial reset API.  It is never
    # used as an episode identity and is never written to the result.
    env.reset(seed=[0] * len(rows))
    env.unwrapped.set_state_dict(_batched_state(rows, device=device))
    observation = env.unwrapped.get_obs()
    return observation


def _termination_reason(*, success: bool, truncated: bool, fail: bool, terminated: bool) -> str:
    if success:
        return "success"
    if truncated:
        return "truncated"
    if fail:
        return "native_fail"
    if terminated:
        return "terminated"
    return "episode_limit"


def run_stackcube_batch(
    *,
    env: Any,
    rows: Sequence[Mapping[str, Any]],
    driver: DiffusionPolicyDriver,
    max_steps: int,
    device: str,
    environment_latencies_ms: list[float],
) -> list[dict[str, Any]]:
    """Run one exact state batch; no reset occurs after an episode finishes."""

    batch_size = len(rows)
    if batch_size <= 0:
        return []
    observation = _reset_exact_state(env, rows, device=device)
    initial_info = env.unwrapped.get_info()
    signals = _signals(initial_info, batch_size)
    ever = {key: signals[key].copy() for key in STAGE_KEYS}
    success_once = signals["success"].copy()
    success_at_end = signals["success"].copy()
    first_success = np.full(batch_size, -1, dtype=np.int32)
    first_success[success_once] = 0
    finished = np.zeros(batch_size, dtype=bool)
    steps = np.zeros(batch_size, dtype=np.int32)
    reasons = ["episode_limit"] * batch_size
    metric_sums = {name: np.zeros(batch_size, dtype=np.float64) for name in ("translation_norm", "rotation_norm", "gripper_abs")}
    jitter_sums = np.zeros(batch_size, dtype=np.float64)
    gripper_flips = np.zeros(batch_size, dtype=np.int32)
    previous_action = np.zeros((batch_size, ACTION_DIM), dtype=np.float32)
    has_previous = np.zeros(batch_size, dtype=bool)
    driver.reset()

    for step_index in range(1, max_steps + 1):
        action = _driver_action(driver, observation)
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        if action.shape != (batch_size, ACTION_DIM) or not np.isfinite(action).all():
            raise ValueError(f"StackCube policy action must be finite {(batch_size, ACTION_DIM)}, got {action.shape}")
        action[finished] = 0.0
        active = ~finished
        measures = _action_measures(action)
        for name in ("translation_norm", "rotation_norm", "gripper_abs"):
            values = measures[name]
            if values is not None:
                metric_sums[name][active] += values[active]
        if np.any(active & has_previous):
            comparable = active & has_previous
            jitter_sums[comparable] += np.linalg.norm(
                action[comparable] - previous_action[comparable], axis=1
            )
            before = np.sign(previous_action[comparable, -1])
            after = np.sign(action[comparable, -1])
            gripper_flips[comparable] += (before != after) & (before != 0) & (after != 0)
        previous_action[active] = action[active]
        has_previous[active] = True
        started = time.perf_counter()
        observation, _, terminated, truncated, info = env.step(action)
        environment_latencies_ms.append((time.perf_counter() - started) * 1000.0)
        signals = _signals(info, batch_size)
        for key in STAGE_KEYS:
            ever[key] |= signals[key]
        active_success = active & signals["success"]
        first_success[(first_success < 0) & active_success] = step_index
        success_once[active] |= signals["success"][active]
        success_at_end[active] = signals["success"][active]
        terminated_array = _to_numpy(terminated).astype(bool, copy=False).reshape(-1)
        truncated_array = _to_numpy(truncated).astype(bool, copy=False).reshape(-1)
        if terminated_array.size != batch_size or truncated_array.size != batch_size:
            raise ValueError("StackCube terminated/truncated signals have the wrong batch size")
        newly_finished = active & (terminated_array | truncated_array)
        for index in np.flatnonzero(newly_finished):
            steps[index] = step_index
            reasons[index] = _termination_reason(
                success=bool(signals["success"][index]),
                truncated=bool(truncated_array[index]),
                fail=bool(signals["fail"][index]),
                terminated=bool(terminated_array[index]),
            )
        finished |= terminated_array | truncated_array

    for index in np.flatnonzero(~finished):
        steps[index] = max_steps
        reasons[index] = "episode_limit"
    result = []
    for index, row in enumerate(rows):
        count = max(int(steps[index]), 1)
        result.append(
            {
                "task": TASK_ID,
                "state_bank_index": int(row["index"]),
                "start_state_fingerprint": str(row["start_state_fingerprint"]),
                "success_once": bool(success_once[index]),
                "success_at_end": bool(success_at_end[index]),
                "success": bool(success_at_end[index]),
                "first_success_step": None if first_success[index] < 0 else int(first_success[index]),
                "steps": int(steps[index]),
                "termination_reason": reasons[index],
                "is_cubeA_grasped": bool(ever["is_cubeA_grasped"][index]),
                "is_cubeA_on_cubeB": bool(ever["is_cubeA_on_cubeB"][index]),
                "is_cubeA_static": bool(ever["is_cubeA_static"][index]),
                "translation_norm_mean": float(metric_sums["translation_norm"][index] / count),
                "rotation_norm_mean": float(metric_sums["rotation_norm"][index] / count),
                "gripper_abs_mean": float(metric_sums["gripper_abs"][index] / count),
                "action_jitter_mean": float(jitter_sums[index] / max(count - 1, 1)),
                "gripper_flip_count": int(gripper_flips[index]),
            }
        )
    return result


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    count = len(rows)
    successes = sum(bool(row[key]) for row in rows)
    return {"num_success": successes, "num_episodes": count, "rate": successes / count if count else None}


def _first_success_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [row["first_success_step"] for row in rows if row["first_success_step"] is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def summarize_stackcube_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty StackCube rollout")
    return {
        "num_episodes": len(rows),
        "success_once": _rate(rows, "success_once"),
        "success_at_end": _rate(rows, "success_at_end"),
        "first_success_step": _first_success_summary(rows),
        "stage_signal_rates": {
            key: sum(bool(row[key]) for row in rows) / len(rows) for key in STAGE_KEYS
        },
        "mean_episode_steps": float(np.mean([row["steps"] for row in rows])),
        "control": {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "translation_norm_mean",
                "rotation_norm_mean",
                "gripper_abs_mean",
                "action_jitter_mean",
                "gripper_flip_count",
            )
        },
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in rows)
            for reason in sorted({str(row["termination_reason"]) for row in rows})
        },
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path, compression="zstd")


def _mean_latency(payloads: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    summaries = [payload.get("latency", {}).get(key, {}) for payload in payloads]
    count = sum(int(item.get("count", 0)) for item in summaries)
    weighted = sum(
        int(item.get("count", 0)) * float(item["mean_ms"])
        for item in summaries
        if item.get("mean_ms") is not None
    )
    return {
        "count": count,
        "mean_ms": weighted / count if count else None,
        "p50_ms": None,
        "p95_ms": None,
        "aggregation_note": "worker means are count-weighted; merged percentiles unavailable",
    }


def _run_worker(arguments: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("This GPU worker owns no state-bank rows")
    device = arguments.device
    # Keep stochastic diffusion sampling reproducible across repeated model
    # selection runs.  Worker indices get independent streams while the same
    # checkpoint/state-bank assignment receives the same stream on rerun.
    _seed_policy(arguments.policy_seed + arguments.worker_index)
    loaded = load_diffusion_policy(
        policy_path=arguments.policy_path,
        dataset_repo_id=arguments.dataset_repo_id,
        dataset_root=arguments.dataset_root,
        n_action_steps=arguments.n_action_steps,
        device=device,
    )
    if int(loaded["action_dim"]) != ACTION_DIM:
        raise ValueError(f"StackCube policy must output {ACTION_DIM}D actions, got {loaded['action_dim']}")
    driver = DiffusionPolicyDriver(
        loaded,
        observation_builder=build_stackcube_policy_observation,
        state_dim=STATE_DIM,
        observation_description="two RGB cameras + 18D qpos/qvel",
    )
    environment_latencies: list[float] = []
    rollout_rows: list[dict[str, Any]] = []
    batch_size = arguments.num_envs or len(rows)
    if batch_size <= 0:
        raise ValueError("num_envs must be positive")
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        env = _make_env(num_envs=len(batch_rows), max_steps=arguments.max_steps)
        try:
            rollout_rows.extend(
                run_stackcube_batch(
                    env=env,
                    rows=batch_rows,
                    driver=driver,
                    max_steps=arguments.max_steps,
                    device=device,
                    environment_latencies_ms=environment_latencies,
                )
            )
        finally:
            env.close()
    output_dir = arguments.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite rollout output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "label": "StackCube GPU closed-loop",
        "system": "visual",
        "task": TASK_ID,
        "control_mode": CONTROL_MODE,
        **require_rollout_backend(M1W_DATA_PROVENANCE, SIM_BACKEND),
        "seed_set": arguments.seed_set,
        "policy_seed": arguments.policy_seed,
        "num_episodes": len(rollout_rows),
        "num_envs": arguments.num_envs or len(rows),
        "max_episode_steps": arguments.max_steps,
        "worker": {"index": arguments.worker_index, "count": arguments.worker_count},
        "evaluation": summarize_stackcube_rows(rollout_rows),
        "latency": {
            "policy_forward": latency_summary(driver.latencies_ms),
            "environment_step": latency_summary(environment_latencies),
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "label": "StackCube GPU closed-loop",
        "system": "visual",
        "task": TASK_ID,
        "control_mode": CONTROL_MODE,
        "sim_backend": SIM_BACKEND,
        **require_rollout_backend(M1W_DATA_PROVENANCE, SIM_BACKEND),
        "policy_path": _portable(arguments.policy_path),
        "dataset": {
            "repo_id": arguments.dataset_repo_id,
            "root": _portable(arguments.dataset_root),
        },
        "state_bank": {
            "path": _portable(arguments.state_bank),
            "seed_set": arguments.seed_set,
            "identity": "start_state_fingerprint",
            "seed_is_not_identity": True,
            "num_rows": len(rollout_rows),
            "fingerprints": [row["start_state_fingerprint"] for row in rollout_rows],
        },
        "policy_seed": arguments.policy_seed,
        "observation": {
            "keys": [BASE_IMAGE_KEY, HAND_IMAGE_KEY, STATE_KEY],
            "camera_shape": [128, 128, 3],
            "state_shape": [STATE_DIM],
            "state_fields": ["agent.qpos[0:9]", "agent.qvel[0:9]"],
        },
        "action": {"control_mode": CONTROL_MODE, "shape": [ACTION_DIM]},
        "worker": {"index": arguments.worker_index, "count": arguments.worker_count},
        "runtime": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "mani_skill": _version("mani_skill"),
            "lerobot": _version("lerobot"),
        },
    }
    write_json_atomic(output_dir / "metrics.json", metrics)
    _write_rows(output_dir / "episodes.parquet", rollout_rows)
    write_json_atomic(output_dir / "run_manifest.json", manifest)
    return metrics


def _visible_gpu_ids(num_gpus: int) -> list[str]:
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    identifiers = (
        [item.strip() for item in mask.split(",")]
        if mask and mask.strip()
        else None
    )
    if identifiers is None:
        import torch

        identifiers = [str(index) for index in range(torch.cuda.device_count())]
    if len(identifiers) < num_gpus:
        raise RuntimeError(f"requested {num_gpus} GPUs but only {len(identifiers)} are visible")
    return identifiers[:num_gpus]


def _worker_command(arguments: argparse.Namespace, *, worker_index: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "robot_practice.experiments.maniskill.eval_stackcube_gpu",
        "--policy-path",
        str(arguments.policy_path),
        "--dataset-repo-id",
        arguments.dataset_repo_id,
        "--dataset-root",
        str(arguments.dataset_root),
        "--state-bank-root",
        str(arguments.state_bank_root),
        "--seed-set",
        arguments.seed_set,
        "--max-steps",
        str(arguments.max_steps),
        "--n-action-steps",
        str(arguments.n_action_steps),
        "--policy-seed",
        str(arguments.policy_seed),
        "--device",
        arguments.device,
        "--worker-index",
        str(worker_index),
        "--worker-count",
        str(arguments.num_gpus),
        "--output-dir",
        str(output_dir),
    ]
    if arguments.num_envs is not None:
        insert_at = command.index("--device")
        command[insert_at:insert_at] = ["--num-envs", str(arguments.num_envs)]
    if arguments.max_episodes is not None:
        insert_at = command.index("--max-steps")
        command[insert_at:insert_at] = ["--max-episodes", str(arguments.max_episodes)]
    return command


def _merge_workers(arguments: argparse.Namespace, worker_dirs: Sequence[Path], expected: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    payloads = [json.loads((directory / "metrics.json").read_text(encoding="utf-8")) for directory in worker_dirs]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload, directory in zip(payloads, worker_dirs):
        worker_rows = pq.read_table(directory / "episodes.parquet").to_pylist()
        for row in worker_rows:
            fingerprint = str(row["start_state_fingerprint"])
            if fingerprint in seen:
                raise ValueError(f"Duplicate rollout state fingerprint across workers: {fingerprint}")
            seen.add(fingerprint)
        rows.extend(worker_rows)
    if len(rows) != expected:
        raise ValueError(f"Merged StackCube rollout has {len(rows)} episodes, expected {expected}")
    rows.sort(key=lambda row: int(row["state_bank_index"]))
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = dict(payloads[0])
    base["num_episodes"] = len(rows)
    base["evaluation"] = summarize_stackcube_rows(rows)
    base["worker"] = {"count": len(worker_dirs), "worker_outputs": [_portable(item) for item in worker_dirs]}
    base["latency"] = {
        "policy_forward": _mean_latency(payloads, "policy_forward"),
        "environment_step": _mean_latency(payloads, "environment_step"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "label": "StackCube GPU closed-loop",
        "system": "visual",
        "task": TASK_ID,
        "control_mode": CONTROL_MODE,
        "sim_backend": SIM_BACKEND,
        **require_rollout_backend(M1W_DATA_PROVENANCE, SIM_BACKEND),
        "policy_path": _portable(arguments.policy_path),
        "dataset": {"repo_id": arguments.dataset_repo_id, "root": _portable(arguments.dataset_root)},
        "state_bank": {
            "path": _portable(arguments.state_bank),
            "seed_set": arguments.seed_set,
            "identity": "start_state_fingerprint",
            "seed_is_not_identity": True,
            "num_rows": len(rows),
            "fingerprints": [row["start_state_fingerprint"] for row in rows],
        },
        "policy_seed": arguments.policy_seed,
        "observation": {
            "keys": [BASE_IMAGE_KEY, HAND_IMAGE_KEY, STATE_KEY],
            "camera_shape": [128, 128, 3],
            "state_shape": [STATE_DIM],
            "state_fields": ["agent.qpos[0:9]", "agent.qvel[0:9]"],
        },
        "action": {"control_mode": CONTROL_MODE, "shape": [ACTION_DIM]},
        "execution": {"parallelism": "one_physx_cuda_process_per_gpu", "num_gpus": len(worker_dirs)},
        "worker_outputs": [_portable(item) for item in worker_dirs],
    }
    write_json_atomic(output_dir / "metrics.json", base)
    _write_rows(output_dir / "episodes.parquet", rows)
    write_json_atomic(output_dir / "run_manifest.json", manifest)
    return base


def evaluate_multi_gpu(arguments: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir = arguments.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite rollout output: {output_dir}")
    gpu_ids = _visible_gpu_ids(arguments.num_gpus)
    workers_root = output_dir / "workers"
    workers_root.mkdir(parents=True)
    processes: list[subprocess.Popen[str]] = []
    logs = []
    try:
        for index, gpu_id in enumerate(gpu_ids):
            worker_dir = workers_root / f"gpu_{index}"
            log = (workers_root / f"gpu_{index}.log").open("w", encoding="utf-8")
            logs.append(log)
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": gpu_id, "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"})
            processes.append(
                subprocess.Popen(
                    _worker_command(arguments, worker_index=index, output_dir=worker_dir),
                    cwd=practice_root(),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        print(f"[StackCube GPU rollout] workers 0/{len(processes)}", flush=True)
        for index, process in enumerate(processes, start=1):
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"GPU worker {index - 1} failed; inspect {workers_root / f'gpu_{index - 1}.log'}")
            print(f"\r[StackCube GPU rollout] workers {index}/{len(processes)}", end="", flush=True)
        print()
        return _merge_workers(
            arguments,
            [workers_root / f"gpu_{index}" for index in range(len(processes))],
            len(rows),
        )
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        for log in logs:
            log.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=False)
    parser.add_argument("--dataset-repo-id", default="local/maniskill_stackcube_dcore_7d")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/lerobot/maniskill/stackcube_dcore_7d"))
    parser.add_argument(
        "--state-bank-root",
        type=Path,
        default=Path("outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/state_banks"),
    )
    parser.add_argument("--seed-set", choices=("train", "development", "test"), default="development")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_FORMAL_HORIZON)
    parser.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    parser.add_argument("--policy-seed", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.max_steps <= 0 or arguments.n_action_steps <= 0:
        raise SystemExit("--max-steps and --n-action-steps must be positive")
    if arguments.num_gpus <= 0:
        raise SystemExit("--num-gpus must be positive")
    if arguments.worker_count != 1 and arguments.num_gpus != 1:
        raise SystemExit("hidden worker arguments cannot be combined with --num-gpus")
    state_bank = arguments.state_bank_root.expanduser().resolve() / f"{arguments.seed_set}.jsonl"
    arguments.state_bank = state_bank
    rows = load_state_bank(state_bank, max_episodes=arguments.max_episodes)
    if arguments.validate_only:
        if arguments.policy_path is not None and not arguments.policy_path.expanduser().resolve().is_dir():
            raise SystemExit(f"policy path does not exist: {arguments.policy_path}")
        if not arguments.dataset_root.expanduser().resolve().is_dir():
            raise SystemExit(f"dataset root does not exist: {arguments.dataset_root}")
        print(json.dumps({"valid": True, "task": TASK_ID, "state_bank": _portable(state_bank), "episodes": len(rows), "backend": SIM_BACKEND}, indent=2))
        return 0
    if arguments.policy_path is None:
        raise SystemExit("--policy-path is required unless --validate-only is used")
    if not arguments.dataset_root.expanduser().resolve().is_dir():
        raise SystemExit(f"dataset root does not exist: {arguments.dataset_root}")
    if arguments.worker_count > 1:
        owned = rows[arguments.worker_index :: arguments.worker_count]
        if not owned:
            raise SystemExit(f"worker {arguments.worker_index} owns no state-bank rows")
        result = _run_worker(arguments, owned)
    elif arguments.num_gpus > 1:
        result = evaluate_multi_gpu(arguments, rows)
    else:
        result = _run_worker(arguments, rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
