"""Development checkpoint selection for the generic M2 Action-only Flow.

The selector is deliberately separate from M2-W selection.  It evaluates the
configured development episodes only, using exact initial ``env_states`` from
the replay HDF5.  The simulator backend follows the config: official data is
evaluated with CPU PhysX, while M1-W native data is evaluated with GPU PhysX.
Each candidate is one checkpoint paired with one ``n_action_steps`` value.

This is a selection/evaluation utility, not a training command.  It never
opens the configured test split and never reconstructs an episode from a seed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_practice.evaluation.common import practice_root
from robot_practice.policies.wam_tiny.action_flow import (
    BASE_IMAGE_KEY,
    HAND_IMAGE_KEY,
    STATE_KEY,
    load_action_flow_checkpoint,
)

from .eval_stackcube_gpu import (
    _batched_state,
    _signals,
    _to_numpy,
    build_stackcube_policy_observation,
)
from .m2_base_action_flow import (
    DEFAULT_CONFIG,
    TASK_ID,
    _open_h5_read_only,
    _portable,
    _read_json,
    _rooted,
    _split_ids,
    _write_json,
    load_config,
    validate_config,
)

CONTROL_MODE_BY_DIM = {4: "pd_ee_delta_pos", 7: "pd_ee_delta_pose"}
DEFAULT_MAX_STEPS = 200
DEFAULT_N_ACTION_STEPS = (4, 8)
DEFAULT_GPU_ENVS = 64
ROLLOUT_BACKENDS = {"physx_cpu", "physx_cuda"}


def _resolve_rollout_route(
    config: Mapping[str, Any],
    requested_policy_device: str,
) -> tuple[str, str]:
    """Resolve the simulator and policy device from the experiment contract."""

    backend = str(config["protocol"].get("backend", ""))
    if backend not in ROLLOUT_BACKENDS:
        raise ValueError(
            "M2 base selector only supports protocol.backend=physx_cpu or physx_cuda, "
            f"got {backend!r}"
        )
    requested = str(requested_policy_device).strip().lower()
    if requested == "auto":
        return backend, "cuda" if backend == "physx_cuda" else "cpu"
    expected = "cuda" if backend == "physx_cuda" else "cpu"
    if requested != expected:
        raise ValueError(
            f"policy device {requested!r} conflicts with protocol.backend={backend!r}; "
            f"use --policy-device={expected} or --policy-device=auto"
        )
    return backend, requested


def _state_fingerprint(state: Mapping[str, Mapping[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for group_name in sorted(state):
        for object_name in sorted(state[group_name]):
            value = np.asarray(state[group_name][object_name], dtype=np.float32)
            digest.update(group_name.encode())
            digest.update(b"/")
            digest.update(object_name.encode())
            digest.update(value.shape.__repr__().encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _load_split_rows(config: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    """Read exact initial states for one configured split from replay HDF5."""

    source = _rooted(config["inputs"]["source_h5"])
    rows: list[dict[str, Any]] = []
    with _open_h5_read_only(source) as handle:
        for episode_id in _split_ids(config, split):
            group = handle[f"traj_{episode_id}"]
            state: dict[str, dict[str, np.ndarray]] = {}
            for group_name in ("actors", "articulations"):
                state[group_name] = {}
                node = group[f"env_states/{group_name}"]
                for object_name in node:
                    state[group_name][object_name] = np.asarray(node[object_name][0], dtype=np.float32)
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "state": state,
                    "start_state_fingerprint": _state_fingerprint(state),
                }
            )
    fingerprints = [row["start_state_fingerprint"] for row in rows]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError(f"M2 base {split} start-state fingerprints are not unique")
    return rows


def _load_development_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read exact development initial states from the replay HDF5."""

    return _load_split_rows(config, "development")


def _make_cpu_env(*, control_mode: str, max_steps: int) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        TASK_ID,
        num_envs=1,
        obs_mode="rgb",
        control_mode=control_mode,
        sim_backend="cpu",
        reward_mode="sparse",
        max_episode_steps=max_steps,
    )


def _reset_exact_state(env: Any, row: Mapping[str, Any]) -> Mapping[str, Any]:
    env.reset(seed=[0])
    env.unwrapped.set_state_dict(_batched_state([row], device="cpu"))
    return env.unwrapped.get_obs()


def _make_gpu_env(*, control_mode: str, max_steps: int, num_envs: int) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        TASK_ID,
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode=control_mode,
        sim_backend="gpu",
        reward_mode="sparse",
        max_episode_steps=max_steps,
    )


def _reset_exact_gpu_state(
    env: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> Mapping[str, Any]:
    env.reset(seed=[0] * len(rows))
    env.unwrapped.set_state_dict(_batched_state(rows, device=device))
    return env.unwrapped.get_obs()


def _gpu_termination_reason(*, success: bool, truncated: bool, terminated: bool) -> str:
    if success:
        return "success"
    if truncated:
        return "truncated"
    if terminated:
        return "terminated"
    return "episode_limit"


def _print_gpu_progress(
    *,
    prefix: str,
    step_index: int,
    max_steps: int,
    episodes_done: int,
    total_episodes: int,
) -> None:
    width = 24
    fraction = min(max(step_index / max(max_steps, 1), 0.0), 1.0)
    filled = int(fraction * width)
    if filled >= width:
        bar = "=" * width
    else:
        bar = "=" * filled + ">" + " " * (width - filled - 1)
    print(
        f"{prefix} [{bar}] step={step_index}/{max_steps} "
        f"episodes={episodes_done}/{total_episodes}",
        flush=True,
    )


def _evaluate_gpu_batch(
    *,
    env: Any,
    rows: Sequence[Mapping[str, Any]],
    policy: Any,
    max_steps: int,
    device: str,
    expected_dim: int,
    progress_prefix: str | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one exact-state batch without resetting finished episodes."""

    batch_size = len(rows)
    if batch_size <= 0:
        return []
    observation = _reset_exact_gpu_state(env, rows, device=device)
    signals = _signals(env.unwrapped.get_info(), batch_size)
    ever = {key: signals[key].copy() for key in ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")}
    success_once = signals["success"].copy()
    success_at_end = signals["success"].copy()
    first_success = np.full(batch_size, -1, dtype=np.int32)
    first_success[success_once] = 0
    finished = np.zeros(batch_size, dtype=bool)
    steps = np.zeros(batch_size, dtype=np.int32)
    reasons = ["episode_limit"] * batch_size
    history: list[Mapping[str, Any]] = [observation, observation]
    fingerprints = [str(row["start_state_fingerprint"]) for row in rows]
    policy.reset()
    action_clip_elements = np.zeros(batch_size, dtype=np.int64)
    action_total_elements = np.zeros(batch_size, dtype=np.int64)
    total_episodes = int(progress_total or batch_size)
    if progress_prefix is not None:
        _print_gpu_progress(
            prefix=progress_prefix,
            step_index=0,
            max_steps=max_steps,
            episodes_done=progress_offset,
            total_episodes=total_episodes,
        )

    for step_index in range(1, max_steps + 1):
        context = _stack_context(history)
        context = {
            key: value.to(device=device)
            for key, value in context.items()
        }
        action = policy.select_action(context, fingerprints=fingerprints)
        if tuple(action.shape) != (batch_size, expected_dim) or not bool(torch.isfinite(action).all()):
            raise ValueError(
                f"Action Flow policy action must be finite {(batch_size, expected_dim)}, "
                f"got {tuple(action.shape)}"
            )
        active = ~finished
        action[torch.as_tensor(finished, device=action.device)] = 0.0
        action_np = _to_numpy(action).astype(np.float32, copy=False)
        action_clip_elements[active] += np.sum(np.abs(action_np[active]) >= 1.0, axis=1)
        action_total_elements[active] += expected_dim
        observation, _, terminated, truncated, info = env.step(action)
        history.append(observation)
        history = history[-2:]
        signals = _signals(info, batch_size)
        for key in ever:
            ever[key] |= signals[key]
        active_success = active & signals["success"]
        first_success[(first_success < 0) & active_success] = step_index
        success_once[active] |= signals["success"][active]
        success_at_end[active] = signals["success"][active]
        terminated_array = _to_numpy(terminated).astype(bool, copy=False).reshape(-1)
        truncated_array = _to_numpy(truncated).astype(bool, copy=False).reshape(-1)
        if terminated_array.size != batch_size or truncated_array.size != batch_size:
            raise ValueError("GPU terminated/truncated signals have the wrong batch size")
        newly_finished = active & (terminated_array | truncated_array)
        for index in np.flatnonzero(newly_finished):
            steps[index] = step_index
            reasons[index] = _gpu_termination_reason(
                success=bool(signals["success"][index]),
                truncated=bool(truncated_array[index]),
                terminated=bool(terminated_array[index]),
            )
        finished |= terminated_array | truncated_array
        if progress_prefix is not None:
            progress_every = max(1, max_steps // 10)
            if step_index % progress_every == 0 or step_index == max_steps:
                _print_gpu_progress(
                    prefix=progress_prefix,
                    step_index=step_index,
                    max_steps=max_steps,
                    episodes_done=progress_offset + int(finished.sum()),
                    total_episodes=total_episodes,
                )

    for index in np.flatnonzero(~finished):
        steps[index] = max_steps
        reasons[index] = "episode_limit"
    return [
        {
            "episode_id": int(row["episode_id"]),
            "start_state_fingerprint": str(row["start_state_fingerprint"]),
            "success_once": bool(success_once[index]),
            "success_at_end": bool(success_at_end[index]),
            "first_success_step": None if first_success[index] < 0 else int(first_success[index]),
            "is_cubeA_grasped": bool(ever["is_cubeA_grasped"][index]),
            "is_cubeA_on_cubeB": bool(ever["is_cubeA_on_cubeB"][index]),
            "is_cubeA_static": bool(ever["is_cubeA_static"][index]),
            "steps": int(steps[index]),
            "termination_reason": reasons[index],
            "action_clip_elements": int(action_clip_elements[index]),
            "action_total_elements": int(action_total_elements[index]),
        }
        for index, row in enumerate(rows)
    ]


def _stack_context(history: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    frames = [build_stackcube_policy_observation(item) for item in history[-2:]]
    return {
        key: torch.stack([frame[key] for frame in frames], dim=1)
        for key in (BASE_IMAGE_KEY, HAND_IMAGE_KEY, STATE_KEY)
    }


def _one_bool(value: Any) -> bool:
    array = _to_numpy(value).astype(bool, copy=False).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected one scalar signal, got {array.shape}")
    return bool(array[0])


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty M2 base evaluation")
    first_success = [
        float(row["first_success_step"])
        for row in rows
        if row["first_success_step"] is not None
    ]
    return {
        "num_episodes": len(rows),
        "success_once": float(np.mean([bool(row["success_once"]) for row in rows])),
        "success_at_end": float(np.mean([bool(row["success_at_end"]) for row in rows])),
        "ever_grasped": float(np.mean([bool(row["is_cubeA_grasped"]) for row in rows])),
        "ever_on_cube": float(np.mean([bool(row["is_cubeA_on_cubeB"]) for row in rows])),
        "mean_first_success_step": float(np.mean(first_success)) if first_success else None,
        "action_clip_fraction": sum(int(row["action_clip_elements"]) for row in rows)
        / max(sum(int(row["action_total_elements"]) for row in rows), 1),
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in rows)
            for reason in sorted({row["termination_reason"] for row in rows})
        },
    }


def _evaluate_checkpoint(
    checkpoint: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    n_action_steps: int,
    max_steps: int,
    output_dir: str,
    policy_device: str,
    expected_action_dim: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint)
    policy, payload = load_action_flow_checkpoint(
        checkpoint_path,
        device=policy_device,
        n_action_steps=n_action_steps,
    )
    model_device = next(policy.model.parameters()).device
    expected_dim = int(payload["model_config"]["action_dim"])
    if expected_dim != expected_action_dim:
        raise ValueError(
            f"Checkpoint action_dim={expected_dim} does not match config action_dim={expected_action_dim}"
        )
    control_mode = CONTROL_MODE_BY_DIM.get(expected_dim)
    if control_mode is None:
        raise ValueError(f"Unsupported checkpoint action dimension: {expected_dim}")
    env = _make_cpu_env(control_mode=control_mode, max_steps=max_steps)
    episode_rows: list[dict[str, Any]] = []
    try:
        for row_index, row in enumerate(rows):
            observation = _reset_exact_state(env, row)
            info = env.unwrapped.get_info()
            signals = _signals(info, 1)
            success_once = bool(signals["success"][0])
            success_at_end = success_once
            first_success = 0 if success_once else None
            ever = {key: bool(signals[key][0]) for key in ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")}
            history: list[Mapping[str, Any]] = [observation, observation]
            policy.reset()
            terminated_at = None
            termination_reason = "episode_limit"
            action_clip_elements = 0
            action_total_elements = 0
            for step_index in range(1, max_steps + 1):
                context = _stack_context(history)
                context = {
                    key: value.to(device=model_device, non_blocking=model_device.type == "cuda")
                    for key, value in context.items()
                }
                action_tensor = policy.select_action(
                    context,
                    fingerprints=[str(row["start_state_fingerprint"])],
                )
                action = _to_numpy(action_tensor).astype(np.float32, copy=False)
                if action.shape != (1, expected_dim) or not np.isfinite(action).all():
                    raise ValueError(f"Checkpoint action has invalid shape/values: {action.shape}")
                observation, _, terminated, truncated, info = env.step(action)
                history.append(observation)
                history = history[-2:]
                signals = _signals(info, 1)
                success_once |= bool(signals["success"][0])
                success_at_end = bool(signals["success"][0])
                if success_once and first_success is None:
                    first_success = step_index
                for key in ever:
                    ever[key] |= bool(signals[key][0])
                raw_action = action
                action_clip_elements += int(np.sum(np.abs(raw_action) >= 1.0))
                action_total_elements += expected_dim
                terminated_value = _one_bool(terminated)
                truncated_value = _one_bool(truncated)
                if terminated_value or truncated_value:
                    terminated_at = step_index
                    termination_reason = "success" if bool(signals["success"][0]) else (
                        "truncated" if truncated_value else "terminated"
                    )
                    break
            episode_rows.append(
                {
                    "episode_id": int(row["episode_id"]),
                    "start_state_fingerprint": str(row["start_state_fingerprint"]),
                    "success_once": success_once,
                    "success_at_end": success_at_end,
                    "first_success_step": first_success,
                    "is_cubeA_grasped": ever["is_cubeA_grasped"],
                    "is_cubeA_on_cubeB": ever["is_cubeA_on_cubeB"],
                    "is_cubeA_static": ever["is_cubeA_static"],
                    "steps": int(terminated_at or max_steps),
                    "termination_reason": termination_reason,
                    "action_clip_elements": action_clip_elements,
                    "action_total_elements": action_total_elements,
                }
            )
            if (row_index + 1) % 10 == 0:
                print(
                    f"[M2-base-select] checkpoint={checkpoint_path.name} n={n_action_steps} "
                    f"progress={row_index + 1}/{len(rows)}",
                    flush=True,
                )
    finally:
        env.close()
    summary = _summarize_rows(episode_rows)
    metrics = {
        "schema_version": 1,
        "task": TASK_ID,
        "protocol": {
            "control_mode": control_mode,
            "action_dim": expected_dim,
            "sim_backend": "physx_cpu",
            "max_episode_steps": max_steps,
            "partial_reset": False,
            "identity": "start_state_fingerprint",
        },
        "checkpoint": _portable(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "n_action_steps": n_action_steps,
        "policy_device": policy_device,
        "summary": summary,
        "rows": episode_rows,
        "test_access_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target = _rooted(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "metrics.json", metrics)
    return metrics


def _evaluate_checkpoint_gpu(
    checkpoint: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    n_action_steps: int,
    max_steps: int,
    output_dir: str,
    expected_action_dim: int,
    num_envs: int,
    gpu_index: int,
    physical_gpu_id: str,
    progress_label: str = "M2-base-select",
) -> dict[str, Any]:
    """Evaluate one candidate on one visible GPU with exact batched states."""

    if not torch.cuda.is_available():
        raise RuntimeError("GPU M2 base selection requires CUDA")
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"GPU worker index {gpu_index} is outside the visible CUDA range "
            f"[0, {torch.cuda.device_count()})"
        )
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    started = time.perf_counter()
    torch.cuda.set_device(gpu_index)
    device = f"cuda:{gpu_index}"
    checkpoint_path = Path(checkpoint)
    policy, payload = load_action_flow_checkpoint(
        checkpoint_path,
        device=device,
        n_action_steps=n_action_steps,
    )
    expected_dim = int(payload["model_config"]["action_dim"])
    if expected_dim != expected_action_dim:
        raise ValueError(
            f"Checkpoint action_dim={expected_dim} does not match config action_dim={expected_action_dim}"
        )
    control_mode = CONTROL_MODE_BY_DIM.get(expected_dim)
    if control_mode is None:
        raise ValueError(f"Unsupported checkpoint action dimension: {expected_dim}")

    episode_rows: list[dict[str, Any]] = []
    batch_count = (len(rows) + num_envs - 1) // num_envs
    for batch_index, start in enumerate(range(0, len(rows), num_envs), start=1):
        batch_rows = rows[start : start + num_envs]
        env = _make_gpu_env(
            control_mode=control_mode,
            max_steps=max_steps,
            num_envs=len(batch_rows),
        )
        try:
            episode_rows.extend(
                _evaluate_gpu_batch(
                    env=env,
                    rows=batch_rows,
                    policy=policy,
                    max_steps=max_steps,
                    device=device,
                    expected_dim=expected_dim,
                    progress_prefix=(
                        f"[{progress_label} gpu={physical_gpu_id} "
                        f"checkpoint={checkpoint_path.stem} n={n_action_steps} "
                        f"batch={batch_index}/{batch_count}]"
                    ),
                    progress_offset=start,
                    progress_total=len(rows),
                )
            )
        finally:
            env.close()

    metrics = {
        "schema_version": 1,
        "task": TASK_ID,
        "protocol": {
            "control_mode": control_mode,
            "action_dim": expected_dim,
            "sim_backend": "physx_cuda",
            "max_episode_steps": max_steps,
            "partial_reset": False,
            "identity": "start_state_fingerprint",
        },
        "checkpoint": _portable(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "n_action_steps": n_action_steps,
        "policy_device": device,
        "rollout_parallelism": {
            "backend": "physx_cuda",
            "gpu_index": gpu_index,
            "physical_gpu_id": physical_gpu_id,
            "num_envs_per_batch": num_envs,
        },
        "summary": _summarize_rows(episode_rows),
        "rows": episode_rows,
        "test_access_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    target = _rooted(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "metrics.json", metrics)
    return metrics


def _selection_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    summary = candidate["summary"]
    return (
        -float(summary["success_at_end"]),
        -float(summary["success_once"]),
        -float(summary["ever_on_cube"]),
        float(summary["mean_first_success_step"])
        if summary["mean_first_success_step"] is not None
        else float("inf"),
        int(candidate["checkpoint_step"]),
        int(candidate["n_action_steps"]),
    )


def _evaluate_task(
    task: Mapping[str, Any],
    *,
    gpu_index: int | None = None,
    physical_gpu_id: str | None = None,
) -> dict[str, Any]:
    backend = str(task["rollout_backend"])
    if backend == "physx_cpu":
        return _evaluate_checkpoint(
            str(task["checkpoint"]),
            task["rows"],
            n_action_steps=int(task["n_action_steps"]),
            max_steps=int(task["max_steps"]),
            output_dir=str(task["output_dir"]),
            policy_device=str(task["policy_device"]),
            expected_action_dim=int(task["expected_action_dim"]),
        )
    if backend == "physx_cuda":
        if gpu_index is None:
            raise ValueError("GPU rollout task requires a GPU worker index")
        return _evaluate_checkpoint_gpu(
            str(task["checkpoint"]),
            task["rows"],
            n_action_steps=int(task["n_action_steps"]),
            max_steps=int(task["max_steps"]),
            output_dir=str(task["output_dir"]),
            expected_action_dim=int(task["expected_action_dim"]),
            num_envs=int(task["num_envs"]),
            gpu_index=gpu_index,
            physical_gpu_id=physical_gpu_id or str(gpu_index),
            progress_label=str(task.get("progress_label", "M2-base-select")),
        )
    raise ValueError(f"Unsupported M2 base rollout backend: {backend!r}")


def _evaluate_gpu_task_batch(
    tasks: Sequence[Mapping[str, Any]],
    physical_gpu_id: str,
) -> list[dict[str, Any]]:
    """Run candidates in a process that exposes exactly one physical GPU."""

    # ManiSkill calls torch.random.fork_rng() during reset without a devices
    # argument.  Restricting visibility before the first CUDA call makes that
    # call save/restore only this worker's device instead of every GPU in the
    # parent mask.  The evaluator therefore uses logical cuda:0 in each child.
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu_id
    results: list[dict[str, Any]] = []
    for task in tasks:
        results.append(
            _evaluate_task(
                task,
                gpu_index=0,
                physical_gpu_id=physical_gpu_id,
            )
        )
        torch.cuda.empty_cache()
    return results


def _visible_gpu_ids() -> list[str]:
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is not None and mask.strip():
        return [item.strip() for item in mask.split(",") if item.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def _prepare_selection_dir(path: str | Path, *, overwrite: bool) -> Path:
    target = _rooted(path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing existing M2 base selection directory: {_portable(target)}")
        output_root = (practice_root() / "outputs").resolve()
        if not target.resolve().is_relative_to(output_root):
            raise ValueError("--overwrite is restricted to an output directory under outputs/")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)
    return target


def select_development(
    config_path: str | Path,
    *,
    train_root: str | Path,
    output_root: str | Path,
    num_workers: int = 4,
    num_gpus: int | None = None,
    num_envs: int = DEFAULT_GPU_ENVS,
    policy_device: str = "auto",
    max_steps: int = DEFAULT_MAX_STEPS,
    n_action_steps: Sequence[int] = DEFAULT_N_ACTION_STEPS,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config, check_inputs=True)
    rollout_backend, resolved_policy_device = _resolve_rollout_route(config, policy_device)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    candidates_n = tuple(int(value) for value in n_action_steps)
    if any(value not in (4, 8) for value in candidates_n):
        raise ValueError("n_action_steps must contain only 4 or 8")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if rollout_backend == "physx_cuda":
        gpu_count = int(num_gpus or int(config["training"]["world_size"]))
        if gpu_count <= 0:
            raise ValueError("num_gpus must be positive for physx_cuda selection")
        if not torch.cuda.is_available():
            raise RuntimeError("physx_cuda selection requires CUDA")
        visible_gpu_ids = _visible_gpu_ids()
        visible_device_count = torch.cuda.device_count()
        if len(visible_gpu_ids) < gpu_count or visible_device_count < gpu_count:
            raise RuntimeError(
                f"requested {gpu_count} visible GPUs for selection, "
                f"but CUDA exposes {visible_device_count} devices "
                f"from mask {visible_gpu_ids}"
            )
    else:
        gpu_count = None
        if num_gpus is not None:
            raise ValueError("--num-gpus is only valid when protocol.backend=physx_cuda")
    train_dir = _rooted(train_root)
    train_manifest_path = train_dir / "run_manifest.json"
    if not train_manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {_portable(train_manifest_path)}")
    train_manifest = _read_json(train_manifest_path)
    expected_steps = int(config["training"]["optimizer_steps"])
    if train_manifest.get("status") != "complete" or int(train_manifest.get("completed_steps", -1)) != expected_steps:
        raise ValueError(f"M2 base selection requires a complete {expected_steps}-step training run")
    rows = _load_development_rows(config)
    output = _prepare_selection_dir(output_root, overwrite=overwrite)
    checkpoint_steps = tuple(int(value) for value in config["training"]["selection_checkpoint_steps"])
    tasks: list[dict[str, Any]] = []
    for step in checkpoint_steps:
        checkpoint = train_dir / f"checkpoint_step_{step:08d}.pt"
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise FileNotFoundError(f"Missing regular checkpoint: {_portable(checkpoint)}")
        for n_steps in candidates_n:
            candidate_dir = output / "candidates" / f"step_{step:08d}_n{n_steps}"
            tasks.append(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": step,
                    "n_action_steps": n_steps,
                    "max_steps": max_steps,
                    "output_dir": str(candidate_dir),
                    "policy_device": resolved_policy_device,
                    "rollout_backend": rollout_backend,
                    "num_envs": num_envs,
                    "expected_action_dim": int(config["protocol"]["action_dim"]),
                    "rows": rows,
                }
            )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    print(
        f"[M2-base-select] backend={rollout_backend} "
        f"policy_device={resolved_policy_device} candidates={len(tasks)} "
        + (
            f"gpu_workers={gpu_count} num_envs={num_envs}"
            if gpu_count is not None
            else f"cpu_workers={min(num_workers, len(tasks))}"
        ),
        flush=True,
    )

    def record_result(result: Mapping[str, Any], completed: int) -> None:
        results.append(dict(result))
        summary = result["summary"]
        print(
            f"[M2-base-select] candidates={completed}/{len(tasks)} "
            f"step={result['checkpoint_step']} n={result['n_action_steps']} "
            f"success_once={summary['success_once']:.3f} "
            f"success_at_end={summary['success_at_end']:.3f}",
            flush=True,
        )

    if rollout_backend == "physx_cpu":
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(num_workers, len(tasks))
        ) as executor:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                record_result(future.result(), index)
    else:
        assert gpu_count is not None
        task_batches = [tasks[index::gpu_count] for index in range(gpu_count)]
        task_batches = [batch for batch in task_batches if batch]
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(task_batches),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    _evaluate_gpu_task_batch,
                    batch,
                    visible_gpu_ids[gpu_index],
                )
                for gpu_index, batch in enumerate(task_batches)
            ]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                for result in future.result():
                    completed += 1
                    record_result(result, completed)
    results.sort(key=_selection_key)
    ranking = [
        {
            "rank": index,
            "checkpoint": result["checkpoint"],
            "checkpoint_step": result["checkpoint_step"],
            "n_action_steps": result["n_action_steps"],
            "summary": result["summary"],
            "metrics_path": _portable(output / "candidates" / f"step_{result['checkpoint_step']:08d}_n{result['n_action_steps']}" / "metrics.json"),
        }
        for index, result in enumerate(results, start=1)
    ]
    selected = ranking[0]
    selected_copy = output / "selected_checkpoint.pt"
    shutil.copy2(_rooted(selected["checkpoint"]), selected_copy)
    payload = {
        "schema_version": 1,
        "task": TASK_ID,
        "protocol": {
            "control_mode": config["protocol"]["control_mode"],
            "action_dim": int(config["protocol"]["action_dim"]),
            "sim_backend": rollout_backend,
            "max_episode_steps": max_steps,
            "development_episode_count": len(rows),
            "identity": "start_state_fingerprint",
        },
        "policy_device": resolved_policy_device,
        "rollout_parallelism": {
            "backend": rollout_backend,
            "num_gpus": gpu_count,
            "num_envs": num_envs if gpu_count is not None else 1,
            "cpu_workers": num_workers if gpu_count is None else 0,
        },
        "selection_rule": [
            "success_at_end_desc",
            "success_once_desc",
            "ever_on_cube_desc",
            "mean_first_success_step_asc",
            "checkpoint_step_asc",
            "n_action_steps_asc",
        ],
        "selected": selected,
        "selected_checkpoint_copy": _portable(selected_copy),
        "ranking": ranking,
        "candidates": len(results),
        "elapsed_seconds": time.perf_counter() - started,
        "test_access_count": 0,
        "config_sha256": hashlib.sha256(
            json.dumps({key: value for key, value in config.items() if not key.startswith("_")}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _write_json(output / "candidate_metrics.json", {"candidates": ranking, "test_access_count": 0})
    _write_json(output / "selected_checkpoint.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-development")
    select.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    select.add_argument("--train-root", type=Path, required=True)
    select.add_argument("--output-root", type=Path, required=True)
    select.add_argument("--num-workers", type=int, default=4)
    select.add_argument("--num-gpus", type=int, default=None)
    select.add_argument("--num-envs", type=int, default=DEFAULT_GPU_ENVS)
    select.add_argument("--policy-device", choices=("auto", "cpu", "cuda"), default="auto")
    select.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    select.add_argument("--n-action-steps", default="4,8")
    select.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "select-development":
        raise ValueError(f"Unsupported command: {args.command}")
    n_action_steps = tuple(int(value) for value in str(args.n_action_steps).split(",") if value)
    result = select_development(
        args.config,
        train_root=args.train_root,
        output_root=args.output_root,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        num_envs=args.num_envs,
        policy_device=args.policy_device,
        max_steps=args.max_steps,
        n_action_steps=n_action_steps,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["selected"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
