"""Development checkpoint selection for the generic M2 Action-only Flow.

The selector is deliberately separate from M2-W selection.  It evaluates the
official 4D RGB protocol on the configured development episodes only, using
CPU ManiSkill and exact initial ``env_states`` from the replay HDF5.  Each
candidate is one checkpoint paired with one ``n_action_steps`` value.

This is a selection/evaluation utility, not a training command.  It never
opens the configured test split and never reconstructs an episode from a seed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
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


def _load_development_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read only development initial states from the replay HDF5."""

    source = _rooted(config["inputs"]["source_h5"])
    rows: list[dict[str, Any]] = []
    with _open_h5_read_only(source) as handle:
        for episode_id in _split_ids(config, "development"):
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
        raise ValueError("M2 base development start-state fingerprints are not unique")
    return rows


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
    summary = {
        "num_episodes": len(episode_rows),
        "success_once": float(np.mean([row["success_once"] for row in episode_rows])),
        "success_at_end": float(np.mean([row["success_at_end"] for row in episode_rows])),
        "ever_grasped": float(np.mean([row["is_cubeA_grasped"] for row in episode_rows])),
        "ever_on_cube": float(np.mean([row["is_cubeA_on_cubeB"] for row in episode_rows])),
        "mean_first_success_step": float(
            np.mean([row["first_success_step"] for row in episode_rows if row["first_success_step"] is not None])
        )
        if any(row["first_success_step"] is not None for row in episode_rows)
        else None,
        "action_clip_fraction": sum(row["action_clip_elements"] for row in episode_rows)
        / max(sum(row["action_total_elements"] for row in episode_rows), 1),
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in episode_rows)
            for reason in sorted({row["termination_reason"] for row in episode_rows})
        },
    }
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


def _evaluate_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_checkpoint(
        str(task["checkpoint"]),
        task["rows"],
        n_action_steps=int(task["n_action_steps"]),
        max_steps=int(task["max_steps"]),
        output_dir=str(task["output_dir"]),
        policy_device=str(task["policy_device"]),
        expected_action_dim=int(task["expected_action_dim"]),
    )


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
    policy_device: str = "cpu",
    max_steps: int = DEFAULT_MAX_STEPS,
    n_action_steps: Sequence[int] = DEFAULT_N_ACTION_STEPS,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config, check_inputs=True)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    candidates_n = tuple(int(value) for value in n_action_steps)
    if any(value not in (4, 8) for value in candidates_n):
        raise ValueError("n_action_steps must contain only 4 or 8")
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
                    "policy_device": policy_device,
                    "expected_action_dim": int(config["protocol"]["action_dim"]),
                    "rows": rows,
                }
            )
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, len(tasks))) as executor:
        futures = [executor.submit(_evaluate_task, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            summary = result["summary"]
            print(
                f"[M2-base-select] candidates={index}/{len(tasks)} "
                f"step={result['checkpoint_step']} n={result['n_action_steps']} "
                f"success_once={summary['success_once']:.3f} "
                f"success_at_end={summary['success_at_end']:.3f}",
                flush=True,
            )
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
            "sim_backend": "physx_cpu",
            "max_episode_steps": max_steps,
            "development_episode_count": len(rows),
            "identity": "start_state_fingerprint",
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
    select.add_argument("--policy-device", default="cpu")
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
        policy_device=args.policy_device,
        max_steps=args.max_steps,
        n_action_steps=n_action_steps,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["selected"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
