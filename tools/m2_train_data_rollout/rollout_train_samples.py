"""Closed-loop rollout on a random sample of M2 training episodes.

This is intentionally an isolated tool. It uses exact ``env_states[0]`` from
the frozen D_core H5 as the reset state and never reconstructs an episode from
an episode seed. The policy then receives live RGB/qpos/qvel observations and
its own actions are stepped in ManiSkill.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from robot_practice.evaluation.common import practice_root, write_json_atomic
from robot_practice.experiments.maniskill.eval_stackcube_action_flow import (
    _signals,
    _stack_context,
)
from robot_practice.experiments.maniskill.state_identity import state_fingerprint
from robot_practice.policies.wam_tiny.action_flow import load_action_flow_checkpoint

TASK_ID = "StackCube-v1"
ACTION_DIM = 7
DEFAULT_MAX_STEPS = 500


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _portable(path: str | Path) -> str:
    value = _rooted(path).resolve()
    try:
        return value.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    with _rooted(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {_rooted(path)}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = _rooted(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, value)


def _resolve_device(value: str) -> torch.device:
    """Normalize ``cuda`` to the process-local logical device ``cuda:0``."""

    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def _split_episode_ids(split_manifest: Path, split: str) -> tuple[int, ...]:
    if split != "train":
        raise ValueError("This probe is intentionally restricted to the offline train split")
    payload = _read_json(split_manifest)
    values = payload.get(split)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Offline split manifest has no {split!r} episode IDs")
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"Offline split {split!r} contains duplicate episode IDs")
    return result


def _episode_group(handle: h5py.File, episode_id: int) -> h5py.Group:
    name = f"traj_{episode_id}"
    if name not in handle:
        raise KeyError(f"D_core is missing {name}")
    return handle[name]


def _load_sample_rows(
    *,
    d_core_h5: Path,
    split_manifest: Path,
    sample_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    episode_ids = _split_episode_ids(split_manifest, "train")
    if sample_count <= 0 or sample_count > len(episode_ids):
        raise ValueError(f"sample_count must be in [1, {len(episode_ids)}]")
    rng = random.Random(seed)
    selected_ids = sorted(rng.sample(list(episode_ids), sample_count))
    rows: list[dict[str, Any]] = []
    with h5py.File(_rooted(d_core_h5), "r") as handle:
        for episode_id in selected_ids:
            group = _episode_group(handle, episode_id)
            if "env_states" not in group:
                raise ValueError(f"{group.name} has no env_states")
            state = _state_at_zero(group["env_states"])
            rows.append(
                {
                    "episode_id": episode_id,
                    "source_group": group.name,
                    "source_action_count": int(group["actions"].shape[0]),
                    "state": state,
                    # Use the same canonical hash as the M1 state banks. This
                    # keeps Flow source-noise derivation comparable with the
                    # development evaluator; episode IDs are not identities.
                    "start_state_fingerprint": state_fingerprint(state, tolerance=1e-4),
                }
            )
    return rows


def _state_at_zero(node: h5py.Group | h5py.Dataset) -> Any:
    if isinstance(node, h5py.Dataset):
        value = np.asarray(node[0])
        return value.astype(np.float32).tolist()
    return {str(key): _state_at_zero(node[key]) for key in node}


def _state_batch(rows: Sequence[Mapping[str, Any]], *, device: torch.device) -> dict[str, Any]:
    first = rows[0]["state"]
    if not isinstance(first, Mapping):
        raise TypeError("env_states[0] must be a mapping")
    result: dict[str, Any] = {}
    for group_name, group in first.items():
        if not isinstance(group, Mapping):
            raise TypeError(f"state group {group_name!r} must be a mapping")
        result[group_name] = {}
        for object_name in group:
            values = []
            for row in rows:
                current_group = row["state"].get(group_name)
                if not isinstance(current_group, Mapping) or object_name not in current_group:
                    raise ValueError(f"state rows disagree at {group_name}/{object_name}")
                values.append(np.asarray(current_group[object_name], dtype=np.float32))
            result[group_name][object_name] = torch.as_tensor(
                np.stack(values, axis=0), dtype=torch.float32, device=device
            )
    return result


def _make_env(*, num_envs: int, max_steps: int) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        TASK_ID,
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        sim_backend="gpu",
        reward_mode="sparse",
        max_episode_steps=max_steps,
    )


def _reset_exact_state(env: Any, rows: Sequence[Mapping[str, Any]], device: torch.device) -> Any:
    # The reset seed only satisfies Gymnasium's API. The exact state is applied
    # immediately afterwards and is the identity used by this probe.
    env.reset(seed=[0] * len(rows))
    env.unwrapped.set_state_dict(_state_batch(rows, device=device))
    return env.unwrapped.get_obs()


def _bool_array(value: Any, batch_size: int) -> np.ndarray:
    array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
    array = array.astype(bool, copy=False).reshape(-1)
    if array.size == 1 and batch_size != 1:
        array = np.repeat(array, batch_size)
    if array.size != batch_size:
        raise ValueError(f"Expected {batch_size} boolean values, got {array.size}")
    return array


def _episode_result(
    *,
    row: Mapping[str, Any],
    success_once: bool,
    success_at_end: bool,
    ever: Mapping[str, bool],
    first_success_step: int | None,
    steps: int,
    termination_reason: str,
) -> dict[str, Any]:
    return {
        "episode_id": int(row["episode_id"]),
        "source_group": str(row["source_group"]),
        "source_action_count": int(row["source_action_count"]),
        "start_state_fingerprint": str(row["start_state_fingerprint"]),
        "success_once": bool(success_once),
        "success_at_end": bool(success_at_end),
        "first_success_step": first_success_step,
        "ever_grasped": bool(ever["is_cubeA_grasped"]),
        "ever_on_cube": bool(ever["is_cubeA_on_cubeB"]),
        "ever_static": bool(ever["is_cubeA_static"]),
        "steps": int(steps),
        "termination_reason": termination_reason,
    }


def rollout(
    *,
    checkpoint: Path,
    d_core_h5: Path,
    split_manifest: Path,
    output_dir: Path,
    sample_count: int,
    seed: int,
    max_steps: int,
    n_action_steps: int,
    device: str,
) -> dict[str, Any]:
    if n_action_steps not in (4, 8):
        raise ValueError("n_action_steps must be 4 or 8")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    output_dir = _rooted(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {_portable(output_dir)}")
    output_dir.mkdir(parents=True)
    rows = _load_sample_rows(
        d_core_h5=d_core_h5,
        split_manifest=split_manifest,
        sample_count=sample_count,
        seed=seed,
    )
    device_obj = _resolve_device(device)
    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device_obj)
    policy, checkpoint_payload = load_action_flow_checkpoint(
        _rooted(checkpoint), device=device_obj, n_action_steps=n_action_steps
    )
    policy.model.eval()
    env = _make_env(num_envs=len(rows), max_steps=max_steps)
    result_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        observation = _reset_exact_state(env, rows, device_obj)
        info = env.unwrapped.get_info()
        signals = _signals(info, len(rows))
        ever = {key: signals[key].copy() for key in (
            "is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static"
        )}
        success_once = signals["success"].copy()
        success_at_end = signals["success"].copy()
        first_success = np.full(len(rows), -1, dtype=np.int32)
        first_success[success_once] = 0
        finished = np.zeros(len(rows), dtype=bool)
        steps = np.zeros(len(rows), dtype=np.int32)
        reasons = ["episode_limit"] * len(rows)
        history: list[Mapping[str, Any]] = [observation, observation]
        fingerprints = [str(row["start_state_fingerprint"]) for row in rows]
        policy.reset()
        for step_index in range(1, max_steps + 1):
            context = _stack_context(history)
            action = policy.select_action(context, fingerprints=fingerprints)
            action = action.detach().cpu().numpy().astype(np.float32, copy=False)
            if action.shape != (len(rows), ACTION_DIM) or not np.isfinite(action).all():
                raise ValueError(f"Policy action must be finite [{len(rows)},{ACTION_DIM}]")
            action[finished] = 0.0
            action = np.clip(action, -1.0, 1.0)
            active = ~finished
            observation, _, terminated, truncated, info = env.step(action)
            history.append(observation)
            history = history[-2:]
            signals = _signals(info, len(rows))
            for key in ever:
                ever[key] |= signals[key]
            success_once[active] |= signals["success"][active]
            success_at_end[active] = signals["success"][active]
            newly_successful = active & signals["success"] & (first_success < 0)
            first_success[newly_successful] = step_index
            terminated_array = _bool_array(terminated, len(rows))
            truncated_array = _bool_array(truncated, len(rows))
            newly_finished = active & (terminated_array | truncated_array)
            for index in np.flatnonzero(newly_finished):
                steps[index] = step_index
                if signals["success"][index]:
                    reasons[index] = "success"
                elif truncated_array[index]:
                    reasons[index] = "truncated"
                elif signals["fail"][index]:
                    reasons[index] = "native_fail"
                else:
                    reasons[index] = "terminated"
            finished |= newly_finished
            if step_index == 1 or step_index % 10 == 0 or np.all(finished):
                done = int(finished.sum())
                print(
                    f"[M2 train-data rollout] step={step_index}/{max_steps} "
                    f"done={done}/{len(rows)} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if np.all(finished):
                break
        for index, row in enumerate(rows):
            if steps[index] == 0:
                steps[index] = max_steps
            result_rows.append(
                _episode_result(
                    row=row,
                    success_once=bool(success_once[index]),
                    success_at_end=bool(success_at_end[index]),
                    ever={key: bool(value[index]) for key, value in ever.items()},
                    first_success_step=None if first_success[index] < 0 else int(first_success[index]),
                    steps=int(steps[index]),
                    termination_reason=reasons[index],
                )
            )
    finally:
        env.close()
    result_rows.sort(key=lambda item: int(item["episode_id"]))
    summary = {
        "num_episodes": len(result_rows),
        "success_once": sum(bool(row["success_once"]) for row in result_rows) / len(result_rows),
        "success_at_end": sum(bool(row["success_at_end"]) for row in result_rows) / len(result_rows),
        "ever_grasped": sum(bool(row["ever_grasped"]) for row in result_rows) / len(result_rows),
        "ever_on_cube": sum(bool(row["ever_on_cube"]) for row in result_rows) / len(result_rows),
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in result_rows)
            for reason in sorted({row["termination_reason"] for row in result_rows})
        },
    }
    for row in result_rows:
        _write_json(output_dir / f"episode_{int(row['episode_id']):04d}.json", row)
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in result_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": TASK_ID,
        "purpose": "closed_loop_debug_probe_on_training_data",
        "checkpoint": _portable(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "d_core_h5": _portable(d_core_h5),
        "offline_split_manifest": _portable(split_manifest),
        "split": "train",
        "sample_count": sample_count,
        "sampling_seed": seed,
        "sampled_episode_ids": [int(row["episode_id"]) for row in rows],
        "sampled_start_state_fingerprints": [
            str(row["start_state_fingerprint"]) for row in rows
        ],
        "identity": "D_core env_states[0]; episode seed is not used",
        "device": str(device_obj),
        "control_mode": "pd_ee_delta_pose",
        "max_steps": max_steps,
        "n_action_steps": n_action_steps,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "rollout_summary.json", {"manifest": manifest, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return {"manifest": manifest, "summary": summary, "rows": result_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--d-core-h5",
        type=Path,
        default=Path("outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/split_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--n-action-steps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rollout(
        checkpoint=args.checkpoint,
        d_core_h5=args.d_core_h5,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        max_steps=args.max_steps,
        n_action_steps=args.n_action_steps,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
