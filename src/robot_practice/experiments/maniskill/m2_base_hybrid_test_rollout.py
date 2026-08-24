"""Evaluate a native 4D checkpoint with hybrid recorded test observations.

The policy receives 7D-source RGB frames and native-4D-source proprioception
from the same source episode and time index.  Its 4D actions are executed in a
GPU PhysX environment reset to the exact d_core test state.  Because the
recorded observations are teacher-forced, this is deliberately reported as a
hybrid observation diagnostic rather than a normal closed-loop policy score.
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from robot_practice.policies.wam_tiny.action_flow import (
    BASE_IMAGE_KEY,
    HAND_IMAGE_KEY,
    STATE_KEY,
    ActionOnlyFlowPolicy,
    load_action_flow_checkpoint,
)

from . import m2_base_action_flow as runner
from . import m2_base_checkpoint_selection as selector
from .state_identity import start_state_fingerprint

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pos"
DEFAULT_DCORE_CONFIG = Path(
    "configs/experiments/maniskill/m2_base_m1w_rgb_4d_success_at_end.json"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/"
    "formal_seed1000_100k/checkpoint_step_00100000.pt"
)
DEFAULT_NATIVE_FINAL_H5 = Path(
    "outputs/maniskill/m1_wam_native_4d/StackCube-v1/06_final_live_rgb/replay.h5"
)
DEFAULT_NATIVE_FINAL_JSON = Path(
    "outputs/maniskill/m1_wam_native_4d/StackCube-v1/06_final_live_rgb/replay.json"
)
DEFAULT_NATIVE_REPAIR_H5 = Path(
    "outputs/maniskill/m1_wam_native_4d/StackCube-v1/03_gpu_repair/repair.h5"
)
DEFAULT_NATIVE_REPAIR_JSON = Path(
    "outputs/maniskill/m1_wam_native_4d/StackCube-v1/03_gpu_repair/repair.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/maniskill/m2_base_action_flow/"
    "m1w_native_rgb_4d_hybrid_7d_rgb_native4dstate_test"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    target = runner._rooted(path)
    payload = runner._read_json(target)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {runner._portable(target)}")
    return payload


def _metadata_episode_map(path: str | Path) -> dict[int, int]:
    payload = _read_json(path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError(f"Native metadata has no episodes list: {runner._portable(path)}")
    result: dict[int, int] = {}
    for item in episodes:
        if not isinstance(item, Mapping):
            raise TypeError("Native episode metadata entries must be objects")
        source_id = int(item.get("source_episode_id", item["episode_id"]))
        episode_id = int(item["episode_id"])
        if source_id in result:
            raise ValueError(f"Duplicate native source_episode_id={source_id}")
        result[source_id] = episode_id
    return result


def _build_native_source_index(
    final_h5: str | Path,
    final_json: str | Path,
    repair_h5: str | Path,
    repair_json: str | Path,
) -> dict[int, dict[str, Any]]:
    final_map = _metadata_episode_map(final_json)
    repair_map = _metadata_episode_map(repair_json)
    index: dict[int, dict[str, Any]] = {}
    final_path = runner._rooted(final_h5)
    repair_path = runner._rooted(repair_h5)
    with h5py.File(final_path, "r") as handle:
        for source_id, episode_id in final_map.items():
            if f"traj_{episode_id}" in handle:
                index[source_id] = {
                    "h5": final_path,
                    "episode_id": episode_id,
                    "route": "native_final_live_rgb",
                }
    with h5py.File(repair_path, "r") as handle:
        for source_id, episode_id in repair_map.items():
            if source_id in index:
                continue
            if f"traj_{episode_id}" in handle:
                index[source_id] = {
                    "h5": repair_path,
                    "episode_id": episode_id,
                    "route": "native_repair_live_rgb",
                }
    return index


def _source_episode(
    dcore_group: h5py.Group,
    native_handle: h5py.File,
    native_episode_id: int,
    *,
    source_episode_id: int,
) -> dict[str, Any]:
    native_group = native_handle[f"traj_{native_episode_id}"]
    dcore_action_count = int(dcore_group["actions"].shape[0])
    native_action_count = int(native_group["actions"].shape[0])
    dcore_rgb_count = int(dcore_group["obs/sensor_data/base_camera/rgb"].shape[0])
    dcore_hand_rgb_count = int(dcore_group["obs/sensor_data/hand_camera/rgb"].shape[0])
    native_rgb_count = int(native_group["obs/sensor_data/base_camera/rgb"].shape[0])
    native_hand_rgb_count = int(native_group["obs/sensor_data/hand_camera/rgb"].shape[0])
    native_state_count = int(native_group["obs/agent/qpos"].shape[0])
    horizon = min(
        dcore_action_count,
        native_action_count,
        dcore_rgb_count,
        dcore_hand_rgb_count,
        native_rgb_count,
        native_hand_rgb_count,
        native_state_count,
    )
    if horizon <= 0:
        raise ValueError(f"Empty hybrid source episode {source_episode_id}")
    dcore_actions = np.asarray(dcore_group["actions"][:horizon], dtype=np.float32)
    if dcore_actions.shape[1] != 7:
        raise ValueError(f"d_core episode {source_episode_id} is not 7D")
    return {
        "source_episode_id": int(source_episode_id),
        "native_episode_id": int(native_episode_id),
        "horizon": int(horizon),
        "base_rgb": np.asarray(
            dcore_group["obs/sensor_data/base_camera/rgb"][:horizon], dtype=np.uint8
        ),
        "hand_rgb": np.asarray(
            dcore_group["obs/sensor_data/hand_camera/rgb"][:horizon], dtype=np.uint8
        ),
        "qpos": np.asarray(native_group["obs/agent/qpos"][:horizon], dtype=np.float32),
        "qvel": np.asarray(native_group["obs/agent/qvel"][:horizon], dtype=np.float32),
        "projected_actions": dcore_actions[:, [0, 1, 2, 6]],
        "source_action_count": dcore_action_count,
        "native_action_count": native_action_count,
    }


def _hybrid_observation(
    episodes: Sequence[Mapping[str, Any]],
    step_index: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    base_rows: list[np.ndarray] = []
    hand_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    for episode in episodes:
        horizon = int(episode["horizon"])
        current = min(step_index, horizon - 1)
        previous = max(0, current - 1)
        base_rows.append(np.stack((episode["base_rgb"][previous], episode["base_rgb"][current])))
        hand_rows.append(np.stack((episode["hand_rgb"][previous], episode["hand_rgb"][current])))
        state_rows.append(
            np.stack(
                (
                    np.concatenate((episode["qpos"][previous], episode["qvel"][previous])),
                    np.concatenate((episode["qpos"][current], episode["qvel"][current])),
                )
            )
        )
    base = torch.from_numpy(np.stack(base_rows)).permute(0, 1, 4, 2, 3).float() / 255.0
    hand = torch.from_numpy(np.stack(hand_rows)).permute(0, 1, 4, 2, 3).float() / 255.0
    state = torch.from_numpy(np.stack(state_rows)).float()
    return {
        BASE_IMAGE_KEY: base.to(device=device, non_blocking=True),
        HAND_IMAGE_KEY: hand.to(device=device, non_blocking=True),
        STATE_KEY: state.to(device=device, non_blocking=True),
    }


def _make_env(*, num_envs: int, max_steps: int) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        TASK_ID,
        num_envs=num_envs,
        obs_mode="none",
        control_mode=CONTROL_MODE,
        sim_backend="gpu",
        reward_mode="sparse",
        max_episode_steps=max_steps,
    )


def _bool_array(value: Any, batch_size: int) -> np.ndarray:
    array = selector._to_numpy(value).astype(bool, copy=False).reshape(-1)
    if array.size == 1 and batch_size != 1:
        array = np.repeat(array, batch_size)
    if array.size != batch_size:
        raise ValueError(f"Expected {batch_size} boolean values, got {array.size}")
    return array.copy()


def _evaluate_batch(
    *,
    env: Any,
    episodes: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    policy: ActionOnlyFlowPolicy,
    device: torch.device,
) -> list[dict[str, Any]]:
    batch_size = len(episodes)
    _ = selector._reset_exact_gpu_state(env, rows, device=str(device))
    signals = selector._signals(env.unwrapped.get_info(), batch_size)
    ever = {key: signals[key].copy() for key in ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")}
    success_once = signals["success"].copy()
    success_at_end = signals["success"].copy()
    finished = np.zeros(batch_size, dtype=bool)
    source_done = np.zeros(batch_size, dtype=bool)
    first_success = np.full(batch_size, -1, dtype=np.int32)
    steps = np.zeros(batch_size, dtype=np.int32)
    reasons = ["source_horizon"] * batch_size
    action_abs_error = np.zeros(batch_size, dtype=np.float64)
    action_squared_error = np.zeros(batch_size, dtype=np.float64)
    action_error_count = np.zeros(batch_size, dtype=np.int64)
    policy.reset()
    fingerprints = [str(row["start_state_fingerprint"]) for row in rows]
    max_source_horizon = max(int(item["horizon"]) for item in episodes)
    for step_index in range(max_source_horizon):
        observations = _hybrid_observation(episodes, step_index, device=device)
        action_tensor = policy.select_action(observations, fingerprints=fingerprints)
        action = selector._to_numpy(action_tensor).astype(np.float32, copy=False)
        if action.shape != (batch_size, 4) or not np.isfinite(action).all():
            raise ValueError(f"Hybrid policy action has invalid shape/values: {action.shape}")
        active = ~finished & ~source_done
        for index, episode in enumerate(episodes):
            if step_index >= int(episode["horizon"]):
                source_done[index] = True
        active = ~finished & ~source_done
        target = np.stack(
            [np.asarray(episode["projected_actions"][min(step_index, int(episode["horizon"]) - 1)]) for episode in episodes]
        )
        action_abs_error[active] += np.abs(action[active] - target[active]).sum(axis=1)
        action_squared_error[active] += np.square(action[active] - target[active]).sum(axis=1)
        action_error_count[active] += 4
        action[~active] = 0.0
        _, _, terminated, truncated, info = env.step(torch.as_tensor(action, device=device))
        signals = selector._signals(info, batch_size)
        for key in ever:
            ever[key] |= signals[key]
        success_once[active] |= signals["success"][active]
        success_at_end[active] = signals["success"][active]
        newly_successful = active & signals["success"] & (first_success < 0)
        first_success[newly_successful] = step_index + 1
        terminated_array = _bool_array(terminated, batch_size)
        truncated_array = _bool_array(truncated, batch_size)
        newly_finished = active & (terminated_array | truncated_array)
        for index in np.flatnonzero(newly_finished):
            steps[index] = step_index + 1
            reasons[index] = selector._gpu_termination_reason(
                success=bool(signals["success"][index]),
                truncated=bool(truncated_array[index]),
                terminated=bool(terminated_array[index]),
            )
        finished |= terminated_array | truncated_array
        for index in np.flatnonzero(source_done & ~finished):
            if steps[index] == 0:
                steps[index] = min(step_index + 1, int(episodes[index]["horizon"]))
                reasons[index] = "source_horizon"
        if bool(np.all(finished | source_done)):
            break
    for index in range(batch_size):
        if steps[index] == 0:
            steps[index] = int(episodes[index]["horizon"])
            reasons[index] = "source_horizon"
    return [
        {
            "episode_id": int(row["episode_id"]),
            "source_episode_id": int(episodes[index]["source_episode_id"]),
            "native_episode_id": int(episodes[index]["native_episode_id"]),
            "native_source_route": str(episodes[index]["native_source_route"]),
            "source_horizon": int(episodes[index]["horizon"]),
            "dcore_start_state_fingerprint": str(
                episodes[index]["dcore_start_state_fingerprint"]
            ),
            "native_start_state_fingerprint": str(
                episodes[index]["native_start_state_fingerprint"]
            ),
            "state_identity_match": bool(episodes[index]["state_identity_match"]),
            "success_once": bool(success_once[index]),
            "success_at_end": bool(success_at_end[index]),
            "first_success_step": None if first_success[index] < 0 else int(first_success[index]),
            "is_cubeA_grasped": bool(ever["is_cubeA_grasped"][index]),
            "is_cubeA_on_cubeB": bool(ever["is_cubeA_on_cubeB"][index]),
            "is_cubeA_static": bool(ever["is_cubeA_static"][index]),
            "steps": int(steps[index]),
            "termination_reason": reasons[index],
            "projected_action_mae": float(
                action_abs_error[index] / max(int(action_error_count[index]), 1)
            ),
            "projected_action_mse": float(
                action_squared_error[index] / max(int(action_error_count[index]), 1)
            ),
            "action_error_count": int(action_error_count[index]),
        }
        for index, row in enumerate(rows)
    ]


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty hybrid rollout")
    matched_rows = [row for row in rows if bool(row["state_identity_match"])]
    mismatched_rows = [row for row in rows if not bool(row["state_identity_match"])]
    return {
        "num_episodes": len(rows),
        "success_once": float(np.mean([bool(row["success_once"]) for row in rows])),
        "success_at_end": float(np.mean([bool(row["success_at_end"]) for row in rows])),
        "ever_grasped": float(np.mean([bool(row["is_cubeA_grasped"]) for row in rows])),
        "ever_on_cube": float(np.mean([bool(row["is_cubeA_on_cubeB"]) for row in rows])),
        "mean_first_success_step": float(
            np.mean([row["first_success_step"] for row in rows if row["first_success_step"] is not None])
        )
        if any(row["first_success_step"] is not None for row in rows)
        else None,
        "mean_source_horizon": float(np.mean([int(row["source_horizon"]) for row in rows])),
        "projected_action_mae": float(np.mean([float(row["projected_action_mae"]) for row in rows])),
        "projected_action_mse": float(np.mean([float(row["projected_action_mse"]) for row in rows])),
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in rows)
            for reason in sorted({str(row["termination_reason"]) for row in rows})
        },
        "native_source_routes": {
            route: sum(row["native_source_route"] == route for row in rows)
            for route in sorted({str(row["native_source_route"]) for row in rows})
        },
        "state_identity": {
            "matched_episodes": len(matched_rows),
            "mismatched_episodes": len(mismatched_rows),
            "mismatched_episode_ids": [int(row["episode_id"]) for row in mismatched_rows],
            "matched_success_once": float(np.mean([bool(row["success_once"]) for row in matched_rows]))
            if matched_rows
            else None,
            "matched_success_at_end": float(
                np.mean([bool(row["success_at_end"]) for row in matched_rows])
            )
            if matched_rows
            else None,
        },
    }


def run_hybrid_rollout(
    *,
    dcore_config_path: str | Path,
    checkpoint_path: str | Path,
    native_final_h5: str | Path,
    native_final_json: str | Path,
    native_repair_h5: str | Path,
    native_repair_json: str | Path,
    output_root: str | Path,
    num_envs: int,
    max_steps: int,
    overwrite: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid test rollout requires CUDA")
    if num_envs <= 0 or max_steps <= 0:
        raise ValueError("num_envs and max_steps must be positive")
    config = runner.load_config(dcore_config_path)
    validation = runner.validate_config(config, num_gpus=1, check_inputs=True)
    if validation["action_dim"] != 4 or validation["source_action_dim"] != 7:
        raise ValueError("Hybrid rollout requires a 7D source projected to action_dim=4")
    checkpoint = runner._rooted(checkpoint_path)
    policy, checkpoint_payload = load_action_flow_checkpoint(checkpoint, device="cuda:0", n_action_steps=4)
    policy.model.eval()
    device = torch.device("cuda:0")
    test_rows = selector._load_split_rows(config, "test")
    source_index = _build_native_source_index(
        native_final_h5,
        native_final_json,
        native_repair_h5,
        native_repair_json,
    )
    missing = [int(row["episode_id"]) for row in test_rows if int(row["episode_id"]) not in source_index]
    if missing:
        raise FileNotFoundError(f"Native 4D RGB/state is missing test episodes: {missing}")
    dcore_path = runner._rooted(config["inputs"]["source_h5"])
    result_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with h5py.File(dcore_path, "r") as dcore, h5py.File(native_final_h5, "r") as final_h5, h5py.File(
        native_repair_h5, "r"
    ) as repair_h5:
        for start in range(0, len(test_rows), num_envs):
            batch_rows = test_rows[start : start + num_envs]
            episodes: list[dict[str, Any]] = []
            for row in batch_rows:
                source_id = int(row["episode_id"])
                native_source = source_index[source_id]
                native_handle = final_h5 if native_source["route"] == "native_final_live_rgb" else repair_h5
                episode = _source_episode(
                    dcore[f"traj_{source_id}"],
                    native_handle,
                    int(native_source["episode_id"]),
                    source_episode_id=source_id,
                )
                native_group = native_handle[f"traj_{int(native_source['episode_id'])}"]
                dcore_start = start_state_fingerprint(dcore[f"traj_{source_id}"])
                native_start = start_state_fingerprint(native_group)
                episode["native_source_route"] = native_source["route"]
                episode["dcore_start_state_fingerprint"] = dcore_start
                episode["native_start_state_fingerprint"] = native_start
                episode["state_identity_match"] = native_start == dcore_start
                episodes.append(episode)
            env = _make_env(num_envs=len(batch_rows), max_steps=max_steps)
            try:
                result_rows.extend(
                    _evaluate_batch(
                        env=env,
                        episodes=episodes,
                        rows=batch_rows,
                        policy=policy,
                        device=device,
                    )
                )
            finally:
                env.close()
            print(
                f"[M2-base-hybrid-test] episodes={min(start + num_envs, len(test_rows))}/{len(test_rows)}",
                flush=True,
            )
    result_rows.sort(key=lambda item: int(item["episode_id"]))
    target = runner._rooted(output_root)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Hybrid rollout output exists: {runner._portable(target)}")
        outputs_root = (runner.practice_root() / "outputs").resolve()
        if not target.resolve().is_relative_to(outputs_root):
            raise ValueError("--overwrite is restricted to an output directory under outputs/")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "task": TASK_ID,
        "evaluation_kind": "hybrid_teacher_forced_observation_gpu_action_execution",
        "checkpoint": runner._portable(checkpoint),
        "checkpoint_step": int(checkpoint_payload["step"]),
        "source": {
            "dcore_config": runner._portable(runner._rooted(dcore_config_path)),
            "dcore_h5": runner._portable(dcore_path),
            "native_final_h5": runner._portable(runner._rooted(native_final_h5)),
            "native_repair_h5": runner._portable(runner._rooted(native_repair_h5)),
            "rgb": "d_core 7D source RGB",
            "state": "native 4D source qpos+qvel",
            "action_target": "d_core action[:, [0,1,2,6]]",
            "state_identity_check": (
                "canonical env_states[0] fingerprint is recorded per episode; mismatches are retained "
                "in the all-test diagnostic and reported separately"
            ),
        },
        "protocol": {
            "task": TASK_ID,
            "control_mode": CONTROL_MODE,
            "action_dim": 4,
            "sim_backend": "physx_cuda",
            "max_episode_steps": max_steps,
            "n_action_steps": 4,
            "observation_feedback": "recorded_hybrid_teacher_forced",
            "action_execution": "GPU PhysX from exact d_core test initial state",
            "normalization": "native checkpoint normalization for RGB/state/action",
        },
        "rollout_parallelism": {
            "num_gpus": 1,
            "num_envs_per_batch": num_envs,
            "test_episodes": len(test_rows),
        },
        "summary": _summary(result_rows),
        "rows": result_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    runner._write_json(target / "metrics.json", payload)
    runner._write_json(
        target / "manifest.json",
        {
            "schema_version": 1,
            "task": TASK_ID,
            "evaluation_kind": payload["evaluation_kind"],
            "checkpoint": payload["checkpoint"],
            "checkpoint_step": payload["checkpoint_step"],
            "test_episodes": len(result_rows),
            "native_source_routes": payload["summary"]["native_source_routes"],
        },
    )
    print(
        f"[M2-base-hybrid-test] complete episodes={len(result_rows)} "
        f"success_once={payload['summary']['success_once']:.4f} "
        f"success_at_end={payload['summary']['success_at_end']:.4f} "
        f"elapsed={payload['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcore-config", type=Path, default=DEFAULT_DCORE_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--native-final-h5", type=Path, default=DEFAULT_NATIVE_FINAL_H5)
    parser.add_argument("--native-final-json", type=Path, default=DEFAULT_NATIVE_FINAL_JSON)
    parser.add_argument("--native-repair-h5", type=Path, default=DEFAULT_NATIVE_REPAIR_H5)
    parser.add_argument("--native-repair-json", type=Path, default=DEFAULT_NATIVE_REPAIR_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_hybrid_rollout(
        dcore_config_path=args.dcore_config,
        checkpoint_path=args.checkpoint,
        native_final_h5=args.native_final_h5,
        native_final_json=args.native_final_json,
        native_repair_h5=args.native_repair_h5,
        native_repair_json=args.native_repair_json,
        output_root=args.output_root,
        num_envs=args.num_envs,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
