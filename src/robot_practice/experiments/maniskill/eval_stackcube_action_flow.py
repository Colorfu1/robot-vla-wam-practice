"""Development closed-loop evaluator for the M2-W Action-only Flow policy."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_practice.evaluation.common import practice_root, write_json_atomic
from robot_practice.policies.wam_tiny.action_flow import (
    BASE_IMAGE_KEY,
    HAND_IMAGE_KEY,
    STATE_KEY,
    ActionOnlyFlowPolicy,
    load_action_flow_checkpoint,
)

from .eval_stackcube_gpu import (
    _reset_exact_state,
    _signals,
    _to_numpy,
    build_stackcube_policy_observation,
    load_state_bank,
)
from .rollout_contract import M1W_DATA_PROVENANCE, require_rollout_backend

TASK_ID = "StackCube-v1"
FORMAL_HORIZON = 500
PROGRESS_INTERVAL_STEPS = 10


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _portable(path: str | Path) -> str:
    value = _rooted(path).resolve()
    try:
        return value.relative_to(practice_root()).as_posix()
    except ValueError:
        return value.as_posix()


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = _rooted(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, value)


def _worker_log(worker_index: int, message: str) -> None:
    print(f"[M2-W worker={worker_index} pid={os.getpid()}] {message}", flush=True)


def _write_worker_progress(
    output_dir: str | Path,
    *,
    worker_index: int,
    worker_count: int,
    status: str,
    completed_episodes: int,
    total_episodes: int,
    current_batch_episodes: int,
    current_step: int,
    max_steps: int,
    worker_device: str,
    environment_device: str,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "completed_episodes": completed_episodes,
        "total_episodes": total_episodes,
        "current_batch_episodes": current_batch_episodes,
        "current_step": current_step,
        "max_steps": max_steps,
        "worker_device": worker_device,
        "environment_device": environment_device,
    }
    physical_device = os.environ.get("M2_W_PHYSICAL_GPU")
    if physical_device is not None:
        payload["physical_device"] = physical_device
    if error is not None:
        payload["error"] = error
    _write_json(
        _rooted(output_dir) / f"progress_worker_{worker_index:03d}.json",
        payload,
    )


def _make_env(
    num_envs: int,
    max_steps: int,
    *,
    device: torch.device,
    worker_index: int,
) -> Any:
    _worker_log(worker_index, "gym import begin")
    import gymnasium as gym

    _worker_log(worker_index, "gym import complete; ManiSkill import begin")
    import mani_skill.envs  # noqa: F401

    _worker_log(worker_index, "ManiSkill import complete")
    # Use ManiSkill's public backend contract.  The process-level
    # CUDA_VISIBLE_DEVICES mask already selects the physical GPU, so the
    # worker must use logical device 0.  The older explicit
    # physx_cuda:N/sapien_cuda:N spelling can hang during gym.make on some
    # driver/device combinations.
    sim_backend = "gpu" if device.type == "cuda" else "cpu"
    require_rollout_backend(M1W_DATA_PROVENANCE, sim_backend)
    _worker_log(worker_index, f"gym.make begin sim_backend={sim_backend} num_envs={num_envs}")
    environment = gym.make(
        TASK_ID,
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        sim_backend=sim_backend,
        reward_mode="sparse",
        max_episode_steps=max_steps,
    )
    _worker_log(worker_index, "gym.make complete")
    return environment


def _stack_context(observations: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    frames = [build_stackcube_policy_observation(item) for item in observations[-2:]]
    return {
        key: torch.stack([frame[key] for frame in frames], dim=1)
        for key in (BASE_IMAGE_KEY, HAND_IMAGE_KEY, STATE_KEY)
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _bool_array(value: Any, batch_size: int) -> np.ndarray:
    array = _to_numpy(value).astype(bool, copy=False).reshape(-1)
    if array.size == 1 and batch_size != 1:
        array = np.repeat(array, batch_size)
    if array.size != batch_size:
        raise ValueError(f"Expected {batch_size} boolean values, got {array.size}")
    return array


def _reference_actions(
    reference: str,
    *,
    batch_size: int,
    rngs: Sequence[np.random.Generator],
) -> np.ndarray:
    if reference == "zero":
        return np.zeros((batch_size, 7), dtype=np.float32)
    if reference != "iid_gaussian_random":
        raise ValueError(f"Unknown M2-W reference policy {reference!r}")
    return np.stack([np.clip(rng.normal(size=7), -1.0, 1.0) for rng in rngs]).astype(np.float32)


def run_development_rollout(
    *,
    state_bank: str | Path,
    output_dir: str | Path,
    n_action_steps: int = 4,
    checkpoint: str | Path | None = None,
    reference: str | None = None,
    max_episodes: int = 200,
    max_steps: int = FORMAL_HORIZON,
    num_envs: int = 64,
    device: str | None = None,
    worker_index: int = 0,
    worker_count: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Evaluate a candidate/reference on exact development states.

    A worker receives a stable slice of the ordered development bank.  The
    caller merges worker rows; no worker ever loads the state-bank test file.
    """

    if not (checkpoint is None) ^ (reference is None):
        raise ValueError("Provide exactly one of checkpoint or reference")
    if checkpoint is not None and n_action_steps not in (4, 8):
        raise ValueError("M2-W candidate n_action_steps must be 4 or 8")
    if not smoke and max_steps != FORMAL_HORIZON:
        raise ValueError("M2-W closed-loop horizon is fixed at 500")
    if smoke and (max_episodes > 10 or max_steps <= 0 or max_steps > 20):
        raise ValueError("M2-W smoke rollout is limited to at most 10 states and 20 steps")
    if max_episodes <= 0 or num_envs <= 0 or worker_count <= 0 or worker_index < 0:
        raise ValueError("invalid rollout worker/episode arguments")
    state_bank_path = _rooted(state_bank)
    if state_bank_path.name != "development.jsonl":
        raise ValueError(
            "M2-W development rollout accepts only the frozen development.jsonl state bank; "
            "test state banks are forbidden"
        )
    rows = load_state_bank(state_bank_path, max_episodes=None)
    if len(rows) < max_episodes:
        raise ValueError(
            f"development state bank has {len(rows)} rows, expected at least {max_episodes}"
        )
    rows = rows[:max_episodes]
    if worker_index >= worker_count:
        raise ValueError("worker_index must be smaller than worker_count")
    rows = rows[worker_index::worker_count]
    if not rows:
        raise ValueError(f"worker {worker_index} received no development states")

    requested_device = device or (f"cuda:{worker_index}" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M2-W GPU rollout requires CUDA")
    device_obj = torch.device(requested_device)
    if device_obj.type == "cuda":
        _worker_log(
            worker_index,
            "cuda probe before set_device: "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
            f"M2_W_PHYSICAL_GPU={os.environ.get('M2_W_PHYSICAL_GPU')!r} "
            f"requested={device_obj}",
        )
        visible_count = torch.cuda.device_count()
        _worker_log(worker_index, f"cuda visible device_count={visible_count}")
        torch.cuda.set_device(device_obj)
        _worker_log(
            worker_index,
            f"cuda set_device complete current={torch.cuda.current_device()} "
            f"name={torch.cuda.get_device_name(device_obj)}",
        )
    else:
        _worker_log(worker_index, f"using CPU device={device_obj}")
    total_worker_episodes = len(rows)
    environment_device = "pending"
    _write_worker_progress(
        output_dir,
        worker_index=worker_index,
        worker_count=worker_count,
        status="starting",
        completed_episodes=0,
        total_episodes=total_worker_episodes,
        current_batch_episodes=0,
        current_step=0,
        max_steps=max_steps,
        worker_device=str(device_obj),
        environment_device=environment_device,
    )
    policy: ActionOnlyFlowPolicy | None = None
    checkpoint_payload: dict[str, Any] | None = None
    env: Any | None = None
    last_step_index = 0
    try:
        if checkpoint is not None:
            _worker_log(worker_index, f"policy load begin checkpoint={_portable(checkpoint)}")
            policy, checkpoint_payload = load_action_flow_checkpoint(
                _rooted(checkpoint), device=device_obj, n_action_steps=n_action_steps
            )
            policy.model.eval()
            _worker_log(worker_index, "policy load complete")
        batch_capacity = min(num_envs, len(rows))
        _worker_log(
            worker_index,
            f"env creation begin num_envs={batch_capacity} max_steps={max_steps} device={device_obj}",
        )
        env = _make_env(
            batch_capacity,
            max_steps,
            device=device_obj,
            worker_index=worker_index,
        )
        environment_device = str(getattr(env.unwrapped, "device", device_obj))
        _worker_log(worker_index, f"env creation complete environment_device={environment_device}")
    except BaseException as exc:
        _worker_log(worker_index, f"initialization failed: {type(exc).__name__}: {exc}")
        _write_worker_progress(
            output_dir,
            worker_index=worker_index,
            worker_count=worker_count,
            status="failed",
            completed_episodes=0,
            total_episodes=total_worker_episodes,
            current_batch_episodes=0,
            current_step=0,
            max_steps=max_steps,
            worker_device=str(device_obj),
            environment_device=environment_device,
            error=f"{type(exc).__name__}: {exc}",
        )
        if env is not None:
            env.close()
        raise
    _write_worker_progress(
        output_dir,
        worker_index=worker_index,
        worker_count=worker_count,
        status="running",
        completed_episodes=0,
        total_episodes=total_worker_episodes,
        current_batch_episodes=0,
        current_step=0,
        max_steps=max_steps,
        worker_device=str(device_obj),
        environment_device=environment_device,
    )
    environment_latencies_ms: list[float] = []
    chunk_latencies_ms: list[float] = []
    result_rows: list[dict[str, Any]] = []
    failed = False
    try:
        batch_size = batch_capacity
        for start in range(0, len(rows), batch_size):
            real_rows = rows[start : start + batch_size]
            completed_before_batch = len(result_rows)
            last_step_index = 0
            _write_worker_progress(
                output_dir,
                worker_index=worker_index,
                worker_count=worker_count,
                status="running",
                completed_episodes=completed_before_batch,
                total_episodes=total_worker_episodes,
                current_batch_episodes=len(real_rows),
                current_step=0,
                max_steps=max_steps,
                worker_device=str(device_obj),
                environment_device=environment_device,
            )
            batch_rows = real_rows + [real_rows[-1]] * (batch_size - len(real_rows))
            batch_size_actual = len(batch_rows)
            _worker_log(
                worker_index,
                f"reset begin batch_start={start} real_episodes={len(real_rows)} "
                f"batch_size={batch_size_actual}",
            )
            observation = _reset_exact_state(
                env,
                batch_rows,
                device=str(device_obj),
            )
            _worker_log(worker_index, f"reset complete batch_start={start}")
            initial_info = env.unwrapped.get_info()
            signals = _signals(initial_info, batch_size_actual)
            ever = {
                key: signals[key].copy()
                for key in ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")
            }
            success_once = signals["success"].copy()
            success_at_end = signals["success"].copy()
            first_success = np.full(batch_size_actual, -1, dtype=np.int32)
            first_success[success_once] = 0
            finished = np.zeros(batch_size_actual, dtype=bool)
            reasons = ["episode_limit"] * batch_size_actual
            steps = np.zeros(batch_size_actual, dtype=np.int32)
            action_clip_count = np.zeros(batch_size_actual, dtype=np.int64)
            action_count = np.zeros(batch_size_actual, dtype=np.int64)
            previous_action = np.zeros((batch_size_actual, 7), dtype=np.float32)
            has_previous = np.zeros(batch_size_actual, dtype=bool)
            action_jitter = np.zeros(batch_size_actual, dtype=np.float64)
            gripper_flips = np.zeros(batch_size_actual, dtype=np.int32)
            rngs = [
                np.random.default_rng(
                    int.from_bytes(
                        str(row["start_state_fingerprint"]).encode()[:8].ljust(8, b"0"), "big"
                    )
                )
                for row in batch_rows
            ]
            if policy is not None:
                policy.reset()
            history: list[Mapping[str, Any]] = [observation, observation]
            fingerprints = [str(row["start_state_fingerprint"]) for row in batch_rows]
            for step_index in range(1, max_steps + 1):
                last_step_index = step_index
                if policy is not None:
                    context = _stack_context(history)
                    before_chunk = policy.chunk_index
                    started = time.perf_counter()
                    action_tensor = policy.select_action(context, fingerprints=fingerprints)
                    elapsed = (time.perf_counter() - started) * 1000.0
                    if policy.chunk_index != before_chunk:
                        chunk_latencies_ms.append(elapsed)
                    action = _to_numpy(action_tensor).astype(np.float32, copy=False)
                else:
                    action = _reference_actions(
                        reference or "", batch_size=batch_size_actual, rngs=rngs
                    )
                if action.shape != (batch_size_actual, 7) or not np.isfinite(action).all():
                    raise ValueError(
                        f"M2-W environment action must be finite {(batch_size_actual, 7)}, got {action.shape}"
                    )
                action[finished] = 0.0
                raw_action = action.copy()
                action = np.clip(action, -1.0, 1.0)
                action_clip_count += np.sum(raw_action != action, axis=1)
                action_count += 7
                active = ~finished
                if np.any(active & has_previous):
                    comparable = active & has_previous
                    action_jitter[comparable] += np.linalg.norm(
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
                history.append(observation)
                history = history[-2:]
                signals = _signals(info, batch_size_actual)
                for key in ever:
                    ever[key] |= signals[key]
                success_once[active] |= signals["success"][active]
                success_at_end[active] = signals["success"][active]
                newly_successful = active & signals["success"] & (first_success < 0)
                first_success[newly_successful] = step_index
                terminated_array = _bool_array(terminated, batch_size_actual)
                truncated_array = _bool_array(truncated, batch_size_actual)
                newly_finished = active & (terminated_array | truncated_array)
                for item in np.flatnonzero(newly_finished):
                    steps[item] = step_index
                    if signals["success"][item]:
                        reasons[item] = "success"
                    elif truncated_array[item]:
                        reasons[item] = "truncated"
                    elif signals["fail"][item]:
                        reasons[item] = "native_fail"
                    else:
                        reasons[item] = "terminated"
                finished |= newly_finished
                if (
                    step_index % PROGRESS_INTERVAL_STEPS == 0
                    or step_index == max_steps
                    or np.any(newly_finished)
                ):
                    _write_worker_progress(
                        output_dir,
                        worker_index=worker_index,
                        worker_count=worker_count,
                        status="running",
                        completed_episodes=completed_before_batch,
                        total_episodes=total_worker_episodes,
                        current_batch_episodes=len(real_rows),
                        current_step=step_index,
                        max_steps=max_steps,
                        worker_device=str(device_obj),
                        environment_device=environment_device,
                    )
                if np.all(finished):
                    break
            for item, row in enumerate(real_rows):
                result_rows.append(
                    {
                        "index": int(row["index"]),
                        "start_state_fingerprint": str(row["start_state_fingerprint"]),
                        "success_once": bool(success_once[item]),
                        "success_at_end": bool(success_at_end[item]),
                        "first_success_step": None
                        if first_success[item] < 0
                        else int(first_success[item]),
                        "is_cubeA_grasped": bool(ever["is_cubeA_grasped"][item]),
                        "is_cubeA_on_cubeB": bool(ever["is_cubeA_on_cubeB"][item]),
                        "is_cubeA_static": bool(ever["is_cubeA_static"][item]),
                        "steps": int(steps[item] or max_steps),
                        "termination_reason": reasons[item],
                        "action_clip_elements": int(action_clip_count[item]),
                        "action_total_elements": int(action_count[item]),
                        "action_jitter_sum": float(action_jitter[item]),
                        "gripper_flip_count": int(gripper_flips[item]),
                    }
                )
    except BaseException as exc:
        failed = True
        _write_worker_progress(
            output_dir,
            worker_index=worker_index,
            worker_count=worker_count,
            status="failed",
            completed_episodes=len(result_rows),
            total_episodes=total_worker_episodes,
            current_batch_episodes=0,
            current_step=last_step_index,
            max_steps=max_steps,
            worker_device=str(device_obj),
            environment_device=environment_device,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if not failed:
            _write_worker_progress(
                output_dir,
                worker_index=worker_index,
                worker_count=worker_count,
                status="finalizing",
                completed_episodes=len(result_rows),
                total_episodes=total_worker_episodes,
                current_batch_episodes=0,
                current_step=last_step_index,
                max_steps=max_steps,
                worker_device=str(device_obj),
                environment_device=environment_device,
            )
            _worker_log(worker_index, f"env close begin last_step={last_step_index}")
        try:
            env.close()
        except BaseException as exc:
            _write_worker_progress(
                output_dir,
                worker_index=worker_index,
                worker_count=worker_count,
                status="failed",
                completed_episodes=len(result_rows),
                total_episodes=total_worker_episodes,
                current_batch_episodes=0,
                current_step=last_step_index,
                max_steps=max_steps,
                worker_device=str(device_obj),
                environment_device=environment_device,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            if not failed:
                _worker_log(worker_index, "env close complete")
    result_rows.sort(key=lambda item: item["index"])
    summary = {
        "count": len(result_rows),
        "success_at_end": float(np.mean([item["success_at_end"] for item in result_rows])),
        "success_once": float(np.mean([item["success_once"] for item in result_rows])),
        "ever_grasped": float(np.mean([item["is_cubeA_grasped"] for item in result_rows])),
        "ever_on_cube": float(np.mean([item["is_cubeA_on_cubeB"] for item in result_rows])),
        "action_clip_fraction": float(
            sum(item["action_clip_elements"] for item in result_rows)
            / max(sum(item["action_total_elements"] for item in result_rows), 1)
        ),
        "finite_rate": 1.0,
        "policy_latency_p95_ms": _percentile(chunk_latencies_ms, 95),
        "environment_step_latency_p95_ms": _percentile(environment_latencies_ms, 95),
        "amortized_policy_latency_p95_ms": (
            None if not chunk_latencies_ms else _percentile(chunk_latencies_ms, 95) / n_action_steps
        ),
        "chunk_forward_count": len(chunk_latencies_ms),
        "test_access_count": 0,
        "split": "development",
        "state_bank": _portable(state_bank),
        "state_bank_indices": [int(item["index"]) for item in result_rows],
        "flow_solver": "euler",
        "flow_integration_steps": 10,
        "n_action_steps": n_action_steps if checkpoint is not None else None,
        "reference": reference,
        "checkpoint": None if checkpoint is None else _portable(checkpoint),
        "checkpoint_step": None if checkpoint_payload is None else checkpoint_payload.get("step"),
        "smoke": smoke,
        "environment_device": environment_device,
    }
    payload = {
        "schema_version": 1,
        "task": TASK_ID,
        "summary": summary,
        "rows": result_rows,
        "worker": {
            "index": worker_index,
            "count": worker_count,
            "worker_device": str(device_obj),
            "policy_device": str(device_obj) if checkpoint is not None else None,
            "environment_device": environment_device,
            "physical_device": os.environ.get("M2_W_PHYSICAL_GPU"),
        },
    }
    _write_worker_progress(
        output_dir,
        worker_index=worker_index,
        worker_count=worker_count,
        status="writing_metrics",
        completed_episodes=total_worker_episodes,
        total_episodes=total_worker_episodes,
        current_batch_episodes=0,
        current_step=last_step_index,
        max_steps=max_steps,
        worker_device=str(device_obj),
        environment_device=environment_device,
    )
    _worker_log(worker_index, "metrics write begin")
    target = _rooted(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / f"metrics_worker_{worker_index:03d}.json", payload)
    if worker_count == 1:
        _write_json(target / "metrics.json", payload)
    _worker_log(worker_index, "metrics write complete")
    _write_worker_progress(
        output_dir,
        worker_index=worker_index,
        worker_count=worker_count,
        status="complete",
        completed_episodes=total_worker_episodes,
        total_episodes=total_worker_episodes,
        current_batch_episodes=0,
        current_step=last_step_index,
        max_steps=max_steps,
        worker_device=str(device_obj),
        environment_device=environment_device,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", choices=("zero", "iid_gaussian_random"))
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=FORMAL_HORIZON)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_development_rollout(
        state_bank=args.state_bank,
        output_dir=args.output_dir,
        n_action_steps=args.n_action_steps,
        checkpoint=args.checkpoint,
        reference=args.reference,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        num_envs=args.num_envs,
        device=args.device,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        smoke=args.smoke,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
