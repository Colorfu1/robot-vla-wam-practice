"""Exact-state evaluator for the official Diffusion Policy on M1-W StackCube."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .eval_stackcube_gpu import (
    _batched_state,
    _signals,
    _termination_reason,
    _to_numpy,
    load_state_bank,
)
from .official_m1w_adapter import M1WFlattenRGBObservationWrapper
from .official_stackcube_rgb_verify import _load_official_train_module
from .rollout_contract import M1W_DATA_PROVENANCE, require_rollout_backend

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pose"
OBS_HORIZON = 2
ACT_HORIZON = 8
PRED_HORIZON = 16
FORMAL_HORIZON = 500
STAGE_KEYS = ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static")


def _official_args() -> SimpleNamespace:
    return SimpleNamespace(
        obs_horizon=OBS_HORIZON,
        act_horizon=ACT_HORIZON,
        pred_horizon=PRED_HORIZON,
        diffusion_step_embed_dim=64,
        unet_dims=[64, 128, 256],
        n_groups=8,
    )


def _make_exact_env(
    num_envs: int,
    max_steps: int,
    *,
    device: torch.device,
    control_mode: str = CONTROL_MODE,
    rollout_sim_backend: str = "physx_cuda",
):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers import FrameStack
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    if device.index is None:
        raise ValueError("exact evaluator requires an indexed CUDA device")
    require_rollout_backend(M1W_DATA_PROVENANCE, rollout_sim_backend)
    if rollout_sim_backend == "physx_cpu":
        num_envs = 1
    raw_env = gym.make(
        TASK_ID,
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode=control_mode,
        sim_backend=(
            "physx_cpu"
            if rollout_sim_backend == "physx_cpu"
            else f"physx_cuda:{device.index}"
        ),
        render_backend=(
            "sapien_cpu"
            if rollout_sim_backend == "physx_cpu"
            else f"sapien_cuda:{device.index}"
        ),
        reward_mode="sparse",
        max_episode_steps=max_steps,
        render_mode="rgb_array",
        reconfiguration_freq=1,
    )
    flatten = M1WFlattenRGBObservationWrapper(raw_env)
    frame_stack = FrameStack(flatten, num_stack=OBS_HORIZON)
    vector_env = ManiSkillVectorEnv(
        frame_stack,
        ignore_terminations=True,
        record_metrics=True,
    )
    return vector_env, flatten, frame_stack


def _reset_exact(
    env: Any,
    flatten: M1WFlattenRGBObservationWrapper,
    frame_stack: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    batch_size = len(rows)
    env.reset(seed=[0] * batch_size)
    env.unwrapped.set_state_dict(_batched_state(rows, device=str(device)))
    raw_observation = env.unwrapped.get_obs()
    frame = flatten.observation(raw_observation)
    frame_stack.frames.clear()
    for _ in range(OBS_HORIZON):
        frame_stack.frames.append(frame)
    observation = frame_stack.observation(None)
    return observation, env.unwrapped.get_info()


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(bool(row[key]) for row in rows)
    return {
        "num_success": successes,
        "num_episodes": len(rows),
        "rate": successes / len(rows) if rows else None,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first_success = [
        float(row["first_success_step"])
        for row in rows
        if row["first_success_step"] is not None
    ]
    reasons = sorted({str(row["termination_reason"]) for row in rows})
    return {
        "num_episodes": len(rows),
        "success_once": _rate(rows, "success_once"),
        "success_at_end": _rate(rows, "success_at_end"),
        "first_success_step": {
            "count": len(first_success),
            "mean": float(np.mean(first_success)) if first_success else None,
            "median": float(np.median(first_success)) if first_success else None,
            "p90": float(np.percentile(first_success, 90)) if first_success else None,
        },
        "stage_signal_rates": {
            key: float(np.mean([bool(row[key]) for row in rows])) for key in STAGE_KEYS
        },
        "mean_episode_steps": float(np.mean([row["steps"] for row in rows])),
        "termination_reasons": {
            reason: sum(str(row["termination_reason"]) == reason for row in rows)
            for reason in reasons
        },
    }


def evaluate_checkpoint(
    *,
    official_repo: Path,
    checkpoint: Path,
    state_bank: Path,
    output_dir: Path,
    max_episodes: int,
    num_envs: int,
    max_steps: int,
    device: str = "cuda:0",
    rows: Sequence[Mapping[str, Any]] | None = None,
    worker_index: int = 0,
    worker_count: int = 1,
    control_mode: str = CONTROL_MODE,
    action_dim: int = 7,
    label: str = "Official Diffusion Policy M1-W exact-state evaluation",
    rollout_sim_backend: str = "physx_cuda",
) -> dict[str, Any]:
    if max_episodes <= 0 or num_envs <= 0:
        raise ValueError("max-episodes and num-envs must be positive")
    if max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if action_dim <= 0:
        raise ValueError("action-dim must be positive")
    rollout_contract = require_rollout_backend(M1W_DATA_PROVENANCE, rollout_sim_backend)
    num_envs = min(num_envs, max_episodes)
    if rollout_sim_backend == "physx_cpu":
        # CPU ManiSkill physics is single-environment.  Keep the policy on
        # the worker GPU and serialize only the simulator side of this exact
        # state evaluator.
        num_envs = 1
    elif rollout_sim_backend != "physx_cuda":
        raise ValueError(
            "rollout_sim_backend must be physx_cpu or physx_cuda, "
            f"got {rollout_sim_backend!r}"
        )
    if rows is None:
        rows = load_state_bank(state_bank, max_episodes=max_episodes)
    else:
        rows = list(rows)
    if not rows:
        raise ValueError("This evaluator worker owns no state-bank rows")
    if len(rows) > max_episodes:
        rows = rows[:max_episodes]
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise ValueError("The official M1-W exact evaluator requires a CUDA device")
    torch.cuda.set_device(device_obj)
    official = _load_official_train_module(official_repo)
    env, flatten, frame_stack = _make_exact_env(
        num_envs,
        max_steps,
        device=device_obj,
        control_mode=control_mode,
        rollout_sim_backend=rollout_sim_backend,
    )
    agent = official.Agent(env, _official_args()).to(device_obj)
    # These checkpoints are produced by the trusted vendored official
    # trainer.  PyTorch 2.6 defaults to weights_only=True, which rejects the
    # NumPy scalar metadata stored in the official checkpoint payload.
    payload = torch.load(checkpoint, map_location=device_obj, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"official checkpoint must be a dictionary: {checkpoint}")
    state_dict = payload.get("ema_agent")
    if state_dict is None:
        state_dict = payload.get("agent")
    if state_dict is None:
        raise KeyError("checkpoint has neither ema_agent nor agent")
    agent.load_state_dict(state_dict)
    agent.eval()

    episode_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    from mani_skill.utils import common

    try:
        for offset in range(0, len(rows), num_envs):
            batch_rows = rows[offset : offset + num_envs]
            batch_size = len(batch_rows)
            observation, info = _reset_exact(
                env,
                flatten,
                frame_stack,
                batch_rows,
                device=(
                    torch.device("cpu")
                    if rollout_sim_backend == "physx_cpu"
                    else device_obj
                ),
            )
            if rollout_sim_backend == "physx_cpu":
                observation = common.to_tensor(observation, device=device_obj)
            signals = _signals(info, batch_size)
            success_once = signals["success"].copy()
            success_at_end = signals["success"].copy()
            first_success = np.full(batch_size, -1, dtype=np.int32)
            first_success[success_once] = 0
            ever = {key: signals[key].copy() for key in STAGE_KEYS}
            finished = np.zeros(batch_size, dtype=bool)
            steps = np.zeros(batch_size, dtype=np.int32)
            reasons = ["episode_limit"] * batch_size

            action_steps = 0
            with torch.no_grad():
                while action_steps < max_steps and not finished.all():
                    action_seq = agent.get_action(observation)
                    chunk_length = min(
                        int(action_seq.shape[1]), max_steps - action_steps
                    )
                    for chunk_index in range(chunk_length):
                        if finished.all():
                            break
                        current_step = action_steps + 1
                        action = action_seq[:, chunk_index].clone()
                        action[torch.as_tensor(finished, device=action.device)] = 0.0
                        if rollout_sim_backend == "physx_cpu":
                            action = action.detach().cpu()
                        observation, _reward, terminated, truncated, info = env.step(action)
                        if rollout_sim_backend == "physx_cpu":
                            observation = common.to_tensor(observation, device=device_obj)
                        step_signals = _signals(info, batch_size)
                        active = ~finished
                        success_once[active] |= step_signals["success"][active]
                        success_at_end[active] = step_signals["success"][active]
                        for key in STAGE_KEYS:
                            ever[key] |= step_signals[key]
                        newly_success = active & step_signals["success"]
                        first_success[(first_success < 0) & newly_success] = current_step
                        terminated_array = _to_numpy(terminated).astype(
                            bool, copy=False
                        ).reshape(-1)
                        truncated_array = _to_numpy(truncated).astype(bool, copy=False).reshape(-1)
                        newly_finished = active & (terminated_array | truncated_array)
                        for item in np.flatnonzero(newly_finished):
                            steps[item] = current_step
                            reasons[item] = _termination_reason(
                                success=bool(step_signals["success"][item]),
                                truncated=bool(truncated_array[item]),
                                fail=bool(step_signals["fail"][item]),
                                terminated=bool(terminated_array[item]),
                            )
                        finished |= newly_finished
                        action_steps = current_step
                    if finished.all():
                        break

            for index, row in enumerate(batch_rows):
                episode_rows.append(
                    {
                        "state_bank_index": int(row["index"]),
                        "start_state_fingerprint": str(row["start_state_fingerprint"]),
                        "success_once": bool(success_once[index]),
                        "success_at_end": bool(success_at_end[index]),
                        "first_success_step": (
                            None if first_success[index] < 0 else int(first_success[index])
                        ),
                        "is_cubeA_grasped": bool(ever["is_cubeA_grasped"][index]),
                        "is_cubeA_on_cubeB": bool(ever["is_cubeA_on_cubeB"][index]),
                        "is_cubeA_static": bool(ever["is_cubeA_static"][index]),
                        "steps": int(steps[index] or max_steps),
                        "termination_reason": reasons[index],
                    }
                )
            print(
                f"[official-m1w-eval worker={worker_index}] "
                f"progress={len(episode_rows)}/{len(rows)}",
                flush=True,
            )
    finally:
        env.close()

    result = {
        "schema_version": 1,
        "label": label,
        "task": TASK_ID,
        "control_mode": control_mode,
        "obs_mode": "rgb",
        "sim_backend": rollout_sim_backend,
        **rollout_contract,
        "state_definition": "agent.qpos(9) + agent.qvel(9)",
        "action_dim": action_dim,
        "max_episodes": len(episode_rows),
        "num_envs": num_envs,
        "max_episode_steps": max_steps,
        "checkpoint": str(checkpoint),
        "checkpoint_kind": "ema_agent" if "ema_agent" in payload else "agent",
        "state_bank": str(state_bank),
        "state_bank_split": state_bank.stem,
        "identity": "start_state_fingerprint",
        "worker": {"index": worker_index, "count": worker_count},
        "evaluation": _summary(episode_rows),
        "rows": episode_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def _merge_worker_results(
    *,
    worker_dirs: Sequence[Path],
    output_dir: Path,
    expected_state_indices: Sequence[int],
) -> dict[str, Any]:
    payloads = []
    for worker_dir in worker_dirs:
        metrics_path = worker_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"missing worker metrics: {metrics_path}")
        payloads.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    if not payloads:
        raise ValueError("no worker metrics to merge")
    rows = [row for payload in payloads for row in payload["rows"]]
    rows.sort(key=lambda row: int(row["state_bank_index"]))
    actual = [int(row["state_bank_index"]) for row in rows]
    expected = sorted(int(index) for index in expected_state_indices)
    if actual != expected:
        raise ValueError(
            "worker metrics do not cover the expected state-bank partition: "
            f"actual_first={actual[:5]} actual_last={actual[-5:]} "
            f"expected_first={expected[:5]} expected_last={expected[-5:]}"
        )
    first = payloads[0]
    merged = {
        "schema_version": 1,
        "label": first["label"],
        "task": first["task"],
        "control_mode": first["control_mode"],
        "obs_mode": first["obs_mode"],
        "sim_backend": first["sim_backend"],
        "data_provenance": first["data_provenance"],
        "required_rollout_backend": first["required_rollout_backend"],
        "actual_rollout_backend": first["actual_rollout_backend"],
        "state_definition": first["state_definition"],
        "action_dim": first["action_dim"],
        "max_episodes": len(rows),
        "num_envs_per_worker": first["num_envs"],
        "worker_count": len(payloads),
        "max_episode_steps": first["max_episode_steps"],
        "checkpoint": first["checkpoint"],
        "checkpoint_kind": first["checkpoint_kind"],
        "state_bank": first["state_bank"],
        "state_bank_split": first["state_bank_split"],
        "identity": first["identity"],
        "evaluation": _summary(rows),
        "rows": rows,
        "workers": [payload["worker"] for payload in payloads],
        "elapsed_seconds": max(
            float(payload.get("elapsed_seconds", 0.0)) for payload in payloads
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=FORMAL_HORIZON)
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--control-mode", default=CONTROL_MODE)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument(
        "--rollout-sim-backend",
        choices=("physx_cpu", "physx_cuda"),
        default="physx_cuda",
    )
    parser.add_argument(
        "--label",
        default="Official Diffusion Policy M1-W exact-state evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    worker_count = args.worker_count or world_size
    worker_index = args.worker_index if args.worker_index is not None else rank
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    official_repo = args.official_repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    state_bank = args.state_bank.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if world_size > 1:
        import torch.distributed as dist

        if world_size != worker_count:
            raise ValueError("torchrun WORLD_SIZE must equal worker-count")
        if not torch.cuda.is_available():
            raise RuntimeError("multi-GPU exact evaluation requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        all_rows = load_state_bank(state_bank, max_episodes=args.max_episodes)
        worker_rows = all_rows[worker_index::worker_count]
        worker_dir = output_dir / "workers" / f"rank_{worker_index:02d}"
        result = evaluate_checkpoint(
            official_repo=official_repo,
            checkpoint=checkpoint,
            state_bank=state_bank,
            output_dir=worker_dir,
            max_episodes=len(worker_rows),
            num_envs=min(args.num_envs, len(worker_rows)),
            max_steps=args.max_steps,
            device=f"cuda:{local_rank}",
            rows=worker_rows,
            worker_index=worker_index,
            worker_count=worker_count,
            control_mode=args.control_mode,
            action_dim=args.action_dim,
            label=args.label,
            rollout_sim_backend=args.rollout_sim_backend,
        )
        dist.barrier()
        if worker_index == 0:
            result = _merge_worker_results(
                worker_dirs=[
                    output_dir / "workers" / f"rank_{index:02d}"
                    for index in range(worker_count)
                ],
                output_dir=output_dir,
                expected_state_indices=[int(row["index"]) for row in all_rows],
            )
        dist.barrier()
        dist.destroy_process_group()
    else:
        result = evaluate_checkpoint(
            official_repo=official_repo,
            checkpoint=checkpoint,
            state_bank=state_bank,
            output_dir=output_dir,
            max_episodes=args.max_episodes,
            num_envs=args.num_envs,
            max_steps=args.max_steps,
            device="cuda:0",
            worker_index=worker_index,
            worker_count=worker_count,
            control_mode=args.control_mode,
            action_dim=args.action_dim,
            label=args.label,
            rollout_sim_backend=args.rollout_sim_backend,
        )
    print(json.dumps(result["evaluation"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
