"""Evaluate LeRobot checkpoints with the ManiSkill3 paper-style CPU protocol.

This module deliberately keeps the simulator contract separate from the M0
GPU/50-step evaluator.  Every official-protocol episode uses a CPU single-env
worker wrapped by ``CPUGymWrapper(ignore_terminations=True, record_metrics=True)``;
the 10 workers are joined by Gymnasium's ``AsyncVectorEnv``.  Reset is called
without a seed, matching the public ManiSkill baseline protocol.
"""

from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

from .eval_closed_loop import (
    DiffusionPolicyDriver,
    _to_numpy,
    _version,
    latency_summary,
    load_diffusion_policy,
    summarize_rows,
)
from .rollout_contract import OFFICIAL_DATA_PROVENANCE, require_rollout_backend

TASK_ID = "PickCube-v1"
CONTROL_MODE = "pd_ee_delta_pos"
SIM_BACKEND = "physx_cpu"
DATA_PROVENANCE = OFFICIAL_DATA_PROVENANCE
IMAGE_KEY = "observation.images.base_camera"
STATE_KEY = "observation.state"
STATE_FIELDS = (
    "agent.qpos[0:9]",
    "agent.qvel[0:9]",
    "extra.goal_pos[0:3]",
)
MAX_EPISODE_STEPS = 100
NUM_ENVS = 10
NUM_EPISODES = 100
N_OBS_STEPS = 2
N_ACTION_STEPS = 8
POLICY_SEED = 1000
SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)


def _portable(path: str | Path) -> str:
    value = Path(path).expanduser().resolve()
    try:
        return value.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _write_episodes(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(
            pa.Table.from_pylist([dict(row) for row in rows]),
            temporary,
            compression="zstd",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_directory(temporary: Path, output_dir: Path) -> None:
    """Publish an evaluation directory, falling back to a cross-device copy."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation directory: {output_dir}")
    try:
        temporary.rename(output_dir)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        LOGGER.warning(
            "Atomic evaluation rename crossed filesystems; copying %s to %s",
            temporary,
            output_dir,
        )

    try:
        shutil.copytree(temporary, output_dir, symlinks=False)
    except BaseException:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    shutil.rmtree(temporary)


def _signal(info: Mapping[str, Any], key: str, batch_size: int) -> np.ndarray:
    value = info.get(key)
    if value is None:
        return np.zeros(batch_size, dtype=bool)
    result = _to_numpy(value).astype(bool, copy=False).reshape(-1)
    if result.size == 1 and batch_size != 1:
        result = np.repeat(result, batch_size)
    if result.size != batch_size:
        raise ValueError(f"Native info[{key!r}] has {result.size} values, expected {batch_size}")
    return result.copy()


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    count = len(rows)
    successes = sum(bool(row[key]) for row in rows)
    return {"num_success": successes, "num_episodes": count, "rate": successes / count}


def _first_success(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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


def _build_single_official_env(
    *,
    env_index: int,
    task: str = TASK_ID,
    control_mode: str = CONTROL_MODE,
    max_episode_steps: int = MAX_EPISODE_STEPS,
) -> Any:
    """Build exactly one official CPU worker, including its wrapper chain."""

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    from mani_skill.utils.wrappers.frame_stack import FrameStack
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    env = gym.make(
        task,
        reconfiguration_freq=1,
        obs_mode="rgb",
        control_mode=control_mode,
        sim_backend="cpu",
        max_episode_steps=max_episode_steps,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    env = FrameStack(env, num_stack=N_OBS_STEPS)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
    env.action_space.seed(env_index)
    env.observation_space.seed(env_index)
    return env


def make_official_vector_env(
    *,
    num_envs: int = NUM_ENVS,
    task: str = TASK_ID,
    control_mode: str = CONTROL_MODE,
    max_episode_steps: int = MAX_EPISODE_STEPS,
) -> Any:
    """Create the official 10-single-CPU-env AsyncVectorEnv."""

    from gymnasium.vector import AsyncVectorEnv

    factories = [
        partial(
            _build_single_official_env,
            env_index=index,
            task=task,
            control_mode=control_mode,
            max_episode_steps=max_episode_steps,
        )
        for index in range(num_envs)
    ]
    return AsyncVectorEnv(factories, context="forkserver")


def build_policy_observation_from_official(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert FrameStack's flattened output to the repo's 21D LeRobot input.

    ManiSkill's RGB wrapper exposes all non-image state fields in its flattened
    ``state`` vector.  For PickCube RGB those fields are qpos, qvel, is_grasped,
    tcp_pose, and goal_pos.  The student contract intentionally keeps only the
    first 18 and last 3 values, matching the dataset converter.
    """

    import torch

    try:
        rgb = _to_numpy(observation["rgb"])
        state = _to_numpy(observation["state"])
    except KeyError as exc:
        raise KeyError("Official FrameStack observation must contain rgb and state") from exc
    if rgb.ndim == 4:
        # CPUGymWrapper removes the singleton environment axis for the
        # one-env holdout path, leaving [T,H,W,C].
        rgb = rgb[None, ...]
    if rgb.ndim != 5:
        raise ValueError(f"Official RGB observation must be [B,T,H,W,3], got {rgb.shape}")
    rgb = rgb[:, -1]
    if rgb.shape[-1] != 3:
        raise ValueError(f"Official RGB observation must be channel-last, got {rgb.shape}")
    if state.ndim == 2:
        state = state[None, ...]
    if state.ndim != 3:
        raise ValueError(f"Official state observation must be [B,T,D], got {state.shape}")
    state = state[:, -1]
    if state.shape[-1] == 21:
        policy_state = state
    elif state.shape[-1] > 21:
        policy_state = np.concatenate((state[..., :18], state[..., -3:]), axis=-1)
    else:
        raise ValueError(f"Official flattened state has too few fields: {state.shape}")
    if policy_state.shape[-1] != 21 or not np.isfinite(policy_state).all():
        raise ValueError(f"Policy state must be finite [B,21], got {policy_state.shape}")
    return {
        IMAGE_KEY: torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).float() / 255.0,
        STATE_KEY: torch.from_numpy(np.ascontiguousarray(policy_state)).float(),
    }


def _action_metrics(action: np.ndarray) -> dict[str, np.ndarray | None]:
    return {
        "translation_norm": np.linalg.norm(action[:, :3], axis=1),
        "rotation_norm": None,
        "gripper_abs": np.abs(action[:, -1]),
    }


def _make_row(
    *,
    index: int,
    success_once: bool,
    success_at_end: bool,
    fail_once: bool,
    fail_at_end: bool,
    first_success_step: int | None,
    ever: Mapping[str, bool],
    action_sums: Mapping[str, float],
    action_count: int,
    jitter_sum: float,
    gripper_flips: int,
    max_steps: int,
) -> dict[str, Any]:
    denominator = max(action_count, 1)
    return {
        "task": TASK_ID,
        "episode_index": index,
        "success_once": success_once,
        "success_at_end": success_at_end,
        "success": success_at_end,
        "fail_once": fail_once,
        "fail_at_end": fail_at_end,
        "first_success_step": first_success_step,
        "steps": max_steps,
        "termination_reason": "truncated",
        "is_grasped": bool(ever["is_grasped"]),
        "is_obj_placed": bool(ever["is_obj_placed"]),
        "is_robot_static": bool(ever["is_robot_static"]),
        "translation_norm_mean": action_sums["translation_norm"] / denominator,
        "rotation_norm_mean": None,
        "gripper_abs_mean": action_sums["gripper_abs"] / denominator,
        "action_jitter_mean": jitter_sum / max(action_count - 1, 1),
        "gripper_flip_count": gripper_flips,
    }


def _summarize_official_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = summarize_rows(rows)
    result["fail_once"] = _rate(rows, "fail_once")
    result["fail_at_end"] = _rate(rows, "fail_at_end")
    return result


def _save_evaluation(
    output_dir: Path,
    *,
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        complete = all(
            (output_dir / name).is_file()
            for name in ("metrics.json", "episodes.parquet", "run_manifest.json")
        )
        if complete:
            return _read_json(output_dir / "metrics.json")
        raise FileExistsError(f"Refusing to reuse incomplete evaluation output: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.incomplete-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        write_json_atomic(temporary / "metrics.json", dict(metrics))
        _write_episodes(temporary / "episodes.parquet", rows)
        write_json_atomic(temporary / "run_manifest.json", dict(manifest))
        _publish_directory(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return dict(metrics)


def evaluate_checkpoint(
    *,
    policy_path: Path,
    dataset_repo_id: str,
    dataset_root: Path,
    output_dir: Path,
    num_envs: int = NUM_ENVS,
    num_episodes: int = NUM_EPISODES,
    max_steps: int = MAX_EPISODE_STEPS,
    policy_seed: int = POLICY_SEED,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate one checkpoint; the final directory is written atomically."""

    if num_envs <= 0 or num_episodes <= 0 or num_episodes % num_envs != 0:
        raise ValueError("num_episodes must be a positive multiple of num_envs")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    rollout_contract = require_rollout_backend(DATA_PROVENANCE, SIM_BACKEND)
    loaded = load_diffusion_policy(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        n_action_steps=N_ACTION_STEPS,
        device=device,
    )
    if int(loaded["action_dim"]) != 4:
        raise ValueError(
            f"Paper-aligned visual policy must output 4 actions, got {loaded['action_dim']}"
        )
    driver = DiffusionPolicyDriver(loaded)
    np.random.seed(policy_seed)
    try:
        import torch

        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)
    except ImportError:
        pass

    environment = make_official_vector_env(
        num_envs=num_envs,
        max_episode_steps=max_steps,
    )
    progress = tqdm(
        total=num_episodes,
        desc=f"official {output_dir.name}",
        unit="episode",
        dynamic_ncols=True,
    )
    rows: list[dict[str, Any]] = []
    environment_latencies: list[float] = []
    control_frequency_hz = 20.0
    try:
        for batch_start in range(0, num_episodes, num_envs):
            # This intentionally has no seed argument: it matches the official
            # baseline's random evaluation reset protocol.
            observation, _ = environment.reset()
            driver.reset()
            batch_size = num_envs
            success_once = np.zeros(batch_size, dtype=bool)
            success_at_end = np.zeros(batch_size, dtype=bool)
            fail_once = np.zeros(batch_size, dtype=bool)
            fail_at_end = np.zeros(batch_size, dtype=bool)
            ever = {
                key: np.zeros(batch_size, dtype=bool)
                for key in ("is_grasped", "is_obj_placed", "is_robot_static")
            }
            first_success = np.full(batch_size, -1, dtype=np.int32)
            action_sums = {
                "translation_norm": np.zeros(batch_size, dtype=np.float64),
                "gripper_abs": np.zeros(batch_size, dtype=np.float64),
            }
            previous_action = np.zeros((batch_size, 4), dtype=np.float32)
            has_previous = np.zeros(batch_size, dtype=bool)
            jitter = np.zeros(batch_size, dtype=np.float64)
            gripper_flips = np.zeros(batch_size, dtype=np.int32)
            last_truncated = None
            last_terminated = None
            for step in range(1, max_steps + 1):
                prepared = driver.preprocessor(build_policy_observation_from_official(observation))
                action = driver.select_action(prepared)
                if action.shape != (batch_size, 4):
                    raise ValueError(f"Policy action must be {(batch_size, 4)}, got {action.shape}")
                metrics = _action_metrics(action)
                active = np.ones(batch_size, dtype=bool)
                action_sums["translation_norm"] += metrics["translation_norm"]
                action_sums["gripper_abs"] += metrics["gripper_abs"]
                comparable = has_previous & active
                if np.any(comparable):
                    jitter[comparable] += np.linalg.norm(
                        action[comparable] - previous_action[comparable], axis=1
                    )
                    before = np.sign(previous_action[comparable, -1])
                    after = np.sign(action[comparable, -1])
                    gripper_flips[comparable] += (before != after) & (before != 0) & (after != 0)
                previous_action[:] = action
                has_previous[:] = True
                started = time.perf_counter()
                observation, _reward, terminated, truncated, info = environment.step(action)
                environment_latencies.append((time.perf_counter() - started) * 1000.0)
                native_success = _signal(info, "success", batch_size)
                native_fail = _signal(info, "fail", batch_size)
                success_once |= native_success
                success_at_end = native_success
                fail_once |= native_fail
                fail_at_end = native_fail
                for key in ever:
                    ever[key] |= _signal(info, key, batch_size)
                first_success[(first_success < 0) & native_success] = step
                last_terminated = _to_numpy(terminated).astype(bool, copy=False).reshape(-1)
                last_truncated = _to_numpy(truncated).astype(bool, copy=False).reshape(-1)
            if (
                last_truncated is None
                or not bool(last_truncated.any())
                or not bool(last_truncated.all())
            ):
                raise AssertionError(
                    "Official protocol must truncate all 10 envs together at step 100"
                )
            if last_terminated is None or bool(last_terminated.any()):
                raise AssertionError("Official protocol must ignore terminations")
            for index in range(batch_size):
                rows.append(
                    _make_row(
                        index=batch_start + index,
                        success_once=bool(success_once[index]),
                        success_at_end=bool(success_at_end[index]),
                        fail_once=bool(fail_once[index]),
                        fail_at_end=bool(fail_at_end[index]),
                        first_success_step=(
                            None if first_success[index] < 0 else int(first_success[index])
                        ),
                        ever={key: bool(value[index]) for key, value in ever.items()},
                        action_sums={
                            key: float(value[index]) for key, value in action_sums.items()
                        },
                        action_count=max_steps,
                        jitter_sum=float(jitter[index]),
                        gripper_flips=int(gripper_flips[index]),
                        max_steps=max_steps,
                    )
                )
            progress.update(batch_size)
    finally:
        environment.close()
        progress.close()

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "label": "Official ManiSkill3 DP RGB protocol",
        "system": "visual",
        "task": TASK_ID,
        "control_mode": CONTROL_MODE,
        "sim_backend": SIM_BACKEND,
        **rollout_contract,
        "num_episodes": len(rows),
        "num_envs": num_envs,
        "max_episode_steps": max_steps,
        "control_frequency_hz": control_frequency_hz,
        "policy_seed": policy_seed,
        "evaluation": _summarize_official_rows(rows),
        "latency": {
            "policy_forward": latency_summary(driver.latencies_ms),
            "environment_step": latency_summary(environment_latencies),
        },
        "official_protocol": {
            "reset": "eval_envs.reset() without seed",
            "action_space_seeds": list(range(num_envs)),
            "observation_space_seeds": list(range(num_envs)),
            "reconfiguration_freq": 1,
            "ignore_terminations": True,
            "record_metrics": True,
            "frame_stack": N_OBS_STEPS,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "policy_path": _portable(policy_path),
        "policy_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset": {"repo_id": dataset_repo_id, "root": _portable(dataset_root)},
        "task": TASK_ID,
        "control_mode": CONTROL_MODE,
        "sim_backend": SIM_BACKEND,
        **rollout_contract,
        "max_episode_steps": max_steps,
        "observation": {
            "keys": [IMAGE_KEY, STATE_KEY],
            "state_shape": [21],
            "state_fields": list(STATE_FIELDS),
        },
        "action": {
            "shape": [4],
            "n_obs_steps": N_OBS_STEPS,
            "horizon": 16,
            "n_action_steps": N_ACTION_STEPS,
        },
        "success_source": 'native info["success"] only',
        "role_note": (
            "This is a visual student evaluated against a privileged demonstration source."
        ),
        "runtime": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "mani_skill": _version("mani_skill"),
        },
    }
    return _save_evaluation(output_dir, metrics=metrics, rows=rows, manifest=manifest)


def _checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir.expanduser().resolve() / "checkpoints" / f"{int(step):06d}" / "pretrained_model"


def _rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda item: (
            -float(item["success_once_rate"]),
            -float(item["success_at_end_rate"]),
            int(item["checkpoint_step"]),
        ),
    )
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
    return ordered


def summarize_official(
    *,
    run_dir: Path,
    evaluation_root: Path,
    selection_dir: Path,
    checkpoint_steps: Sequence[int],
    repeat_count: int = 0,
) -> dict[str, Any]:
    """Rank completed candidate metrics and publish a copied selected checkpoint."""

    candidates = []
    for step in checkpoint_steps:
        metrics_path = evaluation_root / f"checkpoint_{int(step):06d}" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing official evaluation: {metrics_path}")
        metrics = _read_json(metrics_path)
        evaluation = metrics.get("evaluation", {})
        candidates.append(
            {
                "checkpoint_step": int(step),
                "policy_path": _portable(_checkpoint_path(run_dir, int(step))),
                "metrics_path": _portable(metrics_path),
                "success_once_rate": float(evaluation["success_once"]["rate"]),
                "success_at_end_rate": float(evaluation["success_at_end"]["rate"]),
                "num_episodes": int(metrics["num_episodes"]),
            }
        )
    ranking = _rank_candidates(candidates)
    selected = ranking[0]
    selected_source = _checkpoint_path(run_dir, int(selected["checkpoint_step"]))
    if not selected_source.is_dir():
        raise FileNotFoundError(f"Selected checkpoint is missing: {selected_source}")
    selected_dir = selection_dir.expanduser().resolve() / "selected" / "pretrained_model"
    selected_dir.parent.mkdir(parents=True, exist_ok=True)
    selection_file = selection_dir.expanduser().resolve() / "selected_checkpoint.json"
    if selected_dir.exists():
        existing = _read_json(selection_file) if selection_file.is_file() else {}
        if int(existing.get("selected_checkpoint_step", -1)) != int(selected["checkpoint_step"]):
            raise FileExistsError(
                f"Selected checkpoint already exists with a different winner: {selected_dir}"
            )
        if not all(
            (selected_dir / name).is_file() for name in ("config.json", "model.safetensors")
        ):
            raise FileExistsError(f"Selected checkpoint directory is incomplete: {selected_dir}")
    else:
        shutil.copytree(selected_source, selected_dir, symlinks=False)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "primary_metric": "success_once_rate",
        "selection_rule": "max success_once, then max success_at_end, then earliest checkpoint",
        "selected_checkpoint_step": int(selected["checkpoint_step"]),
        "selected_source": _portable(selected_source),
        "selected_checkpoint": _portable(selected_dir),
        "ranking": ranking,
    }
    if repeat_count:
        repeats = []
        for index in range(repeat_count):
            metrics_path = evaluation_root / "repeats" / f"repeat_{index:02d}" / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing official repeat evaluation: {metrics_path}")
            metrics = _read_json(metrics_path)
            repeats.append(
                {
                    "repeat": index,
                    "metrics_path": _portable(metrics_path),
                    "success_once_rate": float(metrics["evaluation"]["success_once"]["rate"]),
                    "success_at_end_rate": float(metrics["evaluation"]["success_at_end"]["rate"]),
                }
            )
        values = np.asarray([item["success_once_rate"] for item in repeats], dtype=np.float64)
        result["official_best_repeats"] = {
            "count": repeat_count,
            "runs": repeats,
            "success_once_rate_mean": float(np.mean(values)),
            "success_once_rate_min": float(np.min(values)),
            "success_once_rate_max": float(np.max(values)),
            "success_once_rate_range": float(np.max(values) - np.min(values)),
        }
    write_json_atomic(selection_dir.expanduser().resolve() / "official_best.json", result)
    write_json_atomic(selection_file, result)
    return result


def _parse_steps(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or len(set(result)) != len(result):
        raise ValueError("checkpoint steps must be a non-empty unique comma-separated list")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--summarize-and-repeat", action="store_true")
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--checkpoint-steps")
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--dataset-repo-id", default="local/maniskill_pickcube_mp_4d")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/lerobot/maniskill/pickcube_mp_4d"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--policy-seed", type=int, default=POLICY_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat-count", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize or args.summarize_and_repeat:
        if args.run_dir is None or args.evaluation_root is None or args.selection_dir is None:
            raise ValueError(
                "summary modes require --run-dir, --evaluation-root, and --selection-dir"
            )
        steps = _parse_steps(args.checkpoint_steps or "")
        result = summarize_official(
            run_dir=args.run_dir,
            evaluation_root=args.evaluation_root,
            selection_dir=args.selection_dir,
            checkpoint_steps=steps,
        )
        if args.summarize_and_repeat:
            selected_path = args.selection_dir / "selected" / "pretrained_model"
            for index in range(args.repeat_count):
                repeat_dir = args.evaluation_root / "repeats" / f"repeat_{index:02d}"
                evaluate_checkpoint(
                    policy_path=selected_path,
                    dataset_repo_id=args.dataset_repo_id,
                    dataset_root=args.dataset_root,
                    output_dir=repeat_dir,
                    num_envs=args.num_envs,
                    num_episodes=args.num_episodes,
                    max_steps=args.max_steps,
                    policy_seed=args.policy_seed + index,
                    device=args.device,
                )
            result = summarize_official(
                run_dir=args.run_dir,
                evaluation_root=args.evaluation_root,
                selection_dir=args.selection_dir,
                checkpoint_steps=steps,
                repeat_count=args.repeat_count,
            )
        print(
            json.dumps(
                {
                    "selected_checkpoint_step": result["selected_checkpoint_step"],
                    "official_best": result.get("official_best_repeats"),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.policy_path is None or args.output_dir is None:
        raise ValueError("evaluation mode requires --policy-path and --output-dir")
    metrics = evaluate_checkpoint(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        policy_seed=args.policy_seed,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "output": _portable(args.output_dir),
                "success_once": metrics["evaluation"]["success_once"],
                "success_at_end": metrics["evaluation"]["success_at_end"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
