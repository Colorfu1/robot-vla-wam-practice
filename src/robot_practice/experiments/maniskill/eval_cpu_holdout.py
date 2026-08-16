"""Evaluate the selected PickCube 4D baseline on the fixed CPU holdout set.

This is deliberately separate from the official unseeded evaluation.  It uses
one CPU environment and calls ``reset(seed=[k])`` for every k in 3000..3399,
then runs the same 100-step/ignore-terminations protocol once per system.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

from .audit_demos import (
    FINGERPRINT_COMPONENTS,
    _state_fingerprint_from_observation,
    collect_training_fingerprints,
    compare_fingerprints,
)
from .audit_demos import (
    _metadata_path as audit_metadata_path,
)
from .eval_closed_loop import (
    DiffusionPolicyDriver,
    _to_numpy,
    build_privileged_observation,
    latency_summary,
    load_diffusion_policy,
    summarize_rows,
)
from .eval_official_protocol import (
    CONTROL_MODE,
    MAX_EPISODE_STEPS,
    SIM_BACKEND,
    TASK_ID,
    _build_single_official_env,
    _publish_directory,
    build_policy_observation_from_official,
)
from .rollout_contract import OFFICIAL_DATA_PROVENANCE, require_rollout_backend

PPO_ACTION_DIM = 4
HOLDOUT_SIZE = 400
SCHEMA_VERSION = 1
DATA_PROVENANCE = OFFICIAL_DATA_PROVENANCE
ROLLOUT_CONTRACT = require_rollout_backend(DATA_PROVENANCE, SIM_BACKEND)


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


def load_cpu_holdout_split(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("task") != TASK_ID:
        raise ValueError(f"CPU holdout task must be {TASK_ID!r}")
    required = {
        "sim_backend": SIM_BACKEND,
        "num_envs": 1,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "ignore_terminations": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"CPU holdout {key} must be {expected!r}, got {payload.get(key)!r}")
    holdout = payload.get("sets", {}).get("holdout")
    if not isinstance(holdout, Mapping):
        raise TypeError("CPU holdout split must contain sets.holdout")
    start = int(holdout["start"])
    stop = int(holdout["stop"])
    step = int(holdout.get("step", 1))
    count = int(holdout["count"])
    seeds = tuple(range(start, stop, step))
    if len(seeds) != count or len(seeds) != HOLDOUT_SIZE:
        raise ValueError(f"CPU holdout must expand to exactly {HOLDOUT_SIZE} seeds")
    if float(payload.get("state_match_tolerance", 0.0)) <= 0:
        raise ValueError("CPU holdout state_match_tolerance must be positive")
    return {
        "payload": payload,
        "seeds": seeds,
        "tolerance": float(payload["state_match_tolerance"]),
    }


def _scalar_signal(info: Mapping[str, Any], key: str) -> bool:
    value = info.get(key)
    if value is None:
        return False
    return bool(np.asarray(_to_numpy(value), dtype=bool).reshape(-1)[0])


class PPO4DDriver:
    """Load the official 4D PPO actor without changing the checkpoint."""

    def __init__(self, checkpoint: Path, *, device: str = "cuda") -> None:
        import torch
        from torch import nn

        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping):
            raise TypeError("PPO checkpoint must be a state dict")
        actor_state = {
            key.removeprefix("actor_mean."): value
            for key, value in state.items()
            if str(key).startswith("actor_mean.")
        }
        if not actor_state:
            raise KeyError("PPO checkpoint has no actor_mean.* parameters")
        output_dim = int(actor_state["6.weight"].shape[0])
        if output_dim != PPO_ACTION_DIM:
            raise ValueError(f"Expected 4D PPO checkpoint, got output dimension {output_dim}")
        self.device = torch.device(device)
        self.actor = nn.Sequential(
            nn.Linear(42, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, output_dim),
        ).to(self.device)
        self.actor.load_state_dict(actor_state, strict=True)
        self.actor.eval()
        self.latencies_ms: list[float] = []

    def reset(self) -> None:
        self.latencies_ms.clear()

    def select_action(self, observation: Mapping[str, Any]) -> np.ndarray:
        import torch

        state = build_privileged_observation(observation)
        started = time.perf_counter()
        with torch.inference_mode():
            action = self.actor(state.to(self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        return np.clip(action.detach().cpu().numpy(), -1.0, 1.0).astype(np.float32)


def _make_raw_env(system: str) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    obs_mode = "state_dict" if system == "ppo_privileged" else "rgb"
    env = gym.make(
        TASK_ID,
        reconfiguration_freq=1,
        obs_mode=obs_mode,
        control_mode=CONTROL_MODE,
        sim_backend="cpu",
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)


def _prepare_visual_action(
    driver: DiffusionPolicyDriver,
    observation: Mapping[str, Any],
) -> np.ndarray:
    prepared = driver.preprocessor(build_policy_observation_from_official(observation))
    return driver.select_action(prepared)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    count = len(rows)
    number = sum(bool(row[key]) for row in rows)
    return {"num_success": number, "num_episodes": count, "rate": number / count}


def _select_holdout_shard(
    seeds: Sequence[int], *, shard_index: int, shard_count: int
) -> tuple[tuple[int, ...], int]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if len(seeds) % shard_count != 0:
        raise ValueError("Holdout seed count must be divisible by shard_count")
    shard_size = len(seeds) // shard_count
    offset = shard_index * shard_size
    return tuple(seeds[offset : offset + shard_size]), offset


def _merge_latency_summaries(
    shard_metrics: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    summaries = []
    for metrics in shard_metrics:
        latency = metrics.get("latency")
        if not isinstance(latency, Mapping):
            continue
        summary = latency.get(key)
        if isinstance(summary, Mapping):
            summaries.append(summary)
    count = sum(int(summary.get("count", 0)) for summary in summaries)
    weighted_mean = sum(
        int(summary.get("count", 0)) * float(summary["mean_ms"])
        for summary in summaries
        if summary.get("mean_ms") is not None
    )
    return {
        "count": count,
        "mean_ms": None if count == 0 else weighted_mean / count,
        "p50_ms": None,
        "p95_ms": None,
        "quantiles": "unavailable after shard merge",
    }


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return [dict(row) for row in pq.read_table(path).to_pylist()]


def merge_holdout_shards(
    *,
    shards_root: Path,
    split_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Merge complete fixed-seed holdout shards into the canonical output."""

    shards_root = shards_root.expanduser().resolve()
    shard_dirs = sorted(path for path in shards_root.glob("worker_*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No holdout shard directories found under {shards_root}")

    split = load_cpu_holdout_split(split_file)
    expected_seeds = tuple(split["seeds"])
    shard_metrics: list[dict[str, Any]] = []
    shard_manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        required = ("metrics.json", "episodes.parquet", "run_manifest.json")
        if not all((shard_dir / name).is_file() for name in required):
            raise FileNotFoundError(f"Incomplete holdout shard: {shard_dir}")
        metrics = _read_json(shard_dir / "metrics.json")
        manifest = _read_json(shard_dir / "run_manifest.json")
        if metrics.get("system") not in {"visual", "random", "ppo_privileged"}:
            raise ValueError(f"Invalid holdout shard system: {shard_dir}")
        shard_rows = _read_parquet_rows(shard_dir / "episodes.parquet")
        if len(shard_rows) != int(metrics.get("num_episodes", -1)):
            raise ValueError(f"Shard row count disagrees with metrics: {shard_dir}")
        shard_metrics.append(metrics)
        shard_manifests.append(manifest)
        rows.extend(shard_rows)

    for key in ("system", "task", "control_mode", "sim_backend"):
        values = {metrics.get(key) for metrics in shard_metrics}
        if len(values) != 1:
            raise ValueError(f"Holdout shard {key} values disagree: {values}")
    rows.sort(key=lambda row: int(row["episode_index"]))
    expected_indices = list(range(len(expected_seeds)))
    actual_indices = [int(row["episode_index"]) for row in rows]
    actual_seeds = [int(row["seed"]) for row in rows]
    if actual_indices != expected_indices or actual_seeds != list(expected_seeds):
        raise ValueError("Holdout shards do not form the exact expected seed/index set")

    evaluation = summarize_rows(rows)
    evaluation["fail_once"] = _rate(rows, "fail_once")
    evaluation["fail_at_end"] = _rate(rows, "fail_at_end")
    metrics = dict(shard_metrics[0])
    metrics.update(
        {
            "created_at": iso_timestamp(),
            "seeds": list(expected_seeds),
            "seed_shard": {"index": None, "count": len(shard_dirs), "num_episodes": len(rows)},
            "num_episodes": len(rows),
            "evaluation": evaluation,
            "latency": {
                "policy_forward": _merge_latency_summaries(shard_metrics, "policy_forward"),
                "environment_step": _merge_latency_summaries(shard_metrics, "environment_step"),
            },
            "parallel_holdout": {
                "num_shards": len(shard_dirs),
                "shards_root": _portable(shards_root),
            },
        }
    )
    manifest = dict(shard_manifests[0])
    manifest.update(
        {
            "created_at": iso_timestamp(),
            "seeds": list(expected_seeds),
            "seed_shard": {"index": None, "count": len(shard_dirs), "num_episodes": len(rows)},
            "parallel_holdout": {
                "num_shards": len(shard_dirs),
                "shards_root": _portable(shards_root),
            },
        }
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "metrics.json", metrics)
    _write_rows(output_dir / "episodes.parquet", rows)
    write_json_atomic(output_dir / "run_manifest.json", manifest)
    return metrics


def evaluate_holdout(
    *,
    system: str,
    split_file: Path,
    output_dir: Path,
    policy_path: Path | None = None,
    dataset_repo_id: str = "local/maniskill_pickcube_mp_4d",
    dataset_root: Path = Path("datasets/lerobot/maniskill/pickcube_mp_4d"),
    ppo_checkpoint: Path | None = None,
    device: str = "cuda",
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    split = load_cpu_holdout_split(split_file)
    seeds, seed_offset = _select_holdout_shard(
        split["seeds"], shard_index=shard_index, shard_count=shard_count
    )
    if system not in {"visual", "random", "ppo_privileged"}:
        raise ValueError("system must be visual, random, or ppo_privileged")
    driver: Any | None = None
    if system == "visual":
        if policy_path is None:
            raise ValueError("visual holdout requires --policy-path")
        loaded = load_diffusion_policy(
            policy_path=policy_path,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            n_action_steps=8,
            device=device,
        )
        if int(loaded["action_dim"]) != PPO_ACTION_DIM:
            raise ValueError("visual holdout policy must output 4D actions")
        driver = DiffusionPolicyDriver(loaded)
    elif system == "ppo_privileged":
        if ppo_checkpoint is None:
            raise ValueError("ppo_privileged holdout requires --ppo-checkpoint")
        driver = PPO4DDriver(ppo_checkpoint, device=device)

    environment = (
        _build_single_official_env(env_index=0) if system == "visual" else _make_raw_env(system)
    )
    progress = tqdm(
        total=len(seeds),
        desc=f"holdout {output_dir.name}",
        unit="episode",
        dynamic_ncols=True,
    )
    rows: list[dict[str, Any]] = []
    environment_latencies: list[float] = []
    policy_latencies: list[float] = []
    try:
        for episode_index, seed in enumerate(seeds):
            observation, _ = environment.reset(seed=[int(seed)])
            if system == "random":
                # Keep the random reference reproducible while preserving the
                # exact holdout state reset for this episode.
                environment.action_space.seed(int(seed))
            if driver is not None:
                driver.reset()
            success_once = False
            success_at_end = False
            fail_once = False
            fail_at_end = False
            first_success_step: int | None = None
            ever = {key: False for key in ("is_grasped", "is_obj_placed", "is_robot_static")}
            action_sum = {"translation_norm": 0.0, "gripper_abs": 0.0}
            previous_action: np.ndarray | None = None
            jitter_sum = 0.0
            gripper_flips = 0
            for step in range(1, MAX_EPISODE_STEPS + 1):
                if system == "random":
                    action = np.asarray(environment.action_space.sample(), dtype=np.float32)
                elif system == "visual":
                    action = _prepare_visual_action(driver, observation)[0]
                else:
                    action = driver.select_action(observation)[0]
                if action.shape != (PPO_ACTION_DIM,):
                    raise ValueError(f"Holdout action must have shape (4,), got {action.shape}")
                action_sum["translation_norm"] += float(np.linalg.norm(action[:3]))
                action_sum["gripper_abs"] += float(abs(action[-1]))
                if previous_action is not None:
                    jitter_sum += float(np.linalg.norm(action - previous_action))
                    before = np.sign(previous_action[-1])
                    after = np.sign(action[-1])
                    gripper_flips += int(before != after and before != 0 and after != 0)
                previous_action = action.copy()
                started = time.perf_counter()
                observation, _reward, terminated, truncated, info = environment.step(action)
                environment_latencies.append((time.perf_counter() - started) * 1000.0)
                success = _scalar_signal(info, "success")
                fail = _scalar_signal(info, "fail")
                success_once |= success
                success_at_end = success
                fail_once |= fail
                fail_at_end = fail
                for key in ever:
                    ever[key] |= _scalar_signal(info, key)
                if success and first_success_step is None:
                    first_success_step = step
                if step == MAX_EPISODE_STEPS and (not bool(truncated) or bool(terminated)):
                    raise AssertionError(
                        "CPU holdout must truncate at step 100 and ignore terminations"
                    )
            rows.append(
                {
                    "task": TASK_ID,
                    "episode_index": seed_offset + episode_index,
                    "seed": int(seed),
                    "success_once": success_once,
                    "success_at_end": success_at_end,
                    "success": success_at_end,
                    "fail_once": fail_once,
                    "fail_at_end": fail_at_end,
                    "first_success_step": first_success_step,
                    "steps": MAX_EPISODE_STEPS,
                    "termination_reason": "truncated",
                    **ever,
                    "translation_norm_mean": action_sum["translation_norm"] / MAX_EPISODE_STEPS,
                    "rotation_norm_mean": None,
                    "gripper_abs_mean": action_sum["gripper_abs"] / MAX_EPISODE_STEPS,
                    "action_jitter_mean": jitter_sum / (MAX_EPISODE_STEPS - 1),
                    "gripper_flip_count": gripper_flips,
                }
            )
            if driver is not None:
                policy_latencies.extend(driver.latencies_ms)
            progress.update(1)
    finally:
        environment.close()
        progress.close()

    evaluation = summarize_rows(rows)
    evaluation["fail_once"] = _rate(rows, "fail_once")
    evaluation["fail_at_end"] = _rate(rows, "fail_at_end")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "label": "CPU fixed holdout",
        "system": system,
        "task": TASK_ID,
        "control_mode": CONTROL_MODE,
        "sim_backend": SIM_BACKEND,
        **ROLLOUT_CONTRACT,
        "seed_set": "holdout",
        "seeds": list(seeds),
        "seed_shard": {
            "index": shard_index,
            "count": shard_count,
            "num_episodes": len(seeds),
        },
        "num_episodes": len(rows),
        "control_frequency_hz": 20.0,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "evaluation": evaluation,
        "latency": {
            "policy_forward": latency_summary(policy_latencies),
            "environment_step": latency_summary(environment_latencies),
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "system": system,
        "policy_path": None if policy_path is None else _portable(policy_path),
        "ppo_checkpoint": None if ppo_checkpoint is None else _portable(ppo_checkpoint),
        "policy_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "split_file": _portable(split_file),
        "sim_backend": SIM_BACKEND,
        **ROLLOUT_CONTRACT,
        "num_envs": 1,
        "control_frequency_hz": 20.0,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "ignore_terminations": True,
        "reset": "one independent reset(seed=[k]) per episode",
        "seed_shard": {
            "index": shard_index,
            "count": shard_count,
            "num_episodes": len(seeds),
        },
        "random_action_seed": "episode seed" if system == "random" else None,
        "observation": {
            "keys": (
                ["observation.images.base_camera", "observation.state"]
                if system == "visual"
                else (["agent", "extra"] if system == "ppo_privileged" else [])
            ),
            "state_shape": [21]
            if system == "visual"
            else ([42] if system == "ppo_privileged" else None),
        },
        "action": {"shape": [PPO_ACTION_DIM], "control_mode": CONTROL_MODE},
        "success_source": 'native info["success"] only',
    }
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        complete = all(
            (output_dir / name).is_file()
            for name in ("metrics.json", "episodes.parquet", "run_manifest.json")
        )
        if complete:
            return _read_json(output_dir / "metrics.json")
        raise FileExistsError(f"Refusing to reuse incomplete holdout output: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.incomplete-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        write_json_atomic(temporary / "metrics.json", metrics)
        _write_rows(temporary / "episodes.parquet", rows)
        write_json_atomic(temporary / "run_manifest.json", manifest)
        _publish_directory(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return metrics


def _source_seed_records(source_h5: Path) -> dict[int, int]:
    metadata = _read_json(audit_metadata_path(source_h5))
    result = {}
    for item in metadata.get("episodes", []):
        if not isinstance(item, Mapping):
            continue
        value = item.get("episode_seed")
        if value is None:
            reset = item.get("reset_kwargs", {})
            value = reset.get("seed") if isinstance(reset, Mapping) else None
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else None
        if value is not None:
            result[int(item["episode_id"])] = int(value)
    return result


def _reset_fingerprint_cpu(seed: int) -> dict[str, list[float]]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    env = gym.make(
        TASK_ID,
        reconfiguration_freq=1,
        obs_mode="state_dict",
        control_mode=CONTROL_MODE,
        sim_backend="cpu",
    )
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=False)
    try:
        observation, _ = env.reset(seed=[int(seed)])
        fields = _state_fingerprint_from_observation(observation)
        return {
            key: np.asarray(value).reshape(-1)[0:].tolist()
            for key, value in {
                "robot_qpos": fields["robot_qpos"],
                "cube_position": fields["cube_position"],
                "cube_orientation": fields["cube_orientation"],
                "goal_position": fields["goal_position"],
            }.items()
        }
    finally:
        env.close()


def audit_seed_isolation(
    *,
    source_h5: Path,
    split_file: Path,
    output_path: Path,
    sample_size: int = 20,
) -> dict[str, Any]:
    """Check seed reproduction, then fall back to full fingerprint comparison."""

    split = load_cpu_holdout_split(split_file)
    source_h5 = source_h5.expanduser().resolve()
    source_rows = collect_training_fingerprints(source_h5)
    seed_by_episode = _source_seed_records(source_h5)
    if len(source_rows) < sample_size:
        raise ValueError("Source HDF5 has fewer rows than the leakage audit sample")
    rng = np.random.default_rng(20260810)
    sample_indices = sorted(rng.choice(len(source_rows), size=sample_size, replace=False).tolist())
    sample_results = []
    seed_reproduced = True
    for index in sample_indices:
        source_row = source_rows[index]
        seed = seed_by_episode.get(int(source_row["episode_id"]))
        if seed is None:
            seed_reproduced = False
            sample_results.append(
                {"episode_id": source_row["episode_id"], "seed": None, "match": False}
            )
            continue
        reset = _reset_fingerprint_cpu(seed)
        component_errors = {
            key: float(np.max(np.abs(np.asarray(reset[key]) - np.asarray(source_row[key]))))
            for key in FINGERPRINT_COMPONENTS
        }
        match = all(value <= split["tolerance"] for value in component_errors.values())
        seed_reproduced &= match
        sample_results.append(
            {
                "episode_id": source_row["episode_id"],
                "seed": seed,
                "match": match,
                "component_max_abs_error": component_errors,
            }
        )

    demo_seeds = set(seed_by_episode.values())
    holdout_seeds = set(split["seeds"])
    integer_disjoint = not demo_seeds.intersection(holdout_seeds)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "task": TASK_ID,
        "source_h5": _portable(source_h5),
        "split_file": _portable(split_file),
        "tolerance": split["tolerance"],
        "sample_size": sample_size,
        "seed_reproduction_sample": sample_results,
        "seed_reproduced": seed_reproduced,
        "integer_seed_intersection": sorted(demo_seeds.intersection(holdout_seeds)),
        "integer_seed_disjoint": integer_disjoint,
    }
    if seed_reproduced and integer_disjoint:
        result.update(
            {
                "method": "seed_reproduction+integer_disjoint",
                "status": "no_exact_duplicate",
                "num_training_states": len(source_rows),
                "num_holdout_states": len(split["seeds"]),
            }
        )
    else:
        holdout_rows = []
        for index, seed in enumerate(split["seeds"]):
            values = _reset_fingerprint_cpu(int(seed))
            holdout_rows.append(
                {
                    "episode_id": index,
                    "seed": int(seed),
                    "source": "cpu_holdout_reset",
                    **values,
                }
            )
        details, report = compare_fingerprints(
            source_rows,
            holdout_rows,
            tolerance=split["tolerance"],
        )
        result.update(
            {
                "method": "initial_state_fingerprint_fallback",
                "status": report["status"],
                "num_training_states": report["num_training_states"],
                "num_holdout_states": report["num_evaluation_states"],
                "num_exact_duplicates": report["num_exact_duplicates"],
                "nearest_distance": report["nearest_distance"],
                "nearest_component_distances": report["nearest_component_distances"],
                "details": details,
            }
        )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, result)
    return result


def compare_results(
    *,
    official_best: Mapping[str, Path],
    holdout_metrics: Mapping[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "paper_reference": {
            "table": "ManiSkill3 Table III PickCube RGB",
            "dp_1000_demos": 1.00,
            "dp_100_demos": 0.76,
            "note": (
                "Reported paper values are max-over-evaluation-rounds and use "
                "the official DP implementation."
            ),
        },
        "official_best": {},
        "holdout_once": {},
        "references": {},
        "comparability_notes": [
            (
                "official_best is the unseeded official-protocol max over checkpoints "
                "and is the paper-comparison number."
            ),
            (
                "holdout_once is one fixed CPU seed-set evaluation and must not be "
                "compared directly with the paper."
            ),
            (
                "The LeRobot diffusion implementation is intentionally different "
                "from the official DP implementation."
            ),
            (
                "The PPO reference uses privileged 42D state and is not an upper "
                "bound or a pure perception gap."
            ),
        ],
    }
    for arm, path in official_best.items():
        result["official_best"][arm] = _read_json(path)
    for label, path in holdout_metrics.items():
        result["holdout_once"][label] = _read_json(path)
    result["references"] = {
        "random": _read_json(holdout_metrics["random"]),
        "ppo_privileged": _read_json(holdout_metrics["ppo_privileged"]),
    }
    write_json_atomic(output_path.expanduser().resolve(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-source-h5", type=Path)
    mode.add_argument("--system", choices=("visual", "random", "ppo_privileged"))
    mode.add_argument("--compare", action="store_true")
    mode.add_argument("--merge-shards", action="store_true")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("configs/experiments/maniskill/split_cpu.json"),
    )
    parser.add_argument("--source-h5", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--dataset-repo-id", default="local/maniskill_pickcube_mp_4d")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/lerobot/maniskill/pickcube_mp_4d"),
    )
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shards-root", type=Path)
    parser.add_argument("--official-best-1000", type=Path)
    parser.add_argument("--official-best-100", type=Path)
    parser.add_argument("--holdout-1000", type=Path)
    parser.add_argument("--holdout-100", type=Path)
    parser.add_argument("--random-metrics", type=Path)
    parser.add_argument("--ppo-metrics", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_source_h5 is not None:
        if args.output_report is None:
            raise ValueError("leakage audit requires --output-report")
        result = audit_seed_isolation(
            source_h5=args.audit_source_h5,
            split_file=args.split_file,
            output_path=args.output_report,
        )
        print(
            json.dumps(
                {"method": result["method"], "status": result["status"]},
                ensure_ascii=False,
            )
        )
        return 0 if result["status"] == "no_exact_duplicate" else 67
    if args.compare:
        required = {
            "official-best-1000": args.official_best_1000,
            "official-best-100": args.official_best_100,
            "holdout-1000": args.holdout_1000,
            "holdout-100": args.holdout_100,
            "random-metrics": args.random_metrics,
            "ppo-metrics": args.ppo_metrics,
            "output-dir": args.output_dir,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(f"compare mode is missing: {', '.join(missing)}")
        result = compare_results(
            official_best={"1000": args.official_best_1000, "100": args.official_best_100},
            holdout_metrics={
                "diffusion_1000": args.holdout_1000,
                "diffusion_100": args.holdout_100,
                "random": args.random_metrics,
                "ppo_privileged": args.ppo_metrics,
            },
            output_path=args.output_dir,
        )
        print(
            json.dumps(
                {"output": _portable(args.output_dir), "arms": list(result["official_best"])},
                ensure_ascii=False,
            )
        )
        return 0
    if args.merge_shards:
        if args.shards_root is None or args.output_dir is None:
            raise ValueError("merge mode requires --shards-root and --output-dir")
        metrics = merge_holdout_shards(
            shards_root=args.shards_root,
            split_file=args.split_file,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "output": _portable(args.output_dir),
                    "num_episodes": metrics["num_episodes"],
                    "success_once": metrics["evaluation"]["success_once"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.system is None or args.output_dir is None:
        raise ValueError("evaluation mode requires --system and --output-dir")
    metrics = evaluate_holdout(
        system=args.system,
        split_file=args.split_file,
        output_dir=args.output_dir,
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        ppo_checkpoint=args.ppo_checkpoint,
        device=args.device,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(
        json.dumps(
            {
                "output": _portable(args.output_dir),
                "success_once": metrics["evaluation"]["success_once"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
