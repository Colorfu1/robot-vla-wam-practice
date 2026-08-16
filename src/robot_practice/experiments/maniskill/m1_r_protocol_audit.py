"""Formal M1-R audit for the Policy Decorator ManiSkill2/v2 route.

This route is deliberately independent of the ManiSkill3 M1-W route.  It
freezes the pinned Policy Decorator checkout, its public v0 demonstrations,
and the repo-local v2 environment registered by that checkout.  Replay and
state-bank phases are resumable, hard-gated commands; the shell wrapper runs
them in the dedicated ``maniskill2-pidec`` environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .policy_decorator_exact_replay import exact_replay

SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("configs/experiments/maniskill/m1_r_policy_decorator_audit.json")
DEFAULT_OUTPUT = Path("outputs/policy_decorator_ms2/m1_r_protocol_audit")
EXPECTED_STATE_FIELDS = (
    ("agent.qpos", 9),
    ("agent.qvel", 9),
    ("agent.base_pose", 7),
    ("extra.tcp_pose", 7),
    ("extra.peg_pose", 7),
    ("extra.peg_half_size", 3),
    ("extra.box_hole_pose", 7),
    ("extra.box_hole_radius", 1),
)


def _repo_root() -> Path:
    configured = os.environ.get("PRACTICE_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else _repo_root() / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return _relative(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("JSON payload contains NaN or Infinity")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json_dumps(payload), encoding="utf-8"
        )
        try:
            temporary.replace(path)
        except OSError as exc:
            if exc.errno != 18:
                raise
            shutil.copy2(temporary, path)
            temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _read_json(_path(path))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported M1-R schema: {config.get('schema_version')!r}")
    if config.get("protocol_id") != "m1_r_policy_decorator_ms2_v2":
        raise ValueError("Unexpected M1-R protocol_id")
    return config


def _identity_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    identity = config["identity"]
    return (
        _path(str(identity["policy_decorator_root"])),
        _path(str(identity["dataset_h5"])),
        _path(str(identity["dataset_json"])),
    )


def _output_root(override: str | Path | None) -> Path:
    return _path(override) if override is not None else _path(DEFAULT_OUTPUT)


def _cached_gate(path: Path, filename: str) -> dict[str, Any] | None:
    cached_path = path / filename
    if not cached_path.is_file():
        return None
    cached = _read_json(cached_path)
    if not cached.get("gate_pass", False):
        raise RuntimeError(f"Existing artifact is not a hard-gate success: {cached_path}")
    return cached


def _assert_clean_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Incomplete M1-R output exists; refusing overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _episode_ids(handle: h5py.File) -> list[int]:
    names = [name for name in handle if name.startswith("traj_")]
    return sorted(int(name.split("_", 1)[1]) for name in names)


def _source_records(metadata: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    records: dict[int, Mapping[str, Any]] = {}
    for record in metadata.get("episodes", []):
        episode_id = int(record["episode_id"])
        if episode_id in records:
            raise ValueError(f"Duplicate metadata episode_id: {episode_id}")
        records[episode_id] = record
    return records


def _first_success(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool).reshape(-1))
    return None if indices.size == 0 else int(indices[0]) + 1


def _fingerprint(values: np.ndarray, tolerance: float) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    quantized = np.rint(array / tolerance).astype(np.int64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(quantized.tobytes())
    return digest.hexdigest()


def _action_stats(actions: np.ndarray) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "min": values.min(axis=0).tolist() if values.size else [],
        "max": values.max(axis=0).tolist() if values.size else [],
        "mean": values.mean(axis=0).tolist() if values.size else [],
        "std": values.std(axis=0).tolist() if values.size else [],
        "max_abs": float(absolute.max()) if values.size else 0.0,
        "out_of_bounds_count": int(np.sum(absolute > 1.00001)),
    }


def source_audit(
    config: Mapping[str, Any], *, output_root: str | Path | None = None
) -> dict[str, Any]:
    root = _output_root(output_root)
    cached = _cached_gate(root, "source_audit.json")
    if cached is not None:
        return cached
    repo, dataset_h5, dataset_json = _identity_paths(config)
    identity = config["identity"]
    for path in (repo / ".git", dataset_h5, dataset_json):
        if not path.exists():
            raise FileNotFoundError(f"Missing M1-R identity input: {path}")
    commit = _git_head(repo)
    h5_sha = _sha256(dataset_h5)
    json_sha = _sha256(dataset_json)
    if commit != identity["policy_decorator_commit"]:
        raise RuntimeError(f"Policy Decorator commit mismatch: {commit}")
    if h5_sha != identity["dataset_h5_sha256"]:
        raise RuntimeError(f"Dataset H5 hash mismatch: {h5_sha}")
    if json_sha != identity["dataset_json_sha256"]:
        raise RuntimeError(f"Dataset JSON hash mismatch: {json_sha}")

    metadata = _read_json(dataset_json)
    env_info = metadata.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {})
    records = _source_records(metadata)
    expected_count = int(identity["episode_count"])
    lengths: list[int] = []
    first_steps: list[int] = []
    tails: list[int] = []
    fingerprints: list[str] = []
    actions_for_stats: list[np.ndarray] = []
    invalid: list[str] = []
    with h5py.File(dataset_h5, "r") as handle:
        ids = _episode_ids(handle)
        for episode_id in ids:
            group = handle[f"traj_{episode_id}"]
            actions = np.asarray(group["actions"])
            success = np.asarray(group["success"], dtype=bool).reshape(-1)
            obs = np.asarray(group["obs"])
            env_states = np.asarray(group["env_states"])
            record = records.get(episode_id)
            if record is None:
                invalid.append(f"missing_metadata:{episode_id}")
                continue
            if actions.ndim != 2 or actions.shape[1] != 7:
                invalid.append(f"action_shape:{episode_id}:{actions.shape}")
            if obs.shape != (len(actions) + 1, 50):
                invalid.append(f"obs_shape:{episode_id}:{obs.shape}")
            if env_states.shape != (len(actions) + 1, 70):
                invalid.append(f"env_state_shape:{episode_id}:{env_states.shape}")
            if success.shape != (len(actions),):
                invalid.append(f"success_shape:{episode_id}:{success.shape}")
            if not np.isfinite(actions).all() or not np.isfinite(obs).all():
                invalid.append(f"nonfinite:{episode_id}")
            if not bool(success[-1]) or not bool(record.get("info", {}).get("success", False)):
                invalid.append(f"not_successful:{episode_id}")
            if int(record.get("episode_seed")) != int(record.get("reset_kwargs", {}).get("seed")):
                invalid.append(f"seed_metadata_mismatch:{episode_id}")
            first = _first_success(success)
            if first is None:
                invalid.append(f"no_success:{episode_id}")
            else:
                first_steps.append(first)
                tails.append(len(actions) - first)
            lengths.append(len(actions))
            fingerprints.append(_fingerprint(env_states[0], 1e-6))
            actions_for_stats.append(actions)
    all_actions = np.concatenate(actions_for_stats, axis=0) if actions_for_stats else np.empty((0, 7))
    action_stats = _action_stats(all_actions)
    unique_fingerprints = len(set(fingerprints))
    gate_pass = (
        env_info.get("env_id") == identity["dataset_env_id"]
        and env_kwargs.get("control_mode") == config["environment"]["control_mode"]
        and env_kwargs.get("obs_mode") == config["environment"]["obs_mode"]
        and len(ids) == expected_count
        and len(records) == expected_count
        and not invalid
        and unique_fingerprints == expected_count
        and action_stats["finite"]
        and action_stats["out_of_bounds_count"] == 0
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "source_audit",
        "gate_pass": gate_pass,
        "identity": {
            "policy_decorator_commit": commit,
            "dataset_h5": _relative(dataset_h5),
            "dataset_h5_sha256": h5_sha,
            "dataset_json": _relative(dataset_json),
            "dataset_json_sha256": json_sha,
            "dataset_env_id": env_info.get("env_id"),
        },
        "summary": {
            "num_episodes": len(ids),
            "expected_episodes": expected_count,
            "episode_id_range": [min(ids), max(ids)] if ids else None,
            "unique_start_state_fingerprints": unique_fingerprints,
            "trajectory_length": {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "mean": float(np.mean(lengths)) if lengths else None,
            },
            "first_success_step": {
                "min": min(first_steps) if first_steps else None,
                "p50": float(np.percentile(first_steps, 50)) if first_steps else None,
                "p90": float(np.percentile(first_steps, 90)) if first_steps else None,
                "max": max(first_steps) if first_steps else None,
            },
            "post_success_tail": {
                "p50": float(np.percentile(tails, 50)) if tails else None,
                "p90": float(np.percentile(tails, 90)) if tails else None,
                "max": max(tails) if tails else None,
            },
            "action_stats": action_stats,
            "observation_schema": {
                "dim": 50,
                "flatten_order": [f"{name}[0:{size}]" for name, size in EXPECTED_STATE_FIELDS],
            },
        },
        "metadata": {"env_info": env_info, "episodes": len(records)},
        "invalid": invalid,
    }
    _write_json(root / "source_audit.json", result)
    if not gate_pass:
        raise RuntimeError(f"M1-R source audit failed: {invalid[:5]}")
    return result


def _flatten_observation(env: Any) -> tuple[np.ndarray, list[str]]:
    agent = env.unwrapped._get_obs_agent()
    extra = env.unwrapped._get_obs_extra()
    pieces: list[np.ndarray] = []
    fields: list[str] = []
    for prefix, values in (("agent", agent), ("extra", extra)):
        for name, value in values.items():
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            pieces.append(array)
            fields.append(f"{prefix}.{name}[0:{array.size}]")
    return np.concatenate(pieces), fields


def environment_audit(
    config: Mapping[str, Any], *, output_root: str | Path | None = None
) -> dict[str, Any]:
    root = _output_root(output_root)
    cached = _cached_gate(root, "environment_contract.json")
    if cached is not None:
        return cached
    import gymnasium as gym
    import mani_skill2.envs  # noqa: F401
    importlib.import_module("envs.maniskill_fixed")

    environment = config["environment"]
    base_env_kwargs = {
        "obs_mode": environment["obs_mode"],
        "control_mode": environment["control_mode"],
        "bg_name": None,
        "render_mode": environment["render_mode"],
    }
    contracts: dict[str, Any] = {}
    invalid: list[str] = []
    env_ids = (config["identity"]["dataset_env_id"], environment["runtime_env_id"])
    for env_id in env_ids:
        for reward_label, reward_mode in (
            ("reference_default", None),
            ("formal", environment["reward_mode"]),
        ):
            env_kwargs = dict(base_env_kwargs)
            if reward_mode is not None:
                env_kwargs["reward_mode"] = reward_mode
            env = gym.make(env_id, **env_kwargs)
            try:
                observation, _ = env.reset(seed=0)
                flat, fields = _flatten_observation(env)
                spec = gym.spec(env_id)
                action_shape = list(env.action_space.shape or ())
                obs_shape = list(np.asarray(observation).shape)
                arm = env.unwrapped.agent.controller.controllers["arm"]
                arm_config = arm.config
                controller = {
                    "arm_class": type(arm).__name__,
                    "arm_frame": getattr(arm_config, "frame", None),
                    "arm_use_delta": getattr(arm_config, "use_delta", None),
                    "arm_use_target": getattr(arm_config, "use_target", None),
                    "arm_interpolate": getattr(arm_config, "interpolate", None),
                    "arm_normalize_action": getattr(arm_config, "normalize_action", None),
                    "arm_pos_lower": np.asarray(
                        getattr(arm_config, "pos_lower", []), dtype=np.float64
                    ).reshape(-1).tolist(),
                    "arm_pos_upper": np.asarray(
                        getattr(arm_config, "pos_upper", []), dtype=np.float64
                    ).reshape(-1).tolist(),
                    "arm_rot_bound": float(getattr(arm_config, "rot_bound", 0.0)),
                    "arm_action_indices": [0, 1, 2, 3, 4, 5],
                    "gripper_action_indices": [6],
                    "action_layout": [
                        "arm.delta_position[0:3]",
                        "arm.delta_rotvec[0:3]",
                        "gripper.target_position[0:1]",
                    ],
                }
                contract = {
                    "env_id": env_id,
                    "reward_label": reward_label,
                    "requested_reward_mode": reward_mode,
                    "spec_max_episode_steps": spec.max_episode_steps,
                    "time_limit_max_episode_steps": getattr(env, "_max_episode_steps", None),
                    "control_mode": environment["control_mode"],
                    "controller": controller,
                    "action_space": {
                        "shape": action_shape,
                        "low": np.asarray(env.action_space.low).reshape(-1).tolist(),
                        "high": np.asarray(env.action_space.high).reshape(-1).tolist(),
                    },
                    "observation": {
                        "shape": obs_shape,
                        "dtype": str(np.asarray(observation).dtype),
                        "flattened_shape": list(flat.shape),
                        "flatten_order": fields,
                        "max_flatten_error": float(np.max(np.abs(flat - observation))),
                    },
                    "control_frequency_hz": getattr(env.unwrapped, "_control_freq", None),
                    "evaluate_keys": sorted(str(key) for key in env.unwrapped.evaluate()),
                    "reward_mode": getattr(env.unwrapped, "_reward_mode", None),
                }
                contracts[f"{env_id}:{reward_label}"] = contract
                if action_shape != [7] or obs_shape != [50]:
                    invalid.append(f"shape:{env_id}:{reward_label}")
                if contract["spec_max_episode_steps"] != environment["max_episode_steps"]:
                    invalid.append(f"horizon:{env_id}:{reward_label}")
                expected_reward = (
                    config["environment"]["reference_default_reward_mode"]
                    if reward_mode is None
                    else environment["reward_mode"]
                )
                if contract["reward_mode"] != expected_reward:
                    invalid.append(f"reward_mode:{env_id}:{reward_label}:{contract['reward_mode']}")
                if contract["observation"]["max_flatten_error"] > 1e-5:
                    invalid.append(f"flatten:{env_id}:{reward_label}")
                if controller["arm_frame"] != "ee" or not controller["arm_use_delta"]:
                    invalid.append(f"controller_frame:{env_id}:{reward_label}")
            finally:
                env.close()
    v2_source = _path(str(config["identity"]["policy_decorator_root"])) / "envs" / "maniskill_fixed.py"
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "environment_audit",
        "gate_pass": not invalid,
        "runtime": contracts.get(f"{environment['runtime_env_id']}:formal", {}),
        "dataset_env": contracts.get(
            f"{config['identity']['dataset_env_id']}:reference_default", {}
        ),
        "formal_environments": {
            env_id: contracts.get(f"{env_id}:formal", {}) for env_id in env_ids
        },
        "reference_default_environments": {
            env_id: contracts.get(f"{env_id}:reference_default", {}) for env_id in env_ids
        },
        "termination": {
            "success_key": "success",
            "success_terminates": config["protocol"]["termination"]["success_terminates"],
            "time_limit_truncates": config["protocol"]["termination"]["time_limit_truncates"],
            "max_episode_steps": environment["max_episode_steps"],
        },
        "v2_registration_source": {"path": _relative(v2_source), "sha256": _sha256(v2_source)},
        "invalid": invalid,
    }
    _write_json(root / "environment_contract.json", result)
    if not result["gate_pass"]:
        raise RuntimeError(f"M1-R environment audit failed: {invalid}")
    return result


def replay_phase(
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None,
    count: int,
    phase_name: str,
) -> dict[str, Any]:
    root = _output_root(output_root)
    suffix = phase_name.removeprefix("replay_")
    replay_dir = root / f"reference_replay_{suffix}"
    manifest_path = replay_dir / "replay_manifest.json"
    if manifest_path.is_file():
        cached = _read_json(manifest_path)
        if not cached.get("gate_pass", False):
            raise RuntimeError(f"Existing replay manifest is not a gate success: {manifest_path}")
        return cached
    _assert_clean_output(replay_dir)
    _, dataset_h5, dataset_json = _identity_paths(config)
    started = time.perf_counter()
    replay = exact_replay(
        dataset_h5=dataset_h5,
        dataset_json=dataset_json,
        output_dir=replay_dir,
        count=count,
        sample_seed=20260811,
        env_id=config["identity"]["dataset_env_id"],
        strict_env_id=config["environment"]["runtime_env_id"],
        reward_mode=config["environment"]["reward_mode"],
    )
    summary = replay["summary"]
    expected = count
    max_error = float(summary["max_initial_state_error"])
    strict_max_error = max(
        float(row["strict"]["initial_state_max_abs_error"])
        for row in replay["episodes"]
    )
    termination = {}
    for name in ("native", "strict"):
        rows = [row[name] for row in replay["episodes"]]
        termination[name] = {
            "terminated_count": sum(bool(row["terminated"]) for row in rows),
            "truncated_count": sum(bool(row["truncated"]) for row in rows),
            "success_once_count": sum(bool(row["replay_success_once"]) for row in rows),
            "final_success_count": sum(bool(row["replay_final_success"]) for row in rows),
        }
    gate_pass = (
        summary["native"]["num_episodes"] == expected
        and summary["strict"]["num_episodes"] == expected
        and summary["native"]["num_success"] == expected
        and summary["strict"]["num_success"] == expected
        and summary["native_success_once"]["num_success"] == expected
        and summary["strict_success_once"]["num_success"] == expected
        and max_error <= float(config["protocol"]["max_initial_state_error"])
        and strict_max_error <= float(config["protocol"]["max_initial_state_error"])
        and all(value["terminated_count"] == expected for value in termination.values())
        and all(value["truncated_count"] == 0 for value in termination.values())
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase_name,
        "gate_pass": gate_pass,
        "expected_episodes": expected,
        "replay_result": _relative(replay_dir / "p1_exact_replay.json"),
        "summary": summary,
        "termination": termination,
        "max_strict_initial_state_error": strict_max_error,
        "elapsed_seconds": time.perf_counter() - started,
        "state_reconstruction": replay["state_reconstruction"],
    }
    _write_json(manifest_path, manifest)
    if not gate_pass:
        raise RuntimeError(
            f"M1-R {phase_name} gate failed: native={summary['native']} "
            f"strict={summary['strict']} max_initial_state_error={max_error} "
            f"strict_initial_state_error={strict_max_error}"
        )
    return manifest


def trajectory_audit(
    config: Mapping[str, Any], *, output_root: str | Path | None = None
) -> dict[str, Any]:
    root = _output_root(output_root)
    cached = _cached_gate(root, "trajectory_audit.json")
    if cached is not None:
        return cached
    source = source_audit(config, output_root=output_root)
    replay_path = root / "reference_replay_full" / "replay_manifest.json"
    if not replay_path.is_file():
        raise FileNotFoundError("Run replay-full before trajectory-audit")
    replay = _read_json(replay_path)
    _, dataset_h5, dataset_json = _identity_paths(config)
    metadata = _read_json(dataset_json)
    records = _source_records(metadata)
    lengths: list[int] = []
    first_steps: list[int] = []
    tails: list[int] = []
    fingerprints: dict[int, str] = {}
    with h5py.File(dataset_h5, "r") as handle:
        for episode_id in sorted(records):
            group = handle[f"traj_{episode_id}"]
            actions = np.asarray(group["actions"])
            success = np.asarray(group["success"], dtype=bool)
            lengths.append(len(actions))
            first = _first_success(success)
            if first is None:
                raise RuntimeError(f"No success in source trajectory {episode_id}")
            first_steps.append(first)
            tails.append(len(actions) - first)
            fingerprints[episode_id] = _fingerprint(np.asarray(group["env_states"])[0], 1e-6)
    environment = config["environment"]
    formal_horizon = int(environment["max_episode_steps"])
    temporal = config["temporal"]
    coverage = sum(length <= formal_horizon for length in lengths) / max(1, len(lengths))
    gate_pass = (
        bool(source.get("gate_pass"))
        and bool(replay.get("gate_pass"))
        and len(lengths) == int(config["identity"]["episode_count"])
        and coverage == 1.0
        and len(set(fingerprints.values())) == len(fingerprints)
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "trajectory_audit",
        "gate_pass": gate_pass,
        "horizon": {
            "formal_horizon": formal_horizon,
            "source": "pinned runtime environment contract",
            "max_source_trajectory_length": max(lengths, default=None),
            "coverage": float(coverage),
        },
        "summary": {
            "episode_count": len(lengths),
            "length_min": min(lengths, default=None),
            "length_p50": float(np.percentile(lengths, 50)) if lengths else None,
            "length_p90": float(np.percentile(lengths, 90)) if lengths else None,
            "length_max": max(lengths, default=None),
            "first_success_p50": float(np.percentile(first_steps, 50)) if first_steps else None,
            "first_success_p90": float(np.percentile(first_steps, 90)) if first_steps else None,
            "post_success_tail_p50": float(np.percentile(tails, 50)) if tails else None,
            "post_success_tail_max": max(tails, default=None),
        },
        "chunk_contract": {
            "n_obs_steps": int(temporal["n_obs_steps"]),
            "horizon": int(temporal["horizon"]),
            "n_action_steps": int(temporal["n_action_steps"]),
            "execution_action_steps": int(temporal["execution_action_steps"]),
            "cross_episode_windows": 0,
            "tail_policy": temporal["tail_policy"],
        },
        "reward_contract": {
            "formal_reward_mode": config["environment"]["reward_mode"],
            "reference_default_reward_mode": config["environment"][
                "reference_default_reward_mode"
            ],
            "replay_evaluation_reward_mode": config["environment"]["reward_mode"],
        },
        "termination": replay.get("termination", {}),
        "identity": {"key": "start_state_fingerprint", "num_unique": len(set(fingerprints.values()))},
    }
    _write_json(root / "trajectory_audit.json", result)
    if not gate_pass:
        raise RuntimeError("M1-R trajectory audit failed")
    return result


def _copy_trimmed_group(source: h5py.Group, destination: h5py.Group, trim_length: int) -> None:
    action_length = int(source["actions"].shape[0])
    for name, dataset in source.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        if dataset.shape and dataset.shape[0] == action_length + 1:
            data = dataset[: trim_length + 1]
        elif dataset.shape and dataset.shape[0] == action_length:
            data = dataset[:trim_length]
        else:
            data = dataset[:]
        destination.create_dataset(name, data=data, dtype=dataset.dtype)
        for key, value in dataset.attrs.items():
            destination[name].attrs[key] = value


def _split_episode_ids(
    fingerprints: Mapping[int, str], *, development_count: int, test_count: int
) -> dict[str, list[int]]:
    ordered = [episode_id for episode_id, _ in sorted(fingerprints.items(), key=lambda item: item[1])]
    if development_count + test_count >= len(ordered):
        raise ValueError("Offline development/test split consumes the complete dataset")
    return {
        "development": ordered[:development_count],
        "test": ordered[development_count : development_count + test_count],
        "train": ordered[development_count + test_count :],
    }


def freeze_data(
    config: Mapping[str, Any], *, output_root: str | Path | None = None
) -> dict[str, Any]:
    root = _output_root(output_root)
    manifest_path = root / "d_core_manifest.json"
    if manifest_path.is_file():
        cached = _read_json(manifest_path)
        if not cached.get("gate_pass", False):
            raise RuntimeError(f"Existing D_core manifest is not a gate success: {manifest_path}")
        return cached
    audit = trajectory_audit(config, output_root=output_root)
    if not audit["gate_pass"]:
        raise RuntimeError("Cannot freeze D_core before trajectory audit passes")
    _, dataset_h5, dataset_json = _identity_paths(config)
    metadata = _read_json(dataset_json)
    records = _source_records(metadata)
    tail = int(config["protocol"]["d_core_success_tail_frames"])
    fingerprints: dict[int, str] = {}
    trims: dict[int, int] = {}
    with h5py.File(dataset_h5, "r") as source:
        for episode_id in sorted(records):
            group = source[f"traj_{episode_id}"]
            fingerprints[episode_id] = _fingerprint(np.asarray(group["env_states"])[0], 1e-6)
            first = _first_success(np.asarray(group["success"], dtype=bool))
            if first is None:
                raise RuntimeError(f"D_core episode has no success: {episode_id}")
            trims[episode_id] = min(int(group["actions"].shape[0]), first + tail)
    split_config = config["splits"]["offline"]
    splits = _split_episode_ids(
        fingerprints,
        development_count=int(split_config["development"]),
        test_count=int(split_config["test"]),
    )
    dcore = root / "d_core"
    _assert_clean_output(dcore)
    with h5py.File(dataset_h5, "r") as source, h5py.File(dcore / "trajectory.h5", "w") as destination:
        for key, value in source.attrs.items():
            destination.attrs[key] = value
        for episode_id in sorted(records):
            _copy_trimmed_group(
                source[f"traj_{episode_id}"],
                destination.create_group(f"traj_{episode_id}"),
                trims[episode_id],
            )
    frozen_metadata = copy.deepcopy(metadata)
    frozen_metadata["env_info"]["env_id"] = config["environment"]["runtime_env_id"]
    frozen_metadata["env_info"]["max_episode_steps"] = config["environment"]["max_episode_steps"]
    frozen_metadata["m1_r"] = {
        "source_env_id": config["identity"]["dataset_env_id"],
        "runtime_env_id": config["environment"]["runtime_env_id"],
        "trim_success_tail_frames": tail,
        "split_key": "start_state_fingerprint",
    }
    for record in frozen_metadata.get("episodes", []):
        record["elapsed_steps"] = trims[int(record["episode_id"])]
    _write_json(dcore / "trajectory.json", frozen_metadata)
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in splits}
    for split_name, ids in splits.items():
        split_file = dcore / "splits" / f"{split_name}.jsonl"
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with split_file.open("w", encoding="utf-8") as stream:
            for episode_id in ids:
                row = {
                    "episode_id": episode_id,
                    "split": split_name,
                    "start_state_fingerprint": fingerprints[episode_id],
                    "trimmed_length": trims[episode_id],
                    "episode_seed": records[episode_id]["episode_seed"],
                }
                stream.write(json_dumps(row))
                split_rows[split_name].append(row)
    leakage = {
        left: {
            right: len(
                {row["start_state_fingerprint"] for row in split_rows[left]}
                & {row["start_state_fingerprint"] for row in split_rows[right]}
            )
            for right in split_rows
            if right != left
        }
        for left in split_rows
    }
    gate_pass = all(value == 0 for left in leakage.values() for value in left.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "freeze_data",
        "gate_pass": gate_pass,
        "d_core": _relative(dcore),
        "episode_count": len(records),
        "frame_count": int(sum(trims.values())),
        "trim_success_tail_frames": tail,
        "splits": {name: len(rows) for name, rows in split_rows.items()},
        "leakage": leakage,
        "identity": {"key": "start_state_fingerprint", "unique": len(set(fingerprints.values()))},
        "source_h5_sha256": _sha256(dataset_h5),
    }
    _write_json(manifest_path, result)
    _write_json(root / "split_leakage_report.json", result)
    if not gate_pass:
        raise RuntimeError("M1-R D_core split leakage gate failed")
    return result


def state_banks(
    config: Mapping[str, Any], *, output_root: str | Path | None = None
) -> dict[str, Any]:
    root = _output_root(output_root)
    bank_root = root / "state_banks"
    manifest_path = bank_root / "state_bank_manifest.json"
    if manifest_path.is_file():
        cached = _read_json(manifest_path)
        if not cached.get("gate_pass", False):
            raise RuntimeError(f"Existing state-bank manifest is not a gate success: {manifest_path}")
        return cached
    _assert_clean_output(bank_root)
    import gymnasium as gym
    import mani_skill2.envs  # noqa: F401
    importlib.import_module("envs.maniskill_fixed")

    environment = config["environment"]
    env = gym.make(
        environment["runtime_env_id"],
        obs_mode=environment["obs_mode"],
        control_mode=environment["control_mode"],
        reward_mode=environment["reward_mode"],
        bg_name=None,
        render_mode=None,
    )
    sets: dict[str, list[dict[str, Any]]] = {}
    invalid: list[str] = []
    try:
        for set_name, spec in config["splits"]["state_banks"].items():
            rows: list[dict[str, Any]] = []
            count = int(spec["count"])
            seed_start = int(spec["seed_start"])
            with (bank_root / f"{set_name}.jsonl").open("w", encoding="utf-8") as stream:
                for index in range(count):
                    seed = seed_start + index
                    observation, _ = env.reset(seed=seed)
                    state = np.asarray(env.unwrapped.get_state(), dtype=np.float32).reshape(-1)
                    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
                    if state.shape != (70,):
                        invalid.append(f"state_shape:{set_name}:{index}:{state.shape}")
                    if obs.shape != (50,):
                        invalid.append(f"observation_shape:{set_name}:{index}:{obs.shape}")
                    if not np.isfinite(state).all() or not np.isfinite(obs).all():
                        invalid.append(f"nonfinite:{set_name}:{index}")
                    row = {
                        "index": index,
                        "declared_seed": seed,
                        "state": state.tolist(),
                        "observation": obs.tolist(),
                        "start_state_fingerprint": _fingerprint(
                            state, float(config["protocol"]["state_match_tolerance"])
                        ),
                    }
                    stream.write(json_dumps(row))
                    rows.append(row)
                    if (index + 1) % 100 == 0 or index + 1 == count:
                        print(f"[{set_name}] {index + 1}/{count}", flush=True)
            sets[set_name] = rows
    finally:
        env.close()
    pairs: dict[str, int] = {}
    names = list(sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = {row["start_state_fingerprint"] for row in sets[left]} & {
                row["start_state_fingerprint"] for row in sets[right]
            }
            pairs[f"{left}__{right}"] = len(overlap)
    within = {
        name: len({row["start_state_fingerprint"] for row in rows}) == len(rows)
        for name, rows in sets.items()
    }
    gate_pass = not invalid and all(value == 0 for value in pairs.values()) and all(within.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "state_banks",
        "gate_pass": gate_pass,
        "runtime_env_id": environment["runtime_env_id"],
        "backend": "physx_cpu",
        "identity": "start_state_fingerprint; declared seed is only a reset recipe",
        "sets": {name: len(rows) for name, rows in sets.items()},
        "within_split_unique": within,
        "cross_split_duplicate_counts": pairs,
        "state_dim": 70,
        "observation_dim": 50,
        "invalid": invalid,
    }
    _write_json(manifest_path, result)
    _write_json(root / "split_leakage_report.json", result)
    if not gate_pass:
        raise RuntimeError(f"M1-R state-bank leakage gate failed: {pairs}")
    return result


def finalize(
    config: Mapping[str, Any], *, output_root: str | Path | None = None
) -> dict[str, Any]:
    root = _output_root(output_root)
    frozen_path = root / "frozen_protocol.json"
    if frozen_path.is_file():
        return _read_json(frozen_path)
    required = {
        "source_audit": root / "source_audit.json",
        "environment_contract": root / "environment_contract.json",
        "replay_full": root / "reference_replay_full" / "replay_manifest.json",
        "trajectory_audit": root / "trajectory_audit.json",
        "d_core": root / "d_core_manifest.json",
        "state_banks": root / "state_banks" / "state_bank_manifest.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    artifacts = {name: _read_json(path) for name, path in required.items() if path.is_file()}
    gate_pass = not missing and all(bool(item.get("gate_pass")) for item in artifacts.values())
    if not gate_pass:
        _write_json(root / "finalize_failure.json", {"missing": missing, "artifacts": artifacts})
        raise RuntimeError(f"M1-R finalize gate failed; missing={missing}")
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": config["protocol_id"],
        "gate_pass": True,
        "identity": copy.deepcopy(config["identity"]),
        "environment": copy.deepcopy(config["environment"]),
        "observation_schema": copy.deepcopy(config["observation_schema"]),
        "temporal": copy.deepcopy(config["temporal"]),
        "splits": copy.deepcopy(config["splits"]),
        "protocol": copy.deepcopy(config["protocol"]),
        "artifacts": {name: _relative(path) for name, path in required.items()},
        "measured": {
            "source": artifacts["source_audit"]["summary"],
            "replay_full": artifacts["replay_full"]["summary"],
            "trajectory": artifacts["trajectory_audit"],
            "d_core": artifacts["d_core"],
            "state_banks": artifacts["state_banks"],
        },
        "frozen_from": "m1_r_protocol_audit",
        "created_at_unix": time.time(),
    }
    _write_json(frozen_path, frozen)
    _write_json(root / "comparison.json", {"schema_version": SCHEMA_VERSION, "gate_pass": True})
    return frozen


def validate_frozen(path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(_path(path))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("M1-R frozen protocol schema mismatch")
    if payload.get("protocol_id") != config["protocol_id"]:
        raise ValueError("M1-R frozen protocol id mismatch")
    if payload.get("identity", {}).get("policy_decorator_commit") != config["identity"]["policy_decorator_commit"]:
        raise ValueError("M1-R frozen commit mismatch")
    if payload.get("environment", {}).get("runtime_env_id") != config["environment"]["runtime_env_id"]:
        raise ValueError("M1-R runtime env mismatch")
    if not payload.get("gate_pass"):
        raise ValueError("M1-R frozen protocol is not a gate success")
    return {"schema_version": SCHEMA_VERSION, "protocol_id": config["protocol_id"], "gate_pass": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "validate-config",
            "source-audit",
            "environment-audit",
            "replay-smoke",
            "replay-full",
            "trajectory-audit",
            "freeze-data",
            "state-banks",
            "finalize",
            "validate-frozen",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--protocol", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config(args.config)
    if args.command == "validate-config":
        result = {"schema_version": SCHEMA_VERSION, "protocol_id": config["protocol_id"], "gate_pass": True}
    elif args.command == "source-audit":
        result = source_audit(config, output_root=args.output_root)
    elif args.command == "environment-audit":
        result = environment_audit(config, output_root=args.output_root)
    elif args.command == "replay-smoke":
        result = replay_phase(config, output_root=args.output_root, count=args.count, phase_name="replay_smoke")
    elif args.command == "replay-full":
        expected = int(config["identity"]["episode_count"])
        if args.count != expected:
            raise ValueError(f"replay-full must use count={expected}")
        result = replay_phase(config, output_root=args.output_root, count=args.count, phase_name="replay_full")
    elif args.command == "trajectory-audit":
        result = trajectory_audit(config, output_root=args.output_root)
    elif args.command == "freeze-data":
        result = freeze_data(config, output_root=args.output_root)
    elif args.command == "state-banks":
        result = state_banks(config, output_root=args.output_root)
    elif args.command == "finalize":
        result = finalize(config, output_root=args.output_root)
    elif args.command == "validate-frozen":
        if args.protocol is None:
            raise ValueError("validate-frozen requires --protocol")
        result = validate_frozen(args.protocol, config)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json_dumps(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
