"""Audit and convert the official PickCube motion-planning replay to 4D LeRobot data.

The replay itself is intentionally not performed here.  The remote pipeline runs
ManiSkill's official ``replay_trajectory`` command first, then this module audits
the resulting HDF5 and creates one immutable LeRobotDataset root.  The module is
specific to the paper-aligned ``pd_ee_delta_pos``/CPU contract so that an RL or
GPU replay cannot be silently substituted.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

from .audit_demos import ACTION_NAMES
from .convert_demos import (
    ACTION_KEY,
    CAMERA_KEY,
    EXCLUDED_STATE_FIELDS,
    STATE_FIELDS,
    STATE_KEY,
    convert_replay,
)

TASK_ID = "PickCube-v1"
CONTROL_MODE = "pd_ee_delta_pos"
SIM_BACKEND = "physx_cpu"
IMAGE_SHAPE = (128, 128, 3)
ACTION_DIM = 4
AUDIT_SCHEMA_VERSION = 1


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _metadata_path(h5_path: Path) -> Path:
    path = h5_path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"Replay metadata is missing: {path}")
    return path


def _episode_id(name: str) -> int:
    if not name.startswith("traj_"):
        raise ValueError(f"Unexpected HDF5 group {name!r}; expected traj_<id>")
    return int(name.removeprefix("traj_"))


def _records(metadata: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = metadata.get("episodes")
    if not isinstance(raw, list):
        raise TypeError("Trajectory metadata episodes must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("Each trajectory metadata entry must be an object")
        episode_id = int(item["episode_id"])
        if episode_id in result:
            raise ValueError(f"Duplicate episode_id={episode_id}")
        result[episode_id] = item
    return result


def _record_seed(record: Mapping[str, Any]) -> int | None:
    value = record.get("episode_seed")
    if value is None:
        reset_kwargs = record.get("reset_kwargs", {})
        value = reset_kwargs.get("seed") if isinstance(reset_kwargs, Mapping) else None
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one seed per CPU replay episode, got {value!r}")
        value = value[0]
    return None if value is None else int(value)


def _replay_to_source_episode_ids(
    replay_records: Mapping[int, Mapping[str, Any]],
    source_records: Mapping[int, Mapping[str, Any]] | None,
) -> dict[int, int]:
    """Recover source ids after replay_trajectory's multi-process renumbering."""

    if source_records is None:
        return {int(episode_id): int(episode_id) for episode_id in replay_records}

    source_by_seed: dict[int, int] = {}
    for source_episode_id, record in source_records.items():
        seed = _record_seed(record)
        if seed is None:
            raise ValueError(
                f"Source episode {source_episode_id} has no seed; cannot undo replay renumbering"
            )
        if seed in source_by_seed:
            raise ValueError(f"Source metadata has duplicate episode seed {seed}")
        source_by_seed[seed] = int(source_episode_id)

    result: dict[int, int] = {}
    used_source_ids: set[int] = set()
    for replay_episode_id, record in replay_records.items():
        seed = _record_seed(record)
        if seed is None or seed not in source_by_seed:
            raise ValueError(
                f"Replay episode {replay_episode_id} has seed {seed!r} absent from source metadata"
            )
        source_episode_id = source_by_seed[seed]
        if source_episode_id in used_source_ids:
            raise ValueError(
                f"Replay metadata maps multiple episodes to source episode {source_episode_id}"
            )
        result[int(replay_episode_id)] = source_episode_id
        used_source_ids.add(source_episode_id)
    return result


def _native_success(episode: h5py.Group, record: Mapping[str, Any]) -> np.ndarray:
    if "success" in episode:
        values = np.asarray(episode["success"], dtype=bool).reshape(-1)
    else:
        values = np.full(int(episode["actions"].shape[0]), bool(record.get("success", False)))
    if values.size != int(episode["actions"].shape[0]):
        raise ValueError("success and actions have different lengths")
    return values


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {key: None for key in ("mean", "p50", "p90", "p95", "max")}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _channel_stats(actions: np.ndarray) -> dict[str, Any]:
    names = ACTION_NAMES[ACTION_DIM]
    absolute = np.abs(actions.astype(np.float64, copy=False))
    channels = {}
    for index, name in enumerate(names):
        channels[name] = {
            **_quantiles(actions[:, index]),
            "abs_p90": float(np.percentile(absolute[:, index], 90)),
            "abs_p99": float(np.percentile(absolute[:, index], 99)),
            "out_of_bounds_count": int(np.sum(absolute[:, index] > 1.0)),
        }
    out_of_bounds_count = int(np.sum(absolute > 1.0))
    return {
        "action_dim": ACTION_DIM,
        "channel_order": list(names),
        "channels": channels,
        "out_of_bounds_count": out_of_bounds_count,
        "out_of_bounds_fraction": float(out_of_bounds_count / absolute.size),
        "absolute_quantiles": {
            "p50": float(np.percentile(absolute, 50)),
            "p90": float(np.percentile(absolute, 90)),
            "p99": float(np.percentile(absolute, 99)),
            "max": float(np.max(absolute)),
        },
    }


def audit_replay(
    replay_h5: Path,
    *,
    source_h5: Path | None = None,
    task: str = TASK_ID,
) -> dict[str, Any]:
    """Audit replay survival, trajectory timing, observations, and 4D actions."""

    replay_h5 = replay_h5.expanduser().resolve()
    if not replay_h5.is_file():
        raise FileNotFoundError(f"Replay HDF5 does not exist: {replay_h5}")
    replay_metadata = _read_json(_metadata_path(replay_h5))
    env_info = replay_metadata.get("env_info")
    if not isinstance(env_info, Mapping) or env_info.get("env_id") != task:
        raise ValueError(f"Replay metadata must describe {task}")
    env_kwargs = env_info.get("env_kwargs")
    if not isinstance(env_kwargs, Mapping):
        raise TypeError("Replay metadata has no env_kwargs")
    for key, expected in (("control_mode", CONTROL_MODE), ("sim_backend", SIM_BACKEND)):
        actual = env_kwargs.get(key)
        if actual != expected:
            raise ValueError(f"Replay {key} must be {expected!r}, got {actual!r}")
    if env_kwargs.get("obs_mode") not in ("rgb", "rgbd"):
        raise ValueError(
            f"Replay obs_mode must contain RGB observations, got {env_kwargs.get('obs_mode')!r}"
        )

    replay_records = _records(replay_metadata)
    source_records: dict[int, Mapping[str, Any]] | None = None
    if source_h5 is not None:
        source_metadata = _read_json(_metadata_path(source_h5))
        source_records = _records(source_metadata)
    replay_to_source = _replay_to_source_episode_ids(replay_records, source_records)

    episode_rows: list[dict[str, Any]] = []
    all_actions: list[np.ndarray] = []
    rgb_shapes: set[tuple[int, ...]] = set()
    state_shapes: set[tuple[int, ...]] = set()
    extra_fields: set[str] = set()
    with h5py.File(replay_h5, "r") as handle:
        for name in sorted(handle, key=_episode_id):
            episode_id = _episode_id(name)
            if episode_id not in replay_records:
                raise ValueError(f"Replay HDF5 has no metadata for {name}")
            episode = handle[name]
            source_episode_id = replay_to_source[episode_id]
            actions = np.asarray(episode["actions"], dtype=np.float64)
            if (
                actions.ndim != 2
                or actions.shape[1] != ACTION_DIM
                or not np.isfinite(actions).all()
            ):
                raise ValueError(f"{name} actions must be finite [T,4], got {actions.shape}")
            obs = episode.get("obs")
            if not isinstance(obs, h5py.Group):
                raise TypeError(f"{name} has no obs group")
            rgb = obs["sensor_data"]["base_camera"]["rgb"]
            qpos = obs["agent"]["qpos"]
            qvel = obs["agent"]["qvel"]
            goal_pos = obs["extra"]["goal_pos"]
            if int(rgb.shape[0]) != actions.shape[0] + 1:
                raise ValueError(f"{name} RGB timeline must have T+1 frames, got {rgb.shape}")
            if tuple(int(value) for value in rgb.shape[1:]) != IMAGE_SHAPE:
                raise ValueError(f"{name} RGB shape must be [T+1,128,128,3], got {rgb.shape}")
            for label, value, width in (
                ("qpos", qpos, 9),
                ("qvel", qvel, 9),
                ("goal_pos", goal_pos, 3),
            ):
                if int(value.shape[0]) != actions.shape[0] + 1 or int(value.shape[-1]) != width:
                    raise ValueError(f"{name} {label} has incompatible shape {value.shape}")
            rgb_shapes.add(tuple(int(value) for value in rgb.shape[1:]))
            state_shapes.add((21,))
            extra_group = obs.get("extra")
            if isinstance(extra_group, h5py.Group):
                extra_fields.update(str(key) for key in extra_group)
            success = _native_success(episode, replay_records[episode_id])
            success_indices = np.flatnonzero(success)
            first_success = None if success_indices.size == 0 else int(success_indices[0]) + 1
            all_actions.append(actions)
            episode_rows.append(
                {
                    "episode_id": source_episode_id,
                    "replay_episode_id": episode_id,
                    "seed": _record_seed(replay_records[episode_id]),
                    "num_action_steps": int(actions.shape[0]),
                    "success_once": bool(np.any(success)),
                    "success_at_end": bool(success[-1]) if success.size else False,
                    "first_success_step": first_success,
                    "pre_success_frame_count": 0 if first_success is None else first_success - 1,
                    "success_tail_frame_count": (
                        0 if first_success is None else int(actions.shape[0]) - first_success + 1
                    ),
                }
            )

    if not all_actions:
        raise ValueError("Replay HDF5 contains no trajectory groups")
    actions = np.concatenate(all_actions, axis=0)
    source_count = len(source_records) if source_records is not None else len(replay_records)
    replayed_count = len(episode_rows)
    episode_rows.sort(key=lambda row: int(row["episode_id"]))
    total_frames = int(actions.shape[0])
    first_success_values = [
        float(row["first_success_step"])
        for row in episode_rows
        if row["first_success_step"] is not None
    ]
    pre_success_frames = sum(int(row["pre_success_frame_count"]) for row in episode_rows)
    survival_rate = replayed_count / source_count if source_count else None
    action_stats = _channel_stats(actions)
    summary = {
        "source_episode_count": source_count,
        "replayed_episode_count": replayed_count,
        "replay_survival_rate": survival_rate,
        "missing_episode_ids": (
            []
            if source_records is None
            else sorted(set(source_records) - {int(row["episode_id"]) for row in episode_rows})
        ),
        "total_frames": total_frames,
        "mean_episode_frames": total_frames / replayed_count if replayed_count else None,
        "replayed_source_episode_ids": [int(row["episode_id"]) for row in episode_rows],
        "success_once_count": sum(bool(row["success_once"]) for row in episode_rows),
        "success_at_end_count": sum(bool(row["success_at_end"]) for row in episode_rows),
        "first_success_step": _quantiles(first_success_values),
        "pre_success_frame_count": pre_success_frames,
        "pre_success_frame_share": pre_success_frames / total_frames if total_frames else None,
        "episode_frame_count": _quantiles([float(row["num_action_steps"]) for row in episode_rows]),
    }
    gates = {
        "replay_survival_rate": {
            "threshold": 0.95,
            "observed": survival_rate,
            "passed": survival_rate is not None and survival_rate >= 0.95,
        },
        "action_out_of_bounds_fraction": {
            "threshold": 0.0,
            "observed": action_stats["out_of_bounds_fraction"],
            "passed": action_stats["out_of_bounds_fraction"] == 0.0,
        },
        "pre_success_frame_share": {
            "threshold": 0.85,
            "observed": summary["pre_success_frame_share"],
            "passed": summary["pre_success_frame_share"] is not None
            and summary["pre_success_frame_share"] >= 0.85,
        },
        "first_success_p90": {
            "threshold": 90.0,
            "observed": summary["first_success_step"]["p90"],
            "passed": summary["first_success_step"]["p90"] is not None
            and summary["first_success_step"]["p90"] <= 90.0,
        },
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": iso_timestamp(),
        "task": task,
        "replay_h5": _repo_relative(replay_h5),
        "replay_metadata": _repo_relative(_metadata_path(replay_h5)),
        "source_h5": None if source_h5 is None else _repo_relative(source_h5),
        "contract": {
            "control_mode": CONTROL_MODE,
            "sim_backend": SIM_BACKEND,
            "obs_mode": "rgb",
            "action_dim": ACTION_DIM,
            "image_shape": list(IMAGE_SHAPE),
            "state_shape": [21],
            "state_fields": list(STATE_FIELDS),
            "excluded_state_fields": list(EXCLUDED_STATE_FIELDS),
        },
        "summary": summary,
        "actions": action_stats,
        "observations": {
            "rgb_shapes": [list(shape) for shape in sorted(rgb_shapes)],
            "state_shapes": [list(shape) for shape in sorted(state_shapes)],
            "extra_fields": sorted(extra_fields),
            "required_extra_fields": ["goal_pos"],
        },
        "episodes": episode_rows,
        "episode_id_mapping": [
            {
                "replay_episode_id": int(replay_episode_id),
                "source_episode_id": int(source_episode_id),
            }
            for replay_episode_id, source_episode_id in sorted(replay_to_source.items())
        ],
        "gates": gates,
        "gate_pass": all(bool(value["passed"]) for value in gates.values()),
    }


def _validate_dataset(
    root: Path,
    *,
    expected_episodes: int,
    expected_repo_id: str,
    expected_source_episode_ids: list[int],
) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    manifest_path = root / "conversion_manifest.json"
    if not info_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset is incomplete: {root}")
    info = _read_json(info_path)
    manifest = _read_json(manifest_path)
    if manifest.get("repo_id") != expected_repo_id:
        raise ValueError(
            f"Dataset repo_id is {manifest.get('repo_id')!r}, expected {expected_repo_id!r}"
        )
    source = manifest.get("source", {})
    env_kwargs = source.get("env_kwargs", {}) if isinstance(source, Mapping) else {}
    if env_kwargs.get("control_mode") != CONTROL_MODE:
        raise ValueError("Dataset manifest does not describe pd_ee_delta_pos data")
    if env_kwargs.get("sim_backend") != SIM_BACKEND:
        raise ValueError("Dataset manifest does not describe physx_cpu data")
    conversion = manifest.get("conversion", {})
    if conversion.get("action_dim") != ACTION_DIM or conversion.get("episode_filter") != "all":
        raise ValueError("Dataset manifest does not describe the paper-aligned 4D contract")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise TypeError("Dataset manifest has no output object")
    episode_map = output.get("episode_map")
    if not isinstance(episode_map, list):
        raise TypeError("Dataset manifest has no output episode map")
    if not all(isinstance(item, Mapping) for item in episode_map):
        raise TypeError("Dataset manifest episode map must contain objects")
    source_episode_ids = [int(item["source_episode_id"]) for item in episode_map]
    if source_episode_ids != expected_source_episode_ids:
        raise ValueError(
            "Dataset episode order does not match source episode_id order: "
            f"got {source_episode_ids[:5]}, expected {expected_source_episode_ids[:5]}"
        )
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise TypeError(f"Dataset {root} has no features")
    expected = {
        CAMERA_KEY: ("video", [128, 128, 3]),
        STATE_KEY: ("float32", [21]),
        ACTION_KEY: ("float32", [4]),
    }
    for key, (dtype, shape) in expected.items():
        feature = features.get(key)
        if not isinstance(feature, Mapping) or feature.get("dtype") != dtype:
            raise ValueError(f"Dataset feature {key} does not have dtype {dtype}")
        if list(feature.get("shape", [])) != shape:
            raise ValueError(f"Dataset feature {key} must have shape {shape}")
    actual_episodes = int(info.get("total_episodes", -1))
    if actual_episodes != expected_episodes:
        raise ValueError(
            f"Dataset episode count is {actual_episodes}, expected {expected_episodes}"
        )
    return {
        "root": _repo_relative(root),
        "info": _repo_relative(info_path),
        "conversion_manifest": _repo_relative(manifest_path),
        "episodes": actual_episodes,
        "frames": int(info.get("total_frames", -1)),
        "source_episode_ids": source_episode_ids,
        "features": {key: dict(features[key]) for key in expected},
    }


def convert_mp_4d(
    *,
    replay_h5: Path,
    source_h5: Path,
    output_dir: Path,
    repo_id: str,
    audit_output: Path,
    action_stats_output: Path | None = None,
    survival_report: Path | None = None,
) -> dict[str, Any]:
    """Audit and create/reuse the one official 4D dataset root."""

    output_dir = output_dir.expanduser().resolve()
    audit_output = audit_output.expanduser().resolve()
    source_h5 = source_h5.expanduser().resolve()
    audit = audit_replay(replay_h5, source_h5=source_h5)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(audit_output, audit)
    if not bool(audit["gate_pass"]):
        failed = [name for name, gate in audit["gates"].items() if not gate["passed"]]
        raise RuntimeError(f"Paper-aligned 4D conversion gates failed: {', '.join(failed)}")

    expected_episodes = int(audit["summary"]["replayed_episode_count"])
    ordered_audit_episodes = sorted(audit["episodes"], key=lambda item: int(item["episode_id"]))
    expected_source_episode_ids = [int(item["episode_id"]) for item in ordered_audit_episodes]
    episode_order = [int(item["replay_episode_id"]) for item in ordered_audit_episodes]
    episode_id_map = {
        int(item["replay_episode_id"]): int(item["episode_id"]) for item in ordered_audit_episodes
    }
    if output_dir.is_dir():
        dataset = _validate_dataset(
            output_dir,
            expected_episodes=expected_episodes,
            expected_repo_id=repo_id,
            expected_source_episode_ids=expected_source_episode_ids,
        )
    else:
        result = convert_replay(
            replay_h5=replay_h5,
            source_h5=source_h5,
            output_dir=output_dir,
            repo_id=repo_id,
            task=TASK_ID,
            fps=20,
            include_failed=True,
            survival_report=survival_report,
            action_stats_output=action_stats_output,
            episode_order=episode_order,
            episode_id_map=episode_id_map,
        )
        dataset = _validate_dataset(
            output_dir,
            expected_episodes=expected_episodes,
            expected_repo_id=repo_id,
            expected_source_episode_ids=expected_source_episode_ids,
        )
        dataset["converter_result"] = result

    if action_stats_output is not None and not action_stats_output.is_file():
        action_stats_output = action_stats_output.expanduser().resolve()
        action_stats_output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(action_stats_output, audit["actions"])
    if survival_report is not None and not survival_report.is_file():
        survival_report = survival_report.expanduser().resolve()
        survival_report.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            survival_report,
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "task": TASK_ID,
                "replays": {CONTROL_MODE: audit["summary"]},
                "updated_at": iso_timestamp(),
            },
        )

    merged = dict(audit)
    merged["dataset"] = dataset
    merged["created_at"] = iso_timestamp()
    write_json_atomic(audit_output, merged)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-h5", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--action-stats-output", type=Path, default=None)
    parser.add_argument("--survival-report", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = convert_mp_4d(
        replay_h5=args.replay_h5,
        source_h5=args.source_h5,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        audit_output=args.audit_output,
        action_stats_output=args.action_stats_output,
        survival_report=args.survival_report,
    )
    print(
        json.dumps(
            {
                "audit": _repo_relative(args.audit_output),
                "gate_pass": result["gate_pass"],
                "episodes": result["summary"]["replayed_episode_count"],
                "frames": result["summary"]["total_frames"],
                "dataset": result.get("dataset", {}).get("root"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
