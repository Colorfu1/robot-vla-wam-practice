"""Validate RGB state materialization against direct RGB trajectory capture.

This is a small final audit for the M1-W StackCube data route.  It first
captures a bounded subset of the native 7D EE trajectories with RGB enabled,
then materializes RGB again from the recorded ``env_states`` and compares the
raw camera frames.  The direct capture is an audit artifact only; it is not
merged into D_core and it does not change the formal training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .gpu_ee_replay import replay as replay_native_ee
from .gpu_state_materializer import materialize as materialize_gpu_state
from .state_identity import start_state_fingerprint

CAMERAS = ("base_camera", "hand_camera")
SCHEMA_VERSION = 1
MIN_COUNT = 20
MAX_COUNT = 50
RGB_MEAN_ABS_ERROR_MAX = 0.1
RGB_MISMATCHED_PIXEL_RATE_MAX = 0.02
RGB_P99_FRAME_MEAN_ABS_ERROR_MAX = 1.0


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON payload contains NaN or Infinity")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        from robot_practice.evaluation.common import practice_root

        return path.resolve().relative_to(practice_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _metadata_episode_ids(path: Path) -> list[int]:
    metadata = _read_json(path)
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError(f"{path} has no episodes list")
    result: list[int] = []
    for row in episodes:
        if not isinstance(row, Mapping):
            raise TypeError(f"{path} contains a non-object episode record")
        source_id = row.get("source_episode_id", row.get("episode_id"))
        result.append(int(source_id))
    if len(set(result)) != len(result):
        raise ValueError(f"{path} contains duplicate source episode IDs")
    return result


def _select_episode_ids(d_core_json: Path, count: int) -> list[int]:
    if not MIN_COUNT <= count <= MAX_COUNT:
        raise ValueError(f"count must be between {MIN_COUNT} and {MAX_COUNT}, got {count}")
    ids = _metadata_episode_ids(d_core_json)
    if len(ids) < count:
        raise ValueError(f"D_core contains {len(ids)} episodes; cannot select {count}")
    return ids[:count]


def _source_id_map(path: Path) -> dict[int, int]:
    metadata = _read_json(path.with_suffix(".json"))
    rows = metadata.get("episodes")
    if not isinstance(rows, list):
        raise TypeError(f"{path.with_suffix('.json')} has no episodes list")
    result: dict[int, int] = {}
    for row in rows:
        target_id = int(row["episode_id"])
        source_id = int(row.get("source_episode_id", target_id))
        if source_id in result:
            raise ValueError(f"Duplicate source episode ID {source_id} in {path}")
        result[source_id] = target_id
    return result


def _h5_node_equal(left: h5py.Group | h5py.Dataset, right: h5py.Group | h5py.Dataset) -> bool:
    if isinstance(left, h5py.Dataset):
        return (
            isinstance(right, h5py.Dataset)
            and left.shape == right.shape
            and np.array_equal(left[()], right[()])
        )
    if not isinstance(right, h5py.Group):
        return False
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left.keys():
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, h5py.Group):
            if not _h5_node_equal(left_value, right_value):
                return False
        elif not _h5_node_equal(left_value, right_value):
            return False
    return True


def _compare_h5(direct_h5: Path, materialized_h5: Path, selected_ids: Sequence[int]) -> dict[str, Any]:
    direct_by_source = _source_id_map(direct_h5)
    materialized_by_source = _source_id_map(materialized_h5)
    expected = {int(value) for value in selected_ids}
    if set(direct_by_source) != expected:
        raise RuntimeError(
            "Direct RGB capture source IDs differ from selection: "
            f"expected={sorted(expected)[:5]}..., got={sorted(direct_by_source)[:5]}..."
        )
    if materialized_by_source != direct_by_source:
        raise RuntimeError("Materialized RGB capture changed the direct source episode mapping")

    direct_fingerprints: dict[int, str] = {}
    materialized_fingerprints: dict[int, str] = {}
    total_frames = 0
    camera_stats: dict[str, dict[str, Any]] = {}
    proprio_stats: dict[str, Any] = {}
    payload_exact_count = 0
    with h5py.File(direct_h5, "r") as direct, h5py.File(materialized_h5, "r") as materialized:
        for source_id in sorted(expected):
            direct_group = direct[f"traj_{direct_by_source[source_id]}"]
            materialized_group = materialized[f"traj_{materialized_by_source[source_id]}"]
            direct_fingerprints[source_id] = start_state_fingerprint(direct_group)
            materialized_fingerprints[source_id] = start_state_fingerprint(materialized_group)
            if direct_fingerprints[source_id] != materialized_fingerprints[source_id]:
                raise RuntimeError(f"Start-state fingerprint changed for source episode {source_id}")
            if np.asarray(direct_group["actions"]).shape != np.asarray(materialized_group["actions"]).shape:
                raise RuntimeError(f"Action timeline changed for source episode {source_id}")
            payload_exact = all(
                _h5_node_equal(direct_group[key], materialized_group[key])
                for key in ("actions", "success", "env_states")
            )
            payload_exact_count += int(payload_exact)

            for key in ("qpos", "qvel"):
                direct_value = np.asarray(direct_group["obs"]["agent"][key])
                materialized_value = np.asarray(materialized_group["obs"]["agent"][key])
                if direct_value.shape != materialized_value.shape:
                    raise RuntimeError(f"{key} shape changed for source episode {source_id}")
                difference = np.abs(direct_value.astype(np.float64) - materialized_value.astype(np.float64))
                entry = proprio_stats.setdefault(
                    key,
                    {"elements": 0, "max_abs_error": 0.0, "mean_abs_error_sum": 0.0},
                )
                entry["elements"] += int(difference.size)
                entry["max_abs_error"] = max(
                    entry["max_abs_error"], float(difference.max(initial=0.0))
                )
                entry["mean_abs_error_sum"] += float(difference.sum())

            for camera in CAMERAS:
                direct_rgb = np.asarray(direct_group["obs"]["sensor_data"][camera]["rgb"])
                materialized_rgb = np.asarray(
                    materialized_group["obs"]["sensor_data"][camera]["rgb"]
                )
                if direct_rgb.shape != materialized_rgb.shape:
                    raise RuntimeError(
                        f"{camera} shape changed for source episode {source_id}: "
                        f"{direct_rgb.shape} vs {materialized_rgb.shape}"
                    )
                difference = np.abs(
                    direct_rgb.astype(np.int16) - materialized_rgb.astype(np.int16)
                )
                pixels = int(np.prod(direct_rgb.shape[:-1]))
                mismatched_pixels = int(np.any(difference != 0, axis=-1).sum())
                entry = camera_stats.setdefault(
                    camera,
                    {
                        "frames": 0,
                        "pixels": 0,
                        "mismatched_pixels": 0,
                        "max_abs_error": 0,
                        "mean_abs_error_sum": 0,
                        "exact_frame_count": 0,
                    },
                )
                entry["frames"] += int(direct_rgb.shape[0])
                entry["pixels"] += pixels
                entry["mismatched_pixels"] += mismatched_pixels
                entry["max_abs_error"] = max(
                    entry["max_abs_error"], int(difference.max(initial=0))
                )
                entry["mean_abs_error_sum"] += int(difference.sum())
                entry["exact_frame_count"] += int(
                    np.all(difference == 0, axis=tuple(range(1, difference.ndim)))
                    .sum()
                )
                frame_mean_error = difference.mean(axis=(1, 2, 3))
                frame_mismatch_rate = np.any(difference != 0, axis=-1).mean(axis=(1, 2))
                entry.setdefault("frame_mean_error_values", []).extend(
                    frame_mean_error.astype(np.float64).tolist()
                )
                entry.setdefault("frame_mismatch_rate_values", []).extend(
                    frame_mismatch_rate.astype(np.float64).tolist()
                )
                total_frames += int(direct_rgb.shape[0])

    for entry in camera_stats.values():
        entry["mismatched_pixel_rate"] = entry["mismatched_pixels"] / max(1, entry["pixels"])
        entry["mean_abs_error"] = entry["mean_abs_error_sum"] / max(
            1, entry["pixels"] * 3
        )
        entry["exact_frame_rate"] = entry["exact_frame_count"] / max(1, entry["frames"])
        entry["p99_frame_mean_abs_error"] = float(
            np.percentile(entry.pop("frame_mean_error_values"), 99)
        )
        entry["p99_frame_mismatched_pixel_rate"] = float(
            np.percentile(entry.pop("frame_mismatch_rate_values"), 99)
        )
        del entry["mean_abs_error_sum"]
    for entry in proprio_stats.values():
        entry["mean_abs_error"] = entry["mean_abs_error_sum"] / max(1, entry["elements"])
        del entry["mean_abs_error_sum"]

    return {
        "episode_count": len(expected),
        "total_observation_frames_per_camera": total_frames // max(1, len(CAMERAS)),
        "selected_episode_ids": sorted(expected),
        "identity": {
            "key": "start_state_fingerprint from env_states frame 0",
            "source_unique_count": len(set(direct_fingerprints.values())),
            "materialized_unique_count": len(set(materialized_fingerprints.values())),
            "exact_match": direct_fingerprints == materialized_fingerprints,
        },
        "rgb": camera_stats,
        "proprioception": proprio_stats,
        "payload_exact_episode_count": payload_exact_count,
        "tolerances": {
            "mean_abs_error_max": RGB_MEAN_ABS_ERROR_MAX,
            "mismatched_pixel_rate_max": RGB_MISMATCHED_PIXEL_RATE_MAX,
            "p99_frame_mean_abs_error_max": RGB_P99_FRAME_MEAN_ABS_ERROR_MAX,
            "comparison_is_not_bitwise_exact": True,
        },
        "gate_pass": (
            direct_fingerprints == materialized_fingerprints
            and payload_exact_count == len(expected)
            and all(
                entry["mean_abs_error"] <= RGB_MEAN_ABS_ERROR_MAX
                and entry["mismatched_pixel_rate"] <= RGB_MISMATCHED_PIXEL_RATE_MAX
                and entry["p99_frame_mean_abs_error"] <= RGB_P99_FRAME_MEAN_ABS_ERROR_MAX
                for entry in camera_stats.values()
            )
            and all(
                entry["max_abs_error"] == 0 and entry["mean_abs_error"] == 0.0
                for entry in proprio_stats.values()
            )
        ),
    }


def _resolve_manifest_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _write_freeze_attestation(
    *, d_core_json: Path, validation_manifest: Path, episode_count: int
) -> Path:
    d_core_h5 = d_core_json.with_name("trajectory.h5")
    if not d_core_h5.is_file():
        raise FileNotFoundError(f"D_core H5 is missing: {d_core_h5}")
    attestation = d_core_json.with_name("rgb_materialization_freeze.json")
    payload = {
        "schema_version": 1,
        "task": "StackCube-v1",
        "status": "frozen",
        "gate": "rgb_materialization_consistency",
        "validation_manifest": _relative(validation_manifest),
        "d_core_h5": _relative(d_core_h5),
        "d_core_json": _relative(d_core_json),
        "d_core_episode_count": int(episode_count),
        "d_core_h5_sha256": _sha256(d_core_h5),
        "d_core_json_sha256": _sha256(d_core_json),
        "note": (
            "RGB validation attests the existing D_core; it does not rewrite or append episodes. "
            "The current M1-W D_core count is the measured count in trajectory.json."
        ),
    }
    if attestation.is_file():
        existing = _read_json(attestation)
        compatible_existing = dict(existing)
        compatible_existing["gate"] = payload["gate"]
        if compatible_existing == payload:
            if existing != payload:
                _write_json(attestation, payload)
        else:
            raise RuntimeError(f"Existing freeze attestation differs: {attestation}")
    else:
        _write_json(attestation, payload)
    return attestation


def validate(
    *,
    source_h5: str | Path,
    d_core_json: str | Path,
    output_dir: str | Path,
    task_id: str = "StackCube-v1",
    count: int = MIN_COUNT,
    num_envs: int | None = None,
) -> dict[str, Any]:
    source_h5 = Path(source_h5).expanduser().resolve()
    d_core_json = Path(d_core_json).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    selected_ids = _select_episode_ids(d_core_json, int(count))
    if num_envs is None:
        num_envs = int(count)
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    manifest_path = output_dir / "validation_manifest.json"
    if manifest_path.is_file():
        result = _read_json(manifest_path)
        if not result.get("gate_pass"):
            direct_h5 = _resolve_manifest_path(result["direct_rgb_h5"])
            materialized_h5 = _resolve_manifest_path(result["materialized_rgb_h5"])
            comparison = _compare_h5(direct_h5, materialized_h5, selected_ids)
            result["comparison"] = comparison
            result["gate_pass"] = bool(comparison["gate_pass"])
            result["formal_d_core_freeze"]["eligible"] = bool(result["gate_pass"])
            _write_json(manifest_path, result)
        if result.get("gate_pass"):
            d_core_metadata = _read_json(d_core_json)
            attestation = _write_freeze_attestation(
                d_core_json=d_core_json,
                validation_manifest=manifest_path,
                episode_count=len(d_core_metadata.get("episodes", [])),
            )
            result["formal_d_core_freeze"]["attestation"] = _relative(attestation)
            _write_json(manifest_path, result)
        return result
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Incomplete RGB validation output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_dir = output_dir / "direct_rgb"
    materialized_dir = output_dir / "materialized_rgb"
    direct_manifest = replay_native_ee(
        source_h5=source_h5,
        output_dir=direct_dir,
        num_envs=int(num_envs),
        episode_ids=selected_ids,
    )
    direct_h5 = _resolve_manifest_path(direct_manifest["replay_h5"])
    materialized_manifest = materialize_gpu_state(
        source_h5=direct_h5,
        output_dir=materialized_dir,
        task_id=task_id,
        num_envs=int(num_envs),
        expected_episodes=int(count),
    )
    materialized_h5 = _resolve_manifest_path(materialized_manifest["replay_h5"])
    comparison = _compare_h5(direct_h5, materialized_h5, selected_ids)
    d_core_metadata = _read_json(d_core_json)
    d_core_episode_count = len(d_core_metadata.get("episodes", []))
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": task_id,
        "route": "direct_rgb_capture_vs_state_materialization",
        "source_h5": _relative(source_h5),
        "d_core_json": _relative(d_core_json),
        "direct_rgb_h5": _relative(direct_h5),
        "materialized_rgb_h5": _relative(materialized_h5),
        "direct_capture": {
            "episode_count": int(count),
            "episode_ids": selected_ids,
            "backend": "physx_cuda",
            "control_mode": "pd_ee_delta_pose",
            "obs_mode": "rgb",
            "env_states_saved": True,
            "action_replay_used": True,
            "success_at_end": direct_manifest.get("successful_at_end"),
            "success_at_end_rate": direct_manifest.get("success_at_end_rate"),
        },
        "materialization": {
            "backend": materialized_manifest.get("backend"),
            "env_step_called": materialized_manifest.get("env_step_called"),
            "action_replay_used": materialized_manifest.get("action_replay_used"),
        },
        "comparison": comparison,
        "formal_d_core_freeze": {
            "eligible": bool(comparison["gate_pass"]),
            "d_core_episode_count": d_core_episode_count,
            "d_core_h5": _relative(d_core_json.with_name("trajectory.h5")),
            "note": (
                "RGB validation does not rewrite D_core; it only gates the existing artifact. "
                "RGB comparison uses explicit render-tolerance thresholds, while env_states, "
                "actions, success and proprioception must match exactly."
            ),
        },
        "gate_pass": bool(comparison["gate_pass"]),
    }
    _write_json(manifest_path, result)
    if not result["gate_pass"]:
        raise RuntimeError(f"RGB materialization validation failed; inspect {manifest_path}")
    attestation = _write_freeze_attestation(
        d_core_json=d_core_json,
        validation_manifest=manifest_path,
        episode_count=d_core_episode_count,
    )
    result["formal_d_core_freeze"]["attestation"] = _relative(attestation)
    _write_json(manifest_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--d-core-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="StackCube-v1")
    parser.add_argument("--count", type=int, default=MIN_COUNT)
    parser.add_argument("--num-envs", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate(
        source_h5=args.source_h5,
        d_core_json=args.d_core_json,
        output_dir=args.output_dir,
        task_id=args.task,
        count=args.count,
        num_envs=args.num_envs,
    )
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
