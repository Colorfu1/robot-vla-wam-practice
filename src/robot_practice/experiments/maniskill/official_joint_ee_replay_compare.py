"""Compare official CPU joint-to-EE conversion with CPU/GPU replay.

This is a deliberately narrow diagnostic for StackCube.  The first stage uses
ManiSkill's installed ``replay_trajectory`` implementation to convert the
official ``pd_joint_pos`` demonstrations to native ``pd_ee_delta_pose``
actions on CPU.  The second and third stages replay that *same converted H5*
without a controller conversion, once on CPU and once on GPU.

The three stages are kept in separate directories and each writes a manifest
with episode counts, action dimensions, success-once and success-at-end.  This
prevents source success labels, converter execution success, and downstream
replay success from being conflated.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.evaluation.common import practice_root

from .state_identity import invert_unique_fingerprints, start_state_fingerprint

TASK_ID = "StackCube-v1"
SOURCE_CONTROL_MODE = "pd_joint_pos"
TARGET_CONTROL_MODE = "pd_ee_delta_pose"
TARGET_ACTION_DIM = 7
SCHEMA_VERSION = 1


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            temporary.replace(path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(temporary, path)
            temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(practice_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _copy(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def _ensure_source_pair(source_h5: Path) -> tuple[Path, Path]:
    source_h5 = source_h5.expanduser().resolve()
    source_json = source_h5.with_suffix(".json")
    if not source_h5.is_file() or not source_json.is_file():
        raise FileNotFoundError(f"Missing source H5/JSON pair: {source_h5}")
    metadata = _read_json(source_json)
    env_info = metadata.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {}) if isinstance(env_info, Mapping) else {}
    if env_info.get("env_id") != TASK_ID:
        raise ValueError(f"Expected {TASK_ID}, got {env_info.get('env_id')!r}")
    if env_kwargs.get("control_mode") != SOURCE_CONTROL_MODE:
        raise ValueError(
            f"Expected source control mode {SOURCE_CONTROL_MODE}, got {env_kwargs.get('control_mode')!r}"
        )
    episodes = metadata.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Source has no episodes: {source_json}")
    return source_h5, source_json


def _stage_complete(stage_dir: Path, manifest_name: str) -> dict[str, Any] | None:
    manifest_path = stage_dir / manifest_name
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    artifact = manifest.get("artifact_h5")
    if not isinstance(artifact, str) or not Path(artifact).is_file():
        raise RuntimeError(f"Stage manifest points to a missing artifact: {manifest_path}")
    if manifest.get("gate_pass") is not True:
        raise RuntimeError(f"Existing stage failed its gate; move it aside before rerun: {stage_dir}")
    return manifest


def _prepare_stage(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    if any(stage_dir.iterdir()):
        raise FileExistsError(
            f"Incomplete stage exists; inspect or move it aside before rerun: {stage_dir}"
        )


def _run_official(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONNOUSERSITE", "1")
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        subprocess.run(
            command,
            cwd=practice_root(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _generated_h5(work_dir: Path, input_h5: Path) -> Path:
    candidates = sorted(
        path
        for path in work_dir.glob("*.h5")
        if path.resolve() != input_h5.resolve() and not path.name.endswith(".tmp")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one generated H5 in {work_dir}, found {candidates}")
    generated_json = candidates[0].with_suffix(".json")
    if not generated_json.is_file():
        raise RuntimeError(f"Generated H5 has no adjacent JSON: {candidates[0]}")
    return candidates[0]


def _episode_summary(artifact_h5: Path, expected: int) -> dict[str, Any]:
    metadata = _read_json(artifact_h5.with_suffix(".json"))
    records = metadata.get("episodes", [])
    group_names: list[str] = []
    end_success = 0
    success_once = 0
    action_dims: set[int] = set()
    fingerprints: dict[int, str] = {}
    with h5py.File(artifact_h5, "r") as handle:
        for name in sorted(handle.keys()):
            if not name.startswith("traj_"):
                continue
            group_names.append(name)
            group = handle[name]
            actions = np.asarray(group["actions"])
            if actions.ndim != 2:
                raise ValueError(f"{artifact_h5}:{name} actions are not rank-2: {actions.shape}")
            action_dims.add(int(actions.shape[1]))
            success = np.asarray(group["success"], dtype=bool).reshape(-1)
            if success.size == 0:
                raise ValueError(f"{artifact_h5}:{name} has an empty success series")
            success_once += int(bool(np.any(success)))
            end_success += int(bool(success[-1]))
            episode_id = int(name.removeprefix("traj_"))
            fingerprints[episode_id] = start_state_fingerprint(group, tolerance=1e-4)
    fingerprint_index = invert_unique_fingerprints(fingerprints, label=str(artifact_h5))
    episode_count = len(group_names)
    metadata_count = len(records) if isinstance(records, list) else -1
    if episode_count != metadata_count:
        raise RuntimeError(
            f"H5/JSON episode mismatch for {artifact_h5}: {episode_count} vs {metadata_count}"
        )
    return {
        "artifact_h5": _relative(artifact_h5),
        "artifact_json": _relative(artifact_h5.with_suffix(".json")),
        "episode_count": episode_count,
        "expected_episodes": expected,
        "successful_once": success_once,
        "success_once_rate": success_once / max(1, episode_count),
        "successful_at_end": end_success,
        "success_at_end_rate": end_success / max(1, episode_count),
        "action_dims": sorted(action_dims),
        "unique_start_fingerprints": len(fingerprint_index),
    }


def _write_stage_manifest(stage_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    _write_json(stage_dir / "manifest.json", manifest)
    return manifest


def convert_official_cpu(
    *, source_h5: Path, stage_dir: Path, expected: int, num_envs: int
) -> dict[str, Any]:
    cached = _stage_complete(stage_dir, "manifest.json")
    if cached is not None:
        return cached
    _prepare_stage(stage_dir)
    work_dir = stage_dir / "official_work"
    work_dir.mkdir(parents=True)
    input_h5 = work_dir / "trajectory.h5"
    _copy(source_h5, input_h5)
    _copy(source_h5.with_suffix(".json"), input_h5.with_suffix(".json"))
    command = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(input_h5),
        "--use-first-env-state",
        "--target-control-mode",
        TARGET_CONTROL_MODE,
        "--obs-mode",
        "none",
        "--sim-backend",
        "physx_cpu",
        "--reward-mode",
        "sparse",
        "--save-traj",
        "--record-rewards",
        "--allow-failure",
        "--max-retry",
        "0",
        "--num-envs",
        str(num_envs),
    ]
    log_path = stage_dir / "official_conversion.log"
    _run_official(command, log_path)
    generated_h5 = _generated_h5(work_dir, input_h5)
    artifact_h5 = stage_dir / "converted_ee.h5"
    artifact_json = artifact_h5.with_suffix(".json")
    _copy(generated_h5, artifact_h5)
    _copy(generated_h5.with_suffix(".json"), artifact_json)
    summary = _episode_summary(artifact_h5, expected)
    metadata = _read_json(artifact_json)
    env_kwargs = metadata.get("env_info", {}).get("env_kwargs", {})
    gate_pass = (
        summary["episode_count"] == expected
        and summary["action_dims"] == [TARGET_ACTION_DIM]
        and env_kwargs.get("control_mode") == TARGET_CONTROL_MODE
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "stage": "official_cpu_joint_to_ee_conversion",
        "source_h5": _relative(source_h5),
        "source_control_mode": SOURCE_CONTROL_MODE,
        "target_control_mode": TARGET_CONTROL_MODE,
        "conversion_backend": "physx_cpu",
        "conversion_num_envs": num_envs,
        "command": command,
        "log": _relative(log_path),
        **summary,
        "gate_pass": gate_pass,
    }
    _write_stage_manifest(stage_dir, manifest)
    if not gate_pass:
        raise RuntimeError(f"Official CPU conversion gate failed; inspect {stage_dir}")
    return manifest


def replay_converted(
    *,
    converted_h5: Path,
    stage_dir: Path,
    expected: int,
    backend: str,
    num_envs: int,
) -> dict[str, Any]:
    cached = _stage_complete(stage_dir, "manifest.json")
    if cached is not None:
        return cached
    _prepare_stage(stage_dir)
    work_dir = stage_dir / "official_work"
    work_dir.mkdir(parents=True)
    input_h5 = work_dir / "converted_ee.h5"
    _copy(converted_h5, input_h5)
    _copy(converted_h5.with_suffix(".json"), input_h5.with_suffix(".json"))
    command = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(input_h5),
        "--use-first-env-state",
        "--obs-mode",
        "none",
        "--sim-backend",
        backend,
        "--reward-mode",
        "sparse",
        "--save-traj",
        "--record-rewards",
        "--allow-failure",
        "--max-retry",
        "0",
        "--num-envs",
        str(num_envs),
    ]
    log_path = stage_dir / "replay.log"
    _run_official(command, log_path)
    generated_h5 = _generated_h5(work_dir, input_h5)
    artifact_h5 = stage_dir / "replay.h5"
    _copy(generated_h5, artifact_h5)
    _copy(generated_h5.with_suffix(".json"), artifact_h5.with_suffix(".json"))
    summary = _episode_summary(artifact_h5, expected)
    gate_pass = summary["episode_count"] == expected and summary["action_dims"] == [TARGET_ACTION_DIM]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "stage": f"official_{backend}_replay_of_converted_ee",
        "source_h5": _relative(converted_h5),
        "backend": backend,
        "num_envs": num_envs,
        "controller_conversion_used": False,
        "same_converted_ee_actions": True,
        "command": command,
        "log": _relative(log_path),
        **summary,
        "gate_pass": gate_pass,
    }
    _write_stage_manifest(stage_dir, manifest)
    if not gate_pass:
        raise RuntimeError(f"{backend} replay gate failed; inspect {stage_dir}")
    return manifest


def run_compare(
    *,
    source_h5: str | Path,
    output_root: str | Path,
    conversion_num_envs: int = 10,
    cpu_num_envs: int = 10,
    gpu_num_envs: int = 100,
) -> dict[str, Any]:
    source_h5, _ = _ensure_source_pair(_rooted(source_h5))
    metadata = _read_json(source_h5.with_suffix(".json"))
    expected = len(metadata["episodes"])
    if expected != 1000:
        raise ValueError(f"This comparison is fixed to the 1000-demo StackCube source, got {expected}")
    if min(conversion_num_envs, cpu_num_envs, gpu_num_envs) <= 0:
        raise ValueError("all num_envs values must be positive")
    output_root = _rooted(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conversion = convert_official_cpu(
        source_h5=source_h5,
        stage_dir=output_root / "01_official_cpu_conversion",
        expected=expected,
        num_envs=conversion_num_envs,
    )
    converted_h5 = _rooted(conversion["artifact_h5"])
    cpu = replay_converted(
        converted_h5=converted_h5,
        stage_dir=output_root / "02_cpu_replay",
        expected=expected,
        backend="physx_cpu",
        num_envs=cpu_num_envs,
    )
    gpu = replay_converted(
        converted_h5=converted_h5,
        stage_dir=output_root / "03_gpu_replay",
        expected=expected,
        backend="physx_cuda",
        num_envs=gpu_num_envs,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "source_h5": _relative(source_h5),
        "converted_ee_h5": _relative(converted_h5),
        "stages": {
            "official_cpu_conversion": conversion,
            "cpu_replay_same_ee_actions": cpu,
            "gpu_replay_same_ee_actions": gpu,
        },
        "comparison": {
            "source_episode_count": expected,
            "conversion_episode_count": conversion["episode_count"],
            "cpu_replay_successful_at_end": cpu["successful_at_end"],
            "gpu_replay_successful_at_end": gpu["successful_at_end"],
            "cpu_minus_gpu_successes": cpu["successful_at_end"] - gpu["successful_at_end"],
        },
        "gate_pass": all(
            stage.get("gate_pass") is True for stage in (conversion, cpu, gpu)
        ),
    }
    _write_json(output_root / "comparison_manifest.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-h5",
        type=Path,
        default=Path("datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/maniskill/stackcube_joint_to_ee_cpu_gpu_replay"),
    )
    parser.add_argument("--conversion-num-envs", type=int, default=10)
    parser.add_argument("--cpu-num-envs", type=int, default=10)
    parser.add_argument("--gpu-num-envs", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_compare(
        source_h5=args.source_h5,
        output_root=args.output_root,
        conversion_num_envs=args.conversion_num_envs,
        cpu_num_envs=args.cpu_num_envs,
        gpu_num_envs=args.gpu_num_envs,
    )
    print(
        json.dumps(
            {
                "gate_pass": summary["gate_pass"],
                "comparison": summary["comparison"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
