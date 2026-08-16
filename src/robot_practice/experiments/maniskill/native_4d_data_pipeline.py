"""Build a native 4D StackCube dataset from the original joint demonstrations.

This is an independent data route for the M1-W comparison.  It never projects
the frozen 7D M1-W labels.  Instead it runs ManiSkill's official CPU
``pd_joint_pos -> pd_ee_delta_pos`` conversion, replays those native 4D
actions on GPU, repairs only GPU failures with a position-only online EE
retarget, and finally captures RGB/action/env-state in the same GPU
``env.step`` calls.

The phases are intentionally resumable:

``convert`` -> ``gpu-replay`` -> ``gpu-repair`` -> ``merge`` -> ``capture``.

No phase overwrites a successful artifact.  The module does not create an
environment at import time; the caller is responsible for running the actual
replay phases on the remote machine.
"""

from __future__ import annotations

import argparse
import copy
import errno
import json
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

from .capture_gpu_ee_live_rgb import capture, freeze_layout
from .gpu_ee_repair import repair
from .gpu_ee_replay import replay
from .state_identity import invert_unique_fingerprints, start_state_fingerprint

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pos"
ACTION_DIM = 4
SCHEMA_VERSION = 1


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(practice_root()).as_posix()
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
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
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
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _episode_id(name: str) -> int:
    if not name.startswith("traj_"):
        raise ValueError(f"Unexpected trajectory group name: {name}")
    return int(name.removeprefix("traj_"))


def _success_at_end(group: h5py.Group) -> bool:
    values = np.asarray(group["success"], dtype=bool).reshape(-1)
    return bool(values.size and values[-1])


def load_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(_rooted(path))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported native 4D config schema: {config.get('schema_version')!r}")
    if config.get("task") != TASK_ID:
        raise ValueError(f"This route only supports {TASK_ID}")
    if config.get("control_mode") != CONTROL_MODE or int(config.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError("Native 4D route must use pd_ee_delta_pos with action_dim=4")
    return config


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = _rooted(str(config["output_root"]))
    return {
        "root": root,
        "convert": root / "01_convert_native_4d",
        "baseline": root / "02_gpu_baseline",
        "repair": root / "03_gpu_repair",
        "merge": root / "04_accepted_merge",
        "layout": root / "05_frozen_layout" / "replay_layout.json",
        "capture": root / "06_final_live_rgb",
    }


def _source_h5(config: Mapping[str, Any]) -> Path:
    path = _rooted(str(config["source_h5"]))
    if not path.is_file() or not _metadata_path(path).is_file():
        raise FileNotFoundError(f"Missing joint source H5/JSON pair: {path}")
    return path


def _prepare_input_metadata(source_json: Path, output_json: Path) -> None:
    from .m1_protocol_audit import _prepare_replay_metadata

    _prepare_replay_metadata(
        source_json,
        output_json,
        reward_mode="sparse",
        obs_mode="none",
    )


def _source_episode_count(path: Path) -> int:
    metadata = _read_json(_metadata_path(path))
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Source metadata has no episodes: {_metadata_path(path)}")
    return len(episodes)


def convert_native_4d(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the official CPU joint-to-native-4D conversion once."""

    paths = _paths(config)
    phase_root = paths["convert"]
    output_h5 = phase_root / "replay.h5"
    output_json = _metadata_path(output_h5)
    manifest_path = phase_root / "conversion_manifest.json"
    if manifest_path.is_file() and output_h5.is_file() and output_json.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("gate_pass"):
            return manifest
        raise RuntimeError(f"Existing native 4D conversion failed its gate: {manifest_path}")
    if phase_root.exists() and any(phase_root.iterdir()):
        raise FileExistsError(f"Incomplete conversion exists; inspect or remove: {phase_root}")
    phase_root.mkdir(parents=True, exist_ok=True)
    source = _source_h5(config)
    expected = _source_episode_count(source)
    input_dir = phase_root / "input"
    input_dir.mkdir(parents=True)
    input_h5 = input_dir / "trajectory.h5"
    input_json = input_dir / "trajectory.json"
    shutil.copy2(source, input_h5)
    _prepare_input_metadata(_metadata_path(source), input_json)
    log_path = phase_root / "conversion.log"
    command = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(input_h5),
        "--use-first-env-state",
        "--target-control-mode",
        CONTROL_MODE,
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
        "1",
    ]
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=practice_root(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Native 4D conversion failed; inspect {log_path}") from exc
    candidates = [path for path in input_dir.glob("*.h5") if path.name != input_h5.name]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one converted H5, found {candidates}; inspect {log_path}")
    shutil.copy2(candidates[0], output_h5)
    shutil.copy2(_metadata_path(candidates[0]), output_json)
    metadata = _read_json(output_json)
    converted_control_mode = str(
        metadata.get("env_info", {}).get("env_kwargs", {}).get("control_mode", "")
    )
    if converted_control_mode != CONTROL_MODE:
        raise RuntimeError(
            "ManiSkill converter output metadata does not report the requested target "
            f"controller: {converted_control_mode!r} != {CONTROL_MODE!r}"
        )
    action_dims: set[int] = set()
    successful_at_end = 0
    fingerprints: dict[int, str] = {}
    finite_actions = True
    with h5py.File(output_h5, "r") as handle:
        episode_count = len(handle)
        for name, group in handle.items():
            actions = np.asarray(group["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise RuntimeError(f"{name} actions must be rank-2, got {actions.shape}")
            action_dims.add(int(actions.shape[1]))
            finite_actions = finite_actions and bool(np.isfinite(actions).all())
            successful_at_end += int(_success_at_end(group))
            fingerprints[_episode_id(name)] = start_state_fingerprint(group, tolerance=1e-4)
    invert_unique_fingerprints(fingerprints, label="native 4D conversion")
    metadata["native_ee_conversion"] = {
        "route": "official_cpu_joint_to_native_4d",
        "source_joint_h5": _relative(source),
        "conversion_backend": "physx_cpu",
        "control_mode": CONTROL_MODE,
        "converted_control_mode": converted_control_mode,
        "action_dim": ACTION_DIM,
        "identity": "env_states[0] start-state fingerprint; seed is not identity",
    }
    _write_json(output_json, metadata)
    gate_pass = (
        0 < episode_count <= expected
        and action_dims == {ACTION_DIM}
        and finite_actions
        and len(fingerprints) == episode_count
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "route": "official_cpu_joint_to_native_4d",
        "source_joint_h5": _relative(source),
        "replay_h5": _relative(output_h5),
        "replay_json": _relative(output_json),
        "conversion_backend": "physx_cpu",
        "control_mode": CONTROL_MODE,
        "action_dim": ACTION_DIM,
        "episode_count": episode_count,
        "source_episode_count": expected,
        "conversion_survival_rate": episode_count / max(1, expected),
        "dropped_source_episode_count": expected - episode_count,
        "successful_at_end": successful_at_end,
        "action_dims": sorted(action_dims),
        "finite_actions": finite_actions,
        "unique_start_state_fingerprints": len(fingerprints),
        "command": command,
        "log": _relative(log_path),
        "gate_pass": gate_pass,
    }
    _write_json(manifest_path, manifest)
    if not gate_pass:
        raise RuntimeError(
            "Native 4D conversion gate failed: "
            f"episodes={episode_count}/{expected}, dims={sorted(action_dims)}, "
            f"finite={finite_actions}; inspect {manifest_path}"
        )
    shutil.rmtree(input_dir)
    return manifest


def gpu_baseline(config: Mapping[str, Any], *, num_envs: int) -> dict[str, Any]:
    paths = _paths(config)
    source = paths["convert"] / "replay.h5"
    if not source.is_file():
        raise FileNotFoundError(f"Run convert first: {source}")
    expected = _source_episode_count(source)
    return replay(
        source_h5=source,
        output_dir=paths["baseline"],
        num_envs=num_envs,
        expected_episodes=expected,
        obs_mode="none",
        source_type="m1_w_native_4d_gpu_baseline",
        source_desc=(
            "Live GPU replay of native 4D pd_ee_delta_pos actions produced directly "
            "from the original joint demonstrations; RGB is deferred to final capture"
        ),
        required_success_rate=0.0,
        control_mode=CONTROL_MODE,
        action_dim=ACTION_DIM,
    )


def _write_empty_repair_h5(repair_dir: Path, metadata: Mapping[str, Any]) -> Path:
    repair_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = repair_dir / "repair.h5"
    output_json = _metadata_path(output_h5)
    if not output_h5.exists():
        with h5py.File(output_h5, "w"):
            pass
    if not output_json.exists():
        payload = copy.deepcopy(dict(metadata))
        payload["episodes"] = []
        payload["control_mode"] = CONTROL_MODE
        payload["action_dim"] = ACTION_DIM
        _write_json(output_json, payload)
    return output_h5


def gpu_repair(config: Mapping[str, Any], *, num_envs: int) -> dict[str, Any]:
    paths = _paths(config)
    source = paths["convert"] / "replay.h5"
    baseline = paths["baseline"] / "replay.h5"
    if not source.is_file() or not baseline.is_file():
        raise FileNotFoundError("Run convert and gpu-replay before gpu-repair")
    result = repair(
        offline_ee_h5=source,
        gpu_replay_h5=baseline,
        output_dir=paths["repair"],
        task_id=TASK_ID,
        num_envs=num_envs,
        # Individual repair failures are intentionally passed to accepted
        # merge; the formal gate is the final accepted episode count.
        require_all_success=False,
    )
    repair_h5 = paths["repair"] / "repair.h5"
    if not repair_h5.is_file():
        metadata = _read_json(_metadata_path(source))
        _write_empty_repair_h5(paths["repair"], metadata)
        result = dict(result)
        result["repair_h5"] = _relative(repair_h5)
    return result


def merge_accepted(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create a compact accepted-only artifact and report failed leftovers."""

    paths = _paths(config)
    output_dir = paths["merge"]
    output_h5 = output_dir / "replay.h5"
    output_json = _metadata_path(output_h5)
    manifest_path = output_dir / "merge_manifest.json"
    if manifest_path.is_file() and output_h5.is_file() and output_json.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("gate_pass"):
            return manifest
        raise RuntimeError(f"Existing accepted merge failed its gate: {manifest_path}")
    baseline_h5 = paths["baseline"] / "replay.h5"
    repair_h5 = paths["repair"] / "repair.h5"
    if not baseline_h5.is_file() or not repair_h5.is_file():
        raise FileNotFoundError("Run gpu-replay and gpu-repair before merge")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_metadata = _read_json(_metadata_path(baseline_h5))
    baseline_records = {
        int(row["episode_id"]): row for row in baseline_metadata.get("episodes", [])
    }
    repair_by_fingerprint: dict[str, tuple[int, bool]] = {}
    with h5py.File(repair_h5, "r") as repair_file:
        for name in repair_file:
            group = repair_file[name]
            fingerprint = start_state_fingerprint(group, tolerance=1e-4)
            repair_by_fingerprint[fingerprint] = (_episode_id(name), _success_at_end(group))
    accepted: list[dict[str, Any]] = []
    excluded: list[int] = []
    baseline_fingerprints: dict[int, str] = {}
    with h5py.File(baseline_h5, "r") as baseline_file, h5py.File(output_h5, "w") as destination:
        for name, group in baseline_file.items():
            baseline_id = _episode_id(name)
            fingerprint = start_state_fingerprint(group, tolerance=1e-4)
            baseline_fingerprints[baseline_id] = fingerprint
            if _success_at_end(group):
                selected_file, selected_name = baseline_file, name
                route = "baseline"
            else:
                repair_entry = repair_by_fingerprint.get(fingerprint)
                if repair_entry is None or not repair_entry[1]:
                    excluded.append(
                        int(baseline_records[baseline_id].get("source_episode_id", baseline_id))
                    )
                    continue
                repair_id = repair_entry[0]
                repair_file = h5py.File(repair_h5, "r")
                selected_file, selected_name = repair_file, f"traj_{repair_id}"
                route = "repair"
            try:
                source_record = baseline_records[baseline_id]
                source_id = int(source_record.get("source_episode_id", baseline_id))
                selected_file.copy(selected_name, destination, name=f"traj_{source_id}")
                destination[f"traj_{source_id}"].attrs["source_episode_id"] = source_id
                destination[f"traj_{source_id}"].attrs["route"] = route
                accepted.append(
                    {
                        "episode_id": source_id,
                        "source_episode_id": source_id,
                        "baseline_episode_id": baseline_id,
                        "route": route,
                        "start_state_fingerprint": fingerprint,
                    }
                )
            finally:
                if route == "repair":
                    selected_file.close()
    merged_metadata = copy.deepcopy(baseline_metadata)
    merged_metadata["episodes"] = accepted
    merged_metadata["env_info"]["env_kwargs"].update(
        obs_mode="none",
        control_mode=CONTROL_MODE,
        sim_backend="physx_cuda",
    )
    merged_metadata["native_4d_data_route"] = {
        "route": "accepted_native_4d_gpu_replay_plus_position_repair",
        "baseline_h5": _relative(baseline_h5),
        "repair_h5": _relative(repair_h5),
        "identity": "env_states[0] start-state fingerprint; seed is not identity",
    }
    _write_json(output_json, merged_metadata)
    action_dims: set[int] = set()
    with h5py.File(output_h5, "r") as handle:
        for group in handle.values():
            action_dims.add(int(group["actions"].shape[1]))
    min_episodes = int(config.get("min_formal_episodes", 900))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "route": "accepted_native_4d_gpu_replay_plus_position_repair",
        "replay_h5": _relative(output_h5),
        "replay_json": _relative(output_json),
        "baseline_episode_count": len(baseline_fingerprints),
        "accepted_episode_count": len(accepted),
        "excluded_source_episode_ids": sorted(excluded),
        "action_dims": sorted(action_dims),
        "minimum_formal_episodes": min_episodes,
        "gate_pass": len(accepted) >= min_episodes and action_dims == {ACTION_DIM},
    }
    _write_json(manifest_path, manifest)
    if not manifest["gate_pass"]:
        raise RuntimeError(
            f"Native 4D accepted merge gate failed: {len(accepted)} accepted, "
            f"minimum={min_episodes}, dims={sorted(action_dims)}; inspect {manifest_path}"
        )
    return manifest


def capture_final(config: Mapping[str, Any], *, num_envs: int) -> dict[str, Any]:
    paths = _paths(config)
    baseline_h5 = paths["baseline"] / "replay.h5"
    repair_h5 = paths["repair"] / "repair.h5"
    if not baseline_h5.is_file() or not repair_h5.is_file():
        raise FileNotFoundError("Run gpu-replay and gpu-repair before capture")
    layout = freeze_layout(
        baseline_h5=baseline_h5,
        repair_h5=repair_h5,
        output=paths["layout"],
        num_envs=num_envs,
        control_mode=CONTROL_MODE,
        action_dim=ACTION_DIM,
        expected_baseline_accepted=None,
        expected_repair_accepted=None,
        expected_accepted=None,
    )
    result = capture(
        baseline_h5=baseline_h5,
        repair_h5=repair_h5,
        layout_path=paths["layout"],
        output_dir=paths["capture"],
        control_mode=CONTROL_MODE,
        action_dim=ACTION_DIM,
    )
    min_episodes = int(config.get("min_formal_episodes", 900))
    if int(result["episode_count"]) < min_episodes:
        raise RuntimeError(
            f"Final native 4D RGB dataset has only {result['episode_count']} episodes; "
            f"minimum={min_episodes}"
        )
    return {"layout": layout["layout"], "capture": result}


def run(config: Mapping[str, Any], *, phase: str, num_envs: int) -> Any:
    if phase == "convert":
        return convert_native_4d(config)
    if phase == "gpu-replay":
        return gpu_baseline(config, num_envs=num_envs)
    if phase == "gpu-repair":
        return gpu_repair(config, num_envs=num_envs)
    if phase == "merge":
        return merge_accepted(config)
    if phase == "capture":
        return capture_final(config, num_envs=num_envs)
    if phase == "all":
        return {
            "convert": convert_native_4d(config),
            "gpu_replay": gpu_baseline(config, num_envs=num_envs),
            "gpu_repair": gpu_repair(config, num_envs=num_envs),
            "merge": merge_accepted(config),
            "capture": capture_final(config, num_envs=num_envs),
        }
    raise ValueError(f"Unknown phase: {phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("convert", "gpu-replay", "gpu-repair", "merge", "capture", "all"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/maniskill/m1_wam_native_4d.json"),
    )
    parser.add_argument("--num-envs", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = run(config, phase=args.phase, num_envs=args.num_envs)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
