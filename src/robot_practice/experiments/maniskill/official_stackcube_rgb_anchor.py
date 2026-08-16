"""Audit the official StackCube RGB Diffusion Policy replay artifact.

This module is intentionally small and does not replay an environment.  The
official ManiSkill replay command creates the HDF5 file; this audit only checks
that the file has the contract expected by the official RGB recipe before the
official training script is launched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pos"
SIM_BACKEND = "physx_cpu"
OBS_MODE = "rgb"
NUM_DEMOS = 100
EXPECTED_ACTION_DIM = 4
EXPECTED_RGB_SHAPE = (128, 128, 3)


def _sorted_trajectory_keys(handle: h5py.File) -> list[str]:
    keys = [key for key in handle if key.startswith("traj_")]
    return sorted(keys, key=lambda key: int(key.split("_")[-1]))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _dataset(group: h5py.Group, path: str) -> np.ndarray:
    node: Any = group
    for part in path.split("/"):
        node = node[part]
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"Expected HDF5 dataset at {path}")
    return np.asarray(node)


def _episode_record(handle: h5py.File, key: str) -> dict[str, Any]:
    episode = handle[key]
    actions = _dataset(episode, "actions")
    rgb = _dataset(episode, "obs/sensor_data/base_camera/rgb")
    success = _dataset(episode, "success").astype(bool, copy=False).reshape(-1)
    if actions.ndim != 2:
        raise ValueError(f"{key}: actions must be [T,A], got {actions.shape}")
    if rgb.ndim != 4:
        raise ValueError(f"{key}: RGB must be [T+1,H,W,C], got {rgb.shape}")
    if success.size != actions.shape[0]:
        raise ValueError(
            f"{key}: success length {success.size} != action length {actions.shape[0]}"
        )
    if rgb.shape[0] != actions.shape[0] + 1:
        raise ValueError(
            f"{key}: RGB length {rgb.shape[0]} != action length + 1 ({actions.shape[0] + 1})"
        )
    if tuple(rgb.shape[1:]) != EXPECTED_RGB_SHAPE:
        raise ValueError(
            f"{key}: expected RGB shape [T+1,128,128,3], got {rgb.shape}"
        )
    return {
        "trajectory_key": key,
        "episode_id": int(key.split("_")[-1]),
        "num_actions": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        "success_once": bool(success.any()),
        "success_at_end": bool(success[-1]) if success.size else False,
        "action_min": np.min(actions, axis=0).astype(float).tolist(),
        "action_max": np.max(actions, axis=0).astype(float).tolist(),
    }


def _metadata_contract(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {"available": False}
    env_info = metadata.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {}) if isinstance(env_info, dict) else {}
    episodes = metadata.get("episodes", [])
    first_episode = episodes[0] if episodes and isinstance(episodes[0], dict) else {}
    return {
        "available": True,
        "env_id": env_info.get("env_id") if isinstance(env_info, dict) else None,
        "obs_mode": env_kwargs.get("obs_mode"),
        "control_mode": env_kwargs.get("control_mode") or first_episode.get("control_mode"),
        "sim_backend": env_kwargs.get("sim_backend"),
        "max_episode_steps": env_info.get("max_episode_steps")
        if isinstance(env_info, dict)
        else None,
        "num_metadata_episodes": len(episodes) if isinstance(episodes, list) else None,
    }


def audit_replay(
    *,
    replay_h5: Path,
    output: Path,
    num_demos: int = NUM_DEMOS,
) -> dict[str, Any]:
    """Audit a replayed official HDF5 and write a JSON gate report."""

    replay_h5 = replay_h5.expanduser().resolve()
    output = output.expanduser().resolve()
    if num_demos <= 0:
        raise ValueError("num_demos must be positive")
    if not replay_h5.is_file():
        raise FileNotFoundError(replay_h5)

    metadata = _read_json(replay_h5.with_suffix(".json"))
    with h5py.File(replay_h5, "r") as handle:
        keys = _sorted_trajectory_keys(handle)
        if len(keys) < num_demos:
            raise ValueError(f"Replay has {len(keys)} trajectories, expected at least {num_demos}")
        records = [_episode_record(handle, key) for key in keys]

    selected = records[:num_demos]
    action_dim_ok = all(record["action_dim"] == EXPECTED_ACTION_DIM for record in records)
    rgb_shape_ok = all(tuple(record["rgb_shape"][1:]) == EXPECTED_RGB_SHAPE for record in records)
    rgb_dtype_ok = all(record["rgb_dtype"] == "uint8" for record in records)
    success_once_ok = all(record["success_once"] for record in records)
    success_at_end_ok = all(record["success_at_end"] for record in records)
    metadata_contract = _metadata_contract(metadata)
    metadata_ok = metadata is None or (
        metadata_contract["env_id"] == TASK_ID
        and metadata_contract["obs_mode"] == OBS_MODE
        and metadata_contract["control_mode"] == CONTROL_MODE
        and metadata_contract["sim_backend"] in {SIM_BACKEND, "cpu"}
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "task": TASK_ID,
        "replay_h5": str(replay_h5),
        "metadata": metadata_contract,
        "contract": {
            "source": "official ManiSkill replay_trajectory",
            "obs_mode": OBS_MODE,
            "control_mode": CONTROL_MODE,
            "sim_backend": SIM_BACKEND,
            "action_dim": EXPECTED_ACTION_DIM,
            "rgb_shape": list(EXPECTED_RGB_SHAPE),
            "training_demos": num_demos,
            "selection_rule": "official train_rgbd.py sorted trajectory keys, first num_demos",
        },
        "replay": {
            "num_episodes": len(records),
            "num_actions": int(sum(record["num_actions"] for record in records)),
            "selected_episode_ids": [record["episode_id"] for record in selected],
            "success_once_count": int(sum(record["success_once"] for record in records)),
            "success_at_end_count": int(sum(record["success_at_end"] for record in records)),
            "mean_episode_actions": float(
                np.mean([record["num_actions"] for record in records])
            ),
            "selected": selected,
        },
        "checks": {
            "metadata_contract": bool(metadata_ok),
            "action_dim": bool(action_dim_ok),
            "rgb_shape": bool(rgb_shape_ok),
            "rgb_dtype": bool(rgb_dtype_ok),
            "all_replayed_episodes_success_once": success_once_ok,
            "all_replayed_episodes_success_at_end": success_at_end_ok,
        },
    }
    report["gate_pass"] = all(report["checks"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--replay-h5", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--num-demos", type=int, default=NUM_DEMOS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_replay(
        replay_h5=args.replay_h5,
        output=args.output,
        num_demos=args.num_demos,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
