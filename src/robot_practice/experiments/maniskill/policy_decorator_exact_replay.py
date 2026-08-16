"""Exact replay of the public Policy Decorator Peg dataset in ManiSkill2.

Run this module with ManiSkill2 0.5.3 and the Policy Decorator repository on
``PYTHONPATH``.  It deliberately uses the recorded reset seed/options and
executes the stored native actions without conversion.  The H5 initial state
is compared with the state produced by that reset, but is not injected into
the environment, so the result tests the actual reference replay contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

CONTROL_MODE = "pd_ee_delta_pose"
DEFAULT_ENV_ID = "PegInsertionSide-v0"
DEFAULT_STRICT_ENV_ID = "PegInsertionSide-v2"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_bool(value: Any) -> bool:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected one scalar boolean, got shape {array.shape}")
    return bool(array[0])


def _array(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _pose(pose: Any) -> dict[str, list[float]]:
    return {"p": _array(pose.p), "q": _array(pose.q)}


def _make_env(env_id: str, env_kwargs: dict[str, Any]) -> Any:
    import gymnasium as gym
    import mani_skill2.envs  # noqa: F401

    return gym.make(env_id, **env_kwargs)


def _snapshot(env: Any) -> dict[str, Any]:
    base = env.unwrapped
    robot = base.agent.robot
    return {
        "state": _array(base.get_state()),
        "robot_qpos": _array(robot.get_qpos()),
        "robot_qvel": _array(robot.get_qvel()),
        "eef_pose": _pose(base.tcp.pose),
        "peg_pose": _pose(base.peg.pose),
        "box_hole_pose": _pose(base.box_hole_pose),
        "peg_half_size": _array(base.peg_half_size),
        "box_hole_radius": float(base.box_hole_radius),
    }


def _evaluate(env: Any) -> dict[str, Any]:
    result = env.unwrapped.evaluate()
    output: dict[str, Any] = {}
    for key, value in result.items():
        if key == "success":
            output[key] = _scalar_bool(value)
        else:
            array = np.asarray(value)
            output[key] = array.tolist() if array.ndim else float(array)
    return output


def _replay_one(
    env: Any,
    *,
    record: dict[str, Any],
    group: h5py.Group,
    expected_initial_state: np.ndarray,
) -> dict[str, Any]:
    reset_kwargs = dict(record.get("reset_kwargs", {}))
    seed = int(record["episode_seed"])
    options = dict(reset_kwargs.get("options", {}))
    env.reset(seed=seed, options=options)
    initial = _snapshot(env)
    state_delta = np.asarray(initial["state"], dtype=np.float64) - expected_initial_state

    actions = np.asarray(group["actions"], dtype=np.float32)
    success_once = False
    first_success_step: int | None = None
    last_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    executed_steps = 0
    for step, action in enumerate(actions, start=1):
        _, _, terminated, truncated, _info = env.step(action)
        last_info = _evaluate(env)
        success = bool(last_info.get("success", False))
        if success and not success_once:
            success_once = True
            first_success_step = step
        executed_steps = step
        if terminated or truncated:
            break

    final = _snapshot(env)
    dataset_success = np.asarray(group["success"], dtype=bool)
    return {
        "episode_id": int(record["episode_id"]),
        "episode_seed": seed,
        "trajectory_length": len(actions),
        "executed_steps": executed_steps,
        "initial_state_max_abs_error": float(np.max(np.abs(state_delta))),
        "initial_state_mean_abs_error": float(np.mean(np.abs(state_delta))),
        "dataset_final_success": bool(dataset_success[-1]),
        "metadata_final_success": bool(record.get("info", {}).get("success", False)),
        "replay_success_once": success_once,
        "replay_final_success": bool(last_info.get("success", False)),
        "first_success_step": first_success_step,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "last_info": last_info,
        "initial": initial,
        "final": final,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_id",
        "episode_seed",
        "trajectory_length",
        "executed_steps",
        "initial_state_max_abs_error",
        "dataset_final_success",
        "metadata_final_success",
        "replay_success_once",
        "replay_final_success",
        "first_success_step",
        "terminated",
        "truncated",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def exact_replay(
    *,
    dataset_h5: Path,
    dataset_json: Path,
    output_dir: Path,
    count: int,
    sample_seed: int,
    env_id: str = DEFAULT_ENV_ID,
    strict_env_id: str = DEFAULT_STRICT_ENV_ID,
    reward_mode: str | None = None,
) -> dict[str, Any]:
    import mani_skill2.envs  # noqa: F401
    importlib.import_module("envs.maniskill_fixed")

    metadata = _read_json(dataset_json)
    records = {int(item["episode_id"]): item for item in metadata["episodes"]}
    env_info = metadata["env_info"]
    env_kwargs = dict(env_info["env_kwargs"])
    if env_kwargs.get("control_mode") != CONTROL_MODE:
        raise ValueError(f"Expected native {CONTROL_MODE}, got {env_kwargs.get('control_mode')}")
    if count <= 0 or count > len(records):
        raise ValueError(f"count must be in [1, {len(records)}]")

    selected_ids = sorted(random.Random(sample_seed).sample(sorted(records), count))
    evaluation_env_kwargs = dict(env_kwargs)
    if reward_mode is not None:
        evaluation_env_kwargs["reward_mode"] = reward_mode
    native_env = _make_env(env_id, evaluation_env_kwargs)
    strict_env = _make_env(strict_env_id, evaluation_env_kwargs)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with h5py.File(dataset_h5, "r") as handle:
            for ordinal, episode_id in enumerate(selected_ids, start=1):
                group = handle[f"traj_{episode_id}"]
                expected_state = np.asarray(group["env_states"][0], dtype=np.float64)
                record = records[episode_id]
                native = _replay_one(
                    native_env,
                    record=record,
                    group=group,
                    expected_initial_state=expected_state,
                )
                strict = _replay_one(
                    strict_env,
                    record=record,
                    group=group,
                    expected_initial_state=expected_state,
                )
                row = {
                    "sample_ordinal": ordinal,
                    "episode_id": episode_id,
                    "episode_seed": int(record["episode_seed"]),
                    "native": native,
                    "strict": strict,
                }
                rows.append(row)
                print(
                    f"[{ordinal:02d}/{count}] episode={episode_id} "
                    f"native={int(native['replay_final_success'])} "
                    f"strict={int(strict['replay_final_success'])} "
                    f"state_err={native['initial_state_max_abs_error']:.2e} "
                    f"steps={native['executed_steps']}/{native['trajectory_length']}",
                    flush=True,
                )
    finally:
        native_env.close()
        strict_env.close()

    def rate(path: tuple[str, str]) -> dict[str, Any]:
        values = [bool(row[path[0]][path[1]]) for row in rows]
        return {
            "num_success": int(sum(values)),
            "num_episodes": len(values),
            "rate": float(sum(values) / len(values)),
        }

    native_only = [
        int(row["episode_id"])
        for row in rows
        if row["native"]["replay_final_success"] and not row["strict"]["replay_final_success"]
    ]
    result = {
        "schema_version": 1,
        "diagnostic": "policy_decorator_ms2_exact_replay",
        "dataset_h5": dataset_h5.as_posix(),
        "dataset_json": dataset_json.as_posix(),
        "dataset_h5_sha256": _sha256(dataset_h5),
        "dataset_env_id": env_info["env_id"],
        "dataset_env_kwargs": env_kwargs,
        "evaluation_env_kwargs": evaluation_env_kwargs,
        "evaluation_reward_mode": evaluation_env_kwargs.get("reward_mode"),
        "evaluation_env_id": env_id,
        "strict_env_id": strict_env_id,
        "selected_episode_ids": selected_ids,
        "sample_seed": sample_seed,
        "state_reconstruction": "recorded seed/options; initial H5 env_state compared, not injected",
        "summary": {
            "native": rate(("native", "replay_final_success")),
            "strict": rate(("strict", "replay_final_success")),
            "native_success_once": rate(("native", "replay_success_once")),
            "strict_success_once": rate(("strict", "replay_success_once")),
            "native_only_final_success_episode_ids": native_only,
            "max_initial_state_error": max(
                float(row["native"]["initial_state_max_abs_error"]) for row in rows
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "episodes": rows,
    }
    _write_json(output_dir / "p1_exact_replay.json", result)
    _write_csv(
        output_dir / "p1_exact_replay.csv",
        [
            {
                **row["native"],
                "episode_id": row["episode_id"],
                "episode_seed": row["episode_seed"],
            }
            for row in rows
        ],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-h5", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260811)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--strict-env-id", default=DEFAULT_STRICT_ENV_ID)
    parser.add_argument("--reward-mode", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exact_replay(
        dataset_h5=args.dataset_h5.expanduser().resolve(),
        dataset_json=args.dataset_json.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        count=args.count,
        sample_seed=args.sample_seed,
        env_id=args.env_id,
        strict_env_id=args.strict_env_id,
        reward_mode=args.reward_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
