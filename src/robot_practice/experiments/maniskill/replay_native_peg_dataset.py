"""Replay native Policy Decorator Peg actions in the local ManiSkill3 task.

This is intentionally a cross-version diagnostic, not an exact ManiSkill2
trajectory replay.  The public Policy Decorator H5 contains ManiSkill2
environment states, while this workspace has ManiSkill3.  We therefore reset
the local task with each recorded episode seed and execute the recorded native
``pd_ee_delta_pose`` actions directly; the H5 ``env_states`` are not restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .corrected_envs import register_corrected_peg_env

DEFAULT_TASK = "PegInsertionSide-v1"
CONTROL_MODE = "pd_ee_delta_pose"
STRICT_ENV = "RobotPracticePegInsertionSideStrict-v1"


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
        raise ValueError(f"Expected one boolean value, got shape {array.shape}")
    return bool(array[0])


def _make_env(env_id: str, max_episode_steps: int) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        env_id,
        obs_mode="none",
        control_mode=CONTROL_MODE,
        sim_backend="physx_cpu",
        reward_mode="sparse",
        reconfiguration_freq=1,
        max_episode_steps=max_episode_steps,
    )


def _evaluate(env: Any) -> tuple[bool, list[float] | None]:
    info = env.unwrapped.evaluate()
    success = _scalar_bool(info.get("success", False))
    relative = info.get("peg_head_pos_at_hole")
    if relative is None:
        return success, None
    array = np.asarray(relative, dtype=np.float64).reshape(-1, 3)
    return success, array[0].tolist()


def _replay_one(env: Any, *, seed: int, actions: np.ndarray) -> dict[str, Any]:
    env.reset(seed=seed)
    success_once = False
    first_success_step: int | None = None
    final_success = False
    relative: list[float] | None = None
    termination_reason = "dataset_end"
    executed_steps = 0

    for index, action in enumerate(actions, start=1):
        _, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        final_success, relative = _evaluate(env)
        executed_steps = index
        if final_success and not success_once:
            success_once = True
            first_success_step = index
        if terminated or truncated:
            termination_reason = "terminated" if terminated else "truncated"
            break

    return {
        "executed_steps": executed_steps,
        "dataset_steps": len(actions),
        "success_once": success_once,
        "success_at_end": final_success,
        "first_success_step": first_success_step,
        "termination_reason": termination_reason,
        "peg_head_pos_at_hole": relative,
    }


def replay_dataset(
    *,
    dataset_h5: Path,
    dataset_json: Path,
    output_dir: Path,
    count: int,
    sample_seed: int,
    max_episode_steps: int,
    task: str = DEFAULT_TASK,
) -> dict[str, Any]:
    metadata = _read_json(dataset_json)
    records = {int(item["episode_id"]): item for item in metadata["episodes"]}
    if metadata["env_info"]["env_kwargs"].get("control_mode") != CONTROL_MODE:
        raise ValueError("Dataset is not a native pd_ee_delta_pose artifact")
    if count <= 0 or count > len(records):
        raise ValueError(f"count must be in [1, {len(records)}]")

    episode_ids = sorted(records)
    selected_ids = sorted(random.Random(sample_seed).sample(episode_ids, count))
    strict_env_id = register_corrected_peg_env()
    native_env = _make_env(task, max_episode_steps)
    strict_env = _make_env(strict_env_id, max_episode_steps)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with h5py.File(dataset_h5, "r") as handle:
            for ordinal, episode_id in enumerate(selected_ids, start=1):
                group = handle[f"traj_{episode_id}"]
                actions = np.asarray(group["actions"], dtype=np.float32)
                record = records[episode_id]
                seed = int(record["episode_seed"])
                native = _replay_one(native_env, seed=seed, actions=actions)
                strict = _replay_one(strict_env, seed=seed, actions=actions)
                row = {
                    "sample_ordinal": ordinal,
                    "episode_id": episode_id,
                    "episode_seed": seed,
                    "source_success_at_end": bool(np.asarray(group["success"])[-1]),
                    "native": native,
                    "strict": strict,
                }
                rows.append(row)
                print(
                    f"[{ordinal:02d}/{count}] episode={episode_id} seed={seed} "
                    f"native={int(native['success_at_end'])} "
                    f"strict={int(strict['success_at_end'])} "
                    f"steps={native['executed_steps']}/{len(actions)}",
                    flush=True,
                )
    finally:
        native_env.close()
        strict_env.close()

    def summary(key: str) -> dict[str, Any]:
        values = [bool(row[key]["success_at_end"]) for row in rows]
        return {
            "num_success": int(sum(values)),
            "num_episodes": len(values),
            "rate": float(sum(values) / len(values)),
        }

    native_only = [
        int(row["episode_id"])
        for row in rows
        if row["native"]["success_at_end"] and not row["strict"]["success_at_end"]
    ]
    payload = {
        "schema_version": 1,
        "diagnostic": "cross_version_native_action_replay",
        "dataset_h5": dataset_h5.as_posix(),
        "dataset_json": dataset_json.as_posix(),
        "dataset_h5_sha256": _sha256(dataset_h5),
        "dataset_env_id": metadata["env_info"]["env_id"],
        "dataset_control_mode": CONTROL_MODE,
        "evaluation_task": task,
        "native_env_id": task,
        "strict_env_id": strict_env_id,
        "sim_backend": "physx_cpu",
        "max_episode_steps": max_episode_steps,
        "state_restore": "not_used; reset by recorded episode seed",
        "sample_seed": sample_seed,
        "selected_episode_ids": selected_ids,
        "summary": {
            "native": summary("native"),
            "strict": summary("strict"),
            "native_only_final_success_episode_ids": native_only,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "episodes": rows,
    }
    _write_json(output_dir / "probe_manifest.json", payload)
    _write_json(output_dir / "episode_results.json", rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-h5", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260811)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--task", default=DEFAULT_TASK)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replay_dataset(
        dataset_h5=args.dataset_h5.expanduser().resolve(),
        dataset_json=args.dataset_json.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        count=args.count,
        sample_seed=args.sample_seed,
        max_episode_steps=args.max_episode_steps,
        task=args.task,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
