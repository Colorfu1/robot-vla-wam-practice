"""Replay accepted StackCube EE candidates on GPU without recording trajectories.

The candidate set is the union of:

* baseline trajectories whose final success flag is true; and
* repaired trajectories whose final success flag is true.

The runner creates an RGB ManiSkill environment and executes every selected
EE action sequence, but deliberately does not use ``RecordEpisode``. It writes
only a compact JSON validation report; no H5, RGB frame, or video is saved.

The historical default contract is 7D ``pd_ee_delta_pose``.  The native M1-W
route passes the 4D ``pd_ee_delta_pos`` contract explicitly.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .gpu_ee_online_retarget_probe import _batch_states, _jsonable, _read_json, _relative
from .state_identity import start_state_fingerprint

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pose"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Candidate:
    source_episode_id: int
    route: str
    h5_path: Path
    group_id: int
    start_state_fingerprint: str
    accepted: bool = True


def _metadata_records(path: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(path.with_suffix(".json"))
    records = payload.get("episodes")
    if not isinstance(records, list):
        raise TypeError(f"Metadata must contain an episodes list: {path.with_suffix('.json')}")
    return {int(record["episode_id"]): dict(record) for record in records}


def _success_at_end(group: h5py.Group) -> bool:
    values = np.asarray(group["success"], dtype=bool).reshape(-1)
    return bool(values.size and values[-1])


def build_candidates(baseline_h5: Path, repair_h5: Path) -> tuple[list[Candidate], dict[str, Any]]:
    baseline_records = _metadata_records(baseline_h5)
    repair_records = _metadata_records(repair_h5)
    candidates: dict[int, Candidate] = {}
    baseline_success = 0
    repair_success = 0
    excluded_repair_source_ids: list[int] = []

    with h5py.File(baseline_h5, "r") as handle:
        for group_id, record in baseline_records.items():
            group = handle[f"traj_{group_id}"]
            source_id = int(record["source_episode_id"])
            fingerprint = start_state_fingerprint(group, tolerance=1e-4)
            expected = str(record["source_start_state_fingerprint"])
            if fingerprint != expected:
                raise RuntimeError(
                    f"Baseline identity mismatch for group {group_id}: {fingerprint} != {expected}"
                )
            if not _success_at_end(group):
                continue
            if source_id in candidates:
                raise ValueError(f"Duplicate baseline source episode: {source_id}")
            candidates[source_id] = Candidate(
                source_episode_id=source_id,
                route="baseline",
                h5_path=baseline_h5,
                group_id=group_id,
                start_state_fingerprint=fingerprint,
            )
            baseline_success += 1

    with h5py.File(repair_h5, "r") as handle:
        for group_id, record in repair_records.items():
            group = handle[f"traj_{group_id}"]
            source_id = int(record["source_episode_id"])
            fingerprint = start_state_fingerprint(group, tolerance=1e-4)
            expected = str(record["source_start_state_fingerprint"])
            if fingerprint != expected:
                raise RuntimeError(
                    f"Repair identity mismatch for group {group_id}: {fingerprint} != {expected}"
                )
            if not _success_at_end(group):
                excluded_repair_source_ids.append(source_id)
                continue
            if source_id in candidates:
                raise ValueError(
                    f"Source episode {source_id} is marked successful in both baseline and repair"
                )
            candidates[source_id] = Candidate(
                source_episode_id=source_id,
                route="repair",
                h5_path=repair_h5,
                group_id=group_id,
                start_state_fingerprint=fingerprint,
            )
            repair_success += 1

    ordered = [candidates[source_id] for source_id in sorted(candidates)]
    fingerprints = [candidate.start_state_fingerprint for candidate in ordered]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Accepted candidate set contains duplicate start-state fingerprints")
    selection = {
        "baseline_successful_at_end": baseline_success,
        "repair_successful_at_end": repair_success,
        "accepted_episode_count": len(ordered),
        "excluded_repair_source_episode_ids": sorted(excluded_repair_source_ids),
        "unique_source_episode_ids": len({candidate.source_episode_id for candidate in ordered}),
        "unique_start_state_fingerprints": len(set(fingerprints)),
    }
    return ordered, selection


def build_carrier_batches(
    baseline_h5: Path,
    repair_h5: Path,
    *,
    num_envs: int,
    expected_baseline_accepted: int | None = 805,
    expected_repair_accepted: int | None = 182,
    expected_accepted: int | None = 987,
) -> tuple[list[list[Candidate]], dict[str, Any]]:
    """Recreate the two layouts that originally established success.

    Failed trajectories remain in the vectorized environment as physics
    carriers, but are marked ``accepted=False`` and never enter the candidate
    success gate.
    """

    batches: list[list[Candidate]] = []
    route_counts: dict[str, dict[str, int]] = {}
    accepted_source_ids: set[int] = set()

    for route, path, ordering in (
        ("baseline", baseline_h5, "baseline_replay_target_episode_id"),
        ("repair", repair_h5, "target_baseline_episode_id"),
    ):
        records = _metadata_records(path)
        route_items: list[Candidate] = []
        with h5py.File(path, "r") as handle:
            for group_id, record in records.items():
                group = handle[f"traj_{group_id}"]
                source_id = int(record["source_episode_id"])
                fingerprint = start_state_fingerprint(group, tolerance=1e-4)
                expected = str(record["source_start_state_fingerprint"])
                if fingerprint != expected:
                    raise RuntimeError(
                        f"{route} identity mismatch for group {group_id}: "
                        f"{fingerprint} != {expected}"
                    )
                accepted = _success_at_end(group)
                if accepted:
                    if source_id in accepted_source_ids:
                        raise ValueError(f"Accepted source episode appears twice: {source_id}")
                    accepted_source_ids.add(source_id)
                route_items.append(
                    Candidate(
                        source_episode_id=source_id,
                        route=route,
                        h5_path=path,
                        group_id=group_id,
                        start_state_fingerprint=fingerprint,
                        accepted=accepted,
                    )
                )
        if route == "baseline":
            # ``gpu_ee_replay.replay`` orders source episodes by length before
            # creating each CUDA batch.  The merged baseline H5 renumbers those
            # executions as traj_0, traj_1, ...; group_id is therefore the
            # authoritative execution order.  Sorting by source ID silently
            # changes GPU PhysX batch composition and can turn a previously
            # successful replay into a failure during RGB capture.
            route_items.sort(key=lambda item: item.group_id)
        else:
            route_items.sort(
                key=lambda item: int(records[item.group_id]["target_baseline_episode_id"])
            )
        batches.extend(
            route_items[start : start + int(num_envs)]
            for start in range(0, len(route_items), int(num_envs))
        )
        route_counts[route] = {
            "episode_executions": len(route_items),
            "accepted": sum(item.accepted for item in route_items),
            "carriers": sum(not item.accepted for item in route_items),
            "batch_count": (len(route_items) + num_envs - 1) // num_envs,
            "ordering": ordering,
        }

    if (
        expected_baseline_accepted is not None
        and route_counts["baseline"]["accepted"] != int(expected_baseline_accepted)
    ):
        raise RuntimeError(f"Expected 805 accepted baseline episodes, got {route_counts}")
    if (
        expected_repair_accepted is not None
        and route_counts["repair"]["accepted"] != int(expected_repair_accepted)
    ):
        raise RuntimeError(f"Expected 182 accepted repair episodes, got {route_counts}")
    if expected_accepted is not None and len(accepted_source_ids) != int(expected_accepted):
        raise RuntimeError(f"Expected 987 unique accepted source ids, got {len(accepted_source_ids)}")
    layout = {
        "name": "original_route_carrier_layout",
        "reason": "GPU PhysX replay depends on vectorized batch composition",
        "num_envs": int(num_envs),
        "routes": route_counts,
        "total_episode_executions": sum(len(batch) for batch in batches),
        "accepted_episode_count": len(accepted_source_ids),
        "carrier_episode_executions": sum(
            sum(not item.accepted for item in batch) for batch in batches
        ),
    }
    return batches, layout


def _state_list(group: h5py.Group) -> list[dict[str, Any]]:
    from mani_skill.trajectory import utils as trajectory_utils

    return trajectory_utils.dict_to_list_of_dicts(group["env_states"])


def _worker(
    serialized_candidates: Sequence[Mapping[str, Any]],
    env_metadata_path: str,
    result_queue: Any,
) -> None:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import torch
        from mani_skill.utils import common

        metadata = _read_json(Path(env_metadata_path))
        candidates = [dict(value) for value in serialized_candidates]
        env_kwargs = dict(metadata["env_info"]["env_kwargs"])
        env_kwargs.update(
            obs_mode="rgb",
            control_mode=CONTROL_MODE,
            sim_backend="physx_cuda",
            render_mode="rgb_array",
            reward_mode="sparse",
            reconfiguration_freq=1,
            num_envs=len(candidates),
        )
        env = gym.make(TASK_ID, **env_kwargs)
        try:
            actions: list[np.ndarray] = []
            initial_states: list[dict[str, Any]] = []
            for candidate in candidates:
                with h5py.File(candidate["h5_path"], "r") as handle:
                    group = handle[f"traj_{int(candidate['group_id'])}"]
                    episode_actions = np.asarray(group["actions"], dtype=np.float32)
                    states = _state_list(group)
                if episode_actions.ndim != 2 or episode_actions.shape[1] != 7:
                    raise ValueError(
                        f"Candidate {candidate['source_episode_id']} actions must be [T,7], "
                        f"got {episode_actions.shape}"
                    )
                if len(states) != len(episode_actions) + 1:
                    raise ValueError(
                        f"Candidate {candidate['source_episode_id']} state/action mismatch: "
                        f"{len(states)} vs {len(episode_actions)}"
                    )
                actions.append(episode_actions)
                initial_states.append(states[0])

            device = env.unwrapped.device
            temporary_seeds = torch.arange(len(candidates), dtype=torch.int64, device=device)
            env.reset(seed=temporary_seeds)
            env.unwrapped.set_state_dict(_batch_states(initial_states))
            env.unwrapped.scene._gpu_fetch_all()

            lengths = np.asarray([len(value) for value in actions], dtype=np.int64)
            success_once = np.zeros(len(candidates), dtype=bool)
            success_at_end = np.zeros(len(candidates), dtype=bool)
            first_success_step: list[int | None] = [None] * len(candidates)
            for step in range(int(lengths.max())):
                active = lengths > step
                action_batch = np.zeros((len(candidates), 7), dtype=np.float32)
                for index, value in enumerate(actions):
                    if active[index]:
                        action_batch[index] = value[step]
                _, _, _, _, info = env.step(torch.as_tensor(action_batch, device=device))
                success = np.asarray(
                    common.to_numpy(
                        info.get(
                            "success",
                            torch.zeros(len(candidates), dtype=torch.bool, device=device),
                        )
                    ),
                    dtype=bool,
                ).reshape(-1)
                for index, is_active in enumerate(active):
                    if not is_active:
                        continue
                    if success[index]:
                        success_once[index] = True
                        if first_success_step[index] is None:
                            first_success_step[index] = step + 1
                    if step == lengths[index] - 1:
                        success_at_end[index] = bool(success[index])

            rows = []
            for index, candidate in enumerate(candidates):
                rows.append(
                    {
                        "source_episode_id": int(candidate["source_episode_id"]),
                        "route": str(candidate["route"]),
                        "source_group_id": int(candidate["group_id"]),
                        "start_state_fingerprint": str(candidate["start_state_fingerprint"]),
                        "accepted": bool(candidate.get("accepted", True)),
                        "action_steps": int(lengths[index]),
                        "success_once": bool(success_once[index]),
                        "success_at_end": bool(success_at_end[index]),
                        "first_success_step": first_success_step[index],
                    }
                )
            result_queue.put({"ok": True, "rows": rows})
        finally:
            env.close()
    except BaseException as exc:
        result_queue.put({"ok": False, "error": repr(exc)})


def _run_batch(candidates: Sequence[Candidate], env_metadata_path: Path) -> list[dict[str, Any]]:
    serialized = [
        {
            "source_episode_id": candidate.source_episode_id,
            "route": candidate.route,
            "h5_path": str(candidate.h5_path),
            "group_id": candidate.group_id,
            "start_state_fingerprint": candidate.start_state_fingerprint,
            "accepted": candidate.accepted,
        }
        for candidate in candidates
    ]
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(serialized, str(env_metadata_path), queue),
    )
    process.start()
    result = queue.get()
    process.join()
    if process.exitcode != 0 or not result.get("ok"):
        raise RuntimeError(
            f"No-save GPU validation failed (exitcode={process.exitcode}): "
            f"{result.get('error', 'unknown error')}"
        )
    return list(result["rows"])


def validate(
    *,
    baseline_h5: str | Path,
    repair_h5: str | Path,
    output: str | Path,
    num_envs: int = 100,
    preserve_route_layout: bool = False,
) -> dict[str, Any]:
    baseline_h5 = Path(baseline_h5).expanduser().resolve()
    repair_h5 = Path(repair_h5).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    for path in (baseline_h5, repair_h5):
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise FileNotFoundError(f"Missing H5/JSON pair: {path}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    candidates, selection = build_candidates(baseline_h5, repair_h5)
    if selection["baseline_successful_at_end"] != 805:
        raise RuntimeError(f"Expected 805 baseline successes, got {selection}")
    if selection["repair_successful_at_end"] != 182:
        raise RuntimeError(f"Expected 182 repair successes, got {selection}")
    if selection["accepted_episode_count"] != 987:
        raise RuntimeError(f"Expected 987 accepted candidates, got {selection}")

    if preserve_route_layout:
        execution_batches, execution_layout = build_carrier_batches(
            baseline_h5, repair_h5, num_envs=int(num_envs)
        )
    else:
        execution_batches = [
            candidates[start : start + int(num_envs)]
            for start in range(0, len(candidates), int(num_envs))
        ]
        execution_layout = {
            "name": "compact_accepted_source_order",
            "num_envs": int(num_envs),
            "total_episode_executions": len(candidates),
            "accepted_episode_count": len(candidates),
            "carrier_episode_executions": 0,
        }

    rows: list[dict[str, Any]] = []
    batch_summaries = []
    for batch_index, batch in enumerate(execution_batches):
        batch_rows = _run_batch(batch, baseline_h5.with_suffix(".json"))
        rows.extend(batch_rows)
        accepted_rows = [row for row in batch_rows if row["accepted"]]
        batch_summaries.append(
            {
                "batch_index": batch_index,
                "route": batch_rows[0]["route"],
                "source_episode_id_min": min(row["source_episode_id"] for row in batch_rows),
                "source_episode_id_max": max(row["source_episode_id"] for row in batch_rows),
                "episode_executions": len(batch_rows),
                "accepted_episode_count": len(accepted_rows),
                "accepted_successful_at_end": sum(
                    row["success_at_end"] for row in accepted_rows
                ),
            }
        )
        print(
            f"batch {batch_index + 1}/{len(execution_batches)} "
            f"({batch_rows[0]['route']}): "
            f"{batch_summaries[-1]['accepted_successful_at_end']}/"
            f"{len(accepted_rows)} accepted end-success",
            flush=True,
        )

    accepted_rows = [row for row in rows if row["accepted"]]
    successful_once = sum(row["success_once"] for row in accepted_rows)
    successful_at_end = sum(row["success_at_end"] for row in accepted_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK_ID,
        "validation": "no_save_gpu_rgb_action_replay",
        "baseline_h5": _relative(baseline_h5),
        "repair_h5": _relative(repair_h5),
        "backend": "physx_cuda",
        "obs_mode": "rgb",
        "control_mode": CONTROL_MODE,
        "num_envs": int(num_envs),
        "execution_layout": execution_layout,
        "trajectory_recording": False,
        "h5_or_video_written": False,
        "selection": selection,
        "replay": {
            "episode_count": len(accepted_rows),
            "successful_once": successful_once,
            "success_once_rate": successful_once / len(accepted_rows),
            "successful_at_end": successful_at_end,
            "success_at_end_rate": successful_at_end / len(accepted_rows),
            "failed_source_episode_ids": [
                row["source_episode_id"]
                for row in accepted_rows
                if not row["success_at_end"]
            ],
        },
        "batches": batch_summaries,
        "episodes": rows,
        "gate_pass": successful_at_end == len(accepted_rows) == 987,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-h5", type=Path, required=True)
    parser.add_argument("--repair-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument(
        "--preserve-route-layout",
        action="store_true",
        help="Replay baseline and repair in their original carrier batch layouts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate(
        baseline_h5=args.baseline_h5,
        repair_h5=args.repair_h5,
        output=args.output,
        num_envs=args.num_envs,
        preserve_route_layout=args.preserve_route_layout,
    )
    print(
        json.dumps(
            {
                "selection": report["selection"],
                "replay": report["replay"],
                "gate_pass": report["gate_pass"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
