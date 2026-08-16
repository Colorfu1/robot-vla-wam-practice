"""Replay an offline native ManiSkill3 EE dataset on GPU with stable IDs.

The official ``replay_trajectory`` utility can reorder batched trajectories and
does not preserve a reliable source-episode mapping for this route.  This
small runner keeps the offline EE actions unchanged, executes them in a
native GPU EE controller, and annotates every recorded group with its source
episode ID.  It is a replay tool, not a controller converter.  The historical
default is 7D ``pd_ee_delta_pose``; native 4D ``pd_ee_delta_pos`` is supported
explicitly as well.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import multiprocessing as mp
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .gpu_ee_online_retarget_probe import _batch_states, _read_json, _relative
from .state_identity import invert_unique_fingerprints, start_state_fingerprint

TASK_ID = "StackCube-v1"
CONTROL_MODE = "pd_ee_delta_pose"
ACTION_DIM = 7
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


def _state_list(group: h5py.Group) -> list[dict[str, Any]]:
    from mani_skill.trajectory import utils as trajectory_utils

    return trajectory_utils.dict_to_list_of_dicts(group["env_states"])


def _replace_initial_recording(recorder: Any, state: Mapping[str, Any]) -> None:
    from mani_skill.utils import common

    if recorder._trajectory_buffer is None:
        return

    def replace(target: Any, replacement: Any) -> None:
        if target is None or replacement is None:
            return
        if isinstance(target, Mapping):
            for key in target:
                if key in replacement:
                    replace(target[key], replacement[key])
            return
        if isinstance(target, np.ndarray) or hasattr(target, "shape"):
            target[-1, ...] = replacement[-1, ...]

    replace(recorder._trajectory_buffer.state, common.batch(state))
    replace(
        recorder._trajectory_buffer.observation,
        common.to_numpy(common.batch(recorder.base_env.get_obs())),
    )


def _project_visual_observation(replay_h5: Path) -> None:
    with h5py.File(replay_h5, "r+") as handle:
        for group in handle.values():
            observation = group.get("obs")
            if observation is None:
                continue
            for forbidden in ("extra", "sensor_param"):
                if forbidden in observation:
                    del observation[forbidden]


def _remove_seed_identity_fields(record: dict[str, Any]) -> None:
    """Remove misleading vector-reset seed fields from derived metadata."""

    record.pop("episode_seed", None)
    reset_kwargs = record.get("reset_kwargs")
    if isinstance(reset_kwargs, dict):
        reset_kwargs.pop("seed", None)


def _worker(
    source_h5: str,
    output_dir: str,
    episode_ids: Sequence[int],
    num_envs: int,
    obs_mode: str,
    source_type: str,
    source_desc: str,
    control_mode: str,
    action_dim: int,
    result_queue: Any,
) -> None:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import torch
        from mani_skill.utils import common
        from mani_skill.utils.wrappers import RecordEpisode

        source_path = Path(source_h5)
        metadata = _read_json(source_path.with_suffix(".json"))
        ids = [int(value) for value in episode_ids]
        if len(ids) > int(num_envs):
            raise ValueError(f"batch size {len(ids)} exceeds num_envs={num_envs}")
        env_kwargs = dict(metadata["env_info"]["env_kwargs"])
        if obs_mode not in {"none", "rgb"}:
            raise ValueError(f"Unsupported live replay obs_mode: {obs_mode}")
        env_kwargs.update(
            obs_mode=obs_mode,
            control_mode=control_mode,
            sim_backend="physx_cuda",
            render_mode="rgb_array",
            reward_mode="sparse",
            reconfiguration_freq=1,
            num_envs=len(ids),
        )
        task_id = str(metadata["env_info"]["env_id"])
        base_env = gym.make(task_id, **env_kwargs)
        recorder = RecordEpisode(
            base_env,
            output_dir=output_dir,
            trajectory_name="replay",
            save_trajectory=True,
            save_video=False,
            save_on_reset=False,
            record_reward=True,
            record_env_state=True,
            source_type=source_type,
            source_desc=source_desc,
        )
        actions_by_id: dict[int, np.ndarray] = {}
        states_by_id: dict[int, list[dict[str, Any]]] = {}
        with h5py.File(source_path, "r") as source_file:
            for episode_id in ids:
                group = source_file[f"traj_{episode_id}"]
                actions = np.asarray(group["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != action_dim:
                    raise ValueError(
                        f"traj_{episode_id} actions must be [T,{action_dim}], got {actions.shape}"
                    )
                states = _state_list(group)
                if len(states) != len(actions) + 1:
                    raise ValueError(f"traj_{episode_id} state/action mismatch")
                actions_by_id[episode_id] = actions
                states_by_id[episode_id] = states

        device = recorder.base_env.unwrapped.device
        # This seed only initializes the temporary GPU env before its exact
        # recorded state is loaded.  It is intentionally not written as an
        # identity key and is not used for cross-file matching.
        reset_seeds = list(range(len(ids)))
        seed_tensor = torch.as_tensor(reset_seeds, dtype=torch.int64, device=device)
        recorder.reset(seed=seed_tensor)
        recorder.base_env.unwrapped.set_state_dict(
            _batch_states([states_by_id[episode_id][0] for episode_id in ids])
        )
        recorder.base_env.unwrapped.scene._gpu_fetch_all()
        _replace_initial_recording(recorder, _batch_states([states_by_id[episode_id][0] for episode_id in ids]))
        lengths = np.asarray([len(actions_by_id[episode_id]) for episode_id in ids], dtype=np.int64)
        max_length = int(lengths.max())
        success_once = np.zeros(len(ids), dtype=bool)
        first_success: list[int | None] = [None] * len(ids)
        final_success = np.zeros(len(ids), dtype=bool)
        flushed = np.zeros(len(ids), dtype=bool)
        for step in range(max_length):
            active = lengths > step
            action = np.zeros((len(ids), action_dim), dtype=np.float32)
            for index, episode_id in enumerate(ids):
                if active[index]:
                    action[index] = actions_by_id[episode_id][step]
            _, _, _, _, info = recorder.step(torch.as_tensor(action, device=device))
            success = common.to_numpy(
                info.get("success", torch.zeros(len(ids), dtype=torch.bool, device=device))
            ).reshape(-1).astype(bool)
            for index, is_active in enumerate(active):
                if not is_active:
                    continue
                if success[index]:
                    success_once[index] = True
                    if first_success[index] is None:
                        first_success[index] = step + 1
                if step == lengths[index] - 1:
                    final_success[index] = bool(success[index])
            to_flush = (step >= lengths - 1) & ~flushed
            if np.any(to_flush):
                indices = np.flatnonzero(to_flush)
                flushed[indices] = True
                recorder.flush_trajectory(env_idxs_to_flush=indices)
        if not np.all(flushed):
            raise RuntimeError(f"Not all replay episodes flushed: {flushed.tolist()}")
        recorder.close()
        replay_h5 = Path(output_dir) / "replay.h5"
        replay_json = replay_h5.with_suffix(".json")
        output_metadata = _read_json(replay_json)
        output_records = output_metadata.get("episodes", [])
        if len(output_records) != len(ids):
            raise RuntimeError(f"Replay metadata count {len(output_records)} != {len(ids)}")
        with h5py.File(source_path, "r") as source_file, h5py.File(replay_h5, "r") as target_file:
            source_fingerprints = {
                episode_id: start_state_fingerprint(source_file[f"traj_{episode_id}"], tolerance=1e-4)
                for episode_id in ids
            }
            source_by_fingerprint = invert_unique_fingerprints(
                source_fingerprints, label="offline EE source batch"
            )
            for record_index, record in enumerate(output_records):
                target_fingerprint = start_state_fingerprint(
                    target_file[f"traj_{int(record['episode_id'])}"], tolerance=1e-4
                )
                source_id = source_by_fingerprint.get(target_fingerprint)
                if source_id is None:
                    raise KeyError(
                        f"GPU replay target {record['episode_id']} has no source start-state match: "
                        f"{target_fingerprint}"
                    )
                record["source_episode_id"] = source_id
                record["source_start_state_fingerprint"] = source_fingerprints[source_id]
                record["target_start_state_fingerprint"] = target_fingerprint
                _remove_seed_identity_fields(record)
            invert_unique_fingerprints(
                {
                    int(record["episode_id"]): str(record["target_start_state_fingerprint"])
                    for record in output_records
                },
                label="GPU replay target batch",
            )
        output_metadata["episodes"] = output_records
        _write_json(replay_json, output_metadata)
        rows = []
        with h5py.File(replay_h5, "r") as target_file:
            for record in output_records:
                group_id = int(record["episode_id"])
                success_series = np.asarray(
                    target_file[f"traj_{group_id}"]["success"], dtype=bool
                ).reshape(-1)
                rows.append(
                    {
                        "target_episode_id": group_id,
                        "source_episode_id": int(record["source_episode_id"]),
                        "source_start_state_fingerprint": record[
                            "source_start_state_fingerprint"
                        ],
                        "target_start_state_fingerprint": record[
                            "target_start_state_fingerprint"
                        ],
                        "success_once": bool(np.any(success_series)),
                        "success_at_end": bool(success_series[-1]),
                        "first_success_step": first_success[
                            ids.index(int(record["source_episode_id"]))
                        ],
                        "action_dim": action_dim,
                    }
                )
        result_queue.put({"ok": True, "replay_h5": str(replay_h5), "rows": rows})
    except BaseException as exc:
        result_queue.put({"ok": False, "error": repr(exc)})


def _run_isolated(
    source_h5: Path,
    output_dir: Path,
    ids: Sequence[int],
    num_envs: int,
    obs_mode: str,
    source_type: str,
    source_desc: str,
    control_mode: str,
    action_dim: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            str(source_h5),
            str(output_dir),
            list(ids),
            int(num_envs),
            str(obs_mode),
            str(source_type),
            str(source_desc),
            str(control_mode),
            int(action_dim),
            queue,
        ),
    )
    process.start()
    result = queue.get()
    process.join()
    if process.exitcode != 0 or not result.get("ok"):
        raise RuntimeError(
            f"GPU offline EE replay failed (exitcode={process.exitcode}): "
            f"{result.get('error', 'unknown error')}"
        )
    return result


def replay(
    *,
    source_h5: str | Path,
    output_dir: str | Path,
    num_envs: int = 100,
    expected_episodes: int | None = None,
    episode_ids: Sequence[int] | None = None,
    obs_mode: str = "rgb",
    source_type: str = "m1_w_gpu_live_baseline_replay",
    source_desc: str = (
        "Live GPU env.step replay of native EE actions; "
        "observations are captured by the same environment execution"
    ),
    required_success_rate: float = 0.0,
    preserve_source_order: bool = False,
    control_mode: str = CONTROL_MODE,
    action_dim: int = ACTION_DIM,
) -> dict[str, Any]:
    source_h5 = Path(source_h5).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_h5.is_file() or not source_h5.with_suffix(".json").is_file():
        raise FileNotFoundError(f"Offline EE H5/JSON pair is missing: {source_h5}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if control_mode not in {"pd_ee_delta_pos", "pd_ee_delta_pose"}:
        raise ValueError(f"Unsupported GPU EE control mode: {control_mode!r}")
    expected_action_dim = 4 if control_mode == "pd_ee_delta_pos" else 7
    if int(action_dim) != expected_action_dim:
        raise ValueError(
            f"{control_mode} requires action_dim={expected_action_dim}, got {action_dim}"
        )
    if obs_mode not in {"none", "rgb"}:
        raise ValueError(f"obs_mode must be 'none' or 'rgb', got {obs_mode!r}")
    if not 0.0 <= float(required_success_rate) <= 1.0:
        raise ValueError("required_success_rate must be in [0, 1]")
    metadata = _read_json(source_h5.with_suffix(".json"))
    available_ids = sorted(int(record["episode_id"]) for record in metadata.get("episodes", []))
    if episode_ids is None:
        ids = available_ids
        if expected_episodes is not None:
            ids = ids[: int(expected_episodes)]
    else:
        ids = [int(value) for value in episode_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("episode_ids must be unique")
        missing = sorted(set(ids) - set(available_ids))
        if missing:
            raise ValueError(f"Requested episode_ids are missing from source: {missing[:10]}")
        if expected_episodes is not None and len(ids) != int(expected_episodes):
            raise ValueError(
                "expected_episodes must equal len(episode_ids) when episode_ids is provided"
            )
    if not ids:
        raise ValueError("Offline EE source contains no episodes")
    with h5py.File(source_h5, "r") as source_file:
        source_fingerprints = {
            episode_id: start_state_fingerprint(
                source_file[f"traj_{episode_id}"], tolerance=1e-4
            )
            for episode_id in ids
        }
        episode_lengths = {
            episode_id: int(source_file[f"traj_{episode_id}"]["actions"].shape[0])
            for episode_id in ids
        }
    invert_unique_fingerprints(source_fingerprints, label="offline EE source")
    # Keep similarly long trajectories in the same CUDA batch by default.  A
    # direct comparison with ManiSkill's official replay utility must preserve
    # its source order and batch boundaries, because GPU PhysX can be
    # sensitive to the vectorized batch in which states are stepped.
    if preserve_source_order:
        execution_ids = list(ids)
        execution_order = "source_episode_order"
    else:
        execution_ids = sorted(
            ids, key=lambda episode_id: (episode_lengths[episode_id], episode_id)
        )
        execution_order = "ascending_episode_length_then_episode_id"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "replay_manifest.json"
    final_h5 = output_dir / "replay.h5"
    if manifest_path.is_file() and final_h5.is_file():
        cached = _read_json(manifest_path)
        if not cached.get("gate_pass"):
            raise RuntimeError(
                f"Existing GPU replay failed its gate; remove and rerun: {output_dir}"
            )
        capture = cached.get("observation_capture", {})
        if capture.get("obs_mode") != obs_mode:
            raise RuntimeError(
                f"Existing replay uses obs_mode={capture.get('obs_mode')!r}, "
                f"requested {obs_mode!r}: remove {output_dir} before rerun"
            )
        if float(cached.get("required_success_rate", 0.0)) != float(required_success_rate):
            raise RuntimeError(
                "Existing replay has a different success gate: "
                f"{cached.get('required_success_rate')} vs {required_success_rate}"
            )
        cached_mode = cached.get("control_mode")
        cached_dim = cached.get("action_dim")
        # Historical 7D manifests predate the explicit contract fields.  They
        # can only be accepted for the historical default; a 4D request must
        # never silently reuse an old 7D artifact.
        if cached_mode is None and int(action_dim) == ACTION_DIM:
            cached_mode = CONTROL_MODE
        if cached_dim is None and int(action_dim) == ACTION_DIM:
            cached_dim = ACTION_DIM
        if cached_mode != control_mode or int(cached_dim or -1) != int(action_dim):
            raise RuntimeError(
                "Existing replay uses a different EE action contract: "
                f"{cached.get('control_mode')}/{cached.get('action_dim')} vs "
                f"{control_mode}/{action_dim}"
            )
        return cached
    if final_h5.exists():
        raise FileExistsError(f"Incomplete GPU EE replay exists: {final_h5}")
    workers_root = output_dir / "batches"
    workers_root.mkdir(parents=True, exist_ok=True)
    results = []
    batch_stats: list[dict[str, int]] = []
    for start in range(0, len(execution_ids), int(num_envs)):
        batch_ids = execution_ids[start : start + int(num_envs)]
        batch_dir = workers_root / f"batch_{start:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_stats.append(
            {
                "start_index": int(start),
                "episode_count": len(batch_ids),
                "total_action_steps": int(sum(episode_lengths[episode_id] for episode_id in batch_ids)),
                "padded_action_steps": int(
                    len(batch_ids) * max(episode_lengths[episode_id] for episode_id in batch_ids)
                    - sum(episode_lengths[episode_id] for episode_id in batch_ids)
                ),
            }
        )
        results.append(
            _run_isolated(
                source_h5,
                batch_dir,
                batch_ids,
                len(batch_ids),
                obs_mode,
                source_type,
                source_desc,
                control_mode,
                int(action_dim),
            )
        )
    from mani_skill.trajectory.merge_trajectory import merge_trajectories

    merge_trajectories(str(final_h5), [item["replay_h5"] for item in results])
    if obs_mode == "rgb":
        _project_visual_observation(final_h5)
    final_json = final_h5.with_suffix(".json")
    merged_metadata = _read_json(final_json)
    merged_records = merged_metadata.get("episodes", [])
    rows = [row for item in results for row in item["rows"]]
    if len(merged_records) != len(rows):
        raise RuntimeError(f"Merged GPU replay metadata count {len(merged_records)} != {len(rows)}")
    # Annotate merged groups by their start-state fingerprint.  Never infer
    # identity from merge order or the destination trajectory number.
    row_by_target_fingerprint = {
        row["target_start_state_fingerprint"]: row
        for item in results
        for row in item["rows"]
    }
    used_fingerprints: set[str] = set()
    with h5py.File(final_h5, "r") as final_file:
        for record in merged_records:
            target_fingerprint = start_state_fingerprint(
                final_file[f"traj_{int(record['episode_id'])}"], tolerance=1e-4
            )
            row = row_by_target_fingerprint.get(target_fingerprint)
            if row is None:
                raise KeyError(
                    f"Merged GPU replay group {record['episode_id']} has no identity match: "
                    f"{target_fingerprint}"
                )
            if target_fingerprint in used_fingerprints:
                raise ValueError(f"Duplicate merged start-state fingerprint: {target_fingerprint}")
            used_fingerprints.add(target_fingerprint)
            record["source_episode_id"] = int(row["source_episode_id"])
            record["source_start_state_fingerprint"] = row[
                "source_start_state_fingerprint"
            ]
            record["target_start_state_fingerprint"] = target_fingerprint
    if len(used_fingerprints) != len(merged_records):
        raise RuntimeError("Merged GPU replay lost or duplicated start-state identities")
    merged_metadata["episodes"] = merged_records
    for record in merged_records:
        _remove_seed_identity_fields(record)
    _write_json(final_json, merged_metadata)
    end_success = 0
    rgb_schema_ok = True
    live_env_state_ok = True
    with h5py.File(final_h5, "r") as handle:
        for record in merged_records:
            group = handle[f"traj_{record['episode_id']}"]
            end_success += int(bool(np.asarray(group["success"], dtype=bool).reshape(-1)[-1]))
            live_env_state_ok = live_env_state_ok and "env_states" in group
            if obs_mode == "rgb":
                observation = group.get("obs")
                if observation is None:
                    rgb_schema_ok = False
                    continue
                names: list[str] = []
                observation.visit(names.append)
                rgb_schema_ok = rgb_schema_ok and any(name.endswith("rgb") for name in names)
                rgb_schema_ok = rgb_schema_ok and any(name.endswith("qpos") for name in names)
                rgb_schema_ok = rgb_schema_ok and any(name.endswith("qvel") for name in names)
    expected = len(ids)
    source_identity_set = set(source_fingerprints.values())
    target_identity_set = set(used_fingerprints)
    identity_gate = (
        len(merged_records) == expected
        and len(target_identity_set) == expected
        and target_identity_set == source_identity_set
    )
    success_gate = end_success / max(1, expected) >= float(required_success_rate)
    gate_pass = identity_gate and success_gate and rgb_schema_ok and live_env_state_ok
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": str(metadata["env_info"]["env_id"]),
        "route": "gpu_live_rgb_replay" if obs_mode == "rgb" else "gpu_live_state_replay",
        "source_h5": _relative(source_h5),
        "replay_h5": _relative(final_h5),
        "replay_json": _relative(final_json),
        "backend": "physx_cuda",
        "control_mode": control_mode,
        "action_dim": int(action_dim),
        "num_envs": int(num_envs),
        "execution_order": execution_order,
        "batch_stats": batch_stats,
        "episode_count": expected,
        "successful_at_end": end_success,
        "success_at_end_rate": end_success / max(1, expected),
        "required_success_rate": float(required_success_rate),
        "failed_source_episode_ids": [
            int(row["source_episode_id"])
            for row in rows
            if not bool(row["success_at_end"])
        ],
        "identity": {
            "key": "start_state_fingerprint from env_states frame 0",
            "seed_is_not_identity": True,
            "source_unique_count": len(source_identity_set),
            "target_unique_count": len(target_identity_set),
            "exact_source_coverage": target_identity_set == source_identity_set,
        },
        "identity_tolerance": 1e-4,
        "observation_capture": {
            "obs_mode": obs_mode,
            "rgb_saved_during_env_step": obs_mode == "rgb",
            "env_states_saved_during_env_step": live_env_state_ok,
            "action_replay_used": True,
            "state_materialization_used": False,
            "controller": f"native GPU {control_mode}",
        },
        "observation_schema_gate": rgb_schema_ok,
        "success_gate": success_gate,
        "gate_pass": gate_pass,
        "episodes": rows,
    }
    _write_json(manifest_path, manifest)
    if not gate_pass:
        raise RuntimeError(
            "GPU live replay gate failed: "
            f"identity={identity_gate}, success={end_success}/{expected} "
            f"(required={required_success_rate}), rgb_schema={rgb_schema_ok}, "
            f"env_states={live_env_state_ok}; inspect {manifest_path}"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--obs-mode", choices=("none", "rgb"), default="rgb")
    parser.add_argument(
        "--control-mode",
        choices=("pd_ee_delta_pos", "pd_ee_delta_pose"),
        default=CONTROL_MODE,
    )
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument(
        "--preserve-source-order",
        action="store_true",
        help="Keep source episode order and batch boundaries for an official replay comparison",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = replay(
        source_h5=args.source_h5,
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        expected_episodes=args.expected_episodes,
        obs_mode=args.obs_mode,
        preserve_source_order=args.preserve_source_order,
        control_mode=args.control_mode,
        action_dim=(
            int(args.action_dim)
            if args.action_dim is not None
            else (4 if args.control_mode == "pd_ee_delta_pos" else 7)
        ),
    )
    print(json.dumps(_jsonable({key: value for key, value in result.items() if key != "episodes"}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
