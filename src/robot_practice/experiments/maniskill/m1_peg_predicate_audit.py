"""M1 PegInsertionSide native-vs-strict success predicate audit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .corrected_envs import CORRECTED_PEG_ENV_ID, register_corrected_peg_env
from .m1_protocol_audit import (
    SCHEMA_VERSION,
    _read_json,
    _success_series,
    _write_json,
    task_config,
    task_root,
)
from .success_predicates import PEG_INSERTION_LOWER_X, peg_insertion_strict


def _geometry_probes() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for half_length in (0.085, 0.125):
        for radius in (0.015, 0.025):
            delta = 1e-4
            values = (
                ("x_lower", PEG_INSERTION_LOWER_X - delta, False),
                ("x_lower", PEG_INSERTION_LOWER_X, True),
                ("x_lower", PEG_INSERTION_LOWER_X + delta, True),
                ("x_upper", half_length - delta, True),
                ("x_upper", half_length, True),
                ("x_upper", half_length + delta, False),
                ("y_radius", radius - delta, True),
                ("y_radius", radius, True),
                ("y_radius", radius + delta, False),
                ("z_radius", radius - delta, True),
                ("z_radius", radius, True),
                ("z_radius", radius + delta, False),
            )
            for axis, value, expected in values:
                relative = np.zeros(3, dtype=np.float64)
                relative[0 if axis.startswith("x") else 1 if axis == "y_radius" else 2] = value
                observed = bool(peg_insertion_strict(relative, radius, half_length))
                cases.append(
                    {
                        "half_length": half_length,
                        "radius": radius,
                        "axis": axis,
                        "value": value,
                        "expected": expected,
                        "observed": observed,
                        "passed": observed == expected,
                    }
                )
    return {
        "boundary_delta_m": 1e-4,
        "cases": cases,
        "num_cases": len(cases),
        "gate_pass": all(case["passed"] for case in cases),
    }


def _state_list(group: h5py.Group) -> list[dict[str, Any]]:
    from mani_skill.trajectory import utils as trajectory_utils

    if "env_states" not in group:
        raise RuntimeError(f"{group.name} has no env_states; Peg continuation cannot be audited")
    return trajectory_utils.dict_to_list_of_dicts(group["env_states"])


def _make_env(env_id: str, control_mode: str) -> Any:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        env_id,
        obs_mode="none",
        control_mode=control_mode,
        sim_backend="physx_cpu",
        reward_mode="sparse",
        reconfiguration_freq=1,
    )


def _evaluate_state(env: Any, state: Mapping[str, Any]) -> tuple[bool, np.ndarray | None]:
    env.unwrapped.set_state_dict(state)
    info = env.unwrapped.evaluate()
    success_value = info.get("success", False)
    success = bool(np.asarray(success_value).reshape(-1)[0])
    relative = info.get("peg_head_pos_at_hole")
    if relative is None:
        relative_value = None
    else:
        relative_value = np.asarray(relative, dtype=np.float64).reshape(-1, 3)[0]
    return success, relative_value


def _reset_for_episode(env: Any, record: Mapping[str, Any]) -> None:
    """Reset with the episode seed before restoring its dynamic state.

    ``env_states`` stores actor/articulation poses, but the randomized peg and
    hole geometry is created during reset and is not encoded in those arrays.
    The native and strict environments therefore must be reset with the same
    episode seed before ``set_state_dict``; otherwise the strict upper bound is
    evaluated against a different peg length/radius.
    """

    reset_kwargs = dict(record.get("reset_kwargs", {}))
    seed = record.get("episode_seed", reset_kwargs.get("seed"))
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(f"Expected one CPU seed, got {seed!r}")
        seed = seed[0]
    if seed is None:
        raise ValueError(f"Episode {record.get('episode_id')} has no seed")
    reset_kwargs["seed"] = int(seed)
    env.reset(**reset_kwargs)


def _replay_confusion(config: dict[str, Any], root: Path) -> dict[str, Any]:
    replay = root / "replay_full" / "replay.h5"
    replay_json = replay.with_suffix(".json")
    if not replay.is_file() or not replay_json.is_file():
        raise FileNotFoundError(f"Peg full replay is required: {replay}")
    metadata = _read_json(replay_json)
    records = {int(item["episode_id"]): item for item in metadata["episodes"]}
    control_mode = str(task_config(config, "PegInsertionSide-v1")["target_control_mode"])
    native_env = _make_env("PegInsertionSide-v1", control_mode)
    strict_env_id = register_corrected_peg_env()
    strict_env = _make_env(strict_env_id, control_mode)
    confusion = {
        "native_true_strict_true": 0,
        "native_true_strict_false": 0,
        "native_false_strict_true": 0,
        "native_false_strict_false": 0,
    }
    rows: list[dict[str, Any]] = []
    strict_labels: dict[str, list[bool]] = {}
    reachable_rows: list[dict[str, Any]] = []
    try:
        with h5py.File(replay, "r") as handle:
            for name in sorted(handle.keys(), key=lambda value: int(value.split("_")[1])):
                episode_id = int(name.split("_")[1])
                group = handle[name]
                native_series = _success_series(group, records[episode_id])
                states = _state_list(group)
                _reset_for_episode(native_env, records[episode_id])
                _reset_for_episode(strict_env, records[episode_id])
                state_offset = 1 if len(states) == len(native_series) + 1 else 0
                strict_series: list[bool] = []
                relative_series: list[list[float] | None] = []
                for index in range(len(native_series)):
                    state = states[min(index + state_offset, len(states) - 1)]
                    strict_value, relative = _evaluate_state(strict_env, state)
                    strict_series.append(strict_value)
                    relative_series.append(None if relative is None else relative.tolist())
                    key = ("native_true_" if bool(native_series[index]) else "native_false_") + (
                        "strict_true" if strict_value else "strict_false"
                    )
                    confusion[key] += 1
                strict_labels[str(episode_id)] = strict_series
                native_indices = np.flatnonzero(native_series)
                native_only = np.flatnonzero(
                    np.asarray(native_series, dtype=bool) & ~np.asarray(strict_series, dtype=bool)
                )
                rows.append(
                    {
                        "episode_id": episode_id,
                        "native_first_success_step": None
                        if native_indices.size == 0
                        else int(native_indices[0]) + 1,
                        "strict_first_success_step": next(
                            (index + 1 for index, value in enumerate(strict_series) if value), None
                        ),
                        "native_only_steps": [int(index) + 1 for index in native_only],
                        "relative_at_steps": relative_series,
                    }
                )
                if native_indices.size:
                    start_index = int(native_indices[0])
                    if start_index < len(states):
                        native_env.reset(
                            seed=records[episode_id].get(
                                "episode_seed",
                                records[episode_id].get("reset_kwargs", {}).get("seed"),
                            )
                        )
                        strict_env.reset(
                            seed=records[episode_id].get(
                                "episode_seed",
                                records[episode_id].get("reset_kwargs", {}).get("seed"),
                            )
                        )
                        start_state = states[min(start_index + state_offset, len(states) - 1)]
                        native_env.unwrapped.set_state_dict(start_state)
                        strict_env.unwrapped.set_state_dict(start_state)
                        continuation_native_only = 0
                        actions = np.asarray(group["actions"], dtype=np.float32)
                        for action in actions[start_index + 1 :]:
                            _, _, _, _, native_info = native_env.step(action)
                            _, _, _, _, strict_info = strict_env.step(action)
                            native_success = bool(
                                np.asarray(native_info.get("success", False)).reshape(-1)[0]
                            )
                            strict_success = bool(
                                np.asarray(strict_info.get("success", False)).reshape(-1)[0]
                            )
                            continuation_native_only += int(native_success and not strict_success)
                        reachable_rows.append(
                            {
                                "episode_id": episode_id,
                                "start_native_success_step": start_index + 1,
                                "continuation_steps": max(0, len(actions) - start_index - 1),
                                "native_only_continuation_steps": continuation_native_only,
                                "action_reachable_native_only": continuation_native_only > 0
                                or bool(native_only.size),
                            }
                        )
    finally:
        native_env.close()
        strict_env.close()
    return {
        "confusion_matrix": confusion,
        "episodes": rows,
        "strict_labels": strict_labels,
        "reachable_continuations": reachable_rows,
        "official_native_only_count": confusion["native_true_strict_false"],
        "reachable_native_only_count": sum(
            row["native_only_continuation_steps"] for row in reachable_rows
        ),
    }


def audit_peg_predicate(
    config: dict[str, Any], *, override_output: str | Path | None = None
) -> dict[str, Any]:
    task = "PegInsertionSide-v1"
    root = task_root(config, task, override_output) / "success_predicate"
    root.mkdir(parents=True, exist_ok=True)
    geometry = _geometry_probes()
    confusion = _replay_confusion(config, root.parent)
    false_positive = int(confusion["official_native_only_count"])
    reachable = int(confusion["reachable_native_only_count"])
    frozen_env_id = CORRECTED_PEG_ENV_ID if false_positive > 0 or reachable > 0 else task
    contract = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "native_predicate": "ManiSkill3 PegInsertionSideEnv.has_peg_inserted",
        "strict_predicate": "-0.015 <= peg_head_pos_at_hole.x <= peg_half_sizes.x and y/z within box_hole_radii",
        "frozen_env_id": frozen_env_id,
        "official_native_only_count": false_positive,
        "reachable_native_only_count": reachable,
        "decision": "strict_repo_local" if frozen_env_id == CORRECTED_PEG_ENV_ID else "native",
        "gate_pass": bool(geometry["gate_pass"]),
    }
    _write_json(root / "geometry_probe.json", geometry)
    _write_json(
        root / "confusion_matrix.json",
        {key: value for key, value in confusion.items() if key != "strict_labels"},
    )
    _write_json(root / "strict_labels.json", confusion["strict_labels"])
    _write_json(
        root / "reachable_continuations.json",
        {"rows": confusion["reachable_continuations"], "native_only_count": reachable},
    )
    _write_json(root / "predicate_contract.json", contract)
    if not contract["gate_pass"]:
        raise RuntimeError("Peg geometry predicate probes failed")
    return contract
