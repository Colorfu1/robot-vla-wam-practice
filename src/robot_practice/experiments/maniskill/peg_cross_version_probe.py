"""P3 matched-state trajectory divergence probe for the Peg task.

The MS2 and MS3 executions are intentionally separate commands.  MS2 is the
reference rollout and decodes each native action into a physical transform;
MS3 consumes the corresponding semantic bridge action.  Initial robot and
task poses are copied explicitly where the two task implementations expose
the same objects.  Geometry is recorded, not silently assumed identical.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.experiments.maniskill.peg_controller_contract import (
    convert_ms2_action_to_ms3,
    rotation_geodesic,
)

MS2_TASK = "PegInsertionSide-v0"
MS3_TASK = "PegInsertionSide-v1"
DEFAULT_SAMPLE_SEED = 20260811
THRESHOLDS = {
    "eef_position_m": 0.005,
    "eef_rotation_rad": 0.05,
    "joint_position_rad": 0.1,
    "peg_head_position_m": 0.005,
    "peg_rotation_rad": 0.05,
}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _array(value: Any, size: int | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64).squeeze()
    if size is not None:
        result = result.reshape(size)
    return result


def _pose(pose: Any) -> dict[str, list[float]]:
    return {"position": _array(pose.p, 3).tolist(), "quaternion_wxyz": _array(pose.q, 4).tolist()}


def _pose_from_dict(value: dict[str, Any], pose_cls: Any) -> Any:
    return pose_cls(value["position"], value["quaternion_wxyz"])


def _scalar_bool(value: Any) -> bool:
    return bool(_array(value).reshape(-1)[0])


def _env_info(base: Any, backend: str) -> dict[str, Any]:
    if backend == "ms2":
        return {
            "task": MS2_TASK,
            "robot_qpos": _array(base.agent.robot.get_qpos()).reshape(-1).tolist(),
            "robot_qvel": _array(base.agent.robot.get_qvel()).reshape(-1).tolist(),
            "eef": _pose(base.tcp.pose),
            "peg": _pose(base.peg.pose),
            "peg_head": _pose(base.peg_head_pose),
            "box": _pose(base.box.pose),
            "box_hole": _pose(base.box_hole_pose),
            "peg_half_size": _array(base.peg_half_size, 3).tolist(),
            "box_hole_radius": float(base.box_hole_radius),
        }
    return {
        "task": MS3_TASK,
        "robot_qpos": _array(base.agent.robot.get_qpos()).reshape(-1).tolist(),
        "robot_qvel": _array(base.agent.robot.get_qvel()).reshape(-1).tolist(),
        "eef": _pose(base.agent.tcp.pose),
        "peg": _pose(base.peg.pose),
        "peg_head": _pose(base.peg_head_pose),
        "box": _pose(base.box.pose),
        "box_hole": _pose(base.box_hole_pose),
        "peg_half_size": _array(base.peg_half_sizes[0], 3).tolist(),
        "box_hole_radius": float(_array(base.box_hole_radii[0])),
    }


def _controller_info(base: Any) -> dict[str, Any]:
    arm = base.agent.controller.controllers["arm"]
    target_pose = getattr(arm, "_target_pose", None)
    return {
        "target_pose": None if target_pose is None else _pose(target_pose),
        "target_qpos": None
        if getattr(arm, "_target_qpos", None) is None
        else _array(arm._target_qpos).reshape(-1).tolist(),
    }


def _evaluate(base: Any, backend: str) -> dict[str, Any]:
    raw = base.evaluate()
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "success":
            result[key] = _scalar_bool(value)
        else:
            result[key] = _jsonable(value)
    if backend == "ms2":
        result["is_grasped"] = _scalar_bool(base.agent.check_grasp(base.peg, max_angle=20))
    else:
        result["is_grasped"] = _scalar_bool(base.agent.is_grasping(base.peg, max_angle=20))
    return result


def _make_env(backend: str) -> Any:
    import gymnasium as gym

    if backend == "ms2":
        import mani_skill2.envs  # noqa: F401

        importlib.import_module("envs.maniskill_fixed")
        return gym.make(
            MS2_TASK,
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            reward_mode="sparse",
            render_mode=None,
        )
    import mani_skill.envs  # noqa: F401

    return gym.make(
        MS3_TASK,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        render_mode=None,
    )


def _reset_ms2(env: Any, record: dict[str, Any]) -> None:
    options = dict(record.get("reset_kwargs", {}).get("options", {}))
    env.reset(seed=int(record["episode_seed"]), options=options)


def _reset_ms3(env: Any, episode: dict[str, Any], seed: int) -> None:
    import sapien

    base = env.unwrapped
    env.reset(seed=seed)
    source_initial = episode["initial"]
    source_peg_head = _pose_from_dict(source_initial["peg_head"], sapien.Pose)
    source_hole = _pose_from_dict(source_initial["box_hole"], sapien.Pose)
    base.agent.robot.set_qpos(np.asarray(source_initial["robot_qpos"], dtype=np.float64)[None, :])
    base.agent.robot.set_qvel(np.asarray(source_initial["robot_qvel"], dtype=np.float64)[None, :])

    # Align the task's physically relevant peg head and hole frames.  The
    # underlying mesh dimensions may differ across versions and are recorded
    # as a partial state match rather than silently overwritten.
    peg_head_offset = sapien.Pose(
        _array(base.peg_head_offsets[0].p, 3), _array(base.peg_head_offsets[0].q, 4)
    )
    box_hole_offset = sapien.Pose(
        _array(base.box_hole_offsets[0].p, 3), _array(base.box_hole_offsets[0].q, 4)
    )
    base.peg.set_pose(source_peg_head * peg_head_offset.inv())
    base.box.set_pose(source_hole * box_hole_offset.inv())


def _select_records(metadata: dict[str, Any], handle: h5py.File) -> tuple[list[int], list[int]]:
    records = {int(item["episode_id"]): item for item in metadata["episodes"]}
    lengths = {
        episode_id: int(handle[f"traj_{episode_id}"]["actions"].shape[0])
        for episode_id in records
    }
    ordered = sorted(records, key=lambda episode_id: (lengths[episode_id], episode_id))
    simple = ordered[:5]
    long = ordered[-5:]
    return simple, long


def _source_step(base: Any, env: Any, action: np.ndarray, before: dict[str, Any]) -> dict[str, Any]:
    bridge = convert_ms2_action_to_ms3(
        action,
        before["eef"]["position"],
        before["eef"]["quaternion_wxyz"],
    )
    env.step(action.astype(np.float32))
    after = _env_info(base, "ms2")
    return {
        "action_ms2": action.tolist(),
        "bridge": bridge,
        "after": after,
        "controller": _controller_info(base),
        "evaluate": _evaluate(base, "ms2"),
    }


def run_ms2(dataset_h5: Path, dataset_json: Path, output: Path) -> None:
    metadata = _read_json(dataset_json)
    env = _make_env("ms2")
    episodes: list[dict[str, Any]] = []
    try:
        with h5py.File(dataset_h5, "r") as handle:
            simple, long = _select_records(metadata, handle)
            selected = [("simple", episode_id) for episode_id in simple] + [
                ("long", episode_id) for episode_id in long
            ]
            records = {int(item["episode_id"]): item for item in metadata["episodes"]}
            for ordinal, (group_name, episode_id) in enumerate(selected, start=1):
                record = records[episode_id]
                _reset_ms2(env, record)
                base = env.unwrapped
                initial = _env_info(base, "ms2")
                actions = np.asarray(handle[f"traj_{episode_id}"]["actions"], dtype=np.float32)
                timesteps: list[dict[str, Any]] = []
                for step, action in enumerate(actions, start=1):
                    before = _env_info(base, "ms2")
                    current = _source_step(base, env, action, before)
                    timesteps.append({"step": step, "before": before, **current})
                    if current["evaluate"].get("success"):
                        break
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "episode_seed": int(record["episode_seed"]),
                        "group": group_name,
                        "dataset_length": len(actions),
                        "executed_steps": len(timesteps),
                        "dataset_final_success": bool(np.asarray(handle[f"traj_{episode_id}"]["success"])[-1]),
                        "initial": initial,
                        "timesteps": timesteps,
                    }
                )
                print(
                    f"[MS2 {ordinal:02d}/{len(selected)}] episode={episode_id} "
                    f"group={group_name} steps={len(timesteps)}/{len(actions)} "
                    f"success={int(bool(timesteps[-1]['evaluate'].get('success')))}",
                    flush=True,
                )
    finally:
        env.close()
    _write_json(
        output,
        {
            "schema_version": 1,
            "diagnostic": "peg_cross_version_trajectory_probe",
            "backend": "ms2",
            "source_dataset": dataset_h5.as_posix(),
            "control_mode": "pd_ee_delta_pose",
            "selected_policy": "five shortest trajectories + five longest trajectories",
            "thresholds": THRESHOLDS,
            "episodes": episodes,
        },
    )


def _target_step(base: Any, env: Any, action: list[float]) -> dict[str, Any]:
    env.step(np.asarray(action, dtype=np.float32))
    return {
        "after": _env_info(base, "ms3"),
        "controller": _controller_info(base),
        "evaluate": _evaluate(base, "ms3"),
    }


def _state_match(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_qpos_max_abs_error": float(
            np.max(np.abs(np.asarray(source["robot_qpos"]) - np.asarray(target["robot_qpos"])))
        ),
        "robot_qvel_max_abs_error": float(
            np.max(np.abs(np.asarray(source["robot_qvel"]) - np.asarray(target["robot_qvel"])))
        ),
        "eef_position_error_m": float(
            np.linalg.norm(np.asarray(source["eef"]["position"]) - np.asarray(target["eef"]["position"]))
        ),
        "eef_rotation_error_rad": rotation_geodesic(
            source["eef"]["quaternion_wxyz"], target["eef"]["quaternion_wxyz"]
        ),
        "peg_head_position_error_m": float(
            np.linalg.norm(
                np.asarray(source["peg_head"]["position"])
                - np.asarray(target["peg_head"]["position"])
            )
        ),
        "box_hole_position_error_m": float(
            np.linalg.norm(
                np.asarray(source["box_hole"]["position"])
                - np.asarray(target["box_hole"]["position"])
            )
        ),
        "geometry_peg_half_size_max_abs_error": float(
            np.max(np.abs(np.asarray(source["peg_half_size"]) - np.asarray(target["peg_half_size"])))
        ),
        "geometry_box_hole_radius_abs_error": abs(
            float(source["box_hole_radius"]) - float(target["box_hole_radius"])
        ),
    }


def _divergence(source: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    return {
        "eef_position_m": float(
            np.linalg.norm(
                np.asarray(source["eef"]["position"]) - np.asarray(target["eef"]["position"])
            )
        ),
        "eef_rotation_rad": rotation_geodesic(
            source["eef"]["quaternion_wxyz"], target["eef"]["quaternion_wxyz"]
        ),
        "joint_position_rad": float(
            np.linalg.norm(
                np.asarray(source["robot_qpos"][:7]) - np.asarray(target["robot_qpos"][:7])
            )
        ),
        "peg_position_m": float(
            np.linalg.norm(
                np.asarray(source["peg"]["position"]) - np.asarray(target["peg"]["position"])
            )
        ),
        "peg_head_position_m": float(
            np.linalg.norm(
                np.asarray(source["peg_head"]["position"])
                - np.asarray(target["peg_head"]["position"])
            )
        ),
        "peg_rotation_rad": rotation_geodesic(
            source["peg"]["quaternion_wxyz"], target["peg"]["quaternion_wxyz"]
        ),
    }


def run_ms3(source_path: Path, output: Path, seed: int) -> None:
    source = _read_json(source_path)
    env = _make_env("ms3")
    episodes: list[dict[str, Any]] = []
    try:
        for ordinal, source_episode in enumerate(source["episodes"], start=1):
            _reset_ms3(env, source_episode, seed + ordinal)
            base = env.unwrapped
            initial = _env_info(base, "ms3")
            match = _state_match(source_episode["initial"], initial)
            timesteps: list[dict[str, Any]] = []
            for source_timestep in source_episode["timesteps"]:
                result = _target_step(base, env, source_timestep["bridge"]["encoded_ms3"]["action"])
                timesteps.append({"step": source_timestep["step"], **result})
                if result["evaluate"].get("success"):
                    break
            rows: list[dict[str, Any]] = []
            for source_timestep, target_timestep in zip(source_episode["timesteps"], timesteps):
                divergence = _divergence(source_timestep["after"], target_timestep["after"])
                row = {
                    "step": source_timestep["step"],
                    "source_action_ms2": source_timestep["action_ms2"],
                    "converted_action_ms3": source_timestep["bridge"]["encoded_ms3"]["action"],
                    "source": source_timestep,
                    "target": target_timestep,
                    "divergence": divergence,
                    "source_success": bool(source_timestep["evaluate"].get("success")),
                    "target_success": bool(target_timestep["evaluate"].get("success")),
                }
                row["diverged"] = any(
                    divergence[key] > THRESHOLDS[key] for key in THRESHOLDS
                )
                rows.append(row)
            first_divergence = next((row["step"] for row in rows if row["diverged"]), None)
            episodes.append(
                {
                    "episode_id": source_episode["episode_id"],
                    "episode_seed": source_episode["episode_seed"],
                    "group": source_episode["group"],
                    "state_match": match,
                    "state_match_partial": bool(
                        match["geometry_peg_half_size_max_abs_error"] > 1e-8
                        or match["geometry_box_hole_radius_abs_error"] > 1e-8
                    ),
                    "source_executed_steps": source_episode["executed_steps"],
                    "target_executed_steps": len(timesteps),
                    "source_final_success": bool(rows[-1]["source_success"]) if rows else False,
                    "target_final_success": bool(rows[-1]["target_success"]) if rows else False,
                    "first_divergence_step": first_divergence,
                    "max_divergence": {
                        key: max((float(row["divergence"][key]) for row in rows), default=0.0)
                        for key in THRESHOLDS
                    },
                    "rows": rows,
                }
            )
            print(
                f"[MS3 {ordinal:02d}/{len(source['episodes'])}] episode={source_episode['episode_id']} "
                f"state_partial={int(episodes[-1]['state_match_partial'])} "
                f"first_divergence={first_divergence} "
                f"success={int(episodes[-1]['target_final_success'])}",
                flush=True,
            )
    finally:
        env.close()
    _write_json(
        output,
        {
            "schema_version": 1,
            "diagnostic": "peg_cross_version_trajectory_probe",
            "backend": "ms3",
            "reference": source_path.as_posix(),
            "control_mode": "pd_ee_delta_pose",
            "thresholds": THRESHOLDS,
            "episodes": episodes,
        },
    )


def summarize(ms2_path: Path, ms3_path: Path, output: Path) -> None:
    ms2 = _read_json(ms2_path)
    ms3 = _read_json(ms3_path)
    episodes = []
    for source, target in zip(ms2["episodes"], ms3["episodes"]):
        episodes.append(
            {
                "episode_id": source["episode_id"],
                "group": source["group"],
                "state_match_partial": target["state_match_partial"],
                "state_match": target["state_match"],
                "source_final_success": target["source_final_success"],
                "target_final_success": target["target_final_success"],
                "first_divergence_step": target["first_divergence_step"],
                "max_divergence": target["max_divergence"],
            }
        )
    _write_json(
        output,
        {
            "schema_version": 1,
            "diagnostic": "peg_cross_version_trajectory_probe",
            "thresholds": THRESHOLDS,
            "summary": {
                "num_episodes": len(episodes),
                "source_success": sum(item["source_final_success"] for item in episodes),
                "target_success": sum(item["target_final_success"] for item in episodes),
                "num_partial_state_matches": sum(item["state_match_partial"] for item in episodes),
            },
            "episodes": episodes,
        },
    )
    print(json.dumps({"num_episodes": len(episodes), "target_success": sum(item["target_final_success"] for item in episodes)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("ms2", "ms3"))
    parser.add_argument("--dataset-h5", type=Path)
    parser.add_argument("--dataset-json", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--ms2", type=Path)
    parser.add_argument("--ms3", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--phase", choices=("trajectory", "summary"), default="trajectory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "summary":
        if args.ms2 is None or args.ms3 is None:
            raise SystemExit("summary requires --ms2 and --ms3")
        summarize(args.ms2, args.ms3, args.output)
        return 0
    if args.backend == "ms2":
        if args.dataset_h5 is None or args.dataset_json is None:
            raise SystemExit("MS2 trajectory requires --dataset-h5 and --dataset-json")
        run_ms2(args.dataset_h5, args.dataset_json, args.output)
    elif args.backend == "ms3":
        if args.source is None:
            raise SystemExit("MS3 trajectory requires --source")
        run_ms3(args.source, args.output, args.seed)
    else:
        raise SystemExit("trajectory requires --backend ms2 or --backend ms3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
