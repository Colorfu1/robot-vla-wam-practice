"""Detailed source/B/C trajectory comparison for Peg source episode 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_practice.experiments.maniskill.peg_controller_contract import rotation_geodesic
from robot_practice.experiments.maniskill.peg_conversion_probe import (
    ABSOLUTE_MODE,
    FIXED_DELTA_MODE,
    _absolute_arm_action,
    _action_dict_for_target,
    _base_pose_components,
    _desired_world_pose,
    _metadata_records,
    _read_json,
    _scalar_bool,
    _state_list,
    fixed_delta_action,
    pose_error,
)

SOURCE_EPISODE_ID = 2
POSITION_DIVERGENCE_THRESHOLD_M = 0.005
ROTATION_DIVERGENCE_THRESHOLD_RAD = 0.05
JOINT_TARGET_DIVERGENCE_THRESHOLD_RAD = 0.1


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


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).squeeze()


def _pose(pose: Any) -> dict[str, list[float]]:
    return {"position": _array(pose.p).reshape(3).tolist(), "quaternion_wxyz": _array(pose.q).reshape(4).tolist()}


def _snapshot(base: Any) -> dict[str, Any]:
    return {
        "qpos": _array(base.agent.robot.get_qpos()).reshape(-1).tolist(),
        "qvel": _array(base.agent.robot.get_qvel()).reshape(-1).tolist(),
        "eef": _pose(base.agent.tcp.pose),
        "peg": _pose(base.peg.pose),
        "box_hole": _pose(base.box_hole_pose),
    }


def _pose_error_between(desired: dict[str, Any], actual: dict[str, Any]) -> dict[str, float]:
    position = np.asarray(desired["position"]) - np.asarray(actual["position"])
    return {
        "position_m": float(np.linalg.norm(position)),
        "rotation_rad": rotation_geodesic(
            desired["quaternion_wxyz"], actual["quaternion_wxyz"]
        ),
    }


def _target_pose_world(target_arm: Any) -> dict[str, Any]:
    target_pose = target_arm.root_link.pose * target_arm._target_pose
    return _pose(target_pose)


def _run_mode(source_h5: Path, mode: str) -> dict[str, Any]:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils import common

    metadata = _read_json(source_h5.with_suffix(".json"))
    records = _metadata_records(metadata)
    record = records[SOURCE_EPISODE_ID]
    source_env_kwargs = dict(metadata["env_info"]["env_kwargs"])
    source_env_kwargs.update(
        obs_mode="none",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        reward_mode="sparse",
        reconfiguration_freq=1,
    )
    target_env_kwargs = dict(source_env_kwargs)
    target_env_kwargs["control_mode"] = "pd_ee_delta_pose" if mode == FIXED_DELTA_MODE else "pd_ee_pose"
    source_env = gym.make("PegInsertionSide-v1", **source_env_kwargs)
    target_env = gym.make("PegInsertionSide-v1", **target_env_kwargs)
    rows: list[dict[str, Any]] = []
    try:
        with h5py.File(source_h5, "r") as handle:
            source_group = handle[f"traj_{SOURCE_EPISODE_ID}"]
            states = _state_list(source_group)
            actions = np.asarray(source_group["actions"], dtype=np.float32)
            reset_kwargs = dict(record.get("reset_kwargs", {}))
            source_env.reset(**reset_kwargs)
            target_env.reset(**reset_kwargs)
            source_env.unwrapped.set_state_dict(states[0])
            target_env.unwrapped.set_state_dict(states[0])

            source_controller = source_env.unwrapped.agent.controller
            target_controller = target_env.unwrapped.agent.controller
            source_arm = source_controller.controllers["arm"]
            target_arm = target_controller.controllers["arm"]
            pin_model = source_controller.articulation.create_pinocchio_model()

            for step, source_action in enumerate(actions, start=1):
                source_action = np.asarray(source_action, dtype=np.float32).reshape(-1)
                source_action_dict = source_controller.to_action_dict(source_action)
                source_env.step(
                    common.to_tensor(source_action, device=source_env.unwrapped.device)
                )
                desired_world_pose = _desired_world_pose(
                    source_controller, source_arm, target_arm, pin_model
                )
                source_target_pose = _pose(desired_world_pose)
                source_target_qpos = _array(source_arm._target_qpos).reshape(-1)
                source_actual = _snapshot(source_env.unwrapped)
                source_eval = _jsonable(source_env.unwrapped.evaluate())

                desired_position, desired_quaternion = _base_pose_components(
                    target_arm, desired_world_pose
                )
                inner_steps = 0
                converged = False
                target_eval: dict[str, Any] = {}
                target_action_log: list[list[float]] = []
                while True:
                    current_pose = target_arm.ee_pose_at_base
                    current_position = _array(current_pose.p).reshape(3)
                    current_quaternion = _array(current_pose.q).reshape(4)
                    if mode == FIXED_DELTA_MODE:
                        arm_action, action_info = fixed_delta_action(
                            desired_position,
                            desired_quaternion,
                            current_position,
                            current_quaternion,
                        )
                    else:
                        arm_action = _absolute_arm_action(desired_position, desired_quaternion)
                        _, _, before_position_error, before_rotation_error = pose_error(
                            desired_position,
                            desired_quaternion,
                            current_position,
                            current_quaternion,
                        )
                        action_info = {
                            "position_error_m": before_position_error,
                            "rotation_error_rad": before_rotation_error,
                        }
                    target_action = _action_dict_for_target(
                        target_controller, source_action_dict, arm_action
                    )
                    target_env.step(target_action)
                    target_action_log.append(_array(arm_action).reshape(-1).tolist())
                    inner_steps += 1
                    target_eval = _jsonable(target_env.unwrapped.evaluate())
                    next_pose = target_arm.ee_pose_at_base
                    _, _, after_position_error, after_rotation_error = pose_error(
                        desired_position,
                        desired_quaternion,
                        _array(next_pose.p).reshape(3),
                        _array(next_pose.q).reshape(4),
                    )
                    if mode == ABSOLUTE_MODE or (
                        after_position_error <= 0.002 and after_rotation_error <= 0.01
                    ):
                        converged = True
                        break
                    if inner_steps >= 16:
                        break

                target_actual = _snapshot(target_env.unwrapped)
                target_target_pose = _target_pose_world(target_arm)
                target_target_qpos = _array(target_arm._target_qpos).reshape(-1)
                fk_error = _pose_error_between(source_target_pose, target_actual["eef"])
                actual_error = _pose_error_between(source_actual["eef"], target_actual["eef"])
                rows.append(
                    {
                        "step": step,
                        "source_action": source_action.tolist(),
                        "inner_steps": inner_steps,
                        "inner_converged": converged,
                        "source_actual": source_actual,
                        "source_target": {
                            "eef": source_target_pose,
                            "qpos": source_target_qpos.tolist(),
                        },
                        "converted_actual": target_actual,
                        "converted_target": {
                            "eef": target_target_pose,
                            "qpos": target_target_qpos.tolist(),
                        },
                        "source_evaluate": source_eval,
                        "converted_evaluate": target_eval,
                        "target_action_first": target_action_log[0],
                        "target_action_last": target_action_log[-1],
                        "fk_realization_error": fk_error,
                        "actual_cross_env_error": actual_error,
                        "target_qpos_error_rad": float(
                            np.linalg.norm(target_target_qpos[:7] - source_target_qpos[:7])
                        ),
                        "action_info": action_info,
                    }
                )
    finally:
        source_env.close()
        target_env.close()

    first_fk_divergence = next(
        (
            row["step"]
            for row in rows
            if row["fk_realization_error"]["position_m"] > POSITION_DIVERGENCE_THRESHOLD_M
            or row["fk_realization_error"]["rotation_rad"] > ROTATION_DIVERGENCE_THRESHOLD_RAD
        ),
        None,
    )
    return {
        "mode": mode,
        "source_episode_id": SOURCE_EPISODE_ID,
        "source_seed": int(record["episode_seed"]),
        "source_action_steps": len(actions),
        "source_final_success": _scalar_bool(rows[-1]["source_evaluate"].get("success")),
        "converted_final_success": _scalar_bool(rows[-1]["converted_evaluate"].get("success")),
        "first_fk_realization_divergence_step": first_fk_divergence,
        "max_fk_realization_position_m": max(
            row["fk_realization_error"]["position_m"] for row in rows
        ),
        "max_fk_realization_rotation_rad": max(
            row["fk_realization_error"]["rotation_rad"] for row in rows
        ),
        "max_target_qpos_error_rad": max(row["target_qpos_error_rad"] for row in rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-h5",
        type=Path,
        default=Path("datasets/raw/maniskill_demos/PegInsertionSide-v1/motionplanning/trajectory.h5"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {
        "schema_version": 1,
        "diagnostic": "peg_episode2_source_b_c",
        "thresholds": {
            "fk_position_m": POSITION_DIVERGENCE_THRESHOLD_M,
            "fk_rotation_rad": ROTATION_DIVERGENCE_THRESHOLD_RAD,
            "target_qpos_rad": JOINT_TARGET_DIVERGENCE_THRESHOLD_RAD,
        },
        "modes": {
            FIXED_DELTA_MODE: _run_mode(args.source_h5, FIXED_DELTA_MODE),
            ABSOLUTE_MODE: _run_mode(args.source_h5, ABSOLUTE_MODE),
        },
    }
    _write_json(args.output, result)
    print(json.dumps({mode: {
        key: value for key, value in payload.items() if key != "rows"
    } for mode, payload in result["modes"].items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
