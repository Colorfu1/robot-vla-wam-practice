"""P2 no-contact controller-contract probe for ManiSkill2 and ManiSkill3.

The command is deliberately split into backend-specific invocations because
ManiSkill2 and ManiSkill3 cannot safely share one Python process.  The MS2
probe decodes the native action and writes the independent MS3 action bridge;
the MS3 probe consumes that bridge.  A final compare phase computes realized
motion errors and the pre-registered gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robot_practice.experiments.maniskill.peg_controller_contract import (
    convert_ms2_action_to_ms3,
    pose_delta,
)

TASK_IDS = {"ms2": "PegInsertionSide-v0", "ms3": "PegInsertionSide-v1"}
DEFAULT_SEED = 20260811
ZERO_REPEATS = 12


def _jsonable(value: Any) -> Any:
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


def _cases() -> list[dict[str, Any]]:
    nominal = np.asarray(
        [0.0, math.pi / 8, 0.0, -math.pi * 5 / 8, 0.0, math.pi * 3 / 4, -math.pi / 4, 0.04, 0.04],
        dtype=np.float64,
    )
    yaw_pitch = np.asarray(
        [0.019001769, 0.3896719, -0.002064377, -1.9552834, 0.002880872, 2.3852799, -0.7701774, 0.04, 0.04],
        dtype=np.float64,
    )
    pre_insert = np.asarray(
        [0.030360445, 0.5772996, 0.001473363, -1.8519377, -0.000725951, 2.4270263, -0.8690654, 0.02208951, 0.02208406],
        dtype=np.float64,
    )
    states = {"S0_nominal": nominal, "S1_yaw_pitch": yaw_pitch, "S2_pre_insertion": pre_insert}
    commands: list[tuple[str, np.ndarray]] = []
    for name, index in (("x", 0), ("y", 1), ("z", 2)):
        for sign in (1.0, -1.0):
            action = np.zeros(7, dtype=np.float64)
            action[index] = sign * 0.1  # 0.01 m with the MS2 +/-0.1 m bound
            commands.append((f"{'+' if sign > 0 else '-'}{name}", action))
    for name, index in (("rx", 3), ("ry", 4), ("rz", 5)):
        for sign in (1.0, -1.0):
            action = np.zeros(7, dtype=np.float64)
            action[index] = sign * 0.2  # 0.02 rad with the MS2 0.1 rad bound
            commands.append((f"{'+' if sign > 0 else '-'}{name}", action))
    rows: list[dict[str, Any]] = []
    for state_name, qpos in states.items():
        for command_name, action in commands:
            rows.append(
                {
                    "state_name": state_name,
                    "qpos": qpos.tolist(),
                    "command": command_name,
                    "action_ms2": action.tolist(),
                }
            )
    return rows


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).squeeze()


def _pose_snapshot(position: Any, quaternion: Any) -> dict[str, list[float]]:
    return {"position": _as_array(position).reshape(3).tolist(), "quaternion_wxyz": _as_array(quaternion).reshape(4).tolist()}


def _snapshot(base: Any, backend: str) -> dict[str, Any]:
    if backend == "ms2":
        tcp_pose = base.tcp.pose
        qpos = base.agent.robot.get_qpos()
        qvel = base.agent.robot.get_qvel()
        return {
            "eef": _pose_snapshot(tcp_pose.p, tcp_pose.q),
            "qpos": _as_array(qpos).reshape(-1).tolist(),
            "qvel": _as_array(qvel).reshape(-1).tolist(),
        }
    tcp_pose = base.agent.tcp.pose
    qpos = base.agent.robot.get_qpos()
    qvel = base.agent.robot.get_qvel()
    return {
        "eef": _pose_snapshot(tcp_pose.p, tcp_pose.q),
        "qpos": _as_array(qpos).reshape(-1).tolist(),
        "qvel": _as_array(qvel).reshape(-1).tolist(),
    }


def _controller_snapshot(base: Any, backend: str) -> dict[str, Any]:
    arm = base.agent.controller.controllers["arm"]
    target_pose = getattr(arm, "_target_pose", None)
    if target_pose is None:
        target = None
    else:
        target = _pose_snapshot(target_pose.p, target_pose.q)
    target_qpos = getattr(arm, "_target_qpos", None)
    return {
        "target_pose": target,
        "target_qpos": None if target_qpos is None else _as_array(target_qpos).reshape(-1).tolist(),
        "backend": backend,
    }


def _make_env(backend: str) -> Any:
    import gymnasium as gym

    if backend == "ms2":
        import importlib

        import mani_skill2.envs  # noqa: F401

        importlib.import_module("envs.maniskill_fixed")
        return gym.make(
            TASK_IDS[backend],
            obs_mode="state",
            control_mode="pd_ee_delta_pose",
            reward_mode="sparse",
            render_mode=None,
        )
    import mani_skill.envs  # noqa: F401

    return gym.make(
        TASK_IDS[backend],
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        render_mode=None,
    )


def _move_obstacles_away(base: Any, backend: str) -> None:
    if backend == "ms2":
        from sapien.core import Pose
    else:
        from sapien import Pose

    base.peg.set_pose(Pose([10.0, 10.0, 10.0]))
    base.box.set_pose(Pose([-10.0, -10.0, 10.0]))


def _prepare(base: Any, env: Any, backend: str, qpos: list[float], seed: int) -> None:
    env.reset(seed=seed)
    _move_obstacles_away(base, backend)
    qpos_array = np.asarray(qpos, dtype=np.float64)
    if backend == "ms2":
        base.agent.robot.set_qpos(qpos_array)
        base.agent.robot.set_qvel(np.zeros_like(qpos_array))
    else:
        base.agent.robot.set_qpos(qpos_array[None, :])
        base.agent.robot.set_qvel(np.zeros((1, qpos_array.size), dtype=np.float64))


def _run_step(env: Any, action: list[float]) -> None:
    env.step(np.asarray(action, dtype=np.float32))


def run_backend(backend: str, phase: str, output: Path, input_path: Path | None, seed: int) -> None:
    env = _make_env(backend)
    base = env.unwrapped
    rows: list[dict[str, Any]] = []
    try:
        if phase == "baseline":
            zero = np.zeros(7, dtype=np.float64).tolist()
            for case_index, case in enumerate(_cases()[: 3]):
                for repeat in range(ZERO_REPEATS):
                    _prepare(base, env, backend, case["qpos"], seed + case_index * 100 + repeat)
                    before = _snapshot(base, backend)
                    _run_step(env, zero)
                    after = _snapshot(base, backend)
                    rows.append(
                        {
                            "state_name": case["state_name"],
                            "repeat": repeat,
                            "before": before,
                            "after": after,
                            "realized": pose_delta(
                                before["eef"]["position"],
                                before["eef"]["quaternion_wxyz"],
                                after["eef"]["position"],
                                after["eef"]["quaternion_wxyz"],
                            ),
                        }
                    )
        elif phase == "axis":
            if backend == "ms2":
                for row_index, case in enumerate(_cases()):
                    _prepare(base, env, backend, case["qpos"], seed + row_index)
                    before = _snapshot(base, backend)
                    bridge = convert_ms2_action_to_ms3(
                        case["action_ms2"],
                        before["eef"]["position"],
                        before["eef"]["quaternion_wxyz"],
                    )
                    _run_step(env, case["action_ms2"])
                    after = _snapshot(base, backend)
                    rows.append(
                        {
                            **case,
                            "before": before,
                            "after": after,
                            "controller": _controller_snapshot(base, backend),
                            "realized": pose_delta(
                                before["eef"]["position"],
                                before["eef"]["quaternion_wxyz"],
                                after["eef"]["position"],
                                after["eef"]["quaternion_wxyz"],
                            ),
                            "bridge": bridge,
                        }
                    )
            else:
                if input_path is None:
                    raise ValueError("MS3 axis phase requires --input")
                source_rows = _read_json(input_path)["rows"]
                for row_index, source in enumerate(source_rows):
                    _prepare(base, env, backend, source["qpos"], seed + row_index)
                    before = _snapshot(base, backend)
                    ms3_action = source["bridge"]["encoded_ms3"]["action"]
                    _run_step(env, ms3_action)
                    after = _snapshot(base, backend)
                    rows.append(
                        {
                            "state_name": source["state_name"],
                            "qpos": source["qpos"],
                            "command": source["command"],
                            "action_ms2": source["action_ms2"],
                            "action_ms3": ms3_action,
                            "before": before,
                            "after": after,
                            "controller": _controller_snapshot(base, backend),
                            "realized": pose_delta(
                                before["eef"]["position"],
                                before["eef"]["quaternion_wxyz"],
                                after["eef"]["position"],
                                after["eef"]["quaternion_wxyz"],
                            ),
                            "bridge": source["bridge"],
                        }
                    )
        else:
            raise ValueError(f"Unsupported backend phase: {phase}")
    finally:
        env.close()
    _write_json(
        output,
        {
            "schema_version": 1,
            "diagnostic": "peg_controller_contract_probe",
            "backend": backend,
            "task": TASK_IDS[backend],
            "phase": phase,
            "control_mode": "pd_ee_delta_pose",
            "sim_backend": "physx_cpu",
            "zero_repeats": ZERO_REPEATS if phase == "baseline" else None,
            "rows": rows,
        },
    )
    print(f"wrote {output} rows={len(rows)}", flush=True)


def calibrate_tolerance(ms2_path: Path, ms3_path: Path, output: Path) -> None:
    payloads = [_read_json(ms2_path), _read_json(ms3_path)]
    position_noise = [
        float(row["realized"]["position_norm_m"])
        for payload in payloads
        for row in payload["rows"]
    ]
    rotation_noise = [
        float(row["realized"]["rotation_norm_rad"])
        for payload in payloads
        for row in payload["rows"]
    ]
    max_position_noise = max(position_noise, default=0.0)
    max_rotation_noise = max(rotation_noise, default=0.0)
    tolerance = {
        "schema_version": 1,
        "source": "12 zero-action no-contact repeats across S0/S1/S2 in both backends",
        "max_zero_action_position_noise_m": max_position_noise,
        "max_zero_action_rotation_noise_rad": max_rotation_noise,
        "position_error_tolerance_m": max(0.003, 10.0 * max_position_noise),
        "rotation_error_tolerance_rad": max(0.03, 10.0 * max_rotation_noise),
        "direction_cosine_min": 0.5,
        "scale_ratio_min": 0.25,
        "scale_ratio_max": 4.0,
    }
    _write_json(output, tolerance)
    print(json.dumps(tolerance, indent=2), flush=True)


def _direction_cosine(expected: np.ndarray, realized: np.ndarray) -> float:
    expected_norm = float(np.linalg.norm(expected))
    realized_norm = float(np.linalg.norm(realized))
    if expected_norm < 1e-12 or realized_norm < 1e-12:
        return 0.0
    return float(np.dot(expected, realized) / expected_norm / realized_norm)


def compare(ms2_path: Path, ms3_path: Path, tolerance_path: Path, output: Path) -> None:
    ms2 = _read_json(ms2_path)
    ms3 = _read_json(ms3_path)
    tolerance = _read_json(tolerance_path)
    if len(ms2["rows"]) != len(ms3["rows"]):
        raise ValueError("MS2/MS3 row counts differ")
    rows: list[dict[str, Any]] = []
    for left, right in zip(ms2["rows"], ms3["rows"]):
        if (left["state_name"], left["command"]) != (right["state_name"], right["command"]):
            raise ValueError("MS2/MS3 row ordering differs")
        expected = left["bridge"]["decoded_ms2"]
        expected_position = np.asarray(expected["delta_position_root_m"], dtype=np.float64)
        expected_rotation = np.asarray(expected["delta_rotation_root_rad"], dtype=np.float64)
        ms2_realized = left["realized"]
        ms3_realized = right["realized"]
        ms2_position = np.asarray(ms2_realized["position_root_m"], dtype=np.float64)
        ms3_position = np.asarray(ms3_realized["position_root_m"], dtype=np.float64)
        ms2_rotation = np.asarray(ms2_realized["rotation_root_rad"], dtype=np.float64)
        ms3_rotation = np.asarray(ms3_realized["rotation_root_rad"], dtype=np.float64)
        position_error = float(np.linalg.norm(ms2_position - ms3_position))
        rotation_error = float(np.linalg.norm((
            Rotation.from_rotvec(ms2_rotation).inv() * Rotation.from_rotvec(ms3_rotation)
        ).as_rotvec()))
        row = {
            "state_name": left["state_name"],
            "command": left["command"],
            "expected_delta_position_root_m": expected_position.tolist(),
            "expected_delta_rotation_root_rad": expected_rotation.tolist(),
            "ms2": ms2_realized,
            "ms3": ms3_realized,
            "position_error_ms2_vs_ms3_m": position_error,
            "rotation_error_ms2_vs_ms3_rad": rotation_error,
            "ms2_position_direction_cosine": _direction_cosine(expected_position, ms2_position),
            "ms3_position_direction_cosine": _direction_cosine(expected_position, ms3_position),
            "ms2_rotation_direction_cosine": _direction_cosine(expected_rotation, ms2_rotation),
            "ms3_rotation_direction_cosine": _direction_cosine(expected_rotation, ms3_rotation),
            "position_scale_ratio_ms3_over_ms2": float(
                np.linalg.norm(ms3_position) / max(np.linalg.norm(ms2_position), 1e-12)
            ),
            "rotation_scale_ratio_ms3_over_ms2": float(
                np.linalg.norm(ms3_rotation) / max(np.linalg.norm(ms2_rotation), 1e-12)
            ),
        }
        is_rotation_command = left["command"].endswith(("rx", "ry", "rz"))
        row["checked_dimension"] = "rotation" if is_rotation_command else "translation"
        shared_checks = [
            row["position_error_ms2_vs_ms3_m"] <= tolerance["position_error_tolerance_m"],
            row["rotation_error_ms2_vs_ms3_rad"] <= tolerance["rotation_error_tolerance_rad"],
        ]
        if is_rotation_command:
            command_checks = [
                row["ms2_rotation_direction_cosine"] >= tolerance["direction_cosine_min"],
                row["ms3_rotation_direction_cosine"] >= tolerance["direction_cosine_min"],
                tolerance["scale_ratio_min"]
                <= row["rotation_scale_ratio_ms3_over_ms2"]
                <= tolerance["scale_ratio_max"],
            ]
        else:
            command_checks = [
                row["ms2_position_direction_cosine"] >= tolerance["direction_cosine_min"],
                row["ms3_position_direction_cosine"] >= tolerance["direction_cosine_min"],
                tolerance["scale_ratio_min"]
                <= row["position_scale_ratio_ms3_over_ms2"]
                <= tolerance["scale_ratio_max"],
            ]
        row["pass"] = bool(
            all(shared_checks)
            and all(command_checks)
        )
        rows.append(row)
    result = {
        "schema_version": 1,
        "diagnostic": "peg_controller_contract_probe",
        "gate": {
            "num_commands": len(rows),
            "num_pass": int(sum(bool(row["pass"]) for row in rows)),
            "all_pass": all(bool(row["pass"]) for row in rows),
            "tolerance": tolerance,
            "criteria": "all 36 state-axis commands pass direction, cross-version error, and scale checks",
        },
        "rows": rows,
    }
    _write_json(output, result)
    print(json.dumps(result["gate"], indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "axis", "calibrate", "compare"), required=True)
    parser.add_argument("--backend", choices=("ms2", "ms3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--ms2", type=Path)
    parser.add_argument("--ms3", type=Path)
    parser.add_argument("--tolerance", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase in {"baseline", "axis"}:
        if args.backend is None:
            raise SystemExit("--backend is required for baseline/axis")
        run_backend(args.backend, args.phase, args.output, args.input, args.seed)
    elif args.phase == "calibrate":
        if args.ms2 is None or args.ms3 is None:
            raise SystemExit("--ms2 and --ms3 are required for calibrate")
        calibrate_tolerance(args.ms2, args.ms3, args.output)
    else:
        if args.ms2 is None or args.ms3 is None or args.tolerance is None:
            raise SystemExit("--ms2, --ms3, and --tolerance are required for compare")
        compare(args.ms2, args.ms3, args.tolerance, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
