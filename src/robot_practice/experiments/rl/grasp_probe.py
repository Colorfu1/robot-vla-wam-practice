"""Diagnostic-only passive observations at each robosuite control-step boundary."""

import numpy as np

from robot_practice.experiments.rl.rollout_forensics import record


def square_snapshot(env):
    """Passive endpoint evidence, never a policy input or a grasp-success label."""
    nuts = [nut for nut in env.nuts if nut.name == "SquareNut"]
    if len(nuts) != 1:
        raise ValueError("Expected exactly one SquareNut")
    nut, robot = nuts[0], env.robots[0]
    data = env.sim.data
    return {
        "nut_xyz": np.asarray(data.body_xpos[env.obj_body_id[nut.name]]).tolist(),
        "eef_xyz": np.asarray(data.site_xpos[robot.eef_site_id]).tolist(),
        "finger_q": np.asarray(data.qpos[robot._ref_gripper_joint_pos_indexes]).tolist(),
        "bilateral_contact": bool(env._check_grasp(robot.gripper, nut.contact_geoms)),
    }


def snapshot(env):
    """No forward(), controller update, or state mutation; contacts are endpoint-only."""
    model, data = env.sim.model, env.sim.data
    robot = env.robots[0]
    controller = robot.controller
    eef = np.array(data.site_xpos[robot.eef_site_id], copy=True)
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        contacts.append(
            {
                "geom1": model.geom_id2name(contact.geom1),
                "geom2": model.geom_id2name(contact.geom2),
                "distance": float(contact.dist),
            }
        )
    nuts = {}
    for nut in env.nuts:
        position = np.array(data.body_xpos[env.obj_body_id[nut.name]], copy=True)
        handle = np.array(data.site_xpos[model.site_name2id(nut.important_sites["handle"])])
        nuts[nut.name] = {
            "position": position.tolist(),
            "handle": handle.tolist(),
            "handle_minus_eef": (handle - eef).tolist(),
        }
    fields = {}
    for name in (
        "goal_pos",
        "goal_ori",
        "ee_pos",
        "ee_ori_mat",
        "use_delta",
        "input_min",
        "input_max",
        "output_min",
        "output_max",
        "position_limits",
    ):
        value = getattr(controller, name, None)
        fields[name] = None if value is None else np.asarray(value).tolist()
    return {
        "sim_time": float(data.time),
        "eef_xyz": eef.tolist(),
        "gripper_class": type(robot.gripper).__name__,
        "gripper_qpos": data.qpos[robot._ref_gripper_joint_pos_indexes].tolist(),
        "gripper_internal_action": np.asarray(robot.gripper.current_action).tolist(),
        "joint_qpos": data.qpos[robot._ref_joint_pos_indexes].tolist(),
        "joint_limits": model.jnt_range[robot._ref_joint_indexes].tolist(),
        "actuator_ctrl": np.asarray(data.ctrl).tolist(),
        "actuator_ctrlrange": np.asarray(model.actuator_ctrlrange).tolist(),
        "controller_class": type(controller).__name__,
        "controller": fields,
        "nuts": nuts,
        "contacts_endpoint_only": contacts,
    }


def install(directory, worker_index, env_class=None):
    if env_class is None:
        from robosuite.environments.base import MujocoEnv

        env_class = MujocoEnv
    original = env_class.step
    sequence = 0

    def step(env, action):
        nonlocal sequence
        sequence += 1
        before = snapshot(env)
        result = original(env, action)
        after = snapshot(env)
        record(
            directory,
            "grasp_step",
            worker_index=worker_index,
            control_step=sequence,
            chunk=(sequence - 1) // 4,
            substep=(sequence - 1) % 4,
            action=np.asarray(action).tolist(),
            before=before,
            after=after,
        )
        return result

    env_class.step = step
    print(f"[DEBUG-grasp] worker={worker_index} passive control-step probe installed", flush=True)


def main():
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    total = 0
    for path in sorted(args.directory.glob("process_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [row for row in rows if row.get("event") == "grasp_step"]
        if not rows:
            continue
        total += len(rows)
        first = rows[0]["after"]
        print("WORKER", rows[0]["worker_index"], "control_steps", len(rows))
        print("gripper", first["gripper_class"], "controller", first["controller_class"])
        print(
            "controller_config",
            {
                key: first["controller"][key]
                for key in (
                    "use_delta",
                    "input_min",
                    "input_max",
                    "output_min",
                    "output_max",
                    "position_limits",
                )
            },
        )
        for start in range(0, max(row["chunk"] for row in rows) + 1, 10):
            group = [row for row in rows if start <= row["chunk"] < start + 10]
            if not group:
                continue
            pairs = Counter()
            for row in group:
                pairs.update(
                    {
                        tuple(sorted((c["geom1"] or "?", c["geom2"] or "?")))
                        for c in row["after"]["contacts_endpoint_only"]
                    }
                )
            moves = [
                np.linalg.norm(np.subtract(r["after"]["eef_xyz"], r["before"]["eef_xyz"])) * 1000
                for r in group
            ]
            errors = [
                np.linalg.norm(
                    np.subtract(r["after"]["controller"]["goal_pos"], r["after"]["eef_xyz"])
                )
                * 1000
                for r in group
                if r["after"]["controller"]["goal_pos"] is not None
            ]
            grip = [r["action"][-1] for r in group]
            print(
                f"chunks={start:02d}-{start + 9:02d} n={len(group)} "
                f"move_mm/control_step={np.mean(moves):.4f} "
                f"goal_error_mm={np.mean(errors) if errors else None} "
                f"grip_min_max={[min(grip), max(grip)]}"
            )
            print("  contact_endpoint_counts", pairs.most_common(12))
            print(
                "  last_nut_offsets",
                {k: v["handle_minus_eef"] for k, v in group[-1]["after"]["nuts"].items()},
            )
    if not total:
        raise SystemExit("No grasp_step evidence found")
    print(f"DONE grasp evidence control_steps={total}; contacts sampled at step endpoints only")


if __name__ == "__main__":
    main()
