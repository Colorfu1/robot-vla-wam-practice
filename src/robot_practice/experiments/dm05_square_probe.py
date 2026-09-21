"""Square zero-shot DM05 probe: prepare aligned views, offline inference, or rollout.

Run with robot-practice Python. The DM05 server uses its separate environment.
Example: python src/robot_practice/experiments/dm05_square_probe.py --mode prepare
Then --mode offline; optionally --mode rollout against the same localhost server.
"""

import argparse
import base64
import io
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np


def normalize(value, minimum, maximum):
    return 2 * (value - minimum) / (maximum - minimum + 1e-6) - 1


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_scene(args, trajectory):
    import h5py
    import robosuite

    with np.load(args.npz) as data, np.load(args.normalization) as norm:
        lengths = data["traj_lengths"].astype(int)
        if not 0 <= trajectory < len(lengths):
            raise ValueError("Invalid trajectory index")
        start = int(lengths[:trajectory].sum())
        length = int(lengths[trajectory])
        actions = data["actions"][start : start + length]
        images = data["images"][start : start + length].transpose(0, 2, 3, 1)
        states = data["states"][start : start + length]
        norms = {k: norm[k] for k in norm.files}
    with h5py.File(args.hdf5, "r") as source:
        matches = []
        for name, demo in source["data"].items():
            if demo["actions"].shape != actions.shape:
                continue
            error = np.max(
                np.abs(
                    normalize(demo["actions"][:], norms["action_min"], norms["action_max"])
                    - actions
                )
            )
            if error <= 2e-6:
                matches.append((name, float(error)))
        if len(matches) != 1:
            raise ValueError(f"Expected unique full-action match, got {len(matches)}")
        name, match_error = matches[0]
        demo = source["data"][name]
        sim_states, raw_actions = demo["states"][:], demo["actions"][:]
        xml = demo.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode()
        meta = json.loads(source["data"].attrs["env_args"])
    kw = meta["env_kwargs"]
    if (
        meta["env_name"] != "NutAssemblySquare"
        or kw["control_freq"] != 20
        or not kw["controller_configs"]["control_delta"]
    ):
        raise ValueError("Requires Square 20Hz delta OSC_POSE")
    kw.update(
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=96,
        camera_widths=96,
        camera_depths=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        ignore_done=True,
        reward_shaping=False,
    )
    env = robosuite.make(meta["env_name"], **kw)
    try:
        env.reset_from_xml_string(env.edit_model_xml(xml))
        env.sim.reset()
    except BaseException:
        env.close()
        raise

    def restore(frame):
        env.sim.set_state_from_flattened(sim_states[frame])
        env.sim.forward()
        return env._get_observations(force_update=True)

    def check(frame, obs):
        rgb = np.ascontiguousarray(obs["agentview_image"][::-1])
        state = np.concatenate(
            [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
        )
        errors = {
            "sim_state_error": float(
                np.max(np.abs(env.sim.get_state().flatten() - sim_states[frame]))
            ),
            "normalized_state_error": float(
                np.max(np.abs(normalize(state, norms["obs_min"], norms["obs_max"]) - states[frame]))
            ),
            "image_mae": float(np.mean(np.abs(rgb.astype(float) - images[frame]))),
        }
        if (
            errors["sim_state_error"] > 1e-6
            or errors["normalized_state_error"] > 1e-3
            or errors["image_mae"] > 10
        ):
            raise ValueError(f"Scene restoration gate failed: {errors}")
        return errors

    return (
        env,
        restore,
        check,
        images,
        raw_actions,
        {
            "trajectory": trajectory,
            "demo": name,
            "length": length,
            "action_match_error": match_error,
        },
    )


def state8(obs):
    from robosuite.utils.transform_utils import quat2axisangle

    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"].copy()),
            obs["robot0_gripper_qpos"],
        ]
    )


def predict(session, args, head, wrist, state, seed):
    from PIL import Image

    def encode(rgb):
        stream = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode()

    start = time.perf_counter()
    response = session.post(
        args.url,
        json={
            "observation": {
                "images": {"1": encode(head), "2": encode(wrist)},
                "state": state.tolist(),
                "prompt": args.prompt,
                "robot_type": "Franka",
            },
            "sampling": {"seed": seed},
        },
        timeout=180,
    )
    response.raise_for_status()
    actions = np.asarray(response.json()["actions"], dtype=np.float32)
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise ValueError("Invalid DM05 action response")
    return actions, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "offline", "rollout"], required=True)
    parser.add_argument("--npz", default="datasets/raw/dppo/robomimic/square-img/train.npz")
    parser.add_argument(
        "--normalization", default="datasets/raw/dppo/robomimic/square-img/normalization.npz"
    )
    parser.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    parser.add_argument("--trajectory-ids", default="0,25,50,75")
    parser.add_argument("--frames-per-trajectory", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--url", default="http://127.0.0.1:7891/v1/infer")
    parser.add_argument(
        "--prompt", default="pick up the square nut and place it onto the square peg"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/dm05/square_probe"))
    args = parser.parse_args()
    if args.frames_per_trajectory < 1 or args.max_steps < 1:
        parser.error("frames and max-steps must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    def status(line):
        print(line, flush=True)
        with (args.output / f"{args.mode}_progress.log").open("a") as f:
            f.write(line + "\n")

    trajectories = [int(x) for x in args.trajectory_ids.split(",")]
    if args.mode == "prepare":
        heads, wrists, states, targets, sample_ids, gates, mapping = [], [], [], [], [], [], []
        total = len(trajectories) * args.frames_per_trajectory
        for trajectory in trajectories:
            env, restore, check, images, actions, info = make_scene(args, trajectory)
            mapping.append(info)
            try:
                frames = np.linspace(0, len(actions) - 10, args.frames_per_trajectory, dtype=int)
                if len(actions) < 10 or len(set(frames)) != len(frames):
                    raise ValueError("Too few complete, unique action windows")
                for frame in frames:
                    obs = restore(frame)
                    gates.append(check(frame, obs))
                    heads.append(images[frame])
                    wrists.append(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]))
                    states.append(state8(obs))
                    targets.append(actions[frame : frame + 10])
                    sample_ids.append([trajectory, int(frame)])
                    status(f"STATUS samples={len(heads)}/{total}")
            finally:
                env.close()
        np.savez_compressed(
            args.output / "samples.npz",
            head=np.stack(heads),
            wrist=np.stack(wrists),
            state=np.stack(states),
            target=np.stack(targets),
            sample_ids=np.array(sample_ids),
        )
        write_json(
            args.output / "preparation.json",
            {
                "category": "Smoke",
                "mapping": mapping,
                "samples": total,
                "gates": gates,
                "image_transform": "vertical_flip",
                "head_source": "original NPZ",
                "wrist_source": "restored HDF5 simulation",
            },
        )
        status(f"DONE samples={total}/{total}")
        return

    import requests

    session = requests.Session()
    session.trust_env = False
    if args.mode == "offline":
        with np.load(args.output / "samples.npz") as data:
            target = data["target"]
            predictions, latencies = [], []
            for i in range(len(target)):
                actions, latency = predict(
                    session,
                    args,
                    data["head"][i],
                    data["wrist"][i],
                    data["state"][i],
                    args.seed + i,
                )
                predictions.append(actions)
                latencies.append(latency)
                status(f"STATUS samples={i + 1}/{len(target)} latency_ms={latency:.1f}")
        pred = np.stack(predictions)
        with np.load(args.npz) as train, np.load(args.normalization) as norm:
            mean_action = (train["actions"].mean(0) + 1) / 2 * (
                norm["action_max"] - norm["action_min"] + 1e-6
            ) + norm["action_min"]

        def mse(a, b):
            return float(np.mean((a - b) ** 2))

        result = {
            "category": "Offline / Smoke",
            "checkpoint": "DM05-libero",
            "adaptation": "zero-shot, unchanged LIBERO normalization",
            "samples": len(target),
            "horizon": 10,
            "action_dim": 7,
            "prompt": args.prompt,
            "seed": args.seed,
            "action_units": "raw robosuite controller input, before physical scaling",
            "mse": mse(pred, target),
            "mae": float(np.mean(np.abs(pred - target))),
            "eef_mse": mse(pred[..., :6], target[..., :6]),
            "gripper_mse": mse(pred[..., 6], target[..., 6]),
            "per_dim_mse": np.mean((pred - target) ** 2, axis=(0, 1)).tolist(),
            "per_horizon_mse": np.mean((pred - target) ** 2, axis=(0, 2)).tolist(),
            "clipped_mse": mse(np.clip(pred, -1, 1), target),
            "out_of_range_fraction": float(np.mean(np.abs(pred) > 1)),
            "zero_baseline_mse": mse(np.zeros_like(target), target),
            "train_mean_baseline_mse": mse(mean_action, target),
            "zero_baseline_eef_mse": mse(np.zeros_like(target[..., :6]), target[..., :6]),
            "train_mean_baseline_eef_mse": mse(mean_action[:6], target[..., :6]),
            "gripper_sign_agreement": float(np.mean((pred[..., 6] > 0) == (target[..., 6] > 0))),
            "mean_latency_ms": float(np.mean(latencies)),
            "warm_mean_latency_ms": float(np.mean(latencies[2:])) if len(latencies) > 2 else None,
            "closed_loop_success_rate": None,
        }
        write_json(args.output / "offline_metrics.json", result)
        np.savez_compressed(args.output / "predictions.npz", prediction=pred, target=target)
        status(f"DONE samples={len(target)}/{len(target)} mse={result['mse']:.6f}")
    else:
        import imageio.v2 as imageio

        env, restore, check, _, _, info = make_scene(args, trajectories[0])
        try:
            obs = restore(0)
            gates = check(0, obs)
            if env._check_success():
                raise ValueError("Initial state already successful")
            queue, latencies, frames, actions_log = deque(), [], [], []
            success, calls = False, 0
            for step in range(args.max_steps):
                head = np.ascontiguousarray(obs["agentview_image"][::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
                frames.append(np.concatenate([head, wrist], axis=1))
                if not queue:
                    actions, latency = predict(
                        session, args, head, wrist, state8(obs), args.seed + calls
                    )
                    queue.extend(np.clip(actions, -1, 1))
                    latencies.append(latency)
                    calls += 1
                action = queue.popleft()
                actions_log.append(action)
                obs, _, _, _ = env.step(action)
                success = bool(env._check_success())
                if (step + 1) % 10 == 0 or success:
                    status(
                        f"STATUS episodes=0/1 steps={step + 1}/{args.max_steps} success={success}"
                    )
                if success:
                    break
            imageio.mimsave(args.output / "rollout.mp4", frames, fps=20)
            np.save(args.output / "rollout_actions.npy", np.asarray(actions_log))
            write_json(
                args.output / "rollout_metrics.json",
                {
                    "category": "Closed-loop / Smoke",
                    "checkpoint": "DM05-libero",
                    "initialization": "matched training demonstration frame 0",
                    "mapping": info,
                    "gates": gates,
                    "episodes": 1,
                    "success": success,
                    "steps": step + 1,
                    "max_steps": args.max_steps,
                    "success_criterion": "environment _check_success at any step",
                    "control_frequency_hz": 20,
                    "action_horizon": 10,
                    "prompt": args.prompt,
                    "seed": args.seed,
                    "mean_http_latency_ms": float(np.mean(latencies)),
                },
            )
            status(f"DONE episodes=1/1 success={success}")
        finally:
            env.close()
    session.close()


if __name__ == "__main__":
    main()
