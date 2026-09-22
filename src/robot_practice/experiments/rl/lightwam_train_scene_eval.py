"""Paired demonstration replay and live-policy rollout from verified training scenes.

The simulator runs in the DPPO Python process; LightWAM stays in its own environment.
Only initialization restores a recorded state. Neither rollout uses teacher forcing.
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)
LOW_DIM = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_action(adapter, obs, seed):
    cond = {k: v[None, None] for k, v in obs.items()}
    raw = adapter.predict_action(cond, seed=seed).float().cpu().numpy()[0]
    if raw.shape != (4, 7) or not np.isfinite(raw).all():
        raise ValueError("Invalid model action")
    return raw, np.clip(raw, -1, 1)


def normalize(value, minimum, maximum):
    return 2 * (value - minimum) / (maximum - minimum + 1e-6) - 1


class DPTrainSceneAdapter:
    """BC model weights, official DDIM sampling, live observations only."""

    def __init__(self, model):
        self.model = model

    def predict_action(self, cond, seed):
        import torch

        # Seed once when loading the model, not again at every control chunk.
        with torch.inference_mode():
            inputs = {k: torch.as_tensor(v, device="cuda:0").float() for k, v in cond.items()}
            return self.model(cond=inputs, deterministic=True).trajectories


def load_dp(checkpoint, vendor, output, data):
    import hydra
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl.dppo_square_official import configure

    vendor = Path(vendor).resolve()
    if not (vendor / "model/diffusion/diffusion_eval.py").is_file():
        raise FileNotFoundError(f"Missing DPPO source: {vendor}")
    sys.path.insert(0, str(vendor))
    cfg = configure(vendor, output, Path(data).parent, "eval", checkpoint.resolve())
    OmegaConf.save(cfg, output / f"{checkpoint.stem}_dp_config.yaml")
    model = hydra.utils.instantiate(cfg.model).eval()
    return model, DPTrainSceneAdapter(model)


def match_demo(actions, demos, minimum, maximum, tolerance=2e-6):
    matches = []
    for name in demos:
        raw = demos[name]["actions"]
        if raw.shape != actions.shape:
            continue
        error = float(np.max(np.abs(normalize(raw[:], minimum, maximum) - actions)))
        if np.isfinite(error) and error <= tolerance:
            matches.append((name, error))
    if len(matches) != 1:
        raise ValueError(f"Expected one full-action-sequence match; found {matches}")
    return matches[0]


def summarize(rewards, successes):
    rewards = np.asarray(rewards)
    complete = len(rewards) // 4 * 4
    return {
        "control_steps": len(rewards),
        "task_success_any_step": bool(np.any(successes)),
        "official_chunk_success": bool(
            complete and np.any(rewards[:complete].reshape(-1, 4).sum(axis=1) >= 4)
        ),
        "first_success_step": next((i + 1 for i, s in enumerate(successes) if s), None),
    }


def worker(args):
    from multiprocessing.connection import Client

    connection = Client(args.socket, family="AF_UNIX", authkey=bytes.fromhex(args.auth))
    env = None
    try:
        import h5py

        sys.path.insert(0, args.dppo_root)
        import robomimic.utils.env_utils as EU
        import robomimic.utils.obs_utils as OU
        from env.gym_utils.wrapper.robomimic_image import RobomimicImageWrapper

        OU.initialize_obs_modality_mapping_from_dict(
            {"low_dim": LOW_DIM, "rgb": ["agentview_image"]}
        )
        with (
            h5py.File(args.hdf5, "r") as h5,
            np.load(args.npz) as data,
            np.load(args.normalization) as norm,
        ):
            lengths = data["traj_lengths"].astype(int)
            starts = np.r_[0, np.cumsum(lengths)]
            actions, states, images = data["actions"], data["states"], data["images"]
            selected = {}
            for trajectory in args.trajectory_ids:
                if trajectory < 0 or trajectory >= len(lengths):
                    raise ValueError(f"Invalid trajectory {trajectory}")
                sl = slice(starts[trajectory], starts[trajectory + 1])
                name, error = match_demo(
                    actions[sl], h5["data"], norm["action_min"], norm["action_max"]
                )
                selected[trajectory] = {
                    "demo": name,
                    "action_max_abs_error": error,
                    "length": int(lengths[trajectory]),
                }
            meta = json.loads(h5["data"].attrs["env_args"])
            if meta["env_name"] != "NutAssemblySquare":
                raise ValueError("Expected NutAssemblySquare source")
            kw = meta["env_kwargs"]
            if (
                kw["control_freq"] != 20
                or kw["controller_configs"].get("control_delta") is not True
            ):
                raise ValueError("Expected 20 Hz delta controller")
            kw.update(
                camera_names=["agentview"],
                camera_heights=96,
                camera_widths=96,
                camera_depths=False,
                reward_shaping=False,
            )
            env = EU.create_env_from_metadata(
                meta, render=False, render_offscreen=True, use_image_obs=True
            )
            env.env.hard_reset = False
            wrapper = RobomimicImageWrapper(
                env,
                shape_meta={"obs": {"rgb": {"shape": [3, 96, 96]}, "state": {"shape": [9]}}},
                normalization_path=args.normalization,
                low_dim_keys=LOW_DIM,
                image_keys=["agentview_image"],
            )

            def reset(trajectory, frame=0):
                demo = h5["data"][selected[trajectory]["demo"]]
                xml = demo.attrs["model_file"]
                if isinstance(xml, bytes):
                    xml = xml.decode()
                return wrapper.get_observation(
                    env.reset_to({"model": xml, "states": demo["states"][frame]})
                )

            connection.send({"ok": True, "value": {"mapping": selected, "env_meta": meta}})
            capture = None
            while True:
                command, value = connection.recv()
                if command == "close":
                    break
                if command == "reset":
                    trajectory = int(value)
                    obs = reset(trajectory)
                    result = obs
                elif command == "reset_at":
                    trajectory, frame = map(int, value)
                    obs = reset(trajectory, frame)
                    demo = h5["data"][selected[trajectory]["demo"]]
                    result = {"obs": obs, "sim_state_max_abs_error": float(np.max(np.abs(
                        env.get_state()["states"] - demo["states"][frame])))}
                elif command == "reset_seed":
                    np.random.seed(int(value))
                    obs = wrapper.get_observation(env.reset())
                    result = obs
                elif command == "dual_observation":
                    from robot_practice.experiments.rl.fastwam_dual_data import render_dual_rgb

                    result = {"state": obs["state"].copy(), "rgb": render_dual_rgb(env)}
                elif command == "capture_start":
                    from robot_practice.experiments.rl.scene_capture import SceneCapture

                    capture = SceneCapture(
                        env, value, obs,
                        diagnostic_views=args.diagnostic_views and Path(value).name != "prefix_replay",
                    )
                    demo = h5["data"][selected[trajectory]["demo"]]
                    xml = demo.attrs["model_file"]
                    if isinstance(xml, bytes):
                        xml = xml.decode()
                    (Path(value) / "source_model.xml").write_text(xml)
                    write_json(Path(value) / "env_meta.json", meta)
                    np.savez_compressed(
                        Path(value) / "demonstration.npz", states=demo["states"][:],
                        actions=demo["actions"][:],
                        **{key: norm[key] for key in norm.files},
                    )
                    result = True
                elif command == "capture_stop":
                    result = capture.finish()
                    capture = None
                elif command == "audit":
                    trajectory = int(value)
                    start = int(starts[trajectory])
                    length = int(lengths[trajectory])
                    close = np.flatnonzero(actions[start : start + length, 6] > 0)
                    frames = sorted(
                        {0, length // 2, max(0, int(close[0]) - 1) if len(close) else length - 1}
                    )
                    checks = []
                    from PIL import Image

                    for frame in frames:
                        obs = reset(trajectory, frame)
                        reference = images[start + frame].astype(np.float32)
                        rgb = obs["rgb"].astype(np.float32)
                        if rgb.shape != reference.shape or rgb.shape != (3, 96, 96):
                            raise ValueError(
                                f"Image shape mismatch: {rgb.shape}, {reference.shape}"
                            )
                        demo = h5["data"][selected[trajectory]["demo"]]
                        sim_error = float(
                            np.max(np.abs(env.get_state()["states"] - demo["states"][frame]))
                        )
                        checks.append(
                            {
                                "frame": frame,
                                "sim_state_max_abs_error": sim_error,
                                "proprio_normalized_max_abs_error": float(
                                    np.max(np.abs(obs["state"] - states[start + frame]))
                                ),
                                "image_mae_0_255": float(np.abs(rgb - reference).mean()),
                                "image_vertical_flip_mae": float(
                                    np.abs(rgb[:, ::-1] - reference).mean()
                                ),
                            }
                        )
                        pair = np.concatenate([reference, rgb], axis=2).transpose(1, 2, 0)
                        Image.fromarray(np.clip(pair, 0, 255).astype(np.uint8)).save(
                            Path(args.output)
                            / f"trajectory_{trajectory}_frame_{frame}_dataset_vs_sim.png"
                        )
                    from OpenGL import GL

                    renderer = GL.glGetString(GL.GL_RENDERER).decode()
                    if not any(x in renderer.lower() for x in ("llvmpipe", "softpipe", "software")):
                        raise ValueError(f"Software renderer not verified: {renderer}")
                    result = {"checks": checks, "renderer": renderer}
                elif command == "step":
                    action, expected_frame = value
                    rewards, successes, drift = [], [], []
                    for offset, a in enumerate(action):
                        if a.shape != (7,) or not np.isfinite(a).all():
                            raise ValueError("Invalid action")
                        obs, reward, _, _ = wrapper.step(a)
                        rewards.append(float(reward))
                        successes.append(bool(env.is_success()["task"]))
                        if capture is not None:
                            capture.step(a, obs, float(reward), successes[-1])
                        if expected_frame is not None:
                            frame = expected_frame + offset + 1
                            demo = h5["data"][selected[trajectory]["demo"]]
                            if frame < len(demo["states"]):
                                drift.append(
                                    float(
                                        np.max(
                                            np.abs(
                                                env.get_state()["states"] - demo["states"][frame]
                                            )
                                        )
                                    )
                                )
                    result = {
                        "obs": obs,
                        "rewards": rewards,
                        "successes": successes,
                        "sim_state_errors": drift,
                    }
                else:
                    raise ValueError(command)
                connection.send({"ok": True, "value": result})
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            connection.send({"ok": False, "error": error})
        except (OSError, EOFError):
            pass
        raise
    finally:
        if env is not None:
            env.env.close()
        connection.close()


class Simulator:
    def __init__(self, args, output, python, root):
        self.args, self.output, self.python, self.root = args, output, python, root

    def __enter__(self):
        from multiprocessing.connection import Listener

        self.temp = tempfile.TemporaryDirectory(prefix="train-scene-")
        socket = str(Path(self.temp.name) / "worker.sock")
        auth = os.urandom(32)
        self.listener = Listener(socket, family="AF_UNIX", authkey=auth)
        self.listener._listener._socket.settimeout(180)
        self.log = (self.output / "env_worker.log").open("w")
        env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
        command = [
            str(self.python),
            "-u",
            "-m",
            __name__,
            "--worker",
            "--socket",
            socket,
            "--auth",
            auth.hex(),
            "--dppo-root",
            str(self.root),
            "--output",
            str(self.output),
            "--hdf5",
            self.args.hdf5,
            "--npz",
            self.args.npz,
            "--normalization",
            self.args.normalization,
            "--trajectory-ids",
            ",".join(map(str, self.args.trajectory_ids)),
        ]
        # __name__ is __main__ when launched with -m.
        command[3] = "robot_practice.experiments.rl.lightwam_train_scene_eval"
        if getattr(self.args, "diagnostic_views", False):
            command.append("--diagnostic-views")
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            self.connection = self.listener.accept()
            self.audit = self.receive()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def receive(self):
        if not self.connection.poll(180):
            raise TimeoutError(f"Simulator timeout; inspect {self.output / 'env_worker.log'}")
        result = self.connection.recv()
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result["value"]

    def call(self, command, value):
        self.connection.send((command, value))
        return self.receive()

    def __exit__(self, *_):
        if hasattr(self, "connection"):
            try:
                self.connection.send(("close", None))
            except (OSError, EOFError):
                pass
            self.connection.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.listener.close()
        self.log.close()
        self.temp.cleanup()


def run(args):
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    from robot_practice.experiments.rl.lightwam_base import (
        LightWAMBaseAdapter,
        RobomimicLightWAMWindowDataset,
        _resolve_dppo_runtime,
        load_official_lightwam,
    )
    from robot_practice.experiments.rl.lightwam_base_open_loop_diagnostic import MODEL_DTYPES

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    cfg = OmegaConf.load(args.config)
    if cfg.lightwam.action_horizon != 4 or cfg.action.dim != 7:
        raise ValueError("Requires 4x7 LightWAM actions")
    checkpoints = [] if args.replay_only else [Path(p) for p in args.checkpoints.split(",") if p]
    if not args.replay_only and not checkpoints:
        raise ValueError("Provide --checkpoints or --replay-only")
    if len({p.stem for p in checkpoints}) != len(checkpoints):
        raise ValueError("Checkpoint names must be unique")
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    result = {
        "diagnostic_only": True,
        "initialization": "training demonstration state and XML; see per-scene gates",
        "policy_observation": "live simulator RGB and proprio only",
        "arguments": vars(args),
        "replay": {},
        "models": {},
        "status": "running",
    }
    result["source_sha256"] = {
        "runner": file_hash(__file__),
        "config": file_hash(args.config),
        "normalization": file_hash(args.normalization),
        "hdf5": file_hash(args.hdf5),
        "npz": file_hash(args.npz),
    }
    result["limitations"] = [
        "Training scenes are diagnostics, not held-out generalization.",
        "Demo uses its recorded length; policy uses max_steps; compare within_policy_budget too.",
        "Image MAE threshold is a configurable screen, not proof of policy-input equivalence.",
        "Excluded scenes are not counted as policy failures; report coverage with success rate.",
        "Task success at any control step and official four-step reward criterion are separate.",
    ]
    LOG.info("START phase=restore completed=0 total=%d backend=osmesa", len(args.trajectory_ids))
    write_json(output / "metrics.json", result)
    with np.load(args.npz) as data:
        actions = data["actions"]
        starts = np.r_[0, np.cumsum(data["traj_lengths"])]
    with Simulator(args, output, python, root) as sim:
        result["source"] = sim.audit

        def episode(trajectory, label, adapter=None):
            obs = sim.call("reset", trajectory)
            begin, end = map(int, starts[trajectory : trajectory + 2])
            total = end - begin if adapter is None else args.max_steps
            directory = output / label / f"trajectory_{trajectory}"
            directory.mkdir(parents=True)
            if getattr(args, "capture_scene", False):
                sim.call("capture_start", str(directory))
            rewards, successes, errors, frames, predictions, observations = [], [], [], [], [], []
            raw_predictions = []
            for step in range(0, total, 4):
                frames.append(
                    Image.fromarray(np.clip(obs["rgb"].transpose(1, 2, 0), 0, 255).astype(np.uint8))
                )
                observations.append(obs["state"].copy())
                if adapter is None:
                    action = actions[begin + step : begin + min(step + 4, total)]
                    raw = action
                else:
                    raw, action = policy_action(adapter, obs, args.seed)
                raw_predictions.extend(raw)
                predictions.extend(action)
                reply = sim.call("step", (action, step if adapter is None else None))
                obs = reply["obs"]
                rewards.extend(reply["rewards"])
                successes.extend(reply["successes"])
                errors.extend(reply["sim_state_errors"])
                if step % 40 == 0 or step + len(action) == total:
                    LOG.info(
                        "STATUS phase=%s trajectory=%d completed=%d total=%d",
                        label,
                        trajectory,
                        step + len(action),
                        total,
                    )
            frames[0].save(
                directory / "rollout.gif",
                save_all=True,
                append_images=frames[1:],
                duration=200,
                loop=0,
            )
            np.savez_compressed(
                directory / "rollout.npz",
                actions=np.asarray(predictions),
                actions_raw=np.asarray(raw_predictions),
                proprio=np.asarray(observations),
                rewards=rewards,
                successes=successes,
                replay_state_errors=errors,
            )
            row = summarize(rewards, successes)
            row["within_policy_budget"] = summarize(
                rewards[: args.max_steps], successes[: args.max_steps]
            )
            row["within_400_steps"] = summarize(rewards[:400], successes[:400])
            row["replay_sim_state_error_max"] = max(errors) if errors else None
            if getattr(args, "capture_scene", False):
                row["capture"] = sim.call("capture_stop", None)
                # Reconstruct by replaying the exact executed prefix, without model inference.
                sim.call("reset", trajectory)
                replay_directory = directory / "prefix_replay"
                sim.call("capture_start", str(replay_directory))
                for step in range(0, len(predictions), 4):
                    sim.call("step", (np.asarray(predictions[step:step + 4]), None))
                    if step % 40 == 0 or step + 4 >= len(predictions):
                        LOG.info("STATUS phase=prefix_verify trajectory=%d completed=%d total=%d",
                                 trajectory, min(step + 4, len(predictions)), len(predictions))
                sim.call("capture_stop", None)
                from robot_practice.experiments.rl.scene_capture import compare_traces

                with np.load(directory / "control_trace.npz") as reference, np.load(
                    replay_directory / "control_trace.npz"
                ) as replay:
                    row["prefix_replay_verification"] = compare_traces(reference, replay)
                LOG.info("DONE phase=prefix_verify trajectory=%d passed=%s",
                         trajectory, row["prefix_replay_verification"]["passed"])
            write_json(directory / "metrics.json", row)
            return row

        eligible = []
        for number, trajectory in enumerate(args.trajectory_ids, 1):
            audit = sim.call("audit", trajectory)
            # Initial-state identity gates live rollout. Later snapshots remain diagnostic.
            first = audit["checks"][0]
            valid = (
                first["sim_state_max_abs_error"] <= 1e-8
                and first["proprio_normalized_max_abs_error"] <= args.state_tolerance
                and first["image_mae_0_255"] <= args.image_mae_tolerance
            )
            row = {"audit": audit, "initial_observation_pass": valid}
            row["rollout"] = episode(trajectory, "demonstration")
            row["eligible"] = valid and row["rollout"]["task_success_any_step"]
            result["replay"][str(trajectory)] = row
            if row["eligible"]:
                eligible.append(trajectory)
            write_json(output / "metrics.json", result)
            LOG.info(
                "DONE phase=replay completed=%d total=%d eligible=%s",
                number,
                len(args.trajectory_ids),
                row["eligible"],
            )
        if not eligible:
            result["status"] = "blocked_restoration_or_demonstration_replay"
            write_json(output / "metrics.json", result)
            raise RuntimeError(
                "No verified training scene; inspect audit pairs and replay metrics before evaluating policies"
            )
        if not args.replay_only and getattr(args, "policy", "lightwam") == "lightwam":
            dataset = RobomimicLightWAMWindowDataset(
                args.npz,
                context_path=cfg.lightwam.context,
                prompt=cfg.lightwam.prompt,
                image_size=cfg.lightwam.image_size,
                action_horizon=4,
                video_frames=cfg.lightwam.video_frames,
                normalized=True,
            )
        if not args.replay_only:
            for checkpoint in checkpoints:
                LOG.info("START phase=load checkpoint=%s", checkpoint.name)
                torch.manual_seed(args.seed)
                if getattr(args, "policy", "lightwam") == "dp":
                    model, adapter = load_dp(checkpoint, args.dp_vendor, output, args.npz)
                elif args.policy == "fastwam":
                    from robot_practice.experiments.rl.fastwam_square_eval import load_fastwam

                    cfg.data.npz = args.npz
                    cfg.evaluation.normalization_path = args.normalization
                    model, adapter, provenance = load_fastwam(checkpoint, cfg)
                    result["fastwam_provenance"] = provenance
                else:
                    model = load_official_lightwam(
                        vendor_root=Path(cfg.lightwam.vendor_root),
                        model_config=Path(cfg.lightwam.model_config),
                        local_model_root=Path(cfg.lightwam.local_model_root),
                        device="cuda:0",
                        model_dtype=MODEL_DTYPES[cfg.lightwam.model_dtype],
                        checkpoint=checkpoint,
                        allow_download=False,
                    ).eval()
                    adapter = LightWAMBaseAdapter(
                        model,
                        action_horizon=4,
                        action_dim=7,
                        image_size=cfg.lightwam.image_size,
                        context=dataset.context,
                        context_mask=dataset.context_mask,
                        num_inference_steps=cfg.lightwam.inference_steps,
                        clip_actions=False,
                    ).eval()
                rows = {}
                for number, trajectory in enumerate(eligible, 1):
                    rows[str(trajectory)] = episode(trajectory, checkpoint.stem, adapter)
                    result["models"][checkpoint.stem] = {
                        "checkpoint": str(checkpoint),
                        "policy": getattr(args, "policy", "lightwam"),
                        "weight_variant": "model",
                        "episodes": rows,
                        "evaluated": len(rows),
                        "requested": len(args.trajectory_ids),
                        "task_success_rate": sum(r["task_success_any_step"] for r in rows.values())
                        / len(rows),
                        "official_chunk_success_rate": sum(
                            r["official_chunk_success"] for r in rows.values()
                        )
                        / len(rows),
                    }
                    write_json(output / "metrics.json", result)
                    LOG.info(
                        "DONE phase=%s completed=%d total=%d",
                        checkpoint.stem,
                        number,
                        len(eligible),
                    )
                del adapter, model
                gc.collect()
                torch.cuda.empty_cache()
        result["status"] = (
            "complete"
            if len(eligible) == len(args.trajectory_ids)
            else "partial_invalid_scenes_excluded"
        )
        if getattr(args, "capture_scene", False):
            captured = [r["rollout"] for r in result["replay"].values()]
            captured += [r for m in result["models"].values() for r in m["episodes"].values()]
            result["capture_verification"] = {
                "passed": sum(r["prefix_replay_verification"]["passed"] for r in captured),
                "total": len(captured),
                "direct_snapshot_resume_verified": False,
            }
            if not all(r["prefix_replay_verification"]["passed"] for r in captured):
                result["status"] = "complete_with_prefix_replay_mismatch"
        write_json(output / "metrics.json", result)
        LOG.info(
            "DONE phase=all valid_scenes=%d total=%d replay_only=%s",
            len(eligible),
            len(args.trajectory_ids),
            args.replay_only,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/rl/lightwam_base_robomimic_square.yaml")
    p.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    p.add_argument("--npz", default="datasets/raw/dppo/robomimic/square-img/train.npz")
    p.add_argument(
        "--normalization", default="datasets/raw/dppo/robomimic/square-img/normalization.npz"
    )
    p.add_argument("--trajectory-ids", default="0,10,20,30,40,50,60,70")
    p.add_argument("--checkpoints", default="")
    p.add_argument("--policy", choices=("lightwam", "dp", "fastwam"), default="lightwam")
    p.add_argument("--capture-scene", action="store_true",
                   help="Record every control step and verify executed-prefix reconstruction")
    p.add_argument("--diagnostic-views", action="store_true",
                   help="Extra native cameras and contacts; not policy inputs")
    p.add_argument("--dp-vendor", default="third_party/dppo_official")
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-tolerance", type=float, default=1e-3)
    p.add_argument("--image-mae-tolerance", type=float, default=10.0)
    p.add_argument("--replay-only", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--socket", help=argparse.SUPPRESS)
    p.add_argument("--auth", help=argparse.SUPPRESS)
    p.add_argument("--dppo-root", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.trajectory_ids = [int(x) for x in args.trajectory_ids.split(",")]
    if args.diagnostic_views and not args.worker and not args.capture_scene:
        p.error("--diagnostic-views requires --capture-scene")
    if not args.trajectory_ids or len(set(args.trajectory_ids)) != len(args.trajectory_ids):
        p.error("Trajectory ids must be nonempty and unique")
    if args.max_steps <= 0 or args.max_steps % 4:
        p.error("max-steps must be positive and divisible by 4")
    if args.worker:
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
