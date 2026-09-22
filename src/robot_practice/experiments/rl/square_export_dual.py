"""Render native dual-view training frames from matched demonstration states, no actions."""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--npz", default="datasets/raw/dppo/robomimic/square-img/train.npz")
    p.add_argument("--hdf5", default="datasets/raw/robomimic/square/mh/demo_v141.hdf5")
    p.add_argument("--normalization", default="datasets/raw/dppo/robomimic/square-img/normalization.npz")
    p.add_argument("--max-trajectories", type=int, default=100, help="Use 1 only for a separate smoke export")
    p.add_argument("--render-backend", choices=("osmesa", "egl"), default="osmesa")
    p.add_argument("--skip-renderer-query", action="store_true",
                   help="Diagnostic control: skip PyOpenGL glGetString, not rendering")
    args = p.parse_args()
    os.environ.update(MUJOCO_GL=args.render_backend, PYOPENGL_PLATFORM=args.render_backend)
    import h5py
    import robomimic.utils.env_utils as EU
    import robomimic.utils.obs_utils as OU
    from PIL import Image

    from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL, render_dual_rgb
    from robot_practice.experiments.rl.lightwam_train_scene_eval import (
        LOW_DIM,
        file_hash,
        match_demo,
        normalize,
        write_json,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                        logging.FileHandler(args.output / "run.log")])
    result = {"status": "running", "protocol": PROTOCOL, "mapping": {},
              "requested_backend": args.render_backend,
              "renderer_query": "skipped" if args.skip_renderer_query else "requested",
              "egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
              "source_sha256": {k: file_hash(getattr(args, k)) for k in ("npz", "hdf5", "normalization")},
              "note": "Each state independently restored; no actions executed; no random rollout"}
    write_json(args.output / "manifest.json", result)
    env = None
    try:
        with np.load(args.npz) as source, np.load(args.normalization) as norm, h5py.File(args.hdf5) as h5:
            lengths = source["traj_lengths"][:args.max_trajectories]
            if not 1 <= args.max_trajectories <= len(source["traj_lengths"]):
                raise ValueError("Invalid trajectory limit")
            total = int(lengths.sum())
            required = total * 3 * 224 * 448 + 1024**3
            if shutil.disk_usage(args.output).free < required:
                raise RuntimeError("Insufficient disk for native dual-view memmap")
            states, actions = source["states"][:total], source["actions"][:total]
            for value in (states, actions):
                if not np.isfinite(value).all() or np.abs(value).max() > 1.001:
                    raise ValueError("Source must contain finite normalized states/actions")
            for key, value in (("states", states), ("actions", actions), ("traj_lengths", lengths)):
                np.save(args.output / f"{key}.npy", value)
            images = np.lib.format.open_memmap(args.output / "images.npy", mode="w+", dtype=np.uint8,
                                              shape=(total, 3, 224, 448))
            meta = json.loads(h5["data"].attrs["env_args"])
            if meta["env_name"] != "NutAssemblySquare" or meta["env_kwargs"]["control_freq"] != 20:
                raise ValueError("Requires Square at 20Hz")
            OU.initialize_obs_modality_mapping_from_dict({"low_dim": LOW_DIM, "rgb": ["agentview_image"]})
            meta["env_kwargs"].update(camera_names=["agentview"], camera_heights=96,
                                      camera_widths=96, camera_depths=False, reward_shaping=False)
            env = EU.create_env_from_metadata(meta, render=False, render_offscreen=True, use_image_obs=True)
            env.env.hard_reset = False
            offset = 0
            render_seconds = 0.0
            loop_start = time.perf_counter()
            for trajectory, length in enumerate(lengths):
                name, error = match_demo(actions[offset:offset + length], h5["data"],
                                         norm["action_min"], norm["action_max"])
                demo = h5["data"][name]
                xml = demo.attrs["model_file"]
                if isinstance(xml, bytes):
                    xml = xml.decode()
                env.reset()
                max_state = max_proprio = 0.0
                for frame in range(int(length)):
                    reset = {"states": demo["states"][frame]}
                    if frame == 0:
                        reset["model"] = xml
                    obs = env.reset_to(reset)
                    state_error = float(np.abs(env.get_state()["states"] - reset["states"]).max())
                    proprio = normalize(np.concatenate([obs[k] for k in LOW_DIM]),
                                        norm["obs_min"], norm["obs_max"])
                    proprio_error = float(np.abs(proprio - states[offset + frame]).max())
                    if state_error > 1e-8 or proprio_error > 1e-3:
                        raise ValueError(f"Restore failed at trajectory={trajectory} frame={frame}")
                    max_state, max_proprio = max(max_state, state_error), max(max_proprio, proprio_error)
                    render_start = time.perf_counter()
                    images[offset + frame] = render_dual_rgb(env)
                    render_seconds += time.perf_counter() - render_start
                    if offset == 0 and frame == 0 and not args.skip_renderer_query:
                        from OpenGL import GL

                        result["graphics"] = {
                            key: (GL.glGetString(token) or b"").decode()
                            for key, token in (("vendor", GL.GL_VENDOR),
                                               ("renderer", GL.GL_RENDERER),
                                               ("version", GL.GL_VERSION))
                        }
                        LOG.info("RENDERER backend=%s graphics=%s", args.render_backend,
                                 result["graphics"])
                        write_json(args.output / "manifest.json", result)
                    if frame in (0, min(87, int(length) - 1)):
                        Image.fromarray(images[offset + frame].transpose(1, 2, 0)).save(
                            args.output / f"preview_traj{trajectory:03d}_frame{frame:04d}.png")
                    if frame % 100 == 0 or frame + 1 == length:
                        LOG.info("STATUS phase=render completed=%d total=%d trajectory=%d frame=%d",
                                 offset + frame + 1, total, trajectory, frame)
                images.flush()
                result["mapping"][str(trajectory)] = {"demo": name, "length": int(length),
                    "action_error": error, "state_error": max_state, "proprio_error": max_proprio}
                offset += int(length)
                write_json(args.output / "manifest.json", result)
            del images
            loop_seconds = time.perf_counter() - loop_start
            result["timing"] = {"restore_render_write_seconds": loop_seconds,
                                "frames_per_second": total / loop_seconds,
                                "dual_render_seconds": render_seconds,
                                "scope": "excludes setup, source hashing and final hashing"}
            result["array_sha256"] = {k: file_hash(args.output / f"{k}.npy")
                                      for k in ("images", "states", "actions", "traj_lengths")}
            result.update(status="complete", frames=total, trajectories=len(lengths))
            write_json(args.output / "manifest.json", result)
            LOG.info("DONE phase=all completed=%d total=%d", total, total)
    finally:
        if env is not None:
            env.env.close()


if __name__ == "__main__":
    main()
