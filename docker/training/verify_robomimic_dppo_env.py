"""Dependency/source check for the DPPO Docker layer; no assets are downloaded."""

import argparse
import importlib
import subprocess
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--render-smoke", action="store_true")
    args = parser.parse_args()
    required = ("hydra", "omegaconf", "gdown", "robomimic", "robosuite", "mujoco")
    for name in required:
        importlib.import_module(name)
    expected = [
        args.source / "cfg/robomimic/eval/square/eval_diffusion_mlp_img.yaml",
        args.source / "cfg/robomimic/env_meta/square-img.json",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing DPPO source files: " + ", ".join(missing))
    commit = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_commit = "cc7234ad7ff39a8f32de3af903606723a16f0648"
    if commit != expected_commit:
        raise RuntimeError(f"unexpected DPPO commit: {commit}")
    import robomimic
    import robosuite

    if robomimic.__version__ != "0.3.1" or robosuite.__version__ != "1.4.1":
        raise RuntimeError(
            f"unexpected benchmark versions: robomimic={robomimic.__version__} "
            f"robosuite={robosuite.__version__}"
        )
    if args.render_smoke:
        env = robosuite.make(
            "NutAssemblySquare",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=96,
            camera_widths=96,
            control_freq=20,
            reward_shaping=False,
        )
        try:
            image = env.reset()["agentview_image"]
            if image.shape != (96, 96, 3) or not np.any(image):
                raise RuntimeError(f"offscreen render failed: shape={image.shape}")
        finally:
            env.close()
    print(
        f"dppo_env_ok commit={commit} robomimic={robomimic.__version__} "
        f"robosuite={robosuite.__version__} render_smoke={args.render_smoke} "
        f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
