#!/usr/bin/env python3
"""Verify the package-level contract of the prebuilt RL environment layer."""

from __future__ import annotations

import argparse
import importlib
import sys


STAGES = {
    "s1": (
        ("torch", "torch"),
        ("mani_skill2", "mani_skill2"),
        ("gymnasium", "gymnasium"),
        ("stable_baselines3", "stable_baselines3"),
        ("gdown", "gdown"),
        ("diffusers", "diffusers"),
        ("cv2", "cv2"),
        ("sapien", "sapien"),
    ),
    "s2": (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("accelerate", "accelerate"),
        ("deepspeed", "deepspeed"),
        ("gymnasium", "gymnasium"),
        ("numpy", "numpy"),
        ("transformers", "transformers"),
        ("curobo", "curobo"),
        ("open3d", "open3d"),
        ("warp", "warp"),
        ("sapien", "sapien"),
        ("h5py", "h5py"),
    ),
    "s2-peg": (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("accelerate", "accelerate"),
        ("deepspeed", "deepspeed"),
        ("gymnasium", "gymnasium"),
        ("mani_skill2", "mani_skill2"),
        ("numpy", "numpy"),
        ("transformers", "transformers"),
        ("h5py", "h5py"),
        ("sapien", "sapien"),
        ("stable_baselines3", "stable_baselines3"),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    args = parser.parse_args()

    print(f"python={sys.executable}")
    for label, module_name in STAGES[args.stage]:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised by image builds
            raise SystemExit(f"{args.stage}: import failed for {label}: {exc}") from exc
        version = getattr(module, "__version__", "present")
        print(f"{args.stage}: {label}={version}")

    if args.stage == "s2-peg":
        # ManiSkill2 bundles its Warp 0.3.1-compatible implementation under
        # warp_maniskill and adds it to sys.path while importing envs.  This
        # must be exercised here; importing only the top-level mani_skill2
        # package would miss a newer RobotWin warp-lang shadowing it.
        importlib.import_module("mani_skill2.envs")
        importlib.import_module("warp.sim")

    torch = importlib.import_module("torch")
    if args.stage == "s1":
        if not torch.__version__.startswith("2.1.2"):
            raise SystemExit(f"S1 torch mismatch: {torch.__version__}")
        if torch.version.cuda != "12.1":
            raise SystemExit(f"S1 CUDA runtime mismatch: {torch.version.cuda}")
    else:
        if torch.__version__ != "2.7.1+cu128":
            raise SystemExit(f"S2 torch mismatch: {torch.__version__}")
        torchvision = importlib.import_module("torchvision")
        if torchvision.__version__ != "0.22.1+cu128":
            raise SystemExit(f"S2 torchvision mismatch: {torchvision.__version__}")
        if args.stage == "s2-peg":
            numpy = importlib.import_module("numpy")
            if numpy.__version__ != "1.23.5":
                raise SystemExit(f"S2 Peg NumPy mismatch: {numpy.__version__}")

    print(f"{args.stage}: environment contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
