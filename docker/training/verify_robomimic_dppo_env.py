#!/usr/bin/env python
"""Dependency/source check for the DPPO Docker layer; no assets are downloaded."""

import argparse
import importlib
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
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
    print(f"dppo_env_ok cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
