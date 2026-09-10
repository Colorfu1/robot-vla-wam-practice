#!/usr/bin/env python3
"""Download and validate the two frozen-base DPPO Square assets on LOCAL."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gdown
import numpy as np
import torch

ASSET_IDS = {
    "normalization": "1jq5atfHdu-ZMQ8YaFjwcctbJESBYDgP8",
    "checkpoint": "1miud5SX41xjPoW8yqClRhaFx0Q7EWl3V",
}


def log(log_path: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(line + "\n")


def download_if_needed(name: str, path: Path, log_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        log(log_path, f"STATUS asset={name} action=skip path={path}")
        return
    log(log_path, f"STATUS asset={name} action=download path={path}")
    result = gdown.download(id=ASSET_IDS[name], output=str(path), quiet=False)
    if result is None or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"download failed: {path}")


def validate(normalization_path: Path, checkpoint_path: Path) -> None:
    with np.load(normalization_path) as normalization:
        required = {"obs_min", "obs_max", "action_min", "action_max"}
        missing = sorted(required - set(normalization.files))
        if missing:
            raise RuntimeError("normalization missing keys: " + ", ".join(missing))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(model, dict) or not any(key.startswith("network.") for key in model):
        raise RuntimeError("checkpoint has no supervised network.* weights")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    download_if_needed("normalization", args.normalization, args.log)
    download_if_needed("checkpoint", args.checkpoint, args.log)
    validate(args.normalization, args.checkpoint)
    log(args.log, "DONE completed=2 total=2 validation=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
