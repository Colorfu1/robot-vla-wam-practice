"""Shared constants and JSON configuration loading for the PushT Pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_REPO_ID = "lerobot/pusht"
DATASET_REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
PRETRAINED_POLICY_REPO_ID = "lerobot/diffusion_pusht"
PRETRAINED_POLICY_REVISION = "84a7c23178445c6bbf7e1a884ff497017910f653"
PROTOCOL_NAMESPACE = 20260727
PUSHT_FPS = 10
PUSHT_WORKSPACE_SIZE = 512.0
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8
ACTION_DIM = 2
EXECUTION_SLICE = slice(1, 9)
DEFAULT_SPLIT_SEED = 20260727
DEFAULT_TRAIN_SEED = 1000
DEFAULT_MAX_STEPS = 200_000
DEFAULT_SAVE_FREQUENCY = 25_000
SMOKE_STEPS = 1_000


def practice_root() -> Path:
    """Return the editable repository root without relying on the process cwd."""

    return Path(__file__).resolve().parents[4]


def config_root() -> Path:
    return practice_root() / "configs" / "experiments" / "pusht_diffusion"


def default_split_path() -> Path:
    return config_root() / "split.json"


def default_dataset_root() -> Path:
    return practice_root() / "datasets" / "lerobot" / "lerobot" / "pusht"


def default_output_root() -> Path:
    return practice_root() / "outputs" / "pusht_diffusion"


def load_json_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        with resolved.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {resolved}")
    return payload


def load_named_config(name: str) -> dict[str, Any]:
    if not name or Path(name).name != name or not name.endswith(".json"):
        raise ValueError(f"Configuration name must be one JSON filename, got {name!r}")
    return load_json_object(config_root() / name)
