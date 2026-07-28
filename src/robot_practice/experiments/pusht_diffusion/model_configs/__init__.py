"""Explicit model configurations for the PushT Diffusion experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import draccus
from lerobot.configs.policies import PreTrainedConfig

_RUNTIME_ONLY_FIELDS = {
    "device",
    "license",
    "pretrained_path",
    "private",
    "push_to_hub",
    "repo_id",
    "tags",
}


def model_config_payload(config: PreTrainedConfig) -> dict[str, Any]:
    """Return the canonical, runtime-independent model configuration."""

    payload = draccus.encode(config)
    return {
        key: _json_value(value) for key, value in payload.items() if key not in _RUNTIME_ONLY_FIELDS
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return value


def model_config_fingerprint(config: PreTrainedConfig) -> str:
    """Hash model semantics while ignoring device and checkpoint location."""

    encoded = json.dumps(
        model_config_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["model_config_fingerprint", "model_config_payload"]
