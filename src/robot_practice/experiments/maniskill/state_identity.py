"""Stable identity helpers for ManiSkill replay episodes.

GPU vectorized reset assigns seeds in a batch-dependent way, so a seed written
by a replay utility is not a reliable foreign key.  The initial simulator
state is the identity used by the M1-W route instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import h5py
import numpy as np


def state_fingerprint(value: Any, *, tolerance: float = 1e-4) -> str:
    """Hash an in-memory simulator state without using its reset seed.

    This is the in-memory counterpart of :func:`start_state_fingerprint` and
    is used for GPU state-bank leakage checks.  Mapping keys, leaf paths and
    shapes are part of the digest; numeric leaves are quantized by the fixed
    tolerance before hashing.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    digest = hashlib.sha256()

    def visit(path: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(f"{path}/{key}", item[key])
            return
        if hasattr(item, "detach") and hasattr(item, "cpu") and hasattr(item, "numpy"):
            item = item.detach().cpu().numpy()
        array = np.asarray(item)
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        if np.issubdtype(array.dtype, np.number):
            quantized = np.rint(array.astype(np.float64) / tolerance).astype(np.int64)
            digest.update(quantized.tobytes(order="C"))
        else:
            digest.update(np.ascontiguousarray(array).tobytes(order="C"))

    visit("state", value)
    return digest.hexdigest()


def start_state_fingerprint(
    group: h5py.Group,
    *,
    tolerance: float = 1e-4,
) -> str:
    """Hash every numeric leaf of ``env_states`` at its first frame.

    The quantization makes the identity stable across harmless CPU/GPU float
    drift while preserving the randomized object placement and robot state.
    The HDF5 leaf path and shape are included, so changing the state schema
    cannot silently create the same identity.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    root = group.get("env_states")
    if root is None:
        raise ValueError(f"{group.name} has no env_states")
    digest = hashlib.sha256()

    def visit(path: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            array = np.asarray(value[0])
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(array.shape).encode("ascii"))
            digest.update(b"\0")
            if np.issubdtype(array.dtype, np.number):
                quantized = np.rint(array.astype(np.float64) / tolerance).astype(np.int64)
                digest.update(quantized.tobytes(order="C"))
            else:
                digest.update(np.ascontiguousarray(array).tobytes(order="C"))
            return
        for key in sorted(value.keys()):
            child = value[key]
            visit(f"{path}/{key}", child)

    visit("env_states", root)
    return digest.hexdigest()


def episode_start_fingerprints(
    path: str | Path,
    *,
    tolerance: float = 1e-4,
) -> dict[int, str]:
    """Return ``episode_id -> start-state fingerprint`` for an HDF5 replay."""

    result: dict[int, str] = {}
    with h5py.File(path, "r") as handle:
        for name in handle:
            if not name.startswith("traj_"):
                continue
            episode_id = int(name.split("_", 1)[1])
            result[episode_id] = start_state_fingerprint(handle[name], tolerance=tolerance)
    return result


def invert_unique_fingerprints(
    fingerprints: dict[int, str],
    *,
    label: str,
) -> dict[str, int]:
    """Invert an episode fingerprint map and reject collisions."""

    result: dict[str, int] = {}
    for episode_id, fingerprint in fingerprints.items():
        previous = result.get(fingerprint)
        if previous is not None and previous != episode_id:
            raise ValueError(
                f"{label} has duplicate start-state fingerprint {fingerprint}: "
                f"episodes {previous} and {episode_id}"
            )
        result[fingerprint] = episode_id
    return result
