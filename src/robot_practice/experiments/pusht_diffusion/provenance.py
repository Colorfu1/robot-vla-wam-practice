"""Atomic serialization, hashing, and runtime provenance for PushT experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    detached = tensor.detach().contiguous().cpu()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode()
    return sha256_bytes(header + detached.numpy().tobytes())


def state_dict_hashes(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    return {
        name: sha256_tensor(value)
        for name, value in sorted(state_dict.items())
        if isinstance(value, torch.Tensor)
    }


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_output(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_runtime_provenance(root: str | Path) -> dict[str, Any]:
    repository_root = Path(root).expanduser().resolve()
    commit = _git_output(repository_root, "rev-parse", "HEAD")
    dirty_output = _git_output(repository_root, "status", "--porcelain")
    cuda_available = torch.cuda.is_available()
    return {
        "schema_version": 1,
        "repository": {
            "commit": commit,
            "dirty": bool(dirty_output),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "lerobot": package_version("lerobot"),
            "diffusers": package_version("diffusers"),
            "gym_pusht": package_version("gym-pusht"),
            "pymunk": package_version("pymunk"),
            "numpy": package_version("numpy"),
            "scikit_learn": package_version("scikit-learn"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        },
    }


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination
