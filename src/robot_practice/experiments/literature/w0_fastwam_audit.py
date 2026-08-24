"""Auditable launch wrapper for W0 Fast-WAM on the pinned LIBERO protocol.

This module owns provenance, preflight, local environment/artifact validation,
official command construction, and machine-readable manifests.  It deliberately
does not install packages, create environments, clone repositories, or download
model/data artifacts.  Those inputs must already be present in the runtime image
or in the repository-local ``third_party``/``pretrained`` directories.

The FastWAM implementation remains in the pinned checkout under ``vendor-root``;
this repository does not fork or modify its model code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import signal
import contextlib
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .w0_fastwam_analysis import (
    aggregate_eval_directory,
    compare_causal,
    gate_w0,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/literature/w0_fastwam_reproduction.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/literature/w0_fastwam"
FASTWAM_REPO = "https://github.com/yuantianyuan01/FastWAM.git"
LIBERO_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LEROBOT_REPO = "https://github.com/huggingface/lerobot.git"
FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
UPSTREAM_HF_REVISION = "139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a"
LEROBOT_HF_REVISION = "a784645f9ce367ba6953dd20a7a7f0310c85747c"
UPSTREAM_ENV = "fastwam-upstream"
LEROBOT_ENV = "fastwam-lerobot"
TASK_NAME = "libero_uncond_2cam224_1e-4"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ENVIRONMENT_PACKAGE_CONTRACT: dict[str, dict[str, str]] = {
    UPSTREAM_ENV: {
        "fastwam": "0.1.0",
        "libero": "0.1.0",
        "modelscope": "1.34.0",
        "mujoco": "3.3.2",
        "torch": "2.7.1+cu128",
        "easydict": "1.9",
        "opencv-python": "4.6.0.66",
        "robomimic": "0.2.0",
        "robosuite": "1.4.0",
        "bddl": "1.0.1",
        "future": "0.18.2",
        "matplotlib": "3.5.3",
        "cloudpickle": "2.1.0",
        "gym": "0.25.2",
    },
    LEROBOT_ENV: {
        "lerobot": "0.6.0",
        "torch": "2.11.0+cu128",
        "hf-libero": "0.1.4",
        "hf-egl-probe": "1.0.2",
        "robomimic": "0.2.0",
    },
}
ENVIRONMENT_PYTHON_CONTRACT = {UPSTREAM_ENV: (3, 10), LEROBOT_ENV: (3, 12)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _portable(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            temporary.replace(target)
        except OSError as exc:
            if exc.errno != 18:  # errno.EXDEV, kept dependency-free for this wrapper.
                raise
            shutil.copy2(temporary, target)
            temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _portable(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _rooted(path: str | Path | None, *, default: Path | None = None) -> Path:
    if path is None:
        if default is None:
            raise ValueError("a path is required")
        return default
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return _read_json(_rooted(path, default=DEFAULT_CONFIG))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen W0 contract before any heavyweight operation."""

    errors: list[str] = []
    if config.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    sources = _require_mapping(config.get("sources"), "sources")
    expected_sources = {
        "fastwam_commit": FASTWAM_COMMIT,
        "libero_commit": LIBERO_COMMIT,
        "lerobot_commit": LEROBOT_COMMIT,
        "fastwam_hf_revision": UPSTREAM_HF_REVISION,
        "lerobot_hf_revision": LEROBOT_HF_REVISION,
        "wan_model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "tokenizer_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "common_model_id": "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
    }
    for key, expected in expected_sources.items():
        if sources.get(key) != expected:
            errors.append(f"sources.{key} must be {expected!r}")
    protocol = _require_mapping(config.get("protocol"), "protocol")
    expected_protocol = {
        "task": TASK_NAME,
        "suites": list(SUITES),
        "tasks_per_suite": 10,
        "total_tasks": 40,
        "demos_per_suite": 500,
        "total_demos": 2000,
        "mujoco_version": "3.3.2",
        "camera_names": ["front", "wrist"],
        "camera_resolution": [224, 224],
        "concatenated_image_size": [224, 448],
        "proprio_dim": 8,
        "action_dim": 7,
        "action_horizon": 32,
        "n_action_steps": 10,
        "raw_num_frames": 33,
        "action_video_freq_ratio": 4,
        "video_frames": 9,
        "binarize_gripper": True,
        "text_cfg_scale": 1.0,
        "num_inference_steps": 10,
        "train_seed": 42,
        "eval_seed": 42,
        "eval_trials_per_task": 50,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            errors.append(f"protocol.{key} must be {expected!r}")
    training = _require_mapping(config.get("training"), "training")
    expected_training = {
        "steps": 20000,
        "gpus": 8,
        "per_gpu_batch_size": 16,
        "global_batch_size": 128,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "mixed_precision": "bf16",
        "max_grad_norm": 1.0,
        "action_denoising_steps": 10,
        "cfg_scale": 1.0,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            errors.append(f"training.{key} must be {expected!r}")
    if int(training.get("gpus", 0)) * int(training.get("per_gpu_batch_size", 0)) != int(
        training.get("global_batch_size", -1)
    ):
        errors.append("training global batch size does not equal GPUs * per-GPU batch")
    causal = _require_mapping(config.get("causal_pair"), "causal_pair")
    variants = _require_mapping(causal.get("variants"), "causal_pair.variants")
    for name, expected in (("with_video", 1.0), ("no_video", 0.0)):
        item = _require_mapping(variants.get(name), f"causal_pair.variants.{name}")
        if item.get("lambda_video") != expected:
            errors.append(f"{name}.lambda_video must be {expected}")
    if causal.get("only_declared_difference") != "model.loss.lambda_video":
        errors.append("causal pair must differ only in model.loss.lambda_video")
    known = _require_mapping(config.get("known_artifacts"), "known_artifacts")
    for label in ("upstream_checkpoint", "lerobot_model"):
        item = _require_mapping(known.get(label), f"known_artifacts.{label}")
        digest = str(item.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            errors.append(f"known_artifacts.{label}.sha256 must be lowercase SHA-256")
    if errors:
        raise ValueError("W0 config validation failed:\n- " + "\n- ".join(errors))
    return {
        "schema_version": 1,
        "experiment_id": config.get("experiment_id"),
        "valid": True,
        "fixed_sources": {
            "fastwam_commit": FASTWAM_COMMIT,
            "libero_commit": LIBERO_COMMIT,
            "lerobot_commit": LEROBOT_COMMIT,
            "upstream_hf_revision": UPSTREAM_HF_REVISION,
            "lerobot_hf_revision": LEROBOT_HF_REVISION,
        },
        "fixed_protocol": {
            "tasks": 40,
            "demos": 2000,
            "action": "7D EEF pose + gripper",
            "state": "8D proprio",
            "horizon": 32,
            "execute": 10,
        },
        "causal_variants": {"with_video": 1.0, "no_video": 0.0},
    }


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    dry_run: bool = False,
) -> None:
    printable = " ".join(_shell_quote(str(item)) for item in command)
    print(f"[W0] RUN {printable}", flush=True)
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})
    if log_path is None:
        subprocess.run(list(command), cwd=cwd, env=merged_env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            list(command), cwd=cwd, env=merged_env, stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def _shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "_./:-=,[]" for char in value):
        return value
    return repr(value)


def _conda_run(environment: str, *command: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", environment, *command]


def _default_output(config: Mapping[str, Any], suffix: str) -> Path:
    root = _rooted(_require_mapping(config["paths"], "paths")["default_output_root"])
    return root / suffix


def _config_path(config: Mapping[str, Any], key: str) -> Path:
    return _rooted(_require_mapping(config["paths"], "paths")[key])


def _vendor_root(config: Mapping[str, Any]) -> Path:
    """Resolve the pinned source tree in host and image runtimes alike."""

    override = os.environ.get("W0_VENDOR_ROOT")
    if override:
        return _rooted(override)
    return _config_path(config, "vendor_root")


def _find_preflight(config: Mapping[str, Any]) -> Path | None:
    root = _rooted(_require_mapping(config["paths"], "paths")["preflight_root"])
    path = root / "preflight.json"
    return path if path.exists() else None


def _require_resource_phase(config: Mapping[str, Any], phase: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    path = _find_preflight(config)
    if path is None:
        raise RuntimeError("run --phase=preflight before a resource-gated W0 phase")
    allowed = set(_read_json(path).get("allowed_phases", []))
    if phase not in allowed:
        raise RuntimeError(f"resource preflight does not allow phase={phase}")


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    inventory: list[dict[str, Any]] = []
    for index, line in enumerate(result.stdout.splitlines()):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            memory = int(float(fields[1]))
        except ValueError:
            continue
        inventory.append({"index": index, "name": fields[0], "memory_mib": memory})
    return inventory


def preflight(config: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
    output = _rooted(output_root)
    output.mkdir(parents=True, exist_ok=True)
    inventory = _gpu_inventory()
    disk = shutil.disk_usage(REPO_ROOT)
    package_versions: dict[str, str | None] = {}
    for package in ("torch", "gymnasium", "mujoco", "lerobot"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    min_eval_mib = int(config["resources"]["upstream_eval_min_gpu_gib"]) * 1024
    min_causal_mib = int(config["resources"]["causal_min_gpu_gib"]) * 1024
    enough_eval = bool(inventory) and max(item["memory_mib"] for item in inventory) >= min_eval_mib
    causal_count = int(config["resources"]["causal_gpu_count"])
    same_model_candidates: dict[str, int] = {}
    for item in inventory:
        same_model_candidates[item["name"]] = same_model_candidates.get(item["name"], 0) + 1
    enough_causal = any(
        count >= causal_count
        and all(
            item["memory_mib"] >= min_causal_mib
            for item in inventory
            if item["name"] == model_name
        )
        for model_name, count in same_model_candidates.items()
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "python": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "package_versions": package_versions,
        "gpu_inventory": inventory,
        "disk": {
            "free_gib": round(disk.free / (1024**3), 2),
            "total_gib": round(disk.total / (1024**3), 2),
        },
        "resource_gate": {
            "upstream_eval": enough_eval,
            "lerobot_quantitative": enough_eval,
            "causal_training": enough_causal,
        },
        "causal_same_model_candidates": same_model_candidates,
        "allowed_phases": [
            "validate-config",
            "preflight",
            "validate-environments",
            "validate-artifacts",
            *(["eval-upstream-released"] if enough_eval else []),
            *(["eval-lerobot-released"] if enough_eval else []),
            *(["train-causal-pair", "eval-causal-pair"] if enough_causal else []),
        ],
        "notes": [
            "formal causal training is blocked unless all required GPUs meet the memory gate",
            "this manifest intentionally omits hostname, username, and cache absolute paths",
        ],
    }
    _write_json(output / "preflight.json", payload)
    seeds = {
        "schema_version": 1,
        "seed": int(config["protocol"]["eval_seed"]),
        "trials_per_task": int(config["protocol"]["eval_trials_per_task"]),
        "tasks": [
            {"suite": suite, "task_id": task_id, "trial_indices": list(range(50))}
            for suite in SUITES
            for task_id in range(10)
        ],
    }
    _write_json(output / "libero_eval_seeds.json", seeds)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _require_checkout(*, path: Path, url: str, commit: str) -> dict[str, str]:
    """Validate a pre-existing checkout without changing it."""

    if not path.is_dir() or not (path / ".git").exists():
        raise FileNotFoundError(
            f"missing pinned checkout {_portable(path)}; prepare it before running W0 "
            f"(source: {url}, commit: {commit})"
        )
    actual = _git_head(path)
    if actual != commit:
        raise RuntimeError(
            f"checkout commit mismatch for {_portable(path)}: {actual!r} != {commit}"
        )
    return {"path": _portable(path), "commit": commit, "url": url}


def _conda_env_exists(environment: str) -> bool:
    try:
        subprocess.run(
            ["conda", "run", "-n", environment, "python", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _environment_validation_code(environment: str) -> str:
    expected = ENVIRONMENT_PACKAGE_CONTRACT[environment]
    python_version = ENVIRONMENT_PYTHON_CONTRACT[environment]
    expected_literal = repr(expected)
    python_literal = repr(python_version)
    imports = (
        "import fastwam, libero, modelscope, mujoco, torch"
        if environment == UPSTREAM_ENV
        else "import lerobot, libero, robosuite, robomimic, egl_probe, torch"
    )
    return (
        "import importlib.metadata as metadata, sys; "
        f"expected={expected_literal}; "
        "actual={name: metadata.version(name) for name in expected}; "
        f"assert tuple(sys.version_info[:2]) == {python_literal}, sys.version; "
        "assert actual == expected, (actual, expected); "
        f"{imports}; "
        "print('environment_contract_ok', actual)"
    )


def validate_environments(
    config: Mapping[str, Any], *, vendor_root: str | Path, output_root: str | Path, dry_run: bool
) -> dict[str, Any]:
    """Validate image-provided environments and pinned local checkouts.

    This phase is intentionally side-effect free apart from its JSON manifest:
    it never calls ``git clone``, ``git fetch``, ``conda create`` or ``pip``.
    """

    vendor = _rooted(vendor_root)
    fastwam = vendor / "FastWAM"
    libero = vendor / "LIBERO"
    lerobot = vendor / "lerobot"
    if dry_run:
        checkouts = {
            "FastWAM": {"path": _portable(fastwam), "commit": FASTWAM_COMMIT, "url": FASTWAM_REPO},
            "LIBERO": {"path": _portable(libero), "commit": LIBERO_COMMIT, "url": LIBERO_REPO},
            "lerobot": {"path": _portable(lerobot), "commit": LEROBOT_COMMIT, "url": LEROBOT_REPO},
        }
    else:
        checkouts = {
            "FastWAM": _require_checkout(path=fastwam, url=FASTWAM_REPO, commit=FASTWAM_COMMIT),
            "LIBERO": _require_checkout(path=libero, url=LIBERO_REPO, commit=LIBERO_COMMIT),
            "lerobot": _require_checkout(path=lerobot, url=LEROBOT_REPO, commit=LEROBOT_COMMIT),
        }
    commands = [
        _conda_run(
            UPSTREAM_ENV,
            "python",
            "-c",
            _environment_validation_code(UPSTREAM_ENV),
        ),
        _conda_run(
            LEROBOT_ENV,
            "python",
            "-c",
            _environment_validation_code(LEROBOT_ENV),
        ),
    ]
    if not dry_run:
        for environment in (UPSTREAM_ENV, LEROBOT_ENV):
            if not _conda_env_exists(environment):
                raise RuntimeError(
                    f"missing conda environment {environment!r}; provide it in the runtime image; "
                    "the W0 wrapper will not create or install it"
                )
    for command in commands:
        _run(command, dry_run=dry_run)
    payload = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "complete",
        "validation_only": True,
        "installs_performed": False,
        "downloads_performed": False,
        "vendor_root": _portable(vendor),
        "checkouts": checkouts,
        "environments": {
            UPSTREAM_ENV: {
                "python": "3.10",
                "provided_by": "runtime image or host conda",
                "package_contract": ENVIRONMENT_PACKAGE_CONTRACT[UPSTREAM_ENV],
            },
            LEROBOT_ENV: {
                "python": "3.12",
                "provided_by": "runtime image or host conda",
                "package_contract": ENVIRONMENT_PACKAGE_CONTRACT[LEROBOT_ENV],
            },
        },
        "commands": [[_portable(item) if Path(item).is_absolute() else item for item in cmd] for cmd in commands],
    }
    _write_json(_rooted(output_root) / "environment_manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


# Kept as a source-compatible name for callers from the first draft.  It is now
# validation-only; it does not bootstrap anything.
bootstrap_environments = validate_environments


def _model_runtime_env(
    paths: Mapping[str, Path], *, vendor_root: str | Path | None = None
) -> dict[str, str]:
    """Make the pinned local model layout visible to FastWAM/DiffSynth.

    FastWAM's ``redirect_common_files=true`` replaces the T5/VAE model ids with
    DiffSynth-Studio paths.  The upstream loader resolves those paths below
    ``DIFFSYNTH_MODEL_BASE_PATH``; without this environment variable, a command
    launched from the vendor checkout silently falls back to ``./checkpoints``
    and may try to download a second copy of the models.
    """

    runtime_support = REPO_ROOT / "src/robot_practice/experiments/literature/w0_runtime"
    pythonpath_parts = [str(runtime_support)]
    source_root = _rooted(vendor_root) / "FastWAM/src" if vendor_root is not None else None
    if source_root is None:
        inherited_vendor = os.environ.get("W0_VENDOR_ROOT", "").strip()
        if inherited_vendor:
            source_root = Path(inherited_vendor).expanduser() / "FastWAM/src"
    if source_root is not None and source_root.is_dir():
        # LR-REL's local Wan loader reuses the pinned raw FastWAM component
        # implementations.  The fastwam-lerobot environment intentionally does
        # not install the upstream package, so expose only its source tree.
        pythonpath_parts.insert(0, str(source_root))
    pythonpath = os.pathsep.join(pythonpath_parts)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    runtime_env = {
        "DIFFSYNTH_MODEL_BASE_PATH": str(paths["model_base"]),
        "W0_FASTWAM_MODEL_ROOT": str(paths["model_base"]),
        "W0_FASTWAM_POLICY_ROOT": str(paths["lerobot"]),
        "W0_LIBERO_ASSETS": str(paths["libero_assets"]),
        "PYTHONPATH": pythonpath,
        # Every formal W0 command must use only repository-local inputs.  These
        # flags also prevent third-party loaders from silently reaching HF.
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFSYNTH_SKIP_DOWNLOAD": "1",
    }
    if source_root is not None and source_root.is_dir():
        runtime_env["W0_FASTWAM_SOURCE_ROOT"] = str(source_root)
    return runtime_env


def _require_file(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {_portable(path)}")
    return {"path": _portable(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _require_nonempty_dir(path: Path, label: str) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {_portable(path)}")
    if not any(path.iterdir()):
        raise FileNotFoundError(f"empty {label}: {_portable(path)}")
    return {"path": _portable(path), "nonempty": True}


def _dataset_fingerprint(config: Mapping[str, Any], dataset_root: Path) -> dict[str, Any]:
    """Compare episode/frame counts against the frozen protocol.

    "Directory is non-empty" is too weak a check: this project has already been bitten
    twice by silently divergent vendored trees. The released checkpoint ships
    ``num_episodes=1712`` / ``num_transition=277713``, so a mismatch here means the local
    dataset is not the one the release was trained on and the causal pair loses its
    reference. Raises rather than warns, because a wrong dataset invalidates a 17-hour run.
    """

    protocol = config["protocol"]
    expected_per_suite = protocol["train_episodes_per_suite"]
    suites: dict[str, Any] = {}
    total_episodes = 0
    total_frames = 0
    mismatches: list[str] = []
    for suite in SUITES:
        info_path = dataset_root / f"{suite}_no_noops_lerobot/meta/info.json"
        if not info_path.is_file():
            mismatches.append(f"{suite}: missing {_portable(info_path)}")
            suites[suite] = {"info_json": None}
            continue
        info = _read_json(info_path)
        episodes = int(info.get("total_episodes", -1))
        frames = int(info.get("total_frames", -1))
        total_episodes += max(episodes, 0)
        total_frames += max(frames, 0)
        want = int(expected_per_suite[suite])
        suites[suite] = {
            "episodes": episodes,
            "frames": frames,
            "expected_episodes": want,
            "matches": episodes == want,
        }
        if episodes != want:
            mismatches.append(f"{suite}: episodes={episodes} expected={want}")
    want_episodes = int(protocol["train_episodes"])
    want_frames = int(protocol["train_frames"])
    if total_episodes != want_episodes:
        mismatches.append(f"total episodes={total_episodes} expected={want_episodes}")
    if total_frames != want_frames:
        mismatches.append(f"total frames={total_frames} expected={want_frames}")
    if mismatches:
        raise RuntimeError(
            "LIBERO dataset does not match the frozen protocol: " + "; ".join(mismatches)
        )
    return {
        "per_suite": suites,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "expected_total_episodes": want_episodes,
        "expected_total_frames": want_frames,
        "matches_released_stats": True,
        "val_episodes": int(protocol["val_episodes"]),
        "test_episodes": int(protocol["test_episodes"]),
    }


def validate_artifacts(
    config: Mapping[str, Any],
    *,
    cache_root: str | Path,
    output_root: str | Path,
    vendor_root: str | Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Validate artifacts already materialized inside the repository.

    No HF client, archive extraction, or preprocessing command is invoked here.
    In particular, ActionDiT must already exist; its one-time local generation is
    documented as an explicit preparation command outside this wrapper.
    """

    if not dry_run and _find_preflight(config) is None:
        raise RuntimeError("run --phase=preflight before validating W0 artifacts")
    paths = _artifact_paths(config, cache_root=cache_root)
    known = _require_mapping(config["known_artifacts"], "known_artifacts")
    required_paths = {
        "upstream_checkpoint": paths["upstream_checkpoint"],
        "upstream_stats": paths["upstream_stats"],
        "lerobot_model": paths["lerobot"] / "model.safetensors",
        "dataset_root": paths["dataset"],
        "wan": paths["wan"],
        "tokenizer": paths["tokenizer"],
        "common": paths["common"],
        "action_dit": paths["action_dit"],
        "libero_assets": paths["libero_assets"],
    }
    checks: dict[str, Any] = {}
    if not dry_run:
        upstream_check = _require_file(paths["upstream_checkpoint"], "upstream checkpoint")
        upstream_expected = _require_mapping(known["upstream_checkpoint"], "known_artifacts.upstream_checkpoint")
        if upstream_check["sha256"] != upstream_expected["sha256"]:
            raise ValueError(
                "upstream checkpoint SHA-256 mismatch: "
                f"{upstream_check['sha256']} != {upstream_expected['sha256']}"
            )
        if upstream_check["bytes"] != int(upstream_expected["bytes"]):
            raise ValueError(
                f"upstream checkpoint byte-size mismatch: {upstream_check['bytes']} != "
                f"{upstream_expected['bytes']}"
            )
        checks["upstream_checkpoint"] = {**dict(upstream_expected), "actual": upstream_check}
        checks["upstream_stats"] = _require_file(paths["upstream_stats"], "upstream dataset stats")
        lerobot_check = _require_file(paths["lerobot"] / "model.safetensors", "LeRobot model")
        lerobot_expected = _require_mapping(known["lerobot_model"], "known_artifacts.lerobot_model")
        if lerobot_check["sha256"] != lerobot_expected["sha256"]:
            raise ValueError(
                f"LeRobot model SHA-256 mismatch: {lerobot_check['sha256']} != "
                f"{lerobot_expected['sha256']}"
            )
        checks["lerobot_model"] = {**dict(lerobot_expected), "actual": lerobot_check}
        for key in ("wan", "tokenizer", "common"):
            checks[key] = _require_nonempty_dir(required_paths[key], f"{key} model directory")
        for relative in ("models_t5_umt5-xxl-enc-bf16.safetensors", "Wan2.2_VAE.safetensors"):
            _require_file(paths["common"] / relative, f"DiffSynth common model {relative}")
        checks["libero_assets"] = _require_nonempty_dir(
            paths["libero_assets"], "LIBERO simulator assets"
        )
        _require_file(
            paths["libero_assets"] / "scenes/libero_tabletop_base_style.xml",
            "LIBERO tabletop scene asset",
        )
        checks["action_dit"] = _require_file(paths["action_dit"], "ActionDiT checkpoint")
        _require_nonempty_dir(paths["dataset"], "LIBERO dataset root")
        dataset_checks: dict[str, Any] = {}
        for suite in SUITES:
            dataset_dir = paths["dataset"] / f"{suite}_no_noops_lerobot"
            dataset_checks[suite] = _require_nonempty_dir(dataset_dir, f"{suite} dataset")
        checks["datasets"] = dataset_checks
        checks["dataset_fingerprint"] = _dataset_fingerprint(config, paths["dataset"])
    payload = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "complete",
        "validation_only": True,
        "installs_performed": False,
        "downloads_performed": False,
        "cache_root": _portable(_rooted(cache_root)),
        "dataset_root": _portable(paths["dataset"]),
        "upstream_hf_revision": UPSTREAM_HF_REVISION,
        "lerobot_hf_revision": LEROBOT_HF_REVISION,
        "vendor_root": _portable(_rooted(vendor_root)),
        "required_paths": {key: _portable(value) for key, value in required_paths.items()},
        "checks": checks,
        "manual_preparation_required": [
            "source checkouts under third_party/w0_fastwam or W0_VENDOR_ROOT",
            "conda environments fastwam-upstream and fastwam-lerobot on host or in runtime image",
            "HuggingFace artifacts under pretrained/w0_fastwam",
            "LIBERO simulator assets under pretrained/w0_fastwam/libero_assets",
            "ActionDiT local preprocessing before this phase",
        ],
        "commands": [],
        "note": "This phase never installs, downloads, extracts, or preprocesses artifacts.",
    }
    _write_json(_rooted(output_root) / "artifact_manifest.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


# Kept as a source-compatible name for callers from the first draft.  It is now
# validation-only; it does not download or preprocess anything.
fetch_artifacts = validate_artifacts


def _artifact_paths(
    config: Mapping[str, Any], *, cache_root: str | Path | None = None
) -> dict[str, Path]:
    paths = _require_mapping(config["paths"], "paths")
    cache = _rooted(cache_root if cache_root is not None else paths["cache_root"])
    sources = _require_mapping(config["sources"], "sources")
    model_base = cache / "models"
    return {
        "upstream_checkpoint": cache / "upstream/libero_uncond_2cam224.pt",
        "upstream_stats": cache / "upstream/libero_uncond_2cam224_dataset_stats.json",
        "lerobot": cache / "lerobot",
        "dataset": cache / "dataset/data/libero_mujoco3.3.2",
        "model_base": model_base,
        "libero_assets": cache / "libero_assets",
        "wan": model_base / str(sources["wan_model_id"]),
        "tokenizer": model_base / str(sources["tokenizer_model_id"]),
        "common": model_base / str(sources["common_model_id"]),
        "action_dit": cache / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
    }


DEFAULT_MAX_UPSTREAM_TASKS_PER_GPU = 1


def _worker_eval_command(
    *,
    checkpoint: Path,
    stats: Path,
    action_dit: Path,
    output: Path,
    trials: int,
    trial_start: int = 0,
    suite: str,
    task_id: int,
    gpu_slot: int,
    seed: int,
) -> list[str]:
    return _conda_run(
        UPSTREAM_ENV,
        "python",
        "experiments/libero/eval_libero_single.py",
        f"task={TASK_NAME}",
        f"ckpt={checkpoint}",
        f"EVALUATION.dataset_stats_path={stats}",
        f"model.action_dit_pretrained_path={action_dit}",
        f"EVALUATION.task_suite_name={suite}",
        f"EVALUATION.task_id={task_id}",
        f"EVALUATION.num_trials={trials}",
        f"+EVALUATION.trial_start={trial_start}",
        f"EVALUATION.output_dir={output}",
        "EVALUATION.env_num=1",
        "EVALUATION.replan_steps=10",
        "EVALUATION.visualize_future_video=false",
        "EVALUATION.device=cuda",
        f"seed={seed}",
        f"gpu_id={gpu_slot}",
    )


DEFAULT_EGL_CALIBRATION = "outputs/literature/w0_fastwam/egl_calibration.json"
RENDER_PROBE = "scripts/experiments/literature/probe_w0_render_lifecycle.py"
EGL_PROBE_STEPS = 120
EGL_PROBE_TIMEOUT_S = 240.0


def _parse_egl_device_map(spec: str) -> dict[str, str]:
    """Parse a ``cuda:egl`` mapping such as ``"0:0,7:3"``."""

    mapping: dict[str, str] = {}
    for item in spec.split(","):
        entry = item.strip()
        if not entry:
            continue
        cuda_text, separator, egl_text = entry.partition(":")
        if not separator or not cuda_text.strip().isdigit() or not egl_text.strip().isdigit():
            raise ValueError(f"invalid EGL device map entry: {entry!r} (expected cuda:egl)")
        mapping[cuda_text.strip()] = egl_text.strip()
    return mapping


def _render_fingerprint(runtime_env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Identify the render stack so a cached calibration is not reused blindly.

    A calibration is only valid for the exact GPUs, driver and EGL device list
    it was measured on.  Moving the image to a 4-GPU host changes which
    ``MUJOCO_EGL_DEVICE_ID`` values are usable, so the cache must invalidate
    itself instead of relying on a hand-maintained per-machine map.
    """

    gpus: list[str] = []
    driver = ""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                gpus.append(f"{fields[0]}|{fields[1]}")
                driver = fields[2]
    except (OSError, subprocess.CalledProcessError):
        pass

    egl_count: int | None = None
    code = (
        "import os;os.environ.setdefault('PYOPENGL_PLATFORM','egl');"
        "from mujoco.egl import egl_ext as E;print(len(E.eglQueryDevicesEXT()))"
    )
    child_env = os.environ.copy()
    if runtime_env:
        child_env.update({str(key): str(value) for key, value in runtime_env.items()})
    child_env.pop("CUDA_VISIBLE_DEVICES", None)
    child_env.pop("MUJOCO_EGL_DEVICE_ID", None)
    try:
        probe = subprocess.run(
            _conda_run(UPSTREAM_ENV, "python", "-c", code),
            check=True,
            text=True,
            capture_output=True,
            env=child_env,
            cwd=REPO_ROOT,
        )
        egl_count = int(probe.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        pass

    return {
        "gpus": gpus,
        "gpu_count": len(gpus),
        "driver_version": driver,
        "egl_device_count": egl_count,
    }


def _probe_render_pair(
    *,
    cuda_device: str,
    egl_index: str,
    runtime_env: Mapping[str, str],
    steps: int = EGL_PROBE_STEPS,
    trials: int = 1,
    suite: str | None = None,
    task_id: int | None = None,
) -> bool:
    """Return True when this (CUDA device, EGL index) pair renders without abort.

    The pair must be tried in a child process: a mismatched EGL index aborts
    natively inside ``mjr_readPixels``, which no in-process handler can catch.
    The probe loads no checkpoint, so a cell costs seconds rather than minutes.
    """

    probe_path = REPO_ROOT / RENDER_PROBE
    if not probe_path.is_file():
        raise FileNotFoundError(f"missing render probe: {_portable(probe_path)}")

    child_env = _worker_environment(runtime_env, cuda_device, egl_index)
    command = _conda_run(
        UPSTREAM_ENV,
        "python",
        str(probe_path),
        "--trials", str(trials),
        "--steps", str(steps),
        *([] if suite is None else ["--suite", str(suite)]),
        *([] if task_id is None else ["--task-id", str(task_id)]),
        "--obs-device", "cuda",
        "--cuda-device", "0",
        "--hard-reset", "1",
        "--cuda-init", "before",
        "--report-every", str(max(steps, 1)),
    )
    try:
        result = subprocess.run(
            command,
            env=child_env,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=EGL_PROBE_TIMEOUT_S,
        )
    except OSError as error:
        print(f"[W0-EGL-CALIB] probe could not start: {error}", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(
            f"[W0-EGL-CALIB] probe timed out after {EGL_PROBE_TIMEOUT_S}s "
            f"(cuda={cuda_device} egl={egl_index})",
            flush=True,
        )
        return False
    ok = result.returncode == 0 and "[PROBE] DONE" in result.stdout
    if not ok:
        # A boolean-only probe is undebuggable: a pair rejected for an unrelated reason -- a
        # missing asset, a bad env, an import error -- looks exactly like a genuine render
        # abort, and the caller then either scans for another index or refuses to launch with
        # no way to tell which happened.
        print(
            f"[W0-EGL-CALIB] probe failed cuda={cuda_device} egl={egl_index} "
            f"returncode={result.returncode}\n"
            f"  CUDA_VISIBLE_DEVICES={child_env.get('CUDA_VISIBLE_DEVICES')} "
            f"MUJOCO_EGL_DEVICE_ID={child_env.get('MUJOCO_EGL_DEVICE_ID')}\n"
            f"  stdout tail: {result.stdout[-800:]}\n"
            f"  stderr tail: {result.stderr[-800:]}",
            flush=True,
        )
    return ok


def _autocalibrate_egl_device_map(
    gpu_ids: Sequence[str], *, runtime_env: Mapping[str, str], egl_device_count: int | None
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """Measure a usable EGL render index for every requested CUDA device.

    Identity is tried first so rendering stays on the GPU that already holds the
    policy; only devices whose identity pair aborts fall back to scanning.  This
    keeps the common case at one short probe per GPU and needs no per-machine
    configuration.
    """

    candidate_pool = list(range(egl_device_count if egl_device_count else len(gpu_ids)))
    mapping: dict[str, str] = {}
    safe: dict[str, list[int]] = {}

    for gpu in gpu_ids:
        ordered: list[int] = []
        if int(gpu) in candidate_pool:
            ordered.append(int(gpu))
        ordered.extend(index for index in candidate_pool if index not in ordered)

        chosen: int | None = None
        for egl_index in ordered:
            print(
                f"[W0-EGL-CALIB] probing cuda={gpu} egl={egl_index} steps={EGL_PROBE_STEPS}",
                flush=True,
            )
            if _probe_render_pair(
                cuda_device=gpu, egl_index=str(egl_index), runtime_env=runtime_env
            ):
                chosen = egl_index
                break
            print(
                f"[W0-EGL-CALIB] rejected cuda={gpu} egl={egl_index} "
                "(native abort or probe failure)",
                flush=True,
            )
        if chosen is None:
            raise RuntimeError(
                f"no EGL render device index works for CUDA device {gpu}; "
                f"tried {ordered}. Re-run "
                "scripts/experiments/literature/probe_egl_devices.py to inspect the "
                "EGL device list."
            )
        mapping[gpu] = str(chosen)
        safe[gpu] = [chosen]
        print(f"[W0-EGL-CALIB] accepted cuda={gpu} egl={chosen}", flush=True)

    return mapping, safe


def _normalize_pci(value: str) -> str:
    """Canonicalize a PCI address to ``bus:device.function``.

    The two sides disagree on the domain field and a naive prefix strip silently produced an
    empty join: ``nvidia-smi`` reports ``00000000:65:02.0`` (eight hex digits) while sysfs
    reports ``0000:6b:02.0`` (four). Dropping everything before the last two components
    normalizes both, and leaves an already-short address untouched.
    """

    parts = value.strip().lower().split(":")
    if len(parts) >= 3:
        parts = parts[-2:]
    return ":".join(parts)


def _cuda_pci_by_index() -> dict[str, str]:
    """CUDA device index -> normalized PCI bus id, straight from nvidia-smi."""

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    )
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit():
            out[fields[0]] = _normalize_pci(fields[1])
    return out


def _drm_pci_by_card() -> dict[str, str]:
    """DRM card name -> normalized PCI bus id, from sysfs."""

    out: dict[str, str] = {}
    for card in sorted(pathlib.Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:  # connectors like card1-DP-1
            continue
        try:
            target = (card / "device").resolve().name
        except OSError:
            continue
        out[card.name] = _normalize_pci(target)
    return out


def _egl_drm_by_index(runtime_env: Mapping[str, str]) -> dict[str, str]:
    """EGL list position -> DRM card name, by parsing the EGL enumeration probe.

    Enumeration must happen in a child with ``CUDA_VISIBLE_DEVICES`` unset so the list is
    absolute; the probe script already prints ``index=N ... drm=/dev/dri/cardM``.
    """

    probe = REPO_ROOT / "scripts/experiments/literature/probe_egl_devices.py"
    if not probe.is_file():
        raise FileNotFoundError(f"missing EGL enumeration probe: {_portable(probe)}")
    child_env = dict(os.environ)
    child_env.update(runtime_env)
    child_env.pop("CUDA_VISIBLE_DEVICES", None)
    child_env.pop("MUJOCO_EGL_DEVICE_ID", None)
    result = subprocess.run(
        _conda_run(UPSTREAM_ENV, "python", str(probe)),
        env=child_env,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=EGL_PROBE_TIMEOUT_S,
    )
    out: dict[str, str] = {}
    for match in re.finditer(r"index=(\d+)\s.*?drm=(/dev/dri/card\d+)", result.stdout):
        out[match.group(1)] = pathlib.Path(match.group(2)).name
    if not out:
        raise RuntimeError(
            "EGL enumeration produced no drm= fields; cannot derive the device map.\n"
            f"stdout tail: {result.stdout[-400:]}\nstderr tail: {result.stderr[-400:]}"
        )
    return out


def _pci_derived_egl_device_map(
    gpu_ids: Sequence[str], *, runtime_env: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Identify, per CUDA device, the EGL index that refers to that same physical GPU.

    That index is the one that must **not** be used. Measured as a full 8x8 matrix on an
    8x H20 node: rendering aborts immediately when ``MUJOCO_EGL_DEVICE_ID`` names the same
    physical GPU that holds the CUDA context, and survives on every other index. Exactly one
    cell failed per row, and those cells matched this join cell for cell:

        cuda 0 -> egl 6   cuda 4 -> egl 2
        cuda 1 -> egl 7   cuda 5 -> egl 3
        cuda 2 -> egl 4   cuda 6 -> egl 0
        cuda 3 -> egl 5   cuda 7 -> egl 1

    Callers therefore want ``_egl_indices_to_avoid``, not this map, as the render target.

    Derivation, unchanged: join on PCI bus id.

    This replaces guessing. ``MUJOCO_EGL_DEVICE_ID`` is a position in
    ``eglQueryDevicesEXT()``, whose order is the DRM enumeration order, not CUDA's. Measured on
    an 8x H20 node: CUDA enumerates PCI domains 65,67,69,6b ascending while ``card1..card8``
    enumerate them descending, so the correct map is
    ``{0:6, 1:7, 2:4, 3:5, 4:2, 5:3, 6:0, 7:1}`` -- identity is wrong for **every** device.

    Rendering on the wrong physical GPU mostly works, which is why this went unnoticed: the
    render probe accepted identity and a 2,000-episode evaluation completed, but pixel readback
    occasionally aborts natively inside ``mjr_readPixels``. One evaluation died that way at
    about the 141st cumulative render while the 120-step probe had passed.

    ``EGL_CUDA_DEVICE_NV`` would answer this directly but returns ``EGLError`` for every index
    on driver 535.261.03, so the PCI join is the available deterministic route.
    """

    cuda_pci = _cuda_pci_by_index()
    drm_pci = _drm_pci_by_card()
    egl_drm = _egl_drm_by_index(runtime_env)

    pci_to_cuda = {_normalize_pci(pci): index for index, pci in cuda_pci.items()}
    mapping: dict[str, str] = {}
    egl_to_cuda: dict[str, str] = {}
    for egl_index, card in egl_drm.items():
        pci = drm_pci.get(card)
        pci = None if pci is None else _normalize_pci(pci)
        if pci is None:
            continue
        cuda_index = pci_to_cuda.get(pci)
        if cuda_index is None:
            continue
        egl_to_cuda[egl_index] = cuda_index
        # Later EGL entries can duplicate an earlier card; keep the first, which is the one
        # carrying EGL_NV_device_cuda.
        mapping.setdefault(cuda_index, egl_index)

    missing = [gpu for gpu in gpu_ids if gpu not in mapping]
    report = {
        "cuda_pci": cuda_pci,
        "drm_pci": drm_pci,
        "egl_drm": egl_drm,
        "egl_to_cuda": egl_to_cuda,
        "identity_is_correct": all(mapping.get(gpu) == gpu for gpu in gpu_ids if gpu in mapping),
    }
    if missing:
        raise RuntimeError(
            f"PCI join could not place CUDA devices {missing}. "
            f"cuda_pci={cuda_pci} drm_pci={drm_pci} egl_drm={egl_drm}"
        )
    return {gpu: mapping[gpu] for gpu in gpu_ids}, report


def _worker_environment(
    runtime_env: Mapping[str, str], gpu_visible: str, egl_device_id: str | None = None
) -> dict[str, str]:
    """Preserve the launcher environment while applying W0/GPU overrides.

    ``robosuite.utils.binding_utils`` asserts that ``MUJOCO_EGL_DEVICE_ID``
    appears inside ``CUDA_VISIBLE_DEVICES``.  When the render device differs
    from the compute device, both ids are exposed with the compute device
    first, so ``EVALUATION.device=cuda`` still resolves to ``cuda:0`` on the
    intended GPU while the assertion is satisfied.
    """

    egl_visible = gpu_visible if egl_device_id is None else str(egl_device_id)
    visible_devices = (
        gpu_visible if egl_visible == gpu_visible else f"{gpu_visible},{egl_visible}"
    )
    child_env = os.environ.copy()
    child_env.update({str(key): str(value) for key, value in runtime_env.items()})
    child_env.update(
        {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "MUJOCO_EGL_DEVICE_ID": egl_visible,
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
        }
    )
    return child_env



def _upstream_task_specs(suite: str | None, task_id: int | None) -> list[tuple[str, int]]:
    if task_id is not None:
        if suite is None:
            raise ValueError("--task-id requires --suite")
        return [(suite, int(task_id))]
    selected = [suite] if suite else list(SUITES)
    return [(name, task) for name in selected for task in range(10)]



def _assert_freeze_switch_is_wired(vendor_root: Path) -> None:
    """Refuse to launch when W0_FREEZE_VIDEO_EXPERT is set but the patch is absent.

    The freeze switch lives in a vendored source patch
    (``docker/training/vendor-patches/fastwam-trainer-freeze-video-expert.patch``). If the
    patch is missing, the variable is a **silent no-op**: training runs as FW-NV while the
    manifest claims ``freeze_video_expert: true``, so two different experiments become
    indistinguishable after the fact.

    This is not hypothetical. The container gets recreated routinely here -- three times in one
    day -- and an edit applied inside a live container disappears with it.
    """

    if os.environ.get("W0_FREEZE_VIDEO_EXPERT", "").strip() in ("", "0"):
        return
    trainer = _rooted(vendor_root) / "FastWAM/src/fastwam/trainer.py"
    try:
        source = trainer.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot read the vendored trainer at {trainer}: {error}") from error
    missing = [
        marker
        for marker in ("W0_FREEZE_VIDEO_EXPERT", "if p.requires_grad]")
        if marker not in source
    ]
    if not missing:
        return
    raise RuntimeError(
        "W0_FREEZE_VIDEO_EXPERT is set but the freeze patch is not applied in "
        f"{trainer} (missing: {missing}). Without it the flag does nothing and the run would "
        "silently be FW-NV while its manifest says otherwise. Either use an image built with "
        "docker/training/Dockerfile.w0-fastwam-torchcodec-fix "
        "(w0-fastwam-20260821-fwfz or later), or apply the tracked patch in place:\n"
        f"  git -C {_rooted(vendor_root) / 'FastWAM'} apply "
        "docker/training/vendor-patches/fastwam-trainer-freeze-video-expert.patch"
    )


def _assert_vendor_supports_trial_start(vendor_root: Path, *, trials: int, per_worker: int) -> None:
    """Refuse a shard layout the vendored evaluator cannot honour.

    ``+EVALUATION.trial_start=N`` is a Hydra *addition*: if the evaluator never reads the key
    it is silently ignored, every shard then evaluates ``init_states[0:count]``, and the later
    trials never run. The pinned upstream script does not read it -- support lives in
    ``docker/training/vendor-patches/fastwam-eval-trial-start-and-render-diagnostics.patch``,
    which the current image does not apply.

    The aggregate check would eventually notice (``covered_episodes`` short of
    ``total_episodes``), but only after hours of evaluation. Fail before launching instead.

    ``per_worker >= trials`` keeps every trial of a task in one child, so ``trial_start`` is
    always 0 and the missing support cannot matter. That is the layout the recorded UP-REL
    result used: ``trials_per_worker = trials_per_task = 50``, one source per task, 50 unique
    indices, ``missing_trial_count = 0``.
    """

    if per_worker >= trials:
        return
    evaluator = _rooted(vendor_root) / "FastWAM/experiments/libero/eval_libero_single.py"
    try:
        source = evaluator.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot read the vendored evaluator at {evaluator}: {error}") from error
    if 'cfg.EVALUATION.get("trial_start"' in source:
        return
    raise RuntimeError(
        f"--trials-per-worker={per_worker} would shard {trials} trials per task, but the "
        f"vendored evaluator at {evaluator} never reads EVALUATION.trial_start, so every "
        "shard would re-run the first trials and the rest would never execute. Either use "
        f"--trials-per-worker={trials} (one child per task, trial_start always 0) or apply "
        "docker/training/vendor-patches/"
        "fastwam-eval-trial-start-and-render-diagnostics.patch in the image."
    )



def _upstream_worker_specs(
    *,
    suite: str | None,
    task_id: int | None,
    trials: int,
    trials_per_worker: int,
) -> list[tuple[str, int, int, int]]:
    """Expand task specs into isolated trial shards.

    The LIBERO renderer can abort in native EGL/MuJoCo code after several
    episodes in one interpreter.  A Python exception handler cannot recover
    from SIGABRT, so the stable unit of isolation is one (or a small explicit
    number of) trial(s) per child process.  The parent keeps task-level
    accounting while each child receives a global ``trial_start``.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    if trials_per_worker <= 0:
        raise ValueError("trials_per_worker must be positive")
    specs: list[tuple[str, int, int, int]] = []
    for task_suite, task in _upstream_task_specs(suite, task_id):
        for trial_start in range(0, trials, trials_per_worker):
            trial_count = min(trials_per_worker, trials - trial_start)
            specs.append((task_suite, task, trial_start, trial_count))
    return specs



def _select_egl_render_indices(
    gpu_ids: Sequence[str], *, forbidden: Mapping[str, str]
) -> dict[str, str]:
    """Choose a render index per CUDA device, avoiding that device's own physical GPU.

    Measured rule: rendering on the same physical GPU as the CUDA context aborts immediately;
    any other index survives. Identity is preferred when it is not the forbidden index, because
    on a host whose EGL-to-CUDA permutation has no fixed point -- as the measured 8x H20 node
    does, being four transpositions -- identity is already safe and keeps the map readable.

    The assignment is a bijection: two workers pointed at one EGL index would share a render
    device, which is a separate failure to debug later.

    The candidate set is exactly the EGL indices of the requested GPUs, i.e. the values of
    ``forbidden``. That keeps rendering inside the allocated GPUs -- rendering on a device the
    job was not given would work for GL but is not ours to use -- and it automatically excludes
    the EGL entries that are not CUDA-capable. On the measured node the list has 10 entries
    whose last two duplicate earlier cards and lack ``EGL_NV_device_cuda``; both abort
    immediately.
    """

    usable = sorted({str(value) for value in forbidden.values()}, key=int)
    chosen: dict[str, str] = {}
    taken: set[str] = set()
    for gpu in gpu_ids:
        blocked = forbidden.get(gpu)
        if gpu in usable and gpu != blocked and gpu not in taken:
            pick = gpu
        else:
            pick = next(
                (index for index in usable if index != blocked and index not in taken), None
            )
        if pick is None:
            raise RuntimeError(
                f"no EGL index left for CUDA {gpu}: usable={usable} forbidden={blocked} "
                f"already_taken={sorted(taken)}"
            )
        chosen[gpu] = pick
        taken.add(pick)
    return chosen


def _egl_device_map(
    gpu_ids: Sequence[str],
    *,
    explicit: str | None = None,
    runtime_env: Mapping[str, str] | None = None,
    allow_calibration: bool = False,
) -> tuple[dict[str, str], str]:
    """Resolve which EGL device index each CUDA device must render on.

    ``MUJOCO_EGL_DEVICE_ID`` is a position in ``eglQueryDevicesEXT()``, not a CUDA device id,
    and that list follows DRM enumeration order. Measured on an 8x H20 node: CUDA enumerates
    PCI domains 65,67,69,6b ascending while ``card1..card8`` enumerate them descending, so the
    correct map is ``{0:6, 1:7, 2:4, 3:5, 4:2, 5:3, 6:0, 7:1}`` -- identity is wrong for
    **every** device.

    The map is **re-derived on every run**, never read from cache. Deriving costs one
    ``nvidia-smi`` call, one sysfs walk and one short child process; caching it is what let a
    wrong map survive across runs. The cache file is kept only as a record and to report drift.

    Rendering on the wrong GPU mostly works, which is why this stayed hidden: a render probe of
    any finite length can accept a wrong pair. One evaluation aborted natively inside
    ``mjr_readPixels`` at about the 141st cumulative render after a 120-step probe had passed.
    So the probe is a verification of the derived pair, never the source of it.

    ``EGL_CUDA_DEVICE_NV`` would answer directly but returns ``EGLError`` for every index on
    driver 535.261.03, hence the PCI join.

    Resolution order: explicit override, then PCI derivation (+ probe verification when
    allowed), then the legacy probe scan if derivation is impossible. Identity is returned only
    when no subprocess may run at all, and is labelled so it cannot be mistaken for a measured
    map.
    """

    if explicit is None:
        explicit = os.environ.get("W0_EGL_DEVICE_MAP")
    if explicit:
        mapping = _parse_egl_device_map(explicit)
        missing = [gpu for gpu in gpu_ids if gpu not in mapping]
        if missing:
            raise ValueError(
                f"EGL device map is missing CUDA devices {missing}; got {sorted(mapping)}"
            )
        return {gpu: mapping[gpu] for gpu in gpu_ids}, "explicit"

    report_path = _rooted(DEFAULT_EGL_CALIBRATION)
    previous: dict[str, Any] = {}
    if report_path.is_file():
        try:
            previous = _read_json(report_path)
        except Exception:  # pragma: no cover - a corrupt record must not block a run
            previous = {}

    if runtime_env is None or not allow_calibration:
        # Only reachable where no child process may run (dry-run/config validation). Nothing
        # renders on this path, and the label keeps the value out of a real run's manifest.
        return {gpu: gpu for gpu in gpu_ids}, "identity-unverified"

    fingerprint = _render_fingerprint(runtime_env)

    try:
        derived, join = _pci_derived_egl_device_map(gpu_ids, runtime_env=runtime_env)
    except Exception as error:
        # Falling back silently is how a wrong map reached a real run: the probe accepted
        # identity, which was wrong for all eight devices, and an evaluation aborted natively
        # inside mjr_readPixels hours later. The probe cannot prove a pair is good -- it only
        # fails to reject one within its window -- so it must not be a silent default.
        if os.environ.get("W0_ALLOW_EGL_PROBE_FALLBACK", "").strip() in ("", "0"):
            raise RuntimeError(
                f"PCI derivation failed ({type(error).__name__}: {error}).\n"
                "The render-probe fallback cannot be trusted to find the right map: it accepted "
                "an identity map that was wrong for every device on this host, and the "
                "evaluation aborted natively inside mjr_readPixels only after ~141 cumulative "
                "renders.\nFix the derivation (it joins `nvidia-smi --query-gpu=pci.bus_id`, "
                "/sys/class/drm/card*/device and the EGL device list on PCI address), or pass "
                "an explicit map via --egl-device-map / W0_EGL_DEVICE_MAP, or set "
                "W0_ALLOW_EGL_PROBE_FALLBACK=1 to accept probe-only calibration."
            ) from error
        print(
            f"[W0-EGL-CALIB] PCI derivation unavailable ({type(error).__name__}: {error}); "
            "W0_ALLOW_EGL_PROBE_FALLBACK is set, so falling back to render probing, which can "
            "only reject a pair that aborts inside its window",
            flush=True,
        )
        mapping, safe_indices = _autocalibrate_egl_device_map(
            gpu_ids, runtime_env=runtime_env, egl_device_count=fingerprint.get("egl_device_count")
        )
        _write_json(
            report_path,
            {
                "schema_version": 4,
                "source": "launcher-autocalibration",
                "render_fingerprint": fingerprint,
                "steps_per_cell": EGL_PROBE_STEPS,
                "safe_egl_indices_by_cuda_device": safe_indices,
                "selected_map": mapping,
            },
        )
        return mapping, f"autocalibrated:{_portable(report_path)}"

    forbidden = derived
    derived = _select_egl_render_indices(gpu_ids, forbidden=forbidden)
    join["same_gpu_egl_index"] = forbidden
    join["selected_map"] = derived
    print(
        f"[W0-EGL-CALIB] same-GPU EGL index per CUDA device {forbidden} (must be avoided); "
        f"selected {derived}",
        flush=True,
    )

    # Drift against the previous record is worth naming: it means the host's PCI/DRM/CUDA
    # correspondence changed, which silently invalidates any hand-written override.
    prior_map = previous.get("device_map") or previous.get("selected_map")
    if prior_map and {str(k): str(v) for k, v in prior_map.items()} != derived:
        print(
            f"[W0-EGL-CALIB] map changed since {_portable(report_path)}: "
            f"was {prior_map}, now {derived}",
            flush=True,
        )

    verified: dict[str, list[int]] = {}
    for gpu, egl_index in derived.items():
        if not _probe_render_pair(cuda_device=gpu, egl_index=egl_index, runtime_env=runtime_env):
            raise RuntimeError(
                f"PCI-derived pair (cuda {gpu}, egl {egl_index}) failed the render probe. "
                "The derivation joins nvidia-smi PCI ids, /sys/class/drm and the EGL device "
                "list, so a failure means one of those three disagrees with the others: "
                f"{join}"
            )
        verified[gpu] = [int(egl_index)]
        print(f"[W0-EGL-CALIB] verified cuda={gpu} egl={egl_index}", flush=True)

    _write_json(
        report_path,
        {
            "schema_version": 4,
            "source": "pci-derived",
            "render_fingerprint": fingerprint,
            "steps_per_cell": EGL_PROBE_STEPS,
            "pci_join": join,
            "safe_egl_indices_by_cuda_device": verified,
            "device_map": derived,
        },
    )
    return derived, f"pci-derived:{_portable(report_path)}"



def _visible_gpu_ids(num_gpus: int) -> list[str]:
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if inherited:
        visible = [item.strip() for item in inherited.split(",") if item.strip()]
        if len(visible) < num_gpus:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES exposes fewer GPUs than --num-gpus: "
                f"{len(visible)} < {num_gpus}"
            )
        return visible[:num_gpus]
    return [str(index) for index in range(num_gpus)]


def _prepare_output_directory(output: Path, *, overwrite: bool) -> Path | None:
    """Prepare a fresh evaluation directory without mixing old task results."""

    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)
        return None
    if not output.is_dir():
        raise NotADirectoryError(f"evaluation output is not a directory: {_portable(output)}")
    if not any(output.iterdir()):
        return None
    if not overwrite:
        raise FileExistsError(
            f"evaluation output is non-empty: {_portable(output)}; "
            "use --overwrite to archive it and reuse the same path"
        )
    previous_manifest = output / "run_manifest.json"
    if previous_manifest.is_file():
        manifest = _read_json(previous_manifest)
        status = manifest.get("status")
        running_pids: list[int] = []
        if status == "running":
            candidate_pids: list[int] = []
            launcher_pid = manifest.get("launcher_pid")
            if isinstance(launcher_pid, int):
                candidate_pids.append(launcher_pid)
            for status_file in (output / "task_status").glob("*.json"):
                try:
                    worker = _read_json(status_file)
                except (OSError, TypeError, ValueError):
                    continue
                if worker.get("status") == "running" and isinstance(worker.get("pid"), int):
                    candidate_pids.append(worker["pid"])
            for pid in candidate_pids:
                proc = Path(f"/proc/{pid}")
                if not proc.is_dir():
                    continue
                try:
                    state = next(
                        line.split("\t", 1)[1].strip()
                        for line in (proc / "status").read_text(encoding="utf-8").splitlines()
                        if line.startswith("State:")
                    )
                    command_line = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                        errors="replace"
                    )
                except (OSError, StopIteration, UnicodeError):
                    continue
                if not state.startswith("Z") and (
                    str(output.resolve()) in command_line
                    or "w0_fastwam" in command_line
                    or "eval_libero" in command_line
                ):
                    running_pids.append(pid)
        if running_pids:
            raise RuntimeError(
                f"evaluation is still marked running: {_portable(previous_manifest)}; "
                f"live pids={running_pids}; stop the old job before using --overwrite"
            )
        if status == "running":
            print(
                f"[W0] archiving stale running manifest: {_portable(previous_manifest)}",
                flush=True,
            )
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = output.with_name(f"{output.name}.backup-{stamp}")
    suffix = 1
    while backup.exists():
        backup = output.with_name(f"{output.name}.backup-{stamp}-{suffix}")
        suffix += 1
    shutil.move(str(output), str(backup))
    output.mkdir(parents=True, exist_ok=True)
    return backup


def _run_upstream_workers(
    *,
    checkpoint: Path,
    stats: Path,
    action_dit: Path,
    output: Path,
    trials: int,
    num_gpus: int,
    suite: str | None,
    task_id: int | None,
    seed: int,
    vendor_root: Path,
    runtime_env: Mapping[str, str],
    max_tasks_per_gpu: int,
    trials_per_worker: int,
    egl_device_map: Mapping[str, str] | None = None,
    egl_map_source: str = "caller-supplied",
) -> dict[str, Any]:
    """Run official LIBERO trial workers without tmux or shell windows.

    The upstream manager uses tmux panes as a UI-oriented process supervisor.
    That is fragile in containers and does not provide reliable worker status.
    This launcher owns each child process directly, limits model instances per
    GPU, writes per-trial-shard logs/status files, and fails closed on a worker
    error or missing result JSON.  By default a child owns exactly one trial;
    this is deliberate because a native EGL/MuJoCo SIGABRT cannot be recovered
    by Python cleanup inside a long-lived evaluator process.
    """

    _assert_vendor_supports_trial_start(
        vendor_root, trials=trials, per_worker=trials_per_worker or trials
    )
    if max_tasks_per_gpu <= 0:
        raise ValueError("max_tasks_per_gpu must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if trials_per_worker <= 0:
        raise ValueError("trials_per_worker must be positive")
    output.mkdir(parents=True, exist_ok=True)
    task_specs = _upstream_task_specs(suite, task_id)
    worker_specs = _upstream_worker_specs(
        suite=suite,
        task_id=task_id,
        trials=trials,
        trials_per_worker=trials_per_worker,
    )
    gpu_ids = _visible_gpu_ids(num_gpus)
    if egl_device_map is None:
        egl_device_map, egl_map_source = _egl_device_map(
            gpu_ids, runtime_env=runtime_env, allow_calibration=True
        )
    egl_device_map = {str(key): str(value) for key, value in egl_device_map.items()}
    task_logs = output / "task_logs"
    task_status = output / "task_status"
    worker_root = output / "workers"
    task_logs.mkdir(parents=True, exist_ok=True)
    task_status.mkdir(parents=True, exist_ok=True)
    worker_root.mkdir(parents=True, exist_ok=True)
    run_log = output / "run.log"
    log_stream = run_log.open("w", encoding="utf-8")
    active: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    gpu_load = [0 for _ in gpu_ids]
    pending = list(worker_specs)
    completed = 0
    started = time.time()
    last_report = 0.0

    def emit(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        log_stream.write(line + "\n")
        log_stream.flush()

    def spec_label(spec: tuple[str, int, int, int]) -> str:
        return f"{spec[0]}_task{spec[1]}_trial{spec[2]:04d}_count{spec[3]:04d}"

    def status_path(spec: tuple[str, int, int, int]) -> Path:
        return task_status / f"{spec_label(spec)}.json"

    def worker_output_path(spec: tuple[str, int, int, int]) -> Path:
        return worker_root / spec_label(spec)

    def result_path(spec: tuple[str, int, int, int], gpu_slot: int) -> Path:
        worker_output = worker_output_path(spec)
        return worker_output / spec[0] / f"gpu{gpu_slot}_task{spec[1]}_results.json"

    def terminate_active() -> None:
        for record in active.values():
            process = record["process"]
            if process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 10.0
        while time.time() < deadline and any(
            record["process"].poll() is None for record in active.values()
        ):
            time.sleep(0.1)
        for record in active.values():
            process = record["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def launch(spec: tuple[str, int, int, int], gpu_slot: int) -> None:
        task_suite, task, trial_start, trial_count = spec
        worker_output = worker_output_path(spec)
        worker_output.mkdir(parents=True, exist_ok=True)
        command = _worker_eval_command(
            checkpoint=checkpoint,
            stats=stats,
            action_dit=action_dit,
            output=worker_output,
            trials=trial_count,
            trial_start=trial_start,
            suite=task_suite,
            task_id=task,
            gpu_slot=gpu_slot,
            seed=seed,
        )
        log_path = task_logs / f"{spec_label(spec)}_gpu{gpu_slot}.log"
        stream = log_path.open("w", encoding="utf-8")
        gpu_visible = gpu_ids[gpu_slot]
        egl_visible = egl_device_map[gpu_visible]
        child_env = _worker_environment(runtime_env, gpu_visible, egl_visible)
        process = subprocess.Popen(
            command,
            cwd=vendor_root / "FastWAM",
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        active[spec] = {
            "process": process,
            "gpu_slot": gpu_slot,
            "gpu_visible": gpu_visible,
            "egl_visible": egl_visible,
            "log_path": log_path,
            "log_stream": stream,
            "started_at": time.time(),
        }
        gpu_load[gpu_slot] += 1
        _write_json(
            status_path(spec),
            {
                "status": "running",
                "suite": task_suite,
                "task_id": task,
                "trial_start": trial_start,
                "trial_count": trial_count,
                "pid": process.pid,
                "gpu_slot": gpu_slot,
                "gpu_visible": gpu_visible,
                "egl_visible": egl_visible,
                "log": log_path,
                "output": worker_output,
                "command": command,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        emit(
            f"START suite={task_suite} task={task} trial_start={trial_start} "
            f"trial_count={trial_count} gpu_slot={gpu_slot} "
            f"gpu_visible={gpu_visible} egl_device_id={egl_visible} "
            f"pid={process.pid} log={_portable(log_path)}"
        )

    def find_free_gpu() -> int | None:
        candidates = [index for index, load in enumerate(gpu_load) if load < max_tasks_per_gpu]
        return min(candidates, key=lambda index: gpu_load[index]) if candidates else None

    try:
        emit(
            f"START native Popen launcher tasks={len(task_specs)} workers={len(worker_specs)} "
            f"trials_per_worker={trials_per_worker} gpus={gpu_ids} "
            f"egl_device_map={egl_device_map} egl_map_source={egl_map_source} "
            f"max_tasks_per_gpu={max_tasks_per_gpu}"
        )
        while pending or active:
            for spec, record in list(active.items()):
                return_code = record["process"].poll()
                if return_code is None:
                    continue
                result = result_path(spec, int(record["gpu_slot"]))
                ok = return_code == 0 and result.is_file()
                status = "complete" if ok else "failed"
                _write_json(
                    status_path(spec),
                    {
                        "status": status,
                        "suite": spec[0],
                        "task_id": spec[1],
                        "trial_start": spec[2],
                        "trial_count": spec[3],
                        "pid": record["process"].pid,
                        "gpu_slot": record["gpu_slot"],
                        "gpu_visible": record["gpu_visible"],
                        "egl_visible": record["egl_visible"],
                        "return_code": return_code,
                        "result": result if result.is_file() else None,
                        "log": record["log_path"],
                        "output": worker_output_path(spec),
                        "started_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(record["started_at"])
                        ),
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                record["log_stream"].close()
                gpu_load[int(record["gpu_slot"])] -= 1
                del active[spec]
                if ok:
                    completed += 1
                    emit(
                        f"DONE suite={spec[0]} task={spec[1]} trial_start={spec[2]} "
                        f"trial_count={spec[3]} gpu_slot={record['gpu_slot']} "
                        f"return_code={return_code} progress={completed}/{len(worker_specs)}"
                    )
                else:
                    raise RuntimeError(
                        f"W0 worker failed: suite={spec[0]} task={spec[1]} "
                        f"trial_start={spec[2]} trial_count={spec[3]} "
                        f"gpu_slot={record['gpu_slot']} return_code={return_code}; "
                        f"inspect {_portable(record['log_path'])}"
                    )

            while pending:
                gpu_slot = find_free_gpu()
                if gpu_slot is None:
                    break
                launch(pending.pop(0), gpu_slot)

            now = time.time()
            if now - last_report >= 30.0 or (not pending and not active):
                emit(
                    f"STATUS workers_completed={completed}/{len(worker_specs)} "
                    f"tasks={len(task_specs)} running={len(active)} pending={len(pending)} "
                    f"elapsed_s={int(now - started)} "
                    f"gpu_load={gpu_load}"
                )
                last_report = now
            if pending or active:
                time.sleep(1.0)
    except BaseException:
        terminate_active()
        for record in active.values():
            record["log_stream"].close()
        raise
    finally:
        log_stream.close()

    return {
        "launcher": "repository-popen",
        "tasks": len(task_specs),
        "workers": len(worker_specs),
        "completed": completed,
        "num_gpus": len(gpu_ids),
        "egl_device_map": egl_device_map,
        "egl_map_source": egl_map_source,
        "gpu_ids": gpu_ids,
        "max_tasks_per_gpu": max_tasks_per_gpu,
        "trials_per_worker": trials_per_worker,
    }


def _write_run_manifest(output: Path, payload: Mapping[str, Any]) -> None:
    _write_json(output / "run_manifest.json", payload)


def _resolve_eval_dataset_stats(checkpoint: Path, upstream_stats: Path) -> tuple[Path, dict[str, Any]]:
    """Pick the normalization stats an evaluated checkpoint was actually trained with.

    ``eval_libero_single.py`` prefers an explicit ``EVALUATION.dataset_stats_path`` and only
    falls back to ``<checkpoint parent>/dataset_stats.json``. Passing the upstream file
    unconditionally is therefore wrong for a locally trained checkpoint whose training data
    differ from the released run, and wrong *silently*: actions would be denormalized on the
    wrong scale with no error.

    The check is on values, not paths. Measured on the W0 pair, the two files differ in 6
    fields at the 8th significant digit -- relative difference about 2e-7, which is float32
    machine epsilon (1.19e-7) and comes from a different reduction order over identical data
    (the dataset fingerprint matches the released episode/frame counts exactly). That is not a
    reason to refuse. A real data difference would show up far above this tolerance.
    """

    # The trainer writes dataset_stats.json at the run root, while checkpoints land two levels
    # deeper under checkpoints/weights/. Looking only at checkpoint.parent silently missed the
    # file and fell back to the upstream stats -- a wrong-normalization risk with no error.
    # Bounded to three ancestors so the search cannot wander into a shared parent directory.
    sibling: Path | None = None
    searched: list[str] = []
    probe = checkpoint.parent
    for _ in range(4):
        candidate = probe / "dataset_stats.json"
        searched.append(_portable(candidate))
        if candidate.exists():
            sibling = candidate
            break
        if probe.parent == probe:
            break
        probe = probe.parent
    report: dict[str, Any] = {
        "upstream": _portable(upstream_stats),
        "checkpoint_sibling": _portable(sibling) if sibling is not None else None,
        "searched_for_sibling": searched,
    }
    if sibling is None:
        report["chosen"] = "upstream"
        report["reason"] = "no dataset_stats.json beside the checkpoint or its 3 ancestors"
        return upstream_stats, report

    def _numbers(payload: Any, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                out.update(_numbers(value, f"{prefix}{key}."))
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                out.update(_numbers(value, f"{prefix}{index}."))
        elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
            out[prefix.rstrip(".")] = float(payload)
        return out

    try:
        upstream_numbers = _numbers(json.loads(upstream_stats.read_text(encoding="utf-8")))
        local_numbers = _numbers(json.loads(sibling.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot compare dataset stats: {error}") from error

    only_upstream = sorted(set(upstream_numbers) - set(local_numbers))
    only_local = sorted(set(local_numbers) - set(upstream_numbers))
    # A pure relative test false-alarms on near-zero fields: two stepwise_mean entries that
    # differ by 7.6e-10 in absolute terms report a 2.4e-05 relative difference simply because
    # the values themselves are ~3e-05. Use numpy.isclose semantics -- absolute tolerance OR
    # relative tolerance -- so "different data" is what trips the warning, not "small number".
    absolute_tolerance = 1e-8
    relative_tolerance = 1e-5
    worst_key, worst_rel, worst_abs = None, 0.0, 0.0
    exceeded: list[str] = []
    for key, value in upstream_numbers.items():
        other = local_numbers.get(key)
        if other is None:
            continue
        difference = abs(value - other)
        scale = max(abs(value), abs(other), 1e-12)
        relative = difference / scale
        if relative > worst_rel:
            worst_key, worst_rel = key, relative
        worst_abs = max(worst_abs, difference)
        if difference > absolute_tolerance + relative_tolerance * abs(value):
            exceeded.append(key)

    report.update(
        {
            "chosen": "checkpoint_sibling",
            "reason": "stats beside the checkpoint are the ones training normalized with",
            "max_relative_difference": worst_rel,
            "max_relative_difference_field": worst_key,
            "max_absolute_difference": worst_abs,
            "fields_beyond_tolerance": exceeded[:10],
            "keys_only_in_upstream": only_upstream[:10],
            "keys_only_beside_checkpoint": only_local[:10],
        }
    )
    if exceeded or only_upstream or only_local:
        _report_training_env(
            f"dataset stats differ beyond float32 noise ({len(exceeded)} field(s), max absolute "
            f"{worst_abs:.3e}, max relative {worst_rel:.3e} at {worst_key}); using {sibling} "
            f"because that is what training normalized with"
        )
    return sibling, report


def eval_upstream_released(
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path,
    output_root: str | Path,
    vendor_root: str | Path,
    trials: int,
    num_gpus: int,
    smoke: bool,
    suite: str | None,
    task_id: int | None,
    dry_run: bool,
    overwrite: bool = False,
    max_tasks_per_gpu: int = DEFAULT_MAX_UPSTREAM_TASKS_PER_GPU,
    trials_per_worker: int = 0,
    egl_device_map_spec: str | None = None,
) -> dict[str, Any]:
    _require_resource_phase(config, "eval-upstream-released", dry_run=dry_run)
    paths = _artifact_paths(config)
    checkpoint_path = _rooted(checkpoint)
    stats, stats_report = _resolve_eval_dataset_stats(checkpoint_path, paths["upstream_stats"])
    output = _rooted(output_root)
    if not dry_run and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if smoke:
        trials, num_gpus, suite, task_id = 2, 1, suite or "libero_spatial", 0
    if trials_per_worker <= 0:
        # One child per task: the 12 GiB checkpoint is loaded once instead of
        # once per trial.  Per-trial isolation was only ever a workaround for the
        # EGL render-device mismatch, which the resolved map now prevents.
        trials_per_worker = trials
    gpu_ids = _visible_gpu_ids(num_gpus)
    upstream_runtime_env = _model_runtime_env(paths, vendor_root=vendor_root)
    resolved_egl_map, resolved_egl_source = _egl_device_map(
        gpu_ids,
        explicit=egl_device_map_spec,
        runtime_env=upstream_runtime_env,
        allow_calibration=not dry_run,
    )
    task_specs = _upstream_task_specs(suite, task_id)
    backup = None if dry_run else _prepare_output_directory(output, overwrite=overwrite)
    preview_command = _worker_eval_command(
        checkpoint=checkpoint_path,
        stats=stats,
        action_dit=paths["action_dit"],
        output=output,
        trials=trials,
        trial_start=0,
        suite=task_specs[0][0],
        task_id=task_specs[0][1],
        gpu_slot=0,
        seed=int(config["protocol"]["eval_seed"]),
    )
    _write_run_manifest(
        output,
        {
            "schema_version": 1,
            "variant": "UP-REL",
            "status": "planned" if dry_run else "running",
            "checkpoint": _portable(checkpoint_path),
            # Which normalization stats were used, and how far the two candidates differ.
            # A locally trained checkpoint must be denormalized with the stats training
            # produced; using the upstream file instead is silently wrong.
            "dataset_stats": stats_report,
            "trials_per_task": trials,
            "trials_per_worker": trials_per_worker,
            "num_gpus": num_gpus,
            # The render stack must be recorded, not just the map: a node change silently
            # altered rendering behaviour, and because no eval manifest carried the
            # fingerprint -- and the calibration file had been overwritten -- the first
            # host's driver and GPUs were unrecoverable and the difference could not be
            # explained.
            "render_fingerprint": _render_fingerprint(upstream_runtime_env),
            "egl_device_map": resolved_egl_map,
            "egl_map_source": resolved_egl_source,
            "suite": suite,
            "task_id": task_id,
            "seed": int(config["protocol"]["eval_seed"]),
            "protocol": "official FastWAM LIBERO evaluator",
            "launcher": "repository-popen",
            "process_isolation": "one evaluator child per trial shard",
            "result_layout": "workers/<suite>_task<id>_trial<start>_count<count>/",
            "launcher_pid": os.getpid(),
            "max_tasks_per_gpu": max_tasks_per_gpu,
            "gpu_ids": gpu_ids,
            "task_count": len(task_specs),
            "worker_command_preview": preview_command,
            "overwrite": overwrite,
            "previous_output_backup": _portable(backup) if backup else None,
        },
    )
    if dry_run:
        return {
            "status": "dry_run",
            "launcher": "repository-popen",
            "tasks": len(task_specs),
            "gpu_ids": gpu_ids,
            "worker_command": preview_command,
        }
    try:
        launcher_result = _run_upstream_workers(
            checkpoint=checkpoint_path,
            stats=stats,
            action_dit=paths["action_dit"],
            output=output,
            trials=trials,
            num_gpus=num_gpus,
            suite=suite,
            task_id=task_id,
            seed=int(config["protocol"]["eval_seed"]),
            vendor_root=_rooted(vendor_root),
            runtime_env=upstream_runtime_env,
            max_tasks_per_gpu=max_tasks_per_gpu,
            trials_per_worker=trials_per_worker,
            egl_device_map=resolved_egl_map,
            egl_map_source=resolved_egl_source,
        )
    except BaseException as error:
        manifest = _read_json(output / "run_manifest.json")
        manifest.update({"status": "failed", "error": str(error)})
        _write_run_manifest(output, manifest)
        raise
    manifest = _read_json(output / "run_manifest.json")
    manifest.update({"status": "workers_complete", **launcher_result})
    _write_run_manifest(output, manifest)
    summary = aggregate_eval_directory(
        output,
        expected_task_keys=task_specs,
        expected_trials_per_task=trials,
    )
    summary["variant"] = "UP-REL"
    if smoke:
        summary["status"] = "smoke_complete" if summary.get("num_tasks", 0) else "smoke_incomplete"
    _write_json(output / "summary.json", summary)
    manifest = _read_json(output / "run_manifest.json")
    manifest.update({"status": summary.get("status", "incomplete"), "summary": "summary.json"})
    _write_run_manifest(output, manifest)
    return summary


VISUAL_NORMALIZATION_CHOICES = ("identity", "checkpoint")


def _prepare_lerobot_policy_overlay(
    policy_dir: Path, overlay_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Serve the released LeRobot checkpoint with VISUAL normalization disabled.

    The released ``policy_preprocessor.json`` stores
    ``norm_map.VISUAL = MEAN_STD`` with ``mean=std=0.5``, which maps images from
    ``[0,1]`` to ``[-1,1]``.  The pinned LeRobot FastWAM core then applies the
    same mapping again (``policies/fastwam/wan/modular.py:1086``), so the VAE
    receives ``4x-3`` where the official evaluator feeds ``2x-1``
    (measured: ``max|diff| = 2.0``, mean-abs ratio ``2.0``).  The pinned code's
    own contract is ``VISUAL: IDENTITY`` -- see the dataclass default in
    ``configuration_fastwam.py`` and the NOTE in ``processor_fastwam.py`` -- so
    the checkpoint's saved processor config contradicts the code it runs under.

    Rather than editing a pinned artifact, build a directory of symlinks to it
    and replace only that one JSON.  The weights stay byte-identical and the
    adaptation is explicit and recorded in the run manifest.
    """

    source_name = "policy_preprocessor.json"
    source_file = policy_dir / source_name
    if not source_file.is_file():
        raise FileNotFoundError(f"missing {source_name} in {_portable(policy_dir)}")

    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    overlay_dir.mkdir(parents=True)
    for entry in sorted(policy_dir.iterdir()):
        if entry.name == source_name:
            continue
        (overlay_dir / entry.name).symlink_to(entry.resolve())

    document = json.loads(source_file.read_text(encoding="utf-8"))
    changed: list[dict[str, Any]] = []
    for index, step in enumerate(document.get("steps", [])):
        if step.get("registry_name") != "normalizer_processor":
            continue
        norm_map = step.get("config", {}).get("norm_map", {})
        previous = norm_map.get("VISUAL")
        if previous is None or str(previous).endswith("IDENTITY"):
            continue
        norm_map["VISUAL"] = "IDENTITY"
        changed.append({"step_index": index, "from": previous, "to": "IDENTITY"})

    (overlay_dir / source_name).write_text(
        json.dumps(document, indent=4) + "\n", encoding="utf-8"
    )
    record = {
        "overlay": _portable(overlay_dir),
        "source": _portable(policy_dir),
        "replaced_file": source_name,
        "visual_norm_changes": changed,
        "reason": (
            "checkpoint stores norm_map.VISUAL=MEAN_STD(0.5,0.5) while the pinned "
            "LeRobot FastWAM core already maps [0,1]->[-1,1]; applying both feeds "
            "the VAE 4x-3 instead of the official 2x-1"
        ),
    }
    print(
        f"[W0] LR-REL policy overlay {_portable(overlay_dir)} "
        f"visual_norm_changes={changed}",
        flush=True,
    )
    return overlay_dir, record


def _lerobot_worker_command(
    *,
    policy: Path,
    suite: str,
    task_id: int,
    trials: int,
    episode_length: int,
    output: Path,
    seed: int,
    observation_size: int = 224,
) -> list[str]:
    return _conda_run(
        LEROBOT_ENV,
        "lerobot-eval",
        f"--policy.path={policy}",
        "--policy.device=cuda",
        "--policy.torch_dtype=float32",
        "--policy.n_action_steps=10",
        "--policy.toggle_action_dimensions=[-1]",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--env.task_ids=[{task_id}]",
        f"--env.observation_height={observation_size}",
        f"--env.observation_width={observation_size}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={trials}",
        f"--seed={seed}",
        f"--env.episode_length={episode_length}",
        f"--output_dir={output}",
    )


def _run_lerobot_workers(
    *,
    policy: Path,
    output: Path,
    suites: Sequence[str],
    episode_lengths: Mapping[str, int],
    trials: int,
    seed: int,
    num_gpus: int,
    runtime_env: Mapping[str, str],
    egl_device_map: Mapping[str, str],
    egl_map_source: str,
    observation_size: int = 224,
    max_tasks_per_gpu: int = DEFAULT_MAX_UPSTREAM_TASKS_PER_GPU,
) -> dict[str, Any]:
    """Shard LR-REL over GPUs at (suite, task) granularity, like UP-REL.

    ``lerobot-eval`` is single-process and single-GPU, and ``--eval.n_episodes``
    counts episodes *per task*, so one invocation per suite serialises 500
    episodes.  ``--env.task_ids=[N]`` restricts a run to one task, which makes the
    40 protocol tasks an embarrassingly parallel set: 40 shards across 8 GPUs
    instead of 4 sequential suites on one.
    """

    gpu_ids = _visible_gpu_ids(num_gpus)
    shards = [(suite, task) for suite in suites for task in range(10)]
    output.mkdir(parents=True, exist_ok=True)
    shard_root = output / "shards"
    status_root = output / "task_status"
    log_root = output / "task_logs"
    for directory in (shard_root, status_root, log_root):
        directory.mkdir(parents=True, exist_ok=True)
    run_log = (output / "run.log").open("a", encoding="utf-8")

    def emit(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        run_log.write(line + "\n")
        run_log.flush()

    def label(shard: tuple[str, int]) -> str:
        return f"{shard[0]}_task{shard[1]:02d}"

    active: dict[tuple[str, int], dict[str, Any]] = {}
    gpu_load = [0] * len(gpu_ids)
    pending = list(shards)
    completed = 0
    failures: list[dict[str, Any]] = []
    started = time.time()
    last_report = 0.0

    def launch(shard: tuple[str, int], gpu_slot: int) -> None:
        suite, task_id = shard
        shard_output = shard_root / label(shard)
        shard_output.mkdir(parents=True, exist_ok=True)
        gpu_visible = gpu_ids[gpu_slot]
        egl_visible = egl_device_map[gpu_visible]
        command = _lerobot_worker_command(
            policy=policy,
            suite=suite,
            task_id=task_id,
            trials=trials,
            episode_length=int(episode_lengths[suite]),
            output=shard_output,
            seed=seed,
            observation_size=observation_size,
        )
        log_path = log_root / f"{label(shard)}_gpu{gpu_visible}.log"
        stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=_worker_environment(runtime_env, gpu_visible, egl_visible),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        active[shard] = {
            "process": process,
            "gpu_slot": gpu_slot,
            "gpu_visible": gpu_visible,
            "egl_visible": egl_visible,
            "log_stream": stream,
            "log_path": log_path,
            "output": shard_output,
        }
        gpu_load[gpu_slot] += 1
        emit(
            f"START suite={suite} task={task_id} gpu_visible={gpu_visible} "
            f"egl_device_id={egl_visible} pid={process.pid} log={_portable(log_path)}"
        )

    def free_gpu() -> int | None:
        candidates = [i for i, load in enumerate(gpu_load) if load < max_tasks_per_gpu]
        return min(candidates, key=lambda i: gpu_load[i]) if candidates else None

    emit(
        f"START LR-REL launcher suites={list(suites)} shards={len(shards)} "
        f"trials_per_task={trials} gpus={gpu_ids} egl_device_map={dict(egl_device_map)} "
        f"egl_map_source={egl_map_source}"
    )
    try:
        while pending and (slot := free_gpu()) is not None:
            launch(pending.pop(0), slot)
        while active or pending:
            for shard in list(active):
                record = active[shard]
                code = record["process"].poll()
                if code is None:
                    continue
                record["log_stream"].close()
                gpu_load[int(record["gpu_slot"])] -= 1
                del active[shard]
                _write_json(
                    status_root / f"{label(shard)}.json",
                    {
                        "suite": shard[0],
                        "task_id": shard[1],
                        "return_code": code,
                        "gpu_visible": record["gpu_visible"],
                        "egl_visible": record["egl_visible"],
                        "output": _portable(record["output"]),
                        "log": _portable(record["log_path"]),
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                if code == 0:
                    completed += 1
                    emit(
                        f"DONE suite={shard[0]} task={shard[1]} return_code=0 "
                        f"progress={completed}/{len(shards)}"
                    )
                else:
                    failures.append(
                        {
                            "suite": shard[0],
                            "task_id": shard[1],
                            "return_code": code,
                            "log": _portable(record["log_path"]),
                        }
                    )
                    emit(
                        f"FAIL suite={shard[0]} task={shard[1]} return_code={code} "
                        f"log={_portable(record['log_path'])}"
                    )
            while pending and (slot := free_gpu()) is not None:
                launch(pending.pop(0), slot)
            now = time.time()
            if now - last_report >= 60.0 or (not pending and not active):
                emit(
                    f"STATUS shards_completed={completed}/{len(shards)} "
                    f"running={len(active)} pending={len(pending)} "
                    f"failed={len(failures)} elapsed_s={int(now - started)} "
                    f"gpu_load={gpu_load}"
                )
                last_report = now
            if pending or active:
                time.sleep(2.0)
    except BaseException:
        for record in active.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(record["process"].pid), signal.SIGTERM)
            record["log_stream"].close()
        raise
    finally:
        run_log.close()

    if failures:
        raise RuntimeError(
            f"{len(failures)} LR-REL shard(s) failed; first: {failures[0]}"
        )
    return {
        "launcher": "repository-popen",
        "shards": len(shards),
        "completed": completed,
        "gpu_ids": gpu_ids,
        "shard_root": _portable(shard_root),
    }


def _collect_lerobot_metrics(output: Path, suites: Sequence[str]) -> dict[str, Any]:
    """Aggregate LeRobot's own ``eval_info.json`` reports into the manifest.

    Sequential runs write one report per suite under ``<output>/<suite>/``;
    sharded runs write one per (suite, task) under ``<output>/shards/``.  Both
    layouts are read so the manifest is self-documenting either way, and totals
    are weighted by episode count rather than averaging percentages.
    """

    def read(path: Path) -> tuple[float, int] | None:
        if not path.is_file():
            return None
        overall = _read_json(path).get("overall", {})
        percent = overall.get("pc_success")
        episodes = overall.get("n_episodes")
        if not isinstance(percent, (int, float)) or not isinstance(episodes, int):
            return None
        return float(percent), episodes

    metrics: dict[str, Any] = {}
    grand_success = 0.0
    grand_episodes = 0
    for suite in suites:
        shard_reports = sorted((output / "shards").glob(f"{suite}_task*/eval_info.json"))
        collected: list[tuple[float, int]] = []
        detail: dict[str, Any] = {}
        for report in shard_reports:
            parsed = read(report)
            if parsed is None:
                detail[report.parent.name] = {"status": "unreadable"}
                continue
            collected.append(parsed)
            detail[report.parent.name] = {"pc_success": parsed[0], "n_episodes": parsed[1]}
        if not collected:
            parsed = read(output / suite / "eval_info.json")
            if parsed is None:
                metrics[suite] = {"status": "missing"}
                continue
            collected = [parsed]
        successes = sum(percent / 100.0 * episodes for percent, episodes in collected)
        episodes_total = sum(episodes for _, episodes in collected)
        metrics[suite] = {
            "pc_success": round(100.0 * successes / episodes_total, 2),
            "successes": round(successes),
            "n_episodes": episodes_total,
            "reports": len(collected),
        }
        if detail:
            metrics[suite]["per_task"] = detail
        grand_success += successes
        grand_episodes += episodes_total
    if grand_episodes:
        metrics["aggregate"] = {
            "successes": round(grand_success),
            "n_episodes": grand_episodes,
            "pc_success": round(100.0 * grand_success / grand_episodes, 2),
        }
    return metrics


def eval_lerobot_released(
    config: Mapping[str, Any],
    *,
    policy_path: str | Path,
    output_root: str | Path,
    vendor_root: str | Path,
    trials: int,
    smoke: bool,
    api_smoke: bool,
    dry_run: bool,
    gpu_id: str | None = None,
    egl_device_map_spec: str | None = None,
    visual_normalization: str = "identity",
    suite: str | None = None,
    num_gpus: int = 1,
    observation_size: int = 224,
) -> dict[str, Any]:
    _require_resource_phase(config, "eval-lerobot-released", dry_run=dry_run)
    if suite is not None and suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITES}, got {suite!r}")
    if visual_normalization not in VISUAL_NORMALIZATION_CHOICES:
        raise ValueError(
            f"visual_normalization must be one of {VISUAL_NORMALIZATION_CHOICES}, "
            f"got {visual_normalization!r}"
        )
    output = _rooted(output_root)
    policy = _rooted(policy_path)
    overlay_record: dict[str, Any] | None = None
    if visual_normalization == "identity" and not dry_run:
        policy, overlay_record = _prepare_lerobot_policy_overlay(
            policy, output / "policy_overlay"
        )
    # LR-REL is a single-process, single-GPU evaluator, so restricting it to one
    # suite is how a 2000-episode run is sharded across GPUs instead of taking
    # ten hours sequentially.
    if smoke:
        selected = [suite or "libero_spatial"]
    else:
        selected = [suite] if suite else list(SUITES)
    if smoke:
        trials = 2
    paths = _artifact_paths(config)
    runtime_env = _model_runtime_env(paths, vendor_root=vendor_root)
    episode_lengths = _require_mapping(
        config["protocol"]["eval_episode_length"], "protocol.eval_episode_length"
    )
    seed = int(config["protocol"]["eval_seed"])

    if num_gpus > 1 and not dry_run:
        # Sharded path: one lerobot-eval child per (suite, task), spread over GPUs.
        gpu_ids = _visible_gpu_ids(num_gpus)
        resolved_egl_map, resolved_egl_source = _egl_device_map(
            gpu_ids,
            explicit=egl_device_map_spec,
            runtime_env=runtime_env,
            allow_calibration=True,
        )
        print(
            f"[W0] LR-REL sharded launcher gpus={gpu_ids} "
            f"egl_device_map={resolved_egl_map} source={resolved_egl_source}",
            flush=True,
        )
        launcher_result = _run_lerobot_workers(
            policy=policy,
            output=output,
            suites=selected,
            episode_lengths=episode_lengths,
            trials=trials,
            seed=seed,
            num_gpus=num_gpus,
            runtime_env=runtime_env,
            egl_device_map=resolved_egl_map,
            egl_map_source=resolved_egl_source,
            observation_size=observation_size,
        )
        commands = [
            _lerobot_worker_command(
                policy=policy,
                suite=selected[0],
                task_id=0,
                trials=trials,
                episode_length=int(episode_lengths[selected[0]]),
                output=output / "shards" / f"{selected[0]}_task00",
                seed=seed,
                observation_size=observation_size,
            )
        ]
        compute_gpu = gpu_ids[0]
        render_gpu = resolved_egl_map[compute_gpu]
    else:
        launcher_result = {"launcher": "sequential", "shards": len(selected)}
        compute_gpu = str(gpu_id) if gpu_id is not None else _visible_gpu_ids(1)[0]
        resolved_egl_map, resolved_egl_source = _egl_device_map(
            [compute_gpu],
            explicit=egl_device_map_spec,
            runtime_env=runtime_env,
            allow_calibration=not dry_run,
        )
        render_gpu = resolved_egl_map[compute_gpu]
        child_env = _worker_environment(runtime_env, compute_gpu, render_gpu)
        print(
            f"[W0] LR-REL EGL mapping compute_cuda={compute_gpu} "
            f"render_egl={render_gpu} source={resolved_egl_source} "
            f"CUDA_VISIBLE_DEVICES={child_env['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
        commands = []
        for suite_name in selected:
            suite_output = output / suite_name
            command = _lerobot_worker_command(
                policy=policy,
                suite=suite_name,
                task_id=-1,
                trials=trials,
                episode_length=int(episode_lengths[suite_name]),
                output=suite_output,
                seed=seed,
                observation_size=observation_size,
            )
            # task_id=-1 means "every task in the suite": drop the filter.
            command = [item for item in command if not item.startswith("--env.task_ids=")]
            commands.append(command)
            _run(
                command,
                env=child_env,
                log_path=suite_output / "run.log",
                dry_run=dry_run,
            )
    payload = {
        "schema_version": 1,
        "variant": "LR-REL",
        "visual_normalization": visual_normalization,
        "policy_overlay": overlay_record,
        "lerobot_metrics": (
            {} if dry_run else _collect_lerobot_metrics(output, selected)
        ),
        "status": "dry_run" if dry_run else "command_complete",
        "api_pass": not dry_run,
        "quantitative_eval": bool(not api_smoke and not smoke),
        "suites": selected,
        "suite_filter": suite,
        "num_gpus": num_gpus,
        "observation_size": observation_size,
        "launcher": launcher_result,
        "trials_per_task": trials,
        "compute_cuda_device": compute_gpu,
        "egl_render_device_id": render_gpu,
        # The render stack must be recorded, not just the map: a node change silently
        # altered rendering behaviour, and because no eval manifest carried the
        # fingerprint -- and the calibration file had been overwritten -- the first
        # host's driver and GPUs were unrecoverable and the difference could not be
        # explained.
        "render_fingerprint": _render_fingerprint(runtime_env),
        "egl_device_map": resolved_egl_map,
        "egl_map_source": resolved_egl_source,
        "commands": [[_portable(item) if Path(item).is_absolute() else item for item in cmd] for cmd in commands],
        "note": "LeRobot output is kept separate from the upstream aggregate; inspect its native metrics before parity claims.",
    }
    _write_json(output / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _pair_contract(
    config: Mapping[str, Any],
    dataset_root: Path,
    action_dit: Path,
    *,
    train_seed: int | None = None,
) -> dict[str, Any]:
    """Freeze everything the two arms must share, keyed by the effective seed.

    The seed enters ``init_hash``, so a seed-42 pair and a seed-43 pair produce
    different contracts and cannot be mixed under one output parent.  That is the
    isolation multi-seed runs need: each seed gets its own
    ``pair_contract.json``.
    """

    training = config["training"]
    seed = int(config["protocol"]["train_seed"] if train_seed is None else train_seed)
    contract = {
        "seed": seed,
        "dataset_root": _portable(dataset_root),
        "dataset_order": [f"{suite}_no_noops_lerobot" for suite in SUITES],
        "action_dit_init": _portable(action_dit),
        "model_id": config["sources"]["wan_model_id"],
        "tokenizer_id": config["sources"]["tokenizer_model_id"],
        "batch_size_per_gpu": int(training["per_gpu_batch_size"]),
        "global_batch_size": int(training["global_batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "lr_scheduler_type": training["lr_scheduler_type"],
        "mixed_precision": training["mixed_precision"],
        "max_grad_norm": float(training["max_grad_norm"]),
        "noise_seed": seed,
        "data_order_seed": seed,
        "update_seed": seed,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "schema_version": 1,
        "contract": contract,
        "init_hash": digest,
        "data_hash": hashlib.sha256(encoded + b"/data").hexdigest(),
        "noise_hash": hashlib.sha256(encoded + b"/noise").hexdigest(),
        "update_hash": hashlib.sha256(encoded + b"/update").hexdigest(),
        "global_batch_size": int(training["global_batch_size"]),
    }


def _latest_checkpoint(output: Path, steps: int) -> Path | None:
    expected = output / "checkpoints" / "weights" / f"step_{steps:06d}.pt"
    if expected.exists():
        return expected
    candidates = sorted((output / "checkpoints" / "weights").glob("step_*.pt"))
    return candidates[-1] if candidates else None


def _prepare_text_embeddings(
    *,
    config: Mapping[str, Any],
    dataset: Path,
    text_cache: Path,
    wan: Path,
    tokenizer: Path,
    vendor_root: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    """Run the pinned text-cache preparation exactly once before training."""

    if not dry_run and text_cache.exists() and any(text_cache.glob("*.pt")):
        print(f"[W0] reuse text embedding cache: {_portable(text_cache)}", flush=True)
        return
    dirs = [dataset / f"{suite}_no_noops_lerobot" for suite in SUITES]
    dataset_override = "[" + ",".join(str(path) for path in dirs) + "]"
    command = [
        *_conda_run(
            UPSTREAM_ENV,
            "python",
            "scripts/precompute_text_embeds.py",
            f"task={TASK_NAME}",
            f"data.train.dataset_dirs={dataset_override}",
            f"data.train.text_embedding_cache_dir={text_cache}",
            f"model.model_id={config['sources']['wan_model_id']}",
            f"model.tokenizer_model_id={config['sources']['tokenizer_model_id']}",
            "+overwrite=false",
        )
    ]
    _run(
        command,
        cwd=vendor_root / "FastWAM",
        env=_model_runtime_env(_artifact_paths(config), vendor_root=vendor_root),
        log_path=log_path,
        dry_run=dry_run,
    )


def _report_training_env(message: str) -> None:
    """Announce an env decision on stderr.

    stdout must stay parseable: the launcher evaluates ``conda run`` output, and a line
    containing parentheses once broke activation with ``syntax error near unexpected
    token '('``, which silently fell back to the wrong environment and then failed on a
    missing DeepSpeed.
    """

    sys.stderr.write(f"[W0] {message}\n")
    sys.stderr.flush()


def _training_allocator_env() -> dict[str, str]:
    """Default the causal-pair training run to a fragmentation-tolerant allocator.

    Measured pressure: the single-GPU attempt failed allocating 22.43 GiB while
    reporting ``29.81 GiB is reserved by PyTorch but unallocated``, and the working
    8-GPU run sits at ~74.8 GiB of 95.2 GiB per rank -- roughly 20 GiB of headroom
    for a 20k-step run that still has to write periodic checkpoints.
    ``expandable_segments:True`` grows segments through CUDA virtual memory instead
    of fixed-size blocks.

    This is on by default rather than left to the operator because it is
    numerically inert (allocator behaviour only, unlike gradient checkpointing or a
    different ZeRO stage) while forgetting it costs an OOM hours into training.  It
    is scoped to the training path, so evaluation processes stay identical to the
    runs that produced the recorded UP-REL/LR-REL numbers.

    An operator-supplied value wins: ``_run`` merges this mapping over
    ``os.environ`` and would otherwise discard a deliberate setting.
    """

    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if existing:
        _report_training_env(f"keeping operator PYTORCH_CUDA_ALLOC_CONF={existing}")
        return {}
    return {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def _training_debug_env() -> dict[str, str]:
    """Turn on NCCL FlightRecorder so a hung collective is diagnosable after the fact.

    A 21-minute 8-GPU run died to a 600 s ALLREDUCE timeout in which seven ranks reported
    ``last completed work: 11911`` while rank 3 reported nothing -- i.e. one rank never
    reached the collective. The log then said the stack trace was unavailable "because
    FlightRecorder is disabled". These variables only record collective metadata into a
    ring buffer and dump it when a collective times out; they change no numerics. Since the
    cost of not having them is losing an entire multi-hour run to reproduce the hang, they
    default on rather than living in a runbook.

    ``TORCH_NCCL_DESYNC_DEBUG`` is deliberately not set: it names the desynced rank but
    carries documented per-collective overhead, and throughput on this host is already
    below the measured baseline. Set it manually when a hang is reproducible.

    An operator-supplied value wins, because ``_run`` merges this mapping over
    ``os.environ`` and would otherwise discard a deliberate setting.
    """

    defaults = {
        # Ring buffer of recent collectives, dumped on timeout. Named by the very error
        # message this is meant to answer.
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
        # Include C++ frames; symbolization happens at dump time, not per collective.
        "TORCH_NCCL_TRACE_CPP_STACK": "1",
    }
    env: dict[str, str] = {}
    for key, value in defaults.items():
        existing = os.environ.get(key)
        if existing:
            _report_training_env(f"keeping operator {key}={existing}")
            continue
        env[key] = value
    return env


def _assert_sibling_budget_matches(
    output: Path, variant: str, *, train_epochs: int | None, steps: int
) -> dict[str, Any] | None:
    """Reject a causal pair whose two groups were given different step budgets.

    The two groups (``with_video`` / ``no_video``) are launched as two separate commands
    roughly 19 hours apart. ``--steps`` is always required, so forgetting
    ``--train-epochs`` on the second command silently produces one 10-epoch run and one
    20k-step run -- different cosine schedule lengths, different warmup, comparison void,
    and nothing would surface it until evaluation. ``pair_contract.json`` cannot catch this
    because it pins initialization/data/noise, not the budget. So compare against whichever
    sibling manifest already exists in the same output parent.
    """

    other = "no_video" if variant == "with_video" else "with_video"
    sibling = output.parent / other / "run_manifest.json"
    if not sibling.is_file():
        return None
    manifest = _read_json(sibling)
    want_mode = "epochs" if train_epochs is not None else "max_steps"
    got_mode = manifest.get("step_budget_mode")
    if got_mode is None:
        # Written before budget mode was recorded; infer it from the stored fields.
        got_mode = "epochs" if manifest.get("train_epochs") is not None else "max_steps"
    got_budget = manifest.get("train_epochs") if got_mode == "epochs" else manifest.get("steps")
    want_budget = train_epochs if want_mode == "epochs" else steps
    if got_mode != want_mode or got_budget != want_budget:
        raise ValueError(
            "causal pair groups must share one step budget: "
            f"{other} used {got_mode}={got_budget} but this {variant} run uses "
            f"{want_mode}={want_budget}. Pass the same --train-epochs/--steps, or put "
            "this run under a different --output-root parent."
        )
    return {"sibling": _portable(sibling), "budget_mode": got_mode, "budget": got_budget}


def train_causal_variant(
    config: Mapping[str, Any],
    *,
    variant: str,
    steps: int,
    num_gpus: int,
    output_root: str | Path,
    vendor_root: str | Path,
    dry_run: bool,
    train_seed: int | None = None,
    train_epochs: int | None = None,
    train_num_workers: int | None = None,
) -> dict[str, Any]:
    if variant not in {"with_video", "no_video"}:
        raise ValueError("variant must be with_video or no_video")
    # Multi-seed runs are how the causal claim gets a noise reference: the two arms
    # must share a seed, and each seed needs its own output parent so the pair
    # contract stays per-seed.
    effective_seed = int(
        config["protocol"]["train_seed"] if train_seed is None else train_seed
    )
    # Epoch mode reproduces the upstream recipe exactly. The task config ships
    # num_epochs=10 / max_steps=null, and trainer._estimate_total_train_steps then derives
    # the total from the dataset -- which also sets the cosine schedule length and the 5%
    # warmup. Pinning max_steps instead changes the LR curve, not just the run length, so
    # "20000 steps" is not the same experiment as "10 epochs".
    if train_epochs is not None and train_epochs <= 0:
        raise ValueError("train_epochs must be positive")
    formal = train_epochs is not None or steps >= int(config["training"]["steps"])
    if formal and not dry_run:
        preflight_path = _find_preflight(config)
        if preflight_path is None or not _read_json(preflight_path)["resource_gate"]["causal_training"]:
            raise RuntimeError("causal training requires the preflight 8x>=80GiB GPU gate")
    if steps <= 0 or num_gpus <= 0:
        raise ValueError("steps and num_gpus must be positive")
    paths = _artifact_paths(config)
    output = _rooted(output_root)
    dataset = paths["dataset"]
    action_dit = paths["action_dit"]
    text_cache = dataset.parent.parent / "text_embeds_cache/libero"
    _prepare_text_embeddings(
        config=config,
        dataset=dataset,
        text_cache=text_cache,
        wan=paths["wan"],
        tokenizer=paths["tokenizer"],
        vendor_root=_rooted(vendor_root),
        log_path=_rooted(output_root) / "text_embeds.log",
        dry_run=dry_run,
    )
    pair_path = output.parent / "pair_contract.json"
    _assert_freeze_switch_is_wired(_rooted(vendor_root))
    pair = _pair_contract(config, dataset, action_dit, train_seed=effective_seed)
    if pair_path.exists():
        existing = _read_json(pair_path)
        if existing.get("init_hash") != pair["init_hash"]:
            raise ValueError(
                "existing causal pair contract differs from current config: "
                f"{_portable(pair_path)} has seed="
                f"{existing.get('contract', {}).get('seed')} while this run uses "
                f"seed={effective_seed}. Give each seed its own --output-root parent."
            )
    else:
        _write_json(pair_path, pair)
    budget_match = _assert_sibling_budget_matches(
        output, variant, train_epochs=train_epochs, steps=steps
    )
    lambda_video = float(config["causal_pair"]["variants"][variant]["lambda_video"])
    dirs = [dataset / f"{suite}_no_noops_lerobot" for suite in SUITES]
    dataset_override = "[" + ",".join(str(path) for path in dirs) + "]"
    command = [
        *_conda_run(
            UPSTREAM_ENV,
            "bash",
            "scripts/train_zero1.sh",
            str(num_gpus),
        ),
        f"task={TASK_NAME}",
        f"output_dir={output}",
        # max_steps=null hands step budgeting to num_epochs; both must be passed so the
        # value cannot be inherited from an earlier override or a stale task default.
        *(
            [f"num_epochs={train_epochs}", "max_steps=null"]
            if train_epochs is not None
            else [f"max_steps={steps}"]
        ),
        f"batch_size={config['training']['per_gpu_batch_size']}",
        # num_workers is numerically inert: the sampler order comes from the seed, so the
        # batches are bit-identical regardless of how many processes fetch them. It is
        # exposed because it is the main lever on host RAM. Four runs died to a cgroup OOM
        # (memory.failcnt=25, peak sitting on the 1788.9 GiB limit) in which a SIGKILLed
        # DataLoader worker left one rank stuck in queue.get while the other seven spun in
        # DeepSpeed's gradient allreduce until the 600 s NCCL watchdog killed the job.
        # 8 workers per rank means 64 pyav decoder processes, because the torchcodec fast
        # path is unusable here: it needs FFmpeg 4-7 and the image ships FFmpeg 8.
        f"num_workers={train_num_workers if train_num_workers is not None else config['training']['num_workers']}",
        f"learning_rate={config['training']['learning_rate']}",
        f"weight_decay={config['training']['weight_decay']}",
        f"lr_scheduler_type={config['training']['lr_scheduler_type']}",
        f"mixed_precision={config['training']['mixed_precision']}",
        f"max_grad_norm={config['training']['max_grad_norm']}",
        f"save_every={int(config['training']['save_every']) if train_epochs is not None else min(int(config['training']['save_every']), steps)}",
        # The task config logs every 10 steps, so a 5-step smoke would finish
        # without ever emitting a loss and the run could not report the required
        # train-loss field. Short runs must show per-step progress.
        f"log_every={1 if (train_epochs is None and steps <= 10) else 10}",
        "eval_every=0",
        f"seed={effective_seed}",
        # `lambda_video` is absent from configs/model/fastwam.yaml, where the
        # pinned code defaults it to 1.0 (`runtime.py`:
        # `loss.get("lambda_video", 1.0)`), so Hydra's struct mode needs `+` to add
        # it. `lambda_action` is present in that config and must not take `+`.
        # This is what makes the pair causal: FW-V's explicit 1.0 reproduces the
        # released recipe, FW-NV's 0.0 removes the video term from
        # `loss_total = lambda_video * loss_video + lambda_action * loss_action`.
        f"+model.loss.lambda_video={lambda_video}",
        "model.loss.lambda_action=1.0",
        "model.load_text_encoder=false",
        "model.skip_dit_load_from_pretrain=false",
        f"model.action_dit_pretrained_path={action_dit}",
        f"model.model_id={config['sources']['wan_model_id']}",
        f"model.tokenizer_model_id={config['sources']['tokenizer_model_id']}",
        f"data.train.dataset_dirs={dataset_override}",
        f"data.train.text_embedding_cache_dir={text_cache}",
    ]
    _write_run_manifest(
        output,
        {
            "schema_version": 1,
            "variant": variant,
            "status": "planned" if dry_run else "running",
            "lambda_video": lambda_video,
            "train_seed": effective_seed,
            "train_epochs": train_epochs,
        # Self-describing: FW-FZ is also a no_video run, so the manifest must say which arm
        # it is or the two become indistinguishable after the fact.
        "freeze_video_expert": os.environ.get("W0_FREEZE_VIDEO_EXPERT", "").strip() not in ("", "0"),
        "num_workers": (
            train_num_workers
            if train_num_workers is not None
            else int(config["training"]["num_workers"])
        ),
            "step_budget_mode": "epochs" if train_epochs is not None else "max_steps",
            "sibling_budget_check": budget_match,
            "steps": None if train_epochs is not None else steps,
            "num_gpus": num_gpus,
            "pair_contract": _portable(pair_path),
            "init_hash": pair["init_hash"],
            "data_hash": pair["data_hash"],
            "noise_hash": pair["noise_hash"],
            "update_hash": pair["update_hash"],
            "global_batch_size": pair["global_batch_size"],
            "command": [_portable(item) if Path(item).is_absolute() else item for item in command],
        },
    )
    _run(
        command,
        cwd=_rooted(vendor_root) / "FastWAM",
        env={
            **_model_runtime_env(paths, vendor_root=vendor_root),
            # `import deepspeed` probes `$CUDA_HOME/bin/nvcc -V` at import time and
            # dies with FileNotFoundError when the prefix has no compiler. Only the
            # training launcher imports DeepSpeed, so the CUDA_HOME fallback is
            # scoped here instead of applying to evaluation processes.
            "W0_CUDA_HOME_FALLBACK": "1",
            **_training_allocator_env(),
            **_training_debug_env(),
        },
        log_path=output / "train.log",
        dry_run=dry_run,
    )
    if dry_run:
        return {"status": "dry_run", "command": command}
    checkpoint = _latest_checkpoint(output, steps)
    if checkpoint is None:
        raise FileNotFoundError(f"no FastWAM checkpoint found under {_portable(output)}")
    canonical = output / f"checkpoint_step_{steps:08d}.pt"
    if checkpoint.resolve() != canonical.resolve():
        shutil.copy2(checkpoint, canonical)
    manifest = _read_json(output / "run_manifest.json")
    manifest.update({"status": "complete", "checkpoint": _portable(canonical)})
    _write_run_manifest(output, manifest)
    return manifest


def eval_causal_pair(
    config: Mapping[str, Any],
    *,
    with_video: str | Path,
    no_video: str | Path,
    seed_manifest: str | Path | None,
    pair_contract: str | Path | None,
    output_root: str | Path,
    vendor_root: str | Path,
    trials: int,
    num_gpus: int,
    smoke: bool,
    deployment_evidence: str | Path | None,
    dry_run: bool,
    egl_device_map_spec: str | None = None,
) -> dict[str, Any]:
    _require_resource_phase(config, "eval-causal-pair", dry_run=dry_run)
    output = _rooted(output_root)
    left = _rooted(with_video)
    right = _rooted(no_video)
    # Keep the requested GPU count during smoke.  A one-task, one-GPU smoke only proves that
    # one evaluator can load the checkpoint; it cannot expose multi-process EGL/render races.
    # Ten tasks in one suite give an 8-GPU smoke eight concurrent workers while keeping the
    # episode budget small (2 trials per task per checkpoint).
    smoke_suite = "libero_spatial" if smoke else None
    smoke_task_id = None
    if smoke:
        trials = 2
    if seed_manifest and not dry_run:
        manifest = _read_json(seed_manifest)
        if int(manifest.get("seed", -1)) != int(config["protocol"]["eval_seed"]):
            raise ValueError("causal evaluation seed manifest does not match protocol.eval_seed")
        if int(manifest.get("trials_per_task", -1)) != trials:
            raise ValueError("causal evaluation seed manifest does not match trials-per-task")
    for path in (left, right):
        if not dry_run and not path.exists():
            raise FileNotFoundError(path)
    summaries: dict[str, dict[str, Any]] = {}
    for name, checkpoint in (("with_video", left), ("no_video", right)):
        target = output / name
        result = eval_upstream_released(
            config,
            checkpoint=checkpoint,
            output_root=target,
            vendor_root=vendor_root,
            trials=trials,
            num_gpus=num_gpus,
            smoke=False,
            suite=smoke_suite,
            task_id=smoke_task_id,
            dry_run=dry_run,
            egl_device_map_spec=egl_device_map_spec,
        )
        if not dry_run:
            summaries[name] = result
    if dry_run:
        return {"status": "dry_run"}
    pair = _read_json(pair_contract) if pair_contract else {}
    contract = {
        "equal_init_data_noise_update_batch": bool(
            pair.get("init_hash")
            and pair.get("data_hash")
            and pair.get("noise_hash")
            and pair.get("update_hash")
            and int(pair.get("global_batch_size", -1)) == int(config["training"]["global_batch_size"])
        ),
        "init_hash": pair.get("init_hash"),
        "data_hash": pair.get("data_hash"),
        "noise_hash": pair.get("noise_hash"),
        "update_hash": pair.get("update_hash"),
        "global_batch_size": pair.get("global_batch_size"),
    }
    comparison = compare_causal(
        summaries["with_video"],
        summaries["no_video"],
        repeats=10000,
        seed=20260817,
        contract_fields=contract,
    )
    _write_json(output / "comparison.json", comparison)
    if deployment_evidence:
        deployment = _read_json(deployment_evidence)
    else:
        deployment = {
            "schema_version": 1,
            "status": "pending_runtime_trace",
            "note": "supply measured deployment evidence before gate-w0",
        }
    _write_json(output / "deployment.json", deployment)
    return comparison


def _validate_cli_pins(args: argparse.Namespace) -> None:
    expected = {
        "fastwam_commit": FASTWAM_COMMIT,
        "lerobot_commit": LEROBOT_COMMIT,
        "upstream_revision": UPSTREAM_HF_REVISION,
        "lerobot_revision": LEROBOT_HF_REVISION,
    }
    for attribute, value in expected.items():
        supplied = getattr(args, attribute)
        if supplied != value:
            raise ValueError(f"--{attribute.replace('_', '-')} is frozen to {value}")


def _dispatch(args: argparse.Namespace) -> Any:
    config = load_config(args.config)
    _validate_cli_pins(args)
    if args.phase == "validate-config":
        result = validate_config(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    validate_config(config)
    dry_run = bool(args.dry_run)
    if args.phase == "preflight":
        return preflight(config, args.output_root or _default_output(config, "preflight"))
    if args.phase in {"validate-environments", "bootstrap-environments"}:
        return validate_environments(
            config,
            vendor_root=args.vendor_root or _vendor_root(config),
            output_root=args.output_root or _default_output(config, "environment"),
            dry_run=dry_run,
        )
    if args.phase in {"validate-artifacts", "fetch-artifacts"}:
        if args.skip_action_dit:
            raise ValueError(
                "--skip-action-dit is obsolete: validation requires a precomputed local ActionDiT file"
            )
        return validate_artifacts(
            config,
            cache_root=args.cache_root or _config_path(config, "cache_root"),
            output_root=args.output_root or _default_output(config, "artifacts"),
            vendor_root=args.vendor_root or _vendor_root(config),
            dry_run=dry_run,
        )
    if args.phase == "eval-upstream-released":
        checkpoint = args.checkpoint or _artifact_paths(config)["upstream_checkpoint"]
        return eval_upstream_released(
            config,
            checkpoint=checkpoint,
            output_root=args.output_root or _default_output(config, "upstream_released"),
            vendor_root=args.vendor_root or _vendor_root(config),
            trials=args.trials_per_task,
            num_gpus=args.num_gpus,
            smoke=args.smoke,
            suite=args.suite,
            task_id=args.task_id,
            dry_run=dry_run,
            overwrite=args.overwrite,
            max_tasks_per_gpu=args.max_tasks_per_gpu,
            trials_per_worker=args.trials_per_worker,
            egl_device_map_spec=args.egl_device_map,
        )
    if args.phase == "eval-lerobot-released":
        policy = args.policy_path or _artifact_paths(config)["lerobot"]
        return eval_lerobot_released(
            config,
            policy_path=policy,
            output_root=args.output_root or _default_output(config, "lerobot_released"),
            vendor_root=args.vendor_root or _vendor_root(config),
            trials=args.trials_per_task,
            smoke=args.smoke,
            api_smoke=args.api_smoke,
            dry_run=dry_run,
            gpu_id=args.gpu_id,
            egl_device_map_spec=args.egl_device_map,
            visual_normalization=args.visual_normalization,
            suite=args.suite,
            num_gpus=args.num_gpus,
            observation_size=args.observation_size,
        )
    if args.phase == "train-causal-pair":
        return train_causal_variant(
            config,
            variant=args.variant,
            steps=args.steps,
            num_gpus=args.num_gpus,
            output_root=args.output_root or _default_output(config, f"train/{args.variant}"),
            vendor_root=args.vendor_root or _vendor_root(config),
            dry_run=dry_run,
            train_seed=args.train_seed,
            train_epochs=args.train_epochs,
            train_num_workers=args.train_num_workers,
        )
    if args.phase == "eval-causal-pair":
        if not args.with_video or not args.no_video:
            raise ValueError("eval-causal-pair requires --with-video and --no-video")
        return eval_causal_pair(
            config,
            with_video=args.with_video,
            no_video=args.no_video,
            seed_manifest=args.seed_manifest,
            pair_contract=args.pair_contract,
            output_root=args.output_root or _default_output(config, "causal_eval"),
            vendor_root=args.vendor_root or _vendor_root(config),
            trials=args.trials_per_task,
            num_gpus=args.num_gpus,
            smoke=args.smoke,
            deployment_evidence=args.deployment_evidence,
            dry_run=dry_run,
            egl_device_map_spec=args.egl_device_map,
        )
    if args.phase == "gate-w0":
        if dry_run:
            result = {
                "schema_version": 1,
                "status": "dry_run",
                "required_inputs": ["released", "causal", "deployment"],
                "note": "formal gate was not evaluated",
            }
            _write_json(args.output or _default_output(config, "w0_gate.json"), result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return result
        if not args.released or not args.causal or not args.deployment:
            raise ValueError("gate-w0 requires --released, --causal, and --deployment")
        released = _read_json(args.released)
        causal = _read_json(args.causal)
        deployment = _read_json(args.deployment)
        lerobot_path = args.lerobot or _default_output(config, "lerobot_released") / "summary.json"
        lerobot = _read_json(lerobot_path) if lerobot_path.exists() else None
        result = gate_w0(
            released,
            causal,
            deployment,
            lerobot=lerobot,
            causal_min_gap=float(config["paper_reference"]["causal_min_gap"]),
            suite_reference=config["paper_reference"]["suite_success_rates"],
            suite_tolerance=float(config["paper_reference"]["suite_tolerance"]),
            expected_bootstrap_repeats=args.bootstrap_repeats,
            expected_bootstrap_seed=args.bootstrap_seed,
        )
        _write_json(args.output or _default_output(config, "w0_gate.json"), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if args.phase == "audit-deployment":
        if args.deployment_evidence:
            result = _read_json(args.deployment_evidence)
        else:
            result = {
                "schema_version": 1,
                "status": "pending_runtime_trace",
                "note": "replace with measured one-task deployment trace before gate-w0",
            }
        _write_json(args.output or _default_output(config, "deployment.json"), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    raise ValueError(f"unsupported W0 phase: {args.phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")
    parser.add_argument("--output")
    parser.add_argument("--vendor-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy-path")
    parser.add_argument("--fastwam-commit", default=FASTWAM_COMMIT)
    parser.add_argument("--lerobot-commit", default=LEROBOT_COMMIT)
    parser.add_argument("--upstream-revision", default=UPSTREAM_HF_REVISION)
    parser.add_argument("--lerobot-revision", default=LEROBOT_HF_REVISION)
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument(
        "--trials-per-worker",
        type=int,
        default=0,
        help=(
            "number of sequential trials owned by one evaluator child process. "
            "0 (default) puts every trial of a task in one process, so the 12 GiB "
            "checkpoint is loaded once per task instead of once per trial. Set to 1 "
            "only when a native renderer crash has to be isolated per trial."
        ),
    )
    parser.add_argument(
        "--observation-size",
        type=int,
        default=224,
        help=(
            "LR-REL camera render resolution. 224 (default) renders natively at the "
            "model's input size; 256 matches the official evaluator, which rasterises "
            "at LIBERO_ENV_RESOLUTION=256 and then downsamples to 224."
        ),
    )
    parser.add_argument(
        "--visual-normalization",
        choices=VISUAL_NORMALIZATION_CHOICES,
        default="identity",
        help=(
            "LR-REL image normalization. 'identity' (default) serves the released "
            "checkpoint through a symlink overlay whose policy_preprocessor.json sets "
            "norm_map.VISUAL=IDENTITY, matching the pinned LeRobot core that already "
            "maps [0,1]->[-1,1]. 'checkpoint' keeps the released MEAN_STD(0.5,0.5) and "
            "reproduces the doubled mapping for A/B."
        ),
    )
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument(
        "--egl-device-map",
        default=None,
        help=(
            "override the CUDA->EGL render device mapping as 'cuda:egl' pairs, "
            "e.g. '0:0,1:1,7:3'. MUJOCO_EGL_DEVICE_ID is an eglQueryDevicesEXT "
            "list index, not a CUDA id; unmapped pairs can abort natively inside "
            "mjr_readPixels. Defaults to the measured calibration report, then identity."
        ),
    )
    parser.add_argument(
        "--max-tasks-per-gpu",
        type=int,
        default=DEFAULT_MAX_UPSTREAM_TASKS_PER_GPU,
        help="maximum concurrent evaluator workers per GPU (default: 1 for EGL stability)",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--variant", choices=("with_video", "no_video"), default="with_video")
    parser.add_argument("--suite")
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="physical CUDA device for the single-process LR-REL evaluator; defaults to the first visible GPU",
    )
    parser.add_argument("--with-video")
    parser.add_argument("--no-video")
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help=(
            "reproduce the upstream recipe by budgeting in epochs (pass 10 to match "
            "configs/task/libero_uncond_2cam224_1e-4.yaml num_epochs=10 / max_steps=null). "
            "Overrides --steps and makes the run formal, because the epoch budget also "
            "determines the cosine schedule length and the 5 percent warmup."
        ),
    )
    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader workers per rank (default: the protocol value). Numerically inert, "
            "because the sampler order comes from the seed and the batches are therefore "
            "bit-identical, but it is the main lever on host RAM: a cgroup OOM killing one "
            "worker wedges its rank and takes the whole job down via the NCCL watchdog."
        ),
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help=(
            "training seed for --phase=train-causal-pair; defaults to protocol."
            "train_seed. Both arms of a pair must use the same value, and each seed "
            "needs its own --output-root parent because pair_contract.json is keyed "
            "by seed."
        ),
    )
    parser.add_argument("--seed-manifest")
    parser.add_argument("--pair-contract")
    parser.add_argument("--deployment-evidence")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--api-smoke", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="archive a non-empty evaluation output directory and reuse the same path",
    )
    parser.add_argument(
        "--skip-action-dit",
        action="store_true",
        help="deprecated compatibility flag; validation always requires local ActionDiT",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = _dispatch(build_parser().parse_args(argv))
    if isinstance(result, Mapping) and result.get("status") == "incomplete":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
