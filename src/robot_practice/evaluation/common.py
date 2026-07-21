"""Shared checkpoint, output, serialization, and timing utilities."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import shutil
import subprocess
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import EnvConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

LOGGER = logging.getLogger(__name__)

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
PROCESSOR_CONFIG_FILES = (
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)
T = TypeVar("T")


@dataclass(frozen=True)
class CheckpointManifest:
    path: Path
    policy_config: dict[str, Any]
    train_config: dict[str, Any] | None
    processor_state_files: tuple[Path, ...]


@dataclass(frozen=True)
class LoadedACT:
    manifest: CheckpointManifest
    config: ACTConfig
    policy: torch.nn.Module
    preprocessor: Any
    postprocessor: Any


def practice_root() -> Path:
    """Return the editable-install workspace root."""

    return Path(__file__).resolve().parents[3]


def local_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _processor_state_files(checkpoint: Path, processor_config: Path) -> list[Path]:
    payload = read_json(processor_config)
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"Processor config {processor_config} must contain a steps array")

    states: list[Path] = []
    checkpoint_resolved = checkpoint.resolve()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"{processor_config} step {index} must be an object")
        state_file = step.get("state_file")
        if state_file is None:
            continue
        if not isinstance(state_file, str) or not state_file:
            raise ValueError(
                f"{processor_config} step {index} has an invalid state_file value"
            )
        relative = Path(state_file)
        if relative.is_absolute():
            raise ValueError(
                f"{processor_config} step {index} state_file must be relative: {state_file}"
            )
        state_path = (checkpoint / relative).resolve()
        if not state_path.is_relative_to(checkpoint_resolved):
            raise ValueError(
                f"{processor_config} step {index} state_file escapes checkpoint: {state_file}"
            )
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Processor state file referenced by {processor_config.name} is missing: "
                f"{state_path}"
            )
        states.append(state_path)
    return states


def validate_complete_checkpoint(policy_path: Path) -> CheckpointManifest:
    """Validate a local LeRobot ``pretrained_model`` directory and processor states."""

    path = policy_path.expanduser().resolve()
    if path.is_file():
        raise ValueError(
            f"--policy-path must be a complete pretrained_model directory, not a single "
            f"weight file: {path}"
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete LeRobot checkpoint. Missing: "
            f"{', '.join(missing)}. A model weight file alone is insufficient because "
            "config and normalization processors are required."
        )

    policy_config = read_json(path / "config.json")
    if policy_config.get("type") != "act":
        raise ValueError(
            f"Expected an ACT checkpoint (config type 'act'), got {policy_config.get('type')!r}"
        )

    state_files: list[Path] = []
    for name in PROCESSOR_CONFIG_FILES:
        state_files.extend(_processor_state_files(path, path / name))

    train_path = path / "train_config.json"
    train_config = read_json(train_path) if train_path.is_file() else None
    return CheckpointManifest(
        path=path,
        policy_config=policy_config,
        train_config=train_config,
        processor_state_files=tuple(sorted(set(state_files))),
    )


def load_act_checkpoint(
    manifest: CheckpointManifest,
    *,
    dataset_metadata: Any | None = None,
    env_config: EnvConfig | None = None,
) -> LoadedACT:
    """Load policy weights and saved processors without modifying site-packages."""

    if (dataset_metadata is None) == (env_config is None):
        raise ValueError("Provide exactly one of dataset_metadata or env_config")

    config = PreTrainedConfig.from_pretrained(
        manifest.path,
        local_files_only=True,
    )
    if not isinstance(config, ACTConfig):
        raise TypeError(f"Expected ACTConfig, got {type(config).__name__}")
    config.pretrained_path = manifest.path

    policy = make_policy(
        cfg=config,
        ds_meta=dataset_metadata,
        env_cfg=env_config,
    )
    policy.eval()
    action_feature = config.action_feature
    if action_feature is None or tuple(action_feature.shape) != (7,):
        shape = None if action_feature is None else tuple(action_feature.shape)
        raise ValueError(f"ACT checkpoint action shape must be (7,), got {shape}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(manifest.path),
        preprocessor_overrides={"device_processor": {"device": str(config.device)}},
    )
    return LoadedACT(
        manifest=manifest,
        config=config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def _run_root_from_checkpoint(policy_path: Path) -> Path | None:
    path = policy_path.resolve()
    if (
        path.name == "pretrained_model"
        and path.parent.parent.name == "checkpoints"
        and path.parent.parent.parent.is_dir()
    ):
        return path.parent.parent.parent
    return None


def create_output_directory(
    *,
    policy_path: Path,
    mode: str,
    output_dir: Path | None,
) -> Path:
    """Create an exclusive result leaf and its incomplete marker."""

    if mode not in {"offline", "libero"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    if output_dir is None:
        run_root = _run_root_from_checkpoint(policy_path)
        if run_root is None:
            raise ValueError(
                "--output-dir is required when --policy-path is not under "
                "<run>/checkpoints/<step>/pretrained_model"
            )
        result = run_root / "eval" / mode / local_timestamp()
    else:
        result = output_dir.expanduser().resolve()

    try:
        result.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"Output directory already exists: {result}") from exc
    (result / ".incomplete").write_text(
        "Evaluation did not reach its atomic completion step.\n",
        encoding="utf-8",
    )
    return result


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON payload contains NaN or Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Strict JSON write followed by an atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                _jsonable(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def mark_run_complete(output_dir: Path) -> None:
    (output_dir / ".incomplete").unlink(missing_ok=True)


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def runtime_metadata(device: str | torch.device) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "lerobot": package_version("lerobot"),
        "torch": torch.__version__,
        "rerun": package_version("rerun-sdk"),
        "device": str(device),
    }


def build_run_config(
    *,
    status: str,
    mode: str,
    loaded: LoadedACT,
    input_config: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "mode": mode,
        "created_at": iso_timestamp(),
        "policy": {
            "path": loaded.manifest.path,
            "type": "act",
            "config": loaded.manifest.policy_config,
            "train_config": loaded.manifest.train_config,
        },
        "runtime": runtime_metadata(loaded.config.device),
        "input": input_config,
        "arguments": arguments,
    }


def log_terminal_metrics(metrics: dict[str, Any], logger: logging.Logger) -> None:
    """Print every aggregate together with its short metric annotation."""

    definitions = metrics.get("metric_definitions", {})
    logger.info("%s aggregate metrics:", metrics.get("mode", "evaluation"))
    for name, value in metrics.get("aggregate", {}).items():
        description = definitions.get(name, "")
        annotation = f" — {description}" if description else ""
        logger.info("  %s=%s%s", name, value, annotation)


def synchronize_device(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def timed_call(
    function: Callable[..., T],
    *args: Any,
    device: str | torch.device,
    **kwargs: Any,
) -> tuple[T, float]:
    synchronize_device(device)
    started = time.perf_counter_ns()
    result = function(*args, **kwargs)
    synchronize_device(device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return result, elapsed_ms


class ModelQueryTimer(AbstractContextManager["ModelQueryTimer"]):
    """Temporarily measure real ``predict_action_chunk`` calls inside ACT selection."""

    def __init__(self, policy: torch.nn.Module, device: str | torch.device):
        self.policy = policy
        self.device = device
        self.current_latency_ms: float | None = None
        self._original_bound: Callable[..., Any] | None = None
        self._had_instance_value = False
        self._instance_value: Any = None

    def begin_step(self) -> None:
        self.current_latency_ms = None

    def __enter__(self) -> "ModelQueryTimer":
        self._had_instance_value = "predict_action_chunk" in self.policy.__dict__
        if self._had_instance_value:
            self._instance_value = self.policy.__dict__["predict_action_chunk"]
        self._original_bound = getattr(self.policy, "predict_action_chunk")

        def measured(*args: Any, **kwargs: Any) -> Any:
            if self.current_latency_ms is not None:
                raise RuntimeError(
                    "ACT called predict_action_chunk more than once in one control step"
                )
            if self._original_bound is None:
                raise RuntimeError("ModelQueryTimer is not active")
            result, latency = timed_call(
                self._original_bound,
                *args,
                device=self.device,
                **kwargs,
            )
            self.current_latency_ms = latency
            return result

        setattr(self.policy, "predict_action_chunk", measured)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._had_instance_value:
            setattr(self.policy, "predict_action_chunk", self._instance_value)
        else:
            delattr(self.policy, "predict_action_chunk")
        self._original_bound = None
        return None


def launch_rerun_viewer(recording_path: Path, *, enabled: bool) -> bool:
    """Open a completed recording without making viewer availability part of success."""

    if not enabled:
        return False
    rerun_binary = shutil.which("rerun")
    if rerun_binary is None:
        LOGGER.warning(
            "Rerun viewer executable is unavailable; recording saved at %s",
            recording_path,
        )
        return False
    try:
        subprocess.Popen(  # noqa: S603 - executable is resolved locally with shutil.which
            [rerun_binary, str(recording_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        LOGGER.warning("Could not open Rerun viewer for %s: %s", recording_path, exc)
        return False
    return True
