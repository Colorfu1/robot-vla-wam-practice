"""Shared checkpoint, output, serialization, and timing utilities."""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import platform
import shutil
import stat
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
INCOMPLETE_MARKER_TEXT = "Evaluation did not reach its atomic completion step.\n"
_OUTPUT_DIRECTORY_LOCKS: dict[Path, Any] = {}


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


@dataclass(frozen=True)
class ActionPlanStep:
    """One selected ACT action and the policy-space plan that produced it."""

    model_queried: bool
    plan_step_index: int | None
    queried_chunk_policy: torch.Tensor | None
    queued_actions_policy: torch.Tensor
    discarded_actions_policy: torch.Tensor
    new_prediction_chunk_policy: torch.Tensor | None


class ActionPlanTracker:
    """Track ACT queue progression without performing or altering model inference."""

    def __init__(
        self,
        *,
        n_action_steps: int,
        temporal_ensemble_coeff: float | None,
        action_dim: int = 7,
    ) -> None:
        if n_action_steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.n_action_steps = n_action_steps
        self.temporal_ensemble = temporal_ensemble_coeff is not None
        self.action_dim = action_dim
        self._active_chunk: torch.Tensor | None = None
        self._next_index = 0

    def reset_episode(self) -> None:
        self._active_chunk = None
        self._next_index = 0

    def _validate_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(chunk, torch.Tensor)
            or chunk.ndim != 3
            or chunk.shape[0] != 1
            or chunk.shape[2] != self.action_dim
            or chunk.shape[1] == 0
            or not bool(torch.isfinite(chunk).all())
        ):
            shape = getattr(chunk, "shape", None)
            raise ValueError(
                "queried ACT chunk must have finite shape "
                f"[1,horizon,{self.action_dim}], got {shape}"
            )
        if not self.temporal_ensemble and self.n_action_steps > chunk.shape[1]:
            raise ValueError(
                "n_action_steps cannot exceed the queried ACT chunk length: "
                f"{self.n_action_steps} > {chunk.shape[1]}"
            )
        return chunk.detach().clone()

    def record_step(
        self,
        *,
        selected_action_policy: torch.Tensor,
        queried_chunk_policy: torch.Tensor | None,
    ) -> ActionPlanStep:
        if (
            not isinstance(selected_action_policy, torch.Tensor)
            or selected_action_policy.shape != (1, self.action_dim)
            or not bool(torch.isfinite(selected_action_policy).all())
        ):
            shape = getattr(selected_action_policy, "shape", None)
            raise ValueError(
                "selected ACT action must have finite shape "
                f"[1,{self.action_dim}], got {shape}"
            )

        queried = (
            self._validate_chunk(queried_chunk_policy)
            if queried_chunk_policy is not None
            else None
        )
        if self.temporal_ensemble:
            if queried is None:
                raise RuntimeError(
                    "Temporal Ensemble must query predict_action_chunk every control step"
                )
            empty = queried.new_empty((0, self.action_dim))
            return ActionPlanStep(
                model_queried=True,
                plan_step_index=None,
                queried_chunk_policy=queried,
                queued_actions_policy=empty,
                discarded_actions_policy=empty,
                new_prediction_chunk_policy=queried,
            )

        if queried is not None:
            if self._active_chunk is not None:
                raise RuntimeError(
                    "ACT queried a new chunk before the active action queue was exhausted"
                )
            self._active_chunk = queried[0]
            self._next_index = 0
        elif self._active_chunk is None:
            raise RuntimeError("ACT selected an action without an active or newly queried chunk")

        if self._active_chunk is None:
            raise RuntimeError("ACT action plan tracker has no active chunk")
        plan_step_index = self._next_index
        expected = self._active_chunk[plan_step_index : plan_step_index + 1]
        selected = selected_action_policy.detach()
        if not torch.allclose(expected, selected, rtol=1e-5, atol=1e-6):
            max_error = float(torch.max(torch.abs(expected - selected)).detach().cpu())
            raise RuntimeError(
                "ACT selected action does not match the tracked policy-space queue "
                f"at plan index {plan_step_index}; max_abs_error={max_error:.6g}"
            )

        queued = self._active_chunk[
            plan_step_index + 1 : self.n_action_steps
        ].detach().clone()
        discarded = self._active_chunk[self.n_action_steps :].detach().clone()
        self._next_index += 1
        if self._next_index == self.n_action_steps:
            self._active_chunk = None
            self._next_index = 0

        return ActionPlanStep(
            model_queried=queried is not None,
            plan_step_index=plan_step_index,
            queried_chunk_policy=queried,
            queued_actions_policy=queued,
            discarded_actions_policy=discarded,
            new_prediction_chunk_policy=None,
        )


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
    validate_act_libero_contract(config)

    policy = make_policy(
        cfg=config,
        ds_meta=dataset_metadata,
        env_cfg=env_config,
    )
    policy.eval()
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


def validate_act_libero_contract(config: ACTConfig) -> None:
    """Require the exact state, camera, and action contract used by local LIBERO ACT."""

    expected_inputs = {
        "observation.images.image": (3, 256, 256),
        "observation.images.image2": (3, 256, 256),
        "observation.state": (8,),
    }
    input_features = config.input_features or {}
    if set(input_features) != set(expected_inputs):
        raise ValueError(
            "ACT checkpoint LIBERO inputs must be exactly "
            f"{sorted(expected_inputs)}, got {sorted(input_features)}"
        )
    for key, expected_shape in expected_inputs.items():
        actual_shape = tuple(input_features[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"ACT checkpoint input {key} must have shape {expected_shape}, "
                f"got {actual_shape}"
            )
    action_feature = config.action_feature
    if action_feature is None or tuple(action_feature.shape) != (7,):
        shape = None if action_feature is None else tuple(action_feature.shape)
        raise ValueError(f"ACT checkpoint action shape must be (7,), got {shape}")


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
    suite: str | None = None,
    task_id: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Create an evaluator result leaf and hold its explicit-path lifecycle lock."""

    if mode not in {"offline", "libero"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    if (suite is None) != (task_id is None):
        raise ValueError("suite and task_id must either both be supplied or both be omitted")
    if task_id is not None and task_id < 0:
        raise ValueError("task_id must be non-negative")
    if overwrite and output_dir is None:
        raise ValueError("--overwrite requires an explicit --output-dir")
    if output_dir is None:
        run_root = _run_root_from_checkpoint(policy_path)
        if run_root is None:
            raise ValueError(
                "--output-dir is required when --policy-path is not under "
                "<run>/checkpoints/<step>/pretrained_model"
            )
        result_root = run_root / "eval" / mode
        if suite is not None and task_id is not None:
            result_root = result_root / suite / f"task_{task_id:03d}"
        result = result_root / local_timestamp()
    else:
        requested = output_dir.expanduser().absolute()
        if _path_is_symlink(requested):
            raise ValueError(f"Output directory must not be a symbolic link: {requested}")
        result = requested.resolve()
        _acquire_output_directory_lock(result)

    try:
        if result.exists() or result.is_symlink():
            if not overwrite:
                raise FileExistsError(f"Output directory already exists: {result}")
            _validate_overwrite_target(result, mode=mode)
            LOGGER.warning("[Output] overwriting existing evaluator result: %s", result)
            shutil.rmtree(result)
        result.mkdir(parents=True, exist_ok=False)
        (result / ".incomplete").write_text(
            INCOMPLETE_MARKER_TEXT,
            encoding="utf-8",
        )
    except BaseException:
        release_output_directory_lock(result)
        raise
    return result


def _path_is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _is_regular_non_symlink_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _acquire_output_directory_lock(result: Path) -> None:
    if result in _OUTPUT_DIRECTORY_LOCKS:
        raise RuntimeError(f"Output directory is locked by an active evaluator: {result}")
    result.parent.mkdir(parents=True, exist_ok=True)
    lock_path = result.parent / f".{result.name}.evaluation.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"Cannot open evaluator output lock {lock_path}: {exc}") from exc
    lock_stream = os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise RuntimeError(
            f"Output directory is locked by an active evaluator: {result}"
        ) from exc
    try:
        lock_stream.seek(0)
        lock_stream.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "mode": "evaluation-output",
                "acquired_at": iso_timestamp(),
            },
            lock_stream,
        )
        lock_stream.write("\n")
        lock_stream.flush()
    except BaseException:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()
        raise
    _OUTPUT_DIRECTORY_LOCKS[result] = lock_stream


def release_output_directory_lock(output_dir: Path) -> None:
    """Release an explicit evaluator output lock, if this process owns it."""

    result = output_dir.expanduser().resolve()
    lock_stream = _OUTPUT_DIRECTORY_LOCKS.pop(result, None)
    if lock_stream is None:
        return
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    finally:
        lock_stream.close()


def _valid_run_config_sentinel(path: Path, *, mode: str) -> bool:
    if not _is_regular_non_symlink_file(path):
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return False
    return (
        payload.get("schema_version") == 1
        and payload.get("mode") == mode
        and payload.get("status") in {"incomplete", "smoke", "complete"}
        and isinstance(payload.get("created_at"), str)
        and isinstance(payload.get("policy"), dict)
        and payload["policy"].get("type") == "act"
        and isinstance(payload.get("runtime"), dict)
        and isinstance(payload.get("input"), dict)
        and isinstance(payload.get("arguments"), dict)
    )


def _valid_incomplete_sentinel(path: Path) -> bool:
    if not _is_regular_non_symlink_file(path):
        return False
    try:
        return path.read_text(encoding="utf-8") == INCOMPLETE_MARKER_TEXT
    except OSError:
        return False


def _validate_overwrite_target(result: Path, *, mode: str) -> None:
    if _path_is_symlink(result):
        raise ValueError(f"Output directory must not be a symbolic link: {result}")
    if not result.is_dir():
        raise ValueError(f"Overwrite target must be a directory: {result}")
    if not any(result.iterdir()):
        return
    incomplete_path = result / ".incomplete"
    run_config_path = result / "run_config.json"
    for sentinel in (incomplete_path, run_config_path):
        if _path_is_symlink(sentinel):
            raise ValueError(f"Evaluator output sentinel must not be a symlink: {sentinel}")
        if sentinel.exists() and not _is_regular_non_symlink_file(sentinel):
            raise ValueError(f"Evaluator output sentinel must be a regular file: {sentinel}")
    incomplete_exists = incomplete_path.exists()
    run_config_exists = run_config_path.exists()
    valid_incomplete = _valid_incomplete_sentinel(incomplete_path)
    valid_run_config = _valid_run_config_sentinel(run_config_path, mode=mode)
    if incomplete_exists and not valid_incomplete:
        raise ValueError(f"Invalid evaluator incomplete marker: {incomplete_path}")
    if run_config_exists and not valid_run_config:
        raise ValueError(f"Invalid or mismatched evaluator run config: {run_config_path}")
    if valid_incomplete or valid_run_config:
        return
    raise ValueError(
        "Refusing to overwrite a non-empty directory that is not a recognized "
        f"{mode} evaluator result: {result}"
    )


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
    release_output_directory_lock(output_dir)


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
        self.current_chunk_policy: torch.Tensor | None = None
        self._original_bound: Callable[..., Any] | None = None
        self._had_instance_value = False
        self._instance_value: Any = None

    def begin_step(self) -> None:
        self.current_latency_ms = None
        self.current_chunk_policy = None

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
            if not isinstance(result, torch.Tensor):
                raise TypeError(
                    "ACT predict_action_chunk must return a tensor, "
                    f"got {type(result).__name__}"
                )
            self.current_latency_ms = latency
            self.current_chunk_policy = result.detach().clone()
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
