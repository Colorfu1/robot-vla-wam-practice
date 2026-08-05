"""Train or resume the SmolVLA LIBERO B0 domain-adaptation policy."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

from .b0_training import (
    FreshArguments,
    LaunchInspection,
    MetricsSink,
    ResumeArguments,
    ResumeContext,
    build_run_config,
    build_train_config,
    configure_smolvla_policy,
    identify_base_policy,
    instrument_update_policy,
    load_background_split,
    parse_arguments,
    portable_path,
    resolve_base_policy,
    resolve_repository_path,
    runtime_path,
    runtime_payload,
    utc_now,
    validate_dataset,
    validate_local_vlm_assets,
    validate_resume,
    write_json_atomic,
)

_RUN_ARTIFACT_FILENAMES = ("run_config.json", "train_metrics.jsonl", "train.log")
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


class _TeeStream:
    def __init__(self, terminal: TextIO, log: TextIO, lock: threading.Lock) -> None:
        self._terminal = terminal
        self._log = log
        self._lock = lock

    def write(self, value: str) -> int:
        with self._lock:
            written = self._terminal.write(value)
            self._log.write(value)
            self._log.flush()
        return written

    def flush(self) -> None:
        with self._lock:
            self._terminal.flush()
            self._log.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._terminal, name)


def _staging_path(output_dir: Path, filename: str) -> Path:
    return output_dir.parent / f".{output_dir.name}.{filename}"


def _clear_run_artifacts(output_dir: Path) -> None:
    """Remove a previous fresh run so ``--overwrite`` can start cleanly."""

    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    for filename in _RUN_ARTIFACT_FILENAMES:
        staging = _staging_path(output_dir, filename)
        if staging.exists():
            staging.unlink()


def _publish_staging_files(output_dir: Path, filenames: tuple[str, ...]) -> None:
    if not output_dir.is_dir():
        return
    for filename in filenames:
        staging = _staging_path(output_dir, filename)
        if not staging.exists():
            continue
        destination = output_dir / filename
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite run artifact: {destination}")
        staging.replace(destination)


@contextmanager
def _attempt_artifacts(
    *,
    output_dir: Path,
    attempt_id: int,
    initial_step: int,
    run_config: dict[str, Any] | None,
    resume: bool,
    overwrite: bool,
    enabled: bool,
) -> Any:
    if not enabled:
        yield None
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        run_config_path = output_dir / "run_config.json"
        metrics_path = output_dir / "train_metrics.jsonl"
        log_path = output_dir / "train.log"
        if not run_config_path.is_file() or not metrics_path.is_file() or not log_path.is_file():
            raise FileNotFoundError("Resume run is missing run_config, metrics, or train log")
        metrics_mode = "a"
        log_mode = "a"
    else:
        if overwrite:
            _clear_run_artifacts(output_dir)
        run_config_path = _staging_path(output_dir, "run_config.json")
        metrics_path = _staging_path(output_dir, "train_metrics.jsonl")
        log_path = _staging_path(output_dir, "train.log")
        existing = [path for path in (run_config_path, metrics_path, log_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Previous incomplete B0 artifacts exist: {existing}")
        if run_config is None:
            raise ValueError("Fresh B0 run requires run_config")
        write_json_atomic(run_config_path, run_config)
        metrics_mode = "x"
        log_mode = "x"

    lock = threading.Lock()
    started_monotonic = time.perf_counter()
    stdout: _TeeStream | None = None
    stderr: _TeeStream | None = None
    try:
        with (
            metrics_path.open(metrics_mode, encoding="utf-8", buffering=1) as metrics_stream,
            log_path.open(log_mode, encoding="utf-8", buffering=1) as log_stream,
        ):
            stdout = _TeeStream(sys.stdout, log_stream, lock)
            stderr = _TeeStream(sys.stderr, log_stream, lock)
            sink = MetricsSink(
                stream=metrics_stream,
                attempt_id=attempt_id,
                initial_step=initial_step,
                started_monotonic=started_monotonic,
            )
            with redirect_stdout(stdout), redirect_stderr(stderr):
                print(
                    f"B0 training attempt {attempt_id:03d} starts at step {initial_step}",
                    flush=True,
                )
                yield sink
    finally:
        root_logger = logging.getLogger()
        for handler in tuple(root_logger.handlers):
            if getattr(handler, "stream", None) not in {stdout, stderr}:
                continue
            root_logger.removeHandler(handler)
            handler.close()
        if not resume:
            _publish_staging_files(output_dir, _RUN_ARTIFACT_FILENAMES)


@contextmanager
def _publish_artifacts_on_checkpoint(
    lerobot_train_module: Any,
    *,
    output_dir: Path,
    enabled: bool,
):
    if not enabled:
        yield
        return
    original = lerobot_train_module.save_checkpoint

    def publishing_save_checkpoint(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        _publish_staging_files(output_dir, _RUN_ARTIFACT_FILENAMES)
        return result

    lerobot_train_module.save_checkpoint = publishing_save_checkpoint
    try:
        yield
    finally:
        lerobot_train_module.save_checkpoint = original


@contextmanager
def _override_offline_tokenizer_processor(
    lerobot_train_module: Any,
    tokenizer_path: Path,
):
    """Point LeRobot's independently loaded tokenizer step at local assets."""

    original = lerobot_train_module.make_pre_post_processors
    tokenizer_reference = portable_path(tokenizer_path)

    def make_offline_processors(*args: Any, **kwargs: Any):
        overrides = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in (kwargs.get("preprocessor_overrides") or {}).items()
        }
        tokenizer_override = dict(overrides.get("tokenizer_processor") or {})
        tokenizer_override["tokenizer_name"] = tokenizer_reference
        overrides["tokenizer_processor"] = tokenizer_override
        kwargs["preprocessor_overrides"] = overrides
        return original(*args, **kwargs)

    lerobot_train_module.make_pre_post_processors = make_offline_processors
    try:
        yield
    finally:
        lerobot_train_module.make_pre_post_processors = original


@contextmanager
def _upstream_resume_argv(config_path: Path | None):
    if config_path is None:
        yield
        return
    original = sys.argv
    sys.argv = [
        original[0],
        f"--config_path={config_path}",
        "--resume=true",
    ]
    try:
        yield
    finally:
        sys.argv = original


def _make_accelerator():
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    return Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        cpu=False,
    )


def _validate_accelerator(
    accelerator: Any,
    *,
    expected_processes: int,
    expected_mixed_precision: str,
) -> None:
    import torch

    if accelerator.num_processes != expected_processes:
        raise RuntimeError(
            f"Expected {expected_processes} Accelerate processes, found {accelerator.num_processes}"
        )
    if accelerator.mixed_precision != expected_mixed_precision:
        raise RuntimeError(
            f"Expected mixed precision {expected_mixed_precision}, "
            f"found {accelerator.mixed_precision}"
        )
    if accelerator.device.type != "cuda":
        raise RuntimeError(f"B0 training requires CUDA, got {accelerator.device}")
    with torch.cuda.device(accelerator.device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"CUDA device does not support BF16: {accelerator.device}")


def _validate_offline_base_policy_load(
    pretrained_path: Path,
    *,
    learning_rate: float,
) -> int:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy_config = configure_smolvla_policy(
        pretrained_path,
        device="cpu",
        learning_rate=learning_rate,
    )
    policy = SmolVLAPolicy.from_pretrained(
        pretrained_path,
        config=policy_config,
        local_files_only=True,
        strict=True,
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    del policy
    return parameter_count


def _fresh_training(
    arguments: FreshArguments,
    accelerator: Any,
) -> tuple[Any, Path, int, int, dict[str, Any] | None]:
    split = load_background_split(arguments.split_file)
    dataset_audit = validate_dataset(arguments.dataset_root, split)
    resolved_base, revision = resolve_base_policy(
        arguments.base_policy,
        main_process_first=accelerator.main_process_first,
    )
    if accelerator.is_main_process:
        parameter_count = _validate_offline_base_policy_load(
            resolved_base,
            learning_rate=arguments.learning_rate,
        )
        print(
            f"Offline SmolVLA base-policy preflight passed ({parameter_count:,} parameters)",
            flush=True,
        )
    accelerator.wait_for_everyone()
    policy = configure_smolvla_policy(
        resolved_base,
        device="cuda",
        learning_rate=arguments.learning_rate,
    )
    config = build_train_config(arguments, split, policy)
    run_config = None
    if accelerator.is_main_process:
        base_identity = identify_base_policy(
            arguments.base_policy,
            resolved_base,
            revision,
        )
        run_config = build_run_config(
            arguments=arguments,
            split=split,
            dataset_audit=dataset_audit,
            base_identity=base_identity,
            train_config=config,
        )
    return config, arguments.output_dir, 0, 0, run_config


def _resume_training(
    arguments: ResumeArguments,
) -> tuple[Any, Path, int, int, dict[str, Any] | None, Path, ResumeContext]:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig

    del _SmolVLAConfig
    context = validate_resume(arguments)
    config = TrainPipelineConfig.from_pretrained(context.config_path)
    tokenizer_path = validate_local_vlm_assets(config.policy.vlm_model_name)
    config.policy.vlm_model_name = portable_path(tokenizer_path)
    if config.policy.pretrained_path is not None:
        config.policy.pretrained_path = Path(
            portable_path(resolve_repository_path(config.policy.pretrained_path))
        )
    config.dataset.root = portable_path(context.dataset_audit.dataset_root)
    config.output_dir = Path(portable_path(context.run_dir))
    config.resume = True
    return (
        config,
        context.run_dir,
        context.attempt_id,
        context.saved_step,
        None,
        Path(portable_path(context.config_path)),
        context,
    )


def _peak_cuda_memory(device: Any) -> tuple[int | None, int | None]:
    import torch

    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def run_training(arguments: FreshArguments | ResumeArguments) -> None:
    import torch
    from lerobot.scripts import lerobot_train

    accelerator = _make_accelerator()
    _validate_accelerator(
        accelerator,
        expected_processes=arguments.num_processes,
        expected_mixed_precision=arguments.mixed_precision,
    )
    lerobot_train.register_third_party_plugins()

    resume_config_path: Path | None = None
    resume_context: ResumeContext | None = None
    if isinstance(arguments, FreshArguments):
        config, output_dir, attempt_id, initial_step, run_config = _fresh_training(
            arguments,
            accelerator,
        )
        resume = False
    else:
        (
            config,
            output_dir,
            attempt_id,
            initial_step,
            run_config,
            resume_config_path,
            resume_context,
        ) = _resume_training(arguments)
        resume = True

    accelerator.wait_for_everyone()
    tokenizer_path = validate_local_vlm_assets(config.policy.vlm_model_name)
    started_at = utc_now()
    started_monotonic = time.perf_counter()
    with torch.cuda.device(accelerator.device):
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    training_error: BaseException | None = None
    try:
        with _attempt_artifacts(
            output_dir=output_dir,
            attempt_id=attempt_id,
            initial_step=initial_step,
            run_config=run_config,
            resume=resume,
            overwrite=(arguments.overwrite if isinstance(arguments, FreshArguments) else False),
            enabled=accelerator.is_main_process,
        ) as sink:
            try:
                if sink is not None:
                    start_record: dict[str, Any] = {
                        "record_type": "resume" if resume else "start",
                        "timestamp": utc_now(),
                        "step": initial_step,
                    }
                    if resume_context is not None:
                        start_record.update(
                            {
                                "resume_from_step": resume_context.saved_step,
                                "previous_metrics_tail_step": (
                                    resume_context.previous_metrics_tail_step
                                ),
                            }
                        )
                    sink.write(start_record)
                with (
                    instrument_update_policy(
                        lerobot_train,
                        sink if accelerator.is_main_process else None,
                    ),
                    _publish_artifacts_on_checkpoint(
                        lerobot_train,
                        output_dir=output_dir,
                        enabled=accelerator.is_main_process and not resume,
                    ),
                    _override_offline_tokenizer_processor(
                        lerobot_train,
                        tokenizer_path,
                    ),
                    _upstream_resume_argv(resume_config_path),
                ):
                    lerobot_train.train(config, accelerator=accelerator)
            except BaseException as error:
                if sink is not None:
                    sink.write(
                        {
                            "record_type": "attempt_end",
                            "timestamp": utc_now(),
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                raise
            else:
                if sink is not None:
                    sink.write(
                        {
                            "record_type": "attempt_end",
                            "timestamp": utc_now(),
                            "step": config.steps,
                            "status": "complete",
                        }
                    )
    except BaseException as error:
        training_error = error
        raise
    finally:
        peak_allocated, peak_reserved = _peak_cuda_memory(accelerator.device)
        payload = runtime_payload(
            attempt_id=attempt_id,
            rank=accelerator.process_index,
            device=str(accelerator.device),
            started_at=started_at,
            elapsed_s=time.perf_counter() - started_monotonic,
            status="failed" if training_error is not None else "complete",
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            error=training_error,
        )
        destination = runtime_path(
            output_dir,
            attempt_id=attempt_id,
            rank=accelerator.process_index,
        )
        write_json_atomic(destination, payload)


def main(argv: list[str] | None = None) -> None:
    os.environ.update(_OFFLINE_ENVIRONMENT)
    arguments = parse_arguments(argv)
    if isinstance(arguments, LaunchInspection):
        print(f"{arguments.num_processes}\t{arguments.mixed_precision}")
        return
    run_training(arguments)


if __name__ == "__main__":
    main()
