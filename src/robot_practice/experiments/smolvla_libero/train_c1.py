"""Train or resume SmolVLA LIBERO C1 default non-PEFT adaptation."""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path
from typing import Any

from .b0_training import (
    BackgroundSplit,
    instrument_update_policy,
    load_background_split,
    portable_path,
    resolve_repository_path,
    runtime_path,
    runtime_payload,
    utc_now,
    validate_dataset,
    validate_local_vlm_assets,
    write_json_atomic,
)
from .c1_training import (
    EXPERIMENT_NAME,
    C1FreshArguments,
    C1ResumeArguments,
    C1ResumeContext,
    LaunchInspection,
    background_replay_k,
    base_policy_identity,
    build_run_config,
    build_train_config,
    configure_c1_policy,
    parse_arguments,
    resolve_c1_base_policy,
    validate_resume,
)
from .target_training import (
    SAMPLING_STRATEGY,
    TargetSplit,
    load_target_split,
    override_target_training_sampler,
    validate_target_dataset,
)
from .train_b0 import (
    _attempt_artifacts,
    _override_offline_tokenizer_processor,
    _peak_cuda_memory,
    _publish_artifacts_on_checkpoint,
    _upstream_resume_argv,
)

_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


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
        raise RuntimeError(f"C1 training requires CUDA, got {accelerator.device}")
    with torch.cuda.device(accelerator.device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"CUDA device does not support BF16: {accelerator.device}")


def _validate_offline_c1_policy_load(
    pretrained_path: Path,
    *,
    learning_rate: float,
) -> tuple[int, int]:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy_config = configure_c1_policy(
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
    total = sum(parameter.numel() for parameter in policy.parameters())
    trainable = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    del policy
    return total, trainable


def _fresh_training(
    arguments: C1FreshArguments,
    accelerator: Any,
) -> tuple[Any, Path, int, int, dict[str, Any] | None, TargetSplit, BackgroundSplit | None]:
    split = load_target_split(arguments.split_file)
    dataset_audit = validate_target_dataset(arguments.dataset_root, split)
    background_split: BackgroundSplit | None = None
    replay_k = background_replay_k(arguments)
    if replay_k > 0:
        background_split = load_background_split(arguments.background_split_file)
        validate_dataset(arguments.dataset_root, background_split)
        dataset_audit = dataclasses.replace(
            dataset_audit,
            train_episodes=len(split.train_episodes) + len(background_split.train_episodes),
            train_frames=split.train_frames + background_split.train_frames,
        )
    with accelerator.main_process_first():
        base = resolve_c1_base_policy(
            arguments.base_policy,
            require_selection=arguments.mode == "formal",
        )
    if accelerator.is_main_process:
        total, trainable = _validate_offline_c1_policy_load(
            base.root,
            learning_rate=arguments.learning_rate,
        )
        print(
            f"Offline selected-B0 preflight passed (trainable={trainable:,}, total={total:,})",
            flush=True,
        )
    accelerator.wait_for_everyone()
    policy = configure_c1_policy(
        base.root,
        device="cuda",
        learning_rate=arguments.learning_rate,
    )
    config = build_train_config(arguments, split, policy, background_split)
    run_config = None
    if accelerator.is_main_process:
        run_config = build_run_config(
            arguments=arguments,
            split=split,
            dataset_audit=dataset_audit,
            base_identity=base_policy_identity(arguments, base),
            base_selection=base.selection,
            train_config=config,
            background_split=background_split,
        )
    return config, arguments.output_dir, 0, 0, run_config, split, background_split


def _resume_training(
    arguments: C1ResumeArguments,
) -> tuple[
    Any,
    Path,
    int,
    int,
    dict[str, Any] | None,
    Path,
    C1ResumeContext,
    TargetSplit,
    BackgroundSplit | None,
]:
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
        context.split,
        context.background_split,
    )


def _is_out_of_memory(error: BaseException) -> bool:
    import torch

    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _write_oom_marker(
    *,
    output_dir: Path,
    config: Any,
    accelerator: Any,
    peak_allocated_bytes: int | None,
    peak_reserved_bytes: int | None,
    error: BaseException,
) -> None:
    write_json_atomic(
        output_dir / "N_A.json",
        {
            "schema_version": 1,
            "experiment": EXPERIMENT_NAME,
            "status": "not_available",
            "reason": "oom",
            "failure_stage": "training",
            "timestamp": utc_now(),
            "rank": accelerator.process_index,
            "world_size": accelerator.num_processes,
            "per_gpu_batch_size": config.batch_size,
            "mixed_precision": accelerator.mixed_precision,
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
            "error_type": type(error).__name__,
        },
    )


def run_training(arguments: C1FreshArguments | C1ResumeArguments) -> None:
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
    resume_context: C1ResumeContext | None = None
    if isinstance(arguments, C1FreshArguments):
        (
            config,
            output_dir,
            attempt_id,
            initial_step,
            run_config,
            split,
            background_split,
        ) = _fresh_training(arguments, accelerator)
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
            split,
            background_split,
        ) = _resume_training(arguments)
        resume = True

    if isinstance(arguments, C1FreshArguments):
        replay_k = background_replay_k(arguments)
    else:
        replay_k = int(resume_context.run_config["dataset"].get("background_replay_k", 0))

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
            overwrite=(arguments.overwrite if isinstance(arguments, C1FreshArguments) else False),
            enabled=accelerator.is_main_process,
        ) as sink:
            try:
                if sink is not None:
                    effective_run_config = run_config
                    if effective_run_config is None and resume_context is not None:
                        effective_run_config = resume_context.run_config
                    sampler_name = (
                        effective_run_config.get("dataset", {}).get("sampler", SAMPLING_STRATEGY)
                        if effective_run_config is not None
                        else SAMPLING_STRATEGY
                    )
                    start_record: dict[str, Any] = {
                        "record_type": "resume" if resume else "start",
                        "timestamp": utc_now(),
                        "step": initial_step,
                        "sampler": sampler_name,
                        "background_replay_k": replay_k,
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
                    override_target_training_sampler(
                        lerobot_train,
                        split,
                        seed=config.seed,
                        background_split=background_split,
                        background_replay_k=replay_k,
                        num_processes=arguments.num_processes,
                        batch_size=arguments.batch_size,
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
        if isinstance(arguments, C1FreshArguments):
            formal = arguments.mode == "formal"
        else:
            effective_run_config = resume_context.run_config if resume_context else {}
            training_record = effective_run_config.get("training")
            formal = isinstance(training_record, dict) and training_record.get("mode") == "formal"
        if training_error is not None and formal and _is_out_of_memory(training_error):
            _write_oom_marker(
                output_dir=output_dir,
                config=config,
                accelerator=accelerator,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                error=training_error,
            )
        write_json_atomic(
            runtime_path(
                output_dir,
                attempt_id=attempt_id,
                rank=accelerator.process_index,
            ),
            runtime_payload(
                attempt_id=attempt_id,
                rank=accelerator.process_index,
                device=str(accelerator.device),
                started_at=started_at,
                elapsed_s=time.perf_counter() - started_monotonic,
                status="failed" if training_error is not None else "complete",
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                error=training_error,
            ),
        )


def main(argv: list[str] | None = None) -> None:
    os.environ.update(_OFFLINE_ENVIRONMENT)
    arguments = parse_arguments(argv)
    if isinstance(arguments, LaunchInspection):
        print(f"{arguments.num_processes}\t{arguments.mixed_precision}")
        return
    run_training(arguments)


if __name__ == "__main__":
    main()
