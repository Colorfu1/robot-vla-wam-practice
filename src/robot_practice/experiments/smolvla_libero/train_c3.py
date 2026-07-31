"""Train or resume SmolVLA LIBERO C3 LoRA plus paraphrase adaptation."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .b0_training import (
    instrument_update_policy,
    portable_path,
    runtime_path,
    runtime_payload,
    utc_now,
    validate_local_vlm_assets,
    write_json_atomic,
)
from .c3_training import (
    EXPERIMENT_NAME,
    C3FreshArguments,
    C3ResumeArguments,
    C3ResumeContext,
    LaunchInspection,
    build_run_config,
    build_train_config,
    c3_base_policy_identity,
    configure_c1_policy,
    inspect_peft_policy,
    load_paraphrase_training_spec,
    parse_arguments,
    resolve_c3_base_policy,
    validate_resume,
)
from .paraphrase_training import (
    INSTRUCTION_SAMPLING_CONTRACT,
    InstructionSelectionTracker,
    ParaphraseTrainingSpec,
    aggregate_instruction_counts,
    load_rank_instruction_baseline,
    override_instruction_preprocessor,
    track_successful_instruction_updates,
)
from .target_training import (
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
from .train_c2 import (
    _load_trainable_peft_resume,
    _make_accelerator,
    _validate_accelerator,
    _validate_selected_b0_peft_without_scratch_warning,
)

_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def _fresh_training(
    arguments: C3FreshArguments,
    accelerator: Any,
) -> tuple[
    Any,
    Path,
    int,
    int,
    dict[str, Any] | None,
    TargetSplit,
    ParaphraseTrainingSpec,
]:
    split = load_target_split(arguments.split_file)
    paraphrase_spec = load_paraphrase_training_spec(arguments.paraphrases, split)
    dataset_audit = validate_target_dataset(arguments.dataset_root, split)
    with accelerator.main_process_first():
        base = resolve_c3_base_policy(
            arguments.base_policy,
            require_selection=arguments.mode == "formal",
        )
    peft_inspection = None
    if accelerator.is_main_process:
        peft_inspection = inspect_peft_policy(
            base.root,
            learning_rate=arguments.learning_rate,
            rank=arguments.lora_rank,
        )
        print(
            "Offline selected-B0 + C3 LoRA/paraphrase preflight passed "
            f"(trainable={peft_inspection.trainable_parameters:,}, "
            f"total={peft_inspection.total_parameters:,}, "
            f"rank={arguments.lora_rank}, canonical_p={arguments.canonical_probability})",
            flush=True,
        )
    accelerator.wait_for_everyone()
    policy = configure_c1_policy(
        base.root,
        device="cuda",
        learning_rate=arguments.learning_rate,
    )
    config = build_train_config(arguments, split, policy)
    run_config = None
    if accelerator.is_main_process:
        if peft_inspection is None:
            raise RuntimeError("Main process did not complete the C3 PEFT preflight")
        run_config = build_run_config(
            arguments=arguments,
            split=split,
            dataset_audit=dataset_audit,
            base_identity=c3_base_policy_identity(arguments, base),
            base_selection=base.selection,
            train_config=config,
            peft_inspection=peft_inspection,
            paraphrase_spec=paraphrase_spec,
        )
    return config, arguments.output_dir, 0, 0, run_config, split, paraphrase_spec


def _resume_training(
    arguments: C3ResumeArguments,
) -> tuple[
    Any,
    Path,
    int,
    int,
    dict[str, Any] | None,
    Path,
    C3ResumeContext,
    TargetSplit,
    ParaphraseTrainingSpec,
]:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _SmolVLAConfig

    del _SmolVLAConfig
    context = validate_resume(arguments)
    config = TrainPipelineConfig.from_pretrained(context.config_path)
    tokenizer_path = validate_local_vlm_assets(config.policy.vlm_model_name)
    config.policy.vlm_model_name = portable_path(tokenizer_path)
    adapter_dir = context.checkpoint_dir / "pretrained_model"
    config.policy.pretrained_path = Path(portable_path(adapter_dir))
    config.policy.use_peft = True
    config.dataset.root = portable_path(context.dataset_audit.dataset_root)
    config.output_dir = Path(portable_path(context.run_dir))
    config.resume = True
    config.checkpoint_path = Path(portable_path(context.checkpoint_dir))
    return (
        config,
        context.run_dir,
        context.attempt_id,
        context.saved_step,
        None,
        Path(portable_path(context.config_path)),
        context,
        context.split,
        context.paraphrase_spec,
    )


def _instruction_tracker(
    *,
    config: Any,
    output_dir: Path,
    initial_step: int,
    attempt_id: int,
    split: TargetSplit,
    paraphrase_spec: ParaphraseTrainingSpec,
    accelerator: Any,
) -> InstructionSelectionTracker:
    baseline = load_rank_instruction_baseline(
        run_dir=output_dir,
        spec=paraphrase_spec,
        split=split,
        seed=config.seed,
        step=initial_step,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        batch_size=config.batch_size,
    )
    return InstructionSelectionTracker(
        spec=paraphrase_spec,
        split=split,
        seed=config.seed,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        batch_size=config.batch_size,
        initial_step=initial_step,
        total_steps=config.steps,
        save_freq=config.save_freq,
        output_dir=output_dir,
        attempt_id=attempt_id,
        baseline=baseline,
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


def run_training(arguments: C3FreshArguments | C3ResumeArguments) -> None:
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
    resume_context: C3ResumeContext | None = None
    if isinstance(arguments, C3FreshArguments):
        (
            config,
            output_dir,
            attempt_id,
            initial_step,
            run_config,
            split,
            paraphrase_spec,
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
            paraphrase_spec,
        ) = _resume_training(arguments)
        resume = True

    tracker = _instruction_tracker(
        config=config,
        output_dir=output_dir,
        initial_step=initial_step,
        attempt_id=attempt_id,
        split=split,
        paraphrase_spec=paraphrase_spec,
        accelerator=accelerator,
    )
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
            enabled=accelerator.is_main_process,
        ) as sink:
            try:
                if sink is not None:
                    start_record: dict[str, Any] = {
                        "record_type": "resume" if resume else "start",
                        "timestamp": utc_now(),
                        "step": initial_step,
                        "sampler": "uniform_task_then_uniform_episode",
                        "peft_method": "LORA",
                        "lora_rank": config.peft.r,
                        "instruction_sampling_contract": INSTRUCTION_SAMPLING_CONTRACT,
                        "canonical_probability": paraphrase_spec.canonical_probability,
                        "paraphrases_sha256": paraphrase_spec.sha256,
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
                    track_successful_instruction_updates(lerobot_train, tracker),
                    _publish_artifacts_on_checkpoint(
                        lerobot_train,
                        output_dir=output_dir,
                        enabled=accelerator.is_main_process and not resume,
                    ),
                    _override_offline_tokenizer_processor(
                        lerobot_train,
                        tokenizer_path,
                    ),
                    override_instruction_preprocessor(lerobot_train, tracker),
                    override_target_training_sampler(
                        lerobot_train,
                        split,
                        seed=config.seed,
                    ),
                    _validate_selected_b0_peft_without_scratch_warning(),
                    _load_trainable_peft_resume(lerobot_train, enabled=resume),
                    _upstream_resume_argv(resume_config_path),
                ):
                    lerobot_train.train(config, accelerator=accelerator)
                tracker.ensure_complete()
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    aggregate_instruction_counts(
                        run_dir=output_dir,
                        spec=paraphrase_spec,
                        split=split,
                        seed=config.seed,
                        step=config.steps,
                        world_size=accelerator.num_processes,
                        batch_size=config.batch_size,
                    )
                accelerator.wait_for_everyone()
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
        if isinstance(arguments, C3FreshArguments):
            formal = arguments.mode == "formal"
        else:
            effective_run_config = resume_context.run_config if resume_context else {}
            training_record = effective_run_config.get("training")
            formal = isinstance(training_record, dict) and training_record.get("mode") == "formal"
        if (
            training_error is not None
            and formal
            and accelerator.is_main_process
            and _is_out_of_memory(training_error)
        ):
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
