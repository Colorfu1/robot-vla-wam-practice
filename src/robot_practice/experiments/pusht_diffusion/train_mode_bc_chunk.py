"""Train E5 Mode-BCChunk through the unmodified LeRobot train loop."""

from __future__ import annotations

import json
from collections.abc import Sequence

from lerobot.scripts import lerobot_train

from .e5_training import (
    install_prototype_dataset_adapter,
    install_prototype_processor_adapter,
    parse_mode_train_arguments,
)
from .model_configs.mode_bc_chunk import make_mode_bc_chunk_config
from .training_common import (
    DistributedProcessContext,
    ResolvedTraining,
    TrainArguments,
    build_train_pipeline_config,
    print_training_plan,
    resolve_device,
    validate_training_inputs,
)
from .training_runtime import install_lerobot_runtime_adapters, persistent_training_log

ENTRY_MODULE = "robot_practice.experiments.pusht_diffusion.train_mode_bc_chunk"
MODEL_CONFIG_MODULE = (
    "robot_practice.experiments.pusht_diffusion.model_configs.mode_bc_chunk"
)


def resolve_training(
    argv: Sequence[str] | None = None,
) -> tuple[TrainArguments, ResolvedTraining]:
    arguments, prototype_manifest, num_prototypes = parse_mode_train_arguments(
        argv,
        description=__doc__,
    )
    policy = make_mode_bc_chunk_config(
        device=resolve_device(arguments.requested_device),
        num_prototypes=num_prototypes,
    )
    resolved = build_train_pipeline_config(
        arguments=arguments,
        policy=policy,
        policy_type="mode_bc_chunk",
        entry_module=ENTRY_MODULE,
        model_config_module=MODEL_CONFIG_MODULE,
        prototype_manifest=prototype_manifest,
        random_initialization=True,
    )
    return arguments, resolved


def main(argv: Sequence[str] | None = None) -> None:
    try:
        arguments, resolved = resolve_training(argv)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    process_context = DistributedProcessContext.from_environment()
    if arguments.print_command:
        print_training_plan(resolved, process_context)
        return
    validate_training_inputs(
        resolved,
        check_run_destination=process_context.is_main_process,
    )
    install_lerobot_runtime_adapters()
    install_prototype_dataset_adapter(resolved.prototype_manifest)
    install_prototype_processor_adapter()
    lerobot_train.register_third_party_plugins()
    with persistent_training_log(
        resolved.output_dir,
        enabled=process_context.is_main_process,
    ):
        if process_context.is_main_process:
            print("Resolved configuration:", flush=True)
            print(
                json.dumps(
                    resolved.summary(process_context),
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        lerobot_train.train(resolved.config)


if __name__ == "__main__":
    main()
