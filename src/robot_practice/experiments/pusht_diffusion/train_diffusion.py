"""Train the explicit Diffusion Policy PushT configuration with LeRobot."""

from __future__ import annotations

import json
from collections.abc import Sequence

from lerobot.scripts import lerobot_train

from .model_configs.diffusion import make_diffusion_config
from .training_common import (
    DistributedProcessContext,
    ResolvedTraining,
    TrainArguments,
    build_train_pipeline_config,
    parse_train_arguments,
    print_training_plan,
    resolve_device,
    validate_training_inputs,
)
from .training_runtime import (
    install_lerobot_runtime_adapters,
    persistent_training_log,
)

ENTRY_MODULE = "robot_practice.experiments.pusht_diffusion.train_diffusion"
MODEL_CONFIG_MODULE = "robot_practice.experiments.pusht_diffusion.model_configs.diffusion"


def resolve_training(
    argv: Sequence[str] | None = None,
) -> tuple[TrainArguments, ResolvedTraining]:
    """Resolve Diffusion's explicit model and official LeRobot train config."""

    arguments = parse_train_arguments(argv, description=__doc__)
    policy = make_diffusion_config(device=resolve_device(arguments.requested_device))
    resolved = build_train_pipeline_config(
        arguments=arguments,
        policy=policy,
        policy_type="diffusion",
        entry_module=ENTRY_MODULE,
        model_config_module=MODEL_CONFIG_MODULE,
        random_initialization=True,
    )
    return arguments, resolved


def main(argv: Sequence[str] | None = None) -> None:
    arguments, resolved = resolve_training(argv)
    process_context = DistributedProcessContext.from_environment()
    if arguments.print_command:
        print_training_plan(resolved, process_context)
        return
    validate_training_inputs(
        resolved,
        check_run_destination=process_context.is_main_process,
    )
    install_lerobot_runtime_adapters()
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
