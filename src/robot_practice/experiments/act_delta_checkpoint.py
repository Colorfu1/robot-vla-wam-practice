"""Create an ACT-delta checkpoint from an existing ACT checkpoint."""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

from robot_practice.evaluation.common import validate_complete_checkpoint
from robot_practice.policies.act_delta.configuration_act_delta import ACTDeltaConfig


CHECKPOINT_DIR = Path("checkpoints/000000/pretrained_model")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create",
        help="Create an act_delta checkpoint from a source checkpoint",
    )
    create.add_argument(
        "--source-policy",
        required=True,
        type=Path,
        help="Existing ACT checkpoint to reuse weights from",
    )
    create.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Run root directory; writes <output-dir>/checkpoints/000000/pretrained_model",
    )
    create.add_argument(
        "--delta-loss-weight",
        default=0.1,
        type=float,
        help="Delta action loss weight to write into act_delta config",
    )
    return parser.parse_args(argv)


def _validate_delta_loss_weight(delta_loss_weight: float) -> None:
    if not math.isfinite(delta_loss_weight) or delta_loss_weight < 0:
        raise ValueError("delta-loss-weight must be finite and non-negative")


def _read_checkpoint_type(policy_path: Path) -> ACTConfig:
    manifest = validate_complete_checkpoint(policy_path)
    config = PreTrainedConfig.from_pretrained(manifest.path, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise TypeError(f"Expected ACTConfig, got {type(config).__name__}")
    return config


def create_act_delta_checkpoint(
    *,
    source_policy: Path,
    output_dir: Path,
    delta_loss_weight: float,
) -> Path:
    source_policy = source_policy.expanduser().resolve()
    if not source_policy.is_dir():
        raise FileNotFoundError(f"Source policy directory does not exist: {source_policy}")

    output_root = output_dir.expanduser().resolve()
    output_checkpoint = output_root / CHECKPOINT_DIR
    if output_root.exists() or output_checkpoint.exists():
        raise FileExistsError(f"Output checkpoint directory already exists: {output_checkpoint}")

    source_config = _read_checkpoint_type(source_policy)
    source_data: dict[str, Any] = dict(source_config.__dict__)
    source_data["pretrained_path"] = None
    source_data["delta_loss_weight"] = delta_loss_weight
    source_data.pop("type", None)
    delta_config = ACTDeltaConfig(**source_data)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = f".{output_checkpoint.name}.{os.getpid()}"
    temporary_checkpoint = output_checkpoint.parent / temporary_name
    if temporary_checkpoint.exists():
        raise FileExistsError(f"Temporary path already exists: {temporary_checkpoint}")

    try:
        shutil.copytree(source_policy, temporary_checkpoint)
        delta_config.save_pretrained(temporary_checkpoint)
        validate_complete_checkpoint(temporary_checkpoint)
        temporary_checkpoint.rename(output_checkpoint)
    except BaseException:
        if temporary_checkpoint.exists():
            if temporary_checkpoint.is_dir():
                shutil.rmtree(temporary_checkpoint)
            else:
                temporary_checkpoint.unlink()
        raise

    return output_checkpoint


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "create":
        _validate_delta_loss_weight(args.delta_loss_weight)
        output_dir = args.output_dir.expanduser().resolve()
        output_checkpoint = create_act_delta_checkpoint(
            source_policy=args.source_policy,
            output_dir=output_dir,
            delta_loss_weight=args.delta_loss_weight,
        )
        print(f"Created act_delta checkpoint at: {output_checkpoint}")
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
