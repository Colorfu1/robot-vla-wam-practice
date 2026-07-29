"""Shared E5 mode-conditioned training resolution and dataset installation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from lerobot.scripts import lerobot_train

from robot_practice.policies.mode_conditioning import PROTOTYPE_ID

from .e5_dataset import load_prototype_sidecar, wrap_dataset_with_prototypes
from .training_common import TrainArguments, parse_train_arguments


def parse_mode_train_arguments(
    argv: Sequence[str] | None,
    *,
    description: str,
) -> tuple[TrainArguments, Path, int]:
    """Parse one required sidecar option plus the common PushT training options."""

    raw = list(sys.argv[1:] if argv is None else argv)
    matches = [
        argument
        for argument in raw
        if argument == "--prototype-manifest" or argument.startswith("--prototype-manifest=")
    ]
    if len(matches) != 1:
        suffix = "required" if not matches else "may only be provided once"
        raise ValueError(f"--prototype-manifest is {suffix}")
    option = matches[0]
    if option == "--prototype-manifest":
        index = raw.index(option)
        if index + 1 >= len(raw) or raw[index + 1].startswith("--"):
            raise ValueError("--prototype-manifest requires a path")
        value = raw[index + 1]
        del raw[index : index + 2]
    else:
        value = option.split("=", maxsplit=1)[1]
        raw.remove(option)
    if not value:
        raise ValueError("--prototype-manifest requires a non-empty path")
    manifest = Path(value).expanduser().resolve()
    sidecar = load_prototype_sidecar(manifest)
    arguments = parse_train_arguments(raw, description=description)
    return arguments, sidecar.manifest_path, sidecar.selected_k


def install_prototype_dataset_adapter(prototype_manifest: str | Path) -> None:
    """Wrap LeRobot's locally created dataset without editing LeRobot."""

    manifest = Path(prototype_manifest).expanduser().resolve()
    original: Callable = getattr(
        lerobot_train.make_dataset,
        "__robot_practice_unwrapped__",
        lerobot_train.make_dataset,
    )

    def make_conditioned_dataset(config):
        dataset = original(config)
        return wrap_dataset_with_prototypes(
            dataset,
            prototype_manifest=manifest,
        )

    make_conditioned_dataset.__robot_practice_unwrapped__ = original
    lerobot_train.make_dataset = make_conditioned_dataset


class _PrototypePreservingProcessor:
    """Preserve the categorical sidecar key across LeRobot preprocessing."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if PROTOTYPE_ID not in batch:
            raise KeyError(f"Training batch has no {PROTOTYPE_ID!r}")
        prototype_id = batch[PROTOTYPE_ID]
        processed = self.processor(batch)
        if not isinstance(processed, dict):
            raise TypeError("LeRobot policy preprocessor must return a dictionary")
        device = next(
            (value.device for value in processed.values() if isinstance(value, torch.Tensor)),
            prototype_id.device,
        )
        processed[PROTOTYPE_ID] = prototype_id.to(device=device, dtype=torch.long)
        return processed

    def __getattr__(self, name: str) -> Any:
        if name == "processor":
            raise AttributeError(name)
        return getattr(self.processor, name)


def install_prototype_processor_adapter() -> None:
    """Wrap only the runtime preprocessor used by the E5 train entry."""

    original: Callable = getattr(
        lerobot_train.make_pre_post_processors,
        "__robot_practice_unwrapped__",
        lerobot_train.make_pre_post_processors,
    )

    def make_conditioned_processors(*args, **kwargs):
        preprocessor, postprocessor = original(*args, **kwargs)
        return _PrototypePreservingProcessor(preprocessor), postprocessor

    make_conditioned_processors.__robot_practice_unwrapped__ = original
    lerobot_train.make_pre_post_processors = make_conditioned_processors


__all__ = [
    "install_prototype_dataset_adapter",
    "install_prototype_processor_adapter",
    "parse_mode_train_arguments",
]
