"""Shared RouteField training resolution and runtime dataset installation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from lerobot.scripts import lerobot_train

from robot_practice.policies.route_field_conditioning import ROUTE_FIELD

from .route_field_dataset import load_route_field_sidecar, wrap_dataset_with_route_fields
from .training_common import TrainArguments, parse_train_arguments


def parse_route_field_train_arguments(
    argv: Sequence[str] | None,
    *,
    description: str,
) -> tuple[TrainArguments, Path, tuple[int, ...]]:
    """Parse one required route sidecar plus the common PushT options."""

    raw = list(sys.argv[1:] if argv is None else argv)
    matches = [
        argument
        for argument in raw
        if argument == "--route-manifest" or argument.startswith("--route-manifest=")
    ]
    if len(matches) != 1:
        suffix = "required" if not matches else "may only be provided once"
        raise ValueError(f"--route-manifest is {suffix}")
    option = matches[0]
    if option == "--route-manifest":
        index = raw.index(option)
        if index + 1 >= len(raw) or raw[index + 1].startswith("--"):
            raise ValueError("--route-manifest requires a path")
        value = raw[index + 1]
        del raw[index : index + 2]
    else:
        value = option.split("=", maxsplit=1)[1]
        raw.remove(option)
    if not value:
        raise ValueError("--route-manifest requires a non-empty path")

    sidecar = load_route_field_sidecar(value)
    arguments = parse_train_arguments(raw, description=description)
    from .dataset_split import load_split

    requested = load_split(arguments.split_file).episodes_for_mode(arguments.mode)
    episodes = sidecar.training_episodes(requested)
    if not episodes:
        raise ValueError(
            f"No eligible RouteField episodes remain in the {arguments.mode} split"
        )
    return arguments, sidecar.manifest_path, episodes


def install_route_field_dataset_adapter(route_manifest: str | Path) -> None:
    """Wrap LeRobot's locally created dataset without editing LeRobot."""

    manifest = Path(route_manifest).expanduser().resolve()
    original: Callable = getattr(
        lerobot_train.make_dataset,
        "__robot_practice_unwrapped__",
        lerobot_train.make_dataset,
    )

    def make_conditioned_dataset(config):
        dataset = original(config)
        return wrap_dataset_with_route_fields(dataset, route_manifest=manifest)

    make_conditioned_dataset.__robot_practice_unwrapped__ = original
    lerobot_train.make_dataset = make_conditioned_dataset


class _RouteFieldPreservingProcessor:
    """Preserve the sidecar tensor across LeRobot preprocessing."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        if ROUTE_FIELD not in batch:
            raise KeyError(f"Training batch has no {ROUTE_FIELD!r}")
        route_field = batch[ROUTE_FIELD]
        processed = self.processor(batch)
        if not isinstance(processed, dict):
            raise TypeError("LeRobot policy preprocessor must return a dictionary")
        device = next(
            (value.device for value in processed.values() if isinstance(value, torch.Tensor)),
            route_field.device,
        )
        processed[ROUTE_FIELD] = route_field.to(device=device, dtype=torch.float32)
        return processed

    def __getattr__(self, name: str) -> Any:
        if name == "processor":
            raise AttributeError(name)
        return getattr(self.processor, name)


def install_route_field_processor_adapter() -> None:
    """Wrap only the runtime preprocessor used by the RouteField train entry."""

    original: Callable = getattr(
        lerobot_train.make_pre_post_processors,
        "__robot_practice_unwrapped__",
        lerobot_train.make_pre_post_processors,
    )

    def make_conditioned_processors(*args, **kwargs):
        preprocessor, postprocessor = original(*args, **kwargs)
        return _RouteFieldPreservingProcessor(preprocessor), postprocessor

    make_conditioned_processors.__robot_practice_unwrapped__ = original
    lerobot_train.make_pre_post_processors = make_conditioned_processors


__all__ = [
    "install_route_field_dataset_adapter",
    "install_route_field_processor_adapter",
    "parse_route_field_train_arguments",
]
