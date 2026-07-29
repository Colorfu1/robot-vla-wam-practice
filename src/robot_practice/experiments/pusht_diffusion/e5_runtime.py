"""Runtime helpers for injecting one episode-persistent E5 prototype ID."""

from __future__ import annotations

from typing import Any

import torch

from robot_practice.policies.mode_conditioning import PROTOTYPE_ID


def move_processed_batch(
    value: Any,
    *,
    device: torch.device | str,
) -> Any:
    """Move processed tensors without changing non-tensor metadata."""

    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_processed_batch(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_processed_batch(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_processed_batch(item, device=device) for item in value)
    return value


def add_prototype_id(
    batch: dict[str, Any],
    *,
    prototype_id: int,
    num_prototypes: int,
) -> dict[str, Any]:
    """Copy a processed batch and append one validated categorical condition."""

    if not 0 <= int(prototype_id) <= int(num_prototypes):
        raise ValueError(f"prototype_id must be in [0,{num_prototypes}], got {prototype_id}")
    processed = dict(batch)
    device = next(
        (value.device for value in processed.values() if isinstance(value, torch.Tensor)),
        torch.device("cpu"),
    )
    processed[PROTOTYPE_ID] = torch.tensor(
        [int(prototype_id)],
        device=device,
        dtype=torch.long,
    )
    return processed


class FixedPrototypePreprocessor:
    """Run a checkpoint preprocessor and then attach a fixed rollout condition."""

    def __init__(
        self,
        preprocessor: Any,
        *,
        prototype_id: int,
        num_prototypes: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.prototype_id = int(prototype_id)
        self.num_prototypes = int(num_prototypes)
        self.device = device

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        processed = self.preprocessor(batch)
        if not isinstance(processed, dict):
            raise TypeError("Policy preprocessor must return a dictionary")
        if self.device is not None:
            processed = move_processed_batch(processed, device=self.device)
        return add_prototype_id(
            processed,
            prototype_id=self.prototype_id,
            num_prototypes=self.num_prototypes,
        )

    def __getattr__(self, name: str) -> Any:
        if name == "preprocessor":
            raise AttributeError(name)
        return getattr(self.preprocessor, name)


__all__ = ["FixedPrototypePreprocessor", "add_prototype_id", "move_processed_batch"]
