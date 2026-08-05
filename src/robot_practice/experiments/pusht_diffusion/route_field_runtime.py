"""Runtime helpers for injecting one rollout-persistent RF7 route field."""

from __future__ import annotations

from typing import Any

import torch

from robot_practice.policies.route_field_conditioning import ROUTE_FIELD

from .e5_runtime import move_processed_batch


def add_route_field(
    batch: dict[str, Any],
    *,
    field: torch.Tensor,
) -> dict[str, Any]:
    """Copy a processed batch and append one validated spatial route field."""

    if not isinstance(field, torch.Tensor):
        raise TypeError("route field must be a torch.Tensor")
    source = field.detach().cpu()
    if source.ndim == 3:
        source = source.unsqueeze(0)
    if source.ndim != 4:
        raise ValueError(f"route field must have shape [1,C,H,W], got {tuple(source.shape)}")
    processed = dict(batch)
    if ROUTE_FIELD in processed:
        raise ValueError(f"Processed batch already contains {ROUTE_FIELD!r}")
    device = next(
        (
            value.device
            for value in processed.values()
            if isinstance(value, torch.Tensor)
        ),
        torch.device("cpu"),
    )
    processed[ROUTE_FIELD] = source.to(device=device, dtype=torch.float32)
    return processed


class FixedRouteFieldPreprocessor:
    """Run a checkpoint preprocessor and then attach a fixed route field."""

    def __init__(
        self,
        preprocessor: Any,
        *,
        field: torch.Tensor,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(field, torch.Tensor):
            raise TypeError("field must be a torch.Tensor")
        self.preprocessor = preprocessor
        self.field = field.detach().cpu()
        self.device = device

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        processed = self.preprocessor(batch)
        if not isinstance(processed, dict):
            raise TypeError("Policy preprocessor must return a dictionary")
        if self.device is not None:
            processed = move_processed_batch(processed, device=self.device)
        return add_route_field(processed, field=self.field)

    def __getattr__(self, name: str) -> Any:
        if name == "preprocessor":
            raise AttributeError(name)
        return getattr(self.preprocessor, name)


__all__ = ["FixedRouteFieldPreprocessor", "add_route_field"]
