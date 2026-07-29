"""Shared prototype-ID conditioning for local PushT policy variants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
)
from torch import Tensor, nn

PROTOTYPE_ID = "prototype_id"


def validate_mode_config(config: Any) -> None:
    if int(config.num_prototypes) < 2:
        raise ValueError("num_prototypes must be at least 2")
    if int(config.mode_embedding_dim) <= 0:
        raise ValueError("mode_embedding_dim must be positive")
    probability = float(config.condition_dropout_prob)
    if not 0.0 <= probability < 1.0:
        raise ValueError("condition_dropout_prob must be in [0,1)")


def _base_global_condition_dim(model: DiffusionModel) -> int:
    config = model.config
    dimension = int(config.robot_state_feature.shape[0])
    if config.image_features:
        if config.use_separate_rgb_encoder_per_camera:
            dimension += int(model.rgb_encoder[0].feature_dim) * len(config.image_features)
        else:
            dimension += int(model.rgb_encoder.feature_dim) * len(config.image_features)
    if config.env_state_feature:
        dimension += int(config.env_state_feature.shape[0])
    return dimension * int(config.n_obs_steps)


def normalize_prototype_ids(
    value: Tensor,
    *,
    batch_size: int,
    num_prototypes: int,
) -> Tensor:
    """Return one categorical ID per batch item and validate the NULL row."""

    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim == 0 and batch_size == 1:
        value = value.unsqueeze(0)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(
            f"{PROTOTYPE_ID} must have shape ({batch_size},) or ({batch_size},1), "
            f"got {tuple(value.shape)}"
        )
    if value.is_floating_point() and not torch.equal(value, value.round()):
        raise ValueError(f"{PROTOTYPE_ID} must contain integer categorical IDs")
    ids = value.to(dtype=torch.long)
    minimum = int(ids.min().item())
    maximum = int(ids.max().item())
    if minimum < 0 or maximum > num_prototypes:
        raise ValueError(
            f"{PROTOTYPE_ID} values must be in [0,{num_prototypes}], "
            f"got min={minimum}, max={maximum}"
        )
    return ids


class ModeConditionedDiffusionModel(DiffusionModel):
    """Diffusion model whose global FiLM condition includes one mode embedding."""

    def __init__(self, config: Any):
        validate_mode_config(config)
        super().__init__(config)
        self.mode_embedding = nn.Embedding(
            int(config.num_prototypes) + 1,
            int(config.mode_embedding_dim),
        )
        global_cond_dim = _base_global_condition_dim(self) + int(config.mode_embedding_dim)
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)
        self.last_condition_stats: dict[str, int] = {
            "input_null_count": 0,
            "dropout_to_null_count": 0,
            "effective_null_count": 0,
            "conditional_count": 0,
        }

    @property
    def null_prototype_id(self) -> int:
        return int(self.config.num_prototypes)

    def _condition_ids(self, batch: Mapping[str, Tensor], *, batch_size: int) -> Tensor:
        if PROTOTYPE_ID not in batch:
            raise KeyError(f"Missing required categorical condition {PROTOTYPE_ID!r}")
        ids = normalize_prototype_ids(
            batch[PROTOTYPE_ID],
            batch_size=batch_size,
            num_prototypes=int(self.config.num_prototypes),
        )
        input_null = ids == self.null_prototype_id
        dropped = torch.zeros_like(input_null)
        if self.training and float(self.config.condition_dropout_prob) > 0.0:
            eligible = ~input_null
            dropped = eligible & (
                torch.rand(ids.shape, device=ids.device) < float(self.config.condition_dropout_prob)
            )
            ids = torch.where(
                dropped,
                torch.full_like(ids, self.null_prototype_id),
                ids,
            )
        effective_null = ids == self.null_prototype_id
        self.last_condition_stats = {
            "input_null_count": int(input_null.sum().item()),
            "dropout_to_null_count": int(dropped.sum().item()),
            "effective_null_count": int(effective_null.sum().item()),
            "conditional_count": int((~effective_null).sum().item()),
        }
        return ids

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        observation_condition = super()._prepare_global_conditioning(batch)
        ids = self._condition_ids(batch, batch_size=observation_condition.shape[0])
        embedding = self.mode_embedding(ids).to(dtype=observation_condition.dtype)
        return torch.cat((observation_condition, embedding), dim=-1)


def validate_and_latch_prototype(
    active: Tensor | None,
    batch: Mapping[str, Tensor],
    *,
    num_prototypes: int,
) -> Tensor:
    """Latch one caller-supplied ID for a complete rollout episode."""

    if PROTOTYPE_ID not in batch:
        raise KeyError(f"Missing required categorical condition {PROTOTYPE_ID!r}")
    raw = batch[PROTOTYPE_ID]
    batch_size = int(raw.shape[0]) if raw.ndim else 1
    ids = normalize_prototype_ids(
        raw,
        batch_size=batch_size,
        num_prototypes=num_prototypes,
    ).detach()
    if active is not None and not torch.equal(active.to(device=ids.device), ids):
        raise ValueError(
            "prototype_id changed during an active rollout; call policy.reset() "
            "before selecting a different prototype"
        )
    return ids.clone()


def condition_metrics(stats: Mapping[str, int]) -> dict[str, float]:
    return {f"condition/{name}": float(value) for name, value in stats.items()}


__all__ = [
    "PROTOTYPE_ID",
    "ModeConditionedDiffusionModel",
    "condition_metrics",
    "normalize_prototype_ids",
    "validate_and_latch_prototype",
    "validate_mode_config",
]
