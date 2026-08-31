"""Tensor-level contracts shared by residual-RL smoke and formal runners."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

ACTION_DIM = 14
ARM_DIM = 12
GRIPPER_DIM = 2


def compose_residual_action(
    base_action: torch.Tensor,
    residual: torch.Tensor,
    action_std_arm: torch.Tensor,
    q_min_arm: torch.Tensor,
    q_max_arm: torch.Tensor,
    *,
    res_scale: float = 0.1,
) -> torch.Tensor:
    """Compose a residual with a physical-unit 14D qpos target.

    ``base_action`` may have arbitrary leading dimensions (for example
    ``[num_envs, horizon, 14]``).  The residual has the same leading shape and
    twelve arm channels.  The two gripper channels are copied from the base,
    never produced by the residual actor.
    """

    if not isinstance(base_action, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise TypeError("base_action and residual must be torch.Tensor values")
    if base_action.ndim < 1 or base_action.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"base_action must end in {ACTION_DIM} channels, got {tuple(base_action.shape)}"
        )
    expected_residual_shape = (*base_action.shape[:-1], ARM_DIM)
    if tuple(residual.shape) != expected_residual_shape:
        raise ValueError(
            "residual must match base leading dimensions and have 12 arm channels: "
            f"expected {expected_residual_shape}, got {tuple(residual.shape)}"
        )
    if not torch.isfinite(base_action).all() or not torch.isfinite(residual).all():
        raise ValueError("base_action and residual must be finite")
    if res_scale < 0:
        raise ValueError(f"res_scale must be non-negative, got {res_scale}")

    def vector(value: Any, name: str) -> torch.Tensor:
        result = torch.as_tensor(value, device=base_action.device, dtype=base_action.dtype)
        if result.shape != (ARM_DIM,):
            raise ValueError(f"{name} must have shape ({ARM_DIM},), got {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result

    std = vector(action_std_arm, "action_std_arm")
    q_min = vector(q_min_arm, "q_min_arm")
    q_max = vector(q_max_arm, "q_max_arm")
    if torch.any(q_min > q_max):
        raise ValueError("q_min_arm must not exceed q_max_arm")

    final = base_action.clone()
    arm_target = base_action[..., :ARM_DIM] + res_scale * std * torch.tanh(residual)
    final[..., :ARM_DIM] = torch.maximum(torch.minimum(arm_target, q_max), q_min)
    # Do not assign the last two channels from the actor.  Keeping them from
    # the clone makes the gripper pass-through explicit and gradient-safe.
    return final


def residual_saturation_fraction(residual: torch.Tensor, *, threshold: float = 0.99) -> float:
    """Return the fraction of arm elements at or beyond ``|tanh(r)|`` threshold."""

    if not isinstance(residual, torch.Tensor) or residual.numel() == 0:
        raise ValueError("residual must be a non-empty torch.Tensor")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return float((torch.tanh(residual).abs() >= threshold).to(torch.float32).mean().item())


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    """Put a base module in eval mode and freeze every parameter in-place."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def assert_frozen(module: torch.nn.Module) -> None:
    """Raise if a supposedly frozen module still has trainable parameters."""

    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"base policy is not frozen; trainable parameters: {trainable[:5]}")


def module_parameter_sha256(module: torch.nn.Module) -> str:
    """Hash a module state dict without retaining model data in experiment rows."""

    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        # Use the raw Torch storage rather than ``Tensor.numpy()``.  This keeps
        # the smoke usable with a host that has NumPy 2.x and an older Torch
        # build which cannot initialize the NumPy bridge.
        digest.update(bytes(tensor.detach().cpu().contiguous().untyped_storage()))
    return digest.hexdigest()
