"""Stable initial-noise and DDPM reverse-generator streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ACTION_DIM, HORIZON, PROTOCOL_NAMESPACE
from .provenance import sha256_bytes, sha256_tensor

INITIAL_NOISE_STREAM = 0
REVERSE_NOISE_STREAM = 1


def _uint32_seed(parts: list[int]) -> int:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in parts):
        raise TypeError("Seed components must be integers")
    state = np.random.SeedSequence(parts).generate_state(1, dtype=np.uint32)
    return int(state[0])


def initial_noise_seed(
    *,
    environment_seed: int,
    policy_seed: int,
    replan_index: int,
) -> int:
    return _uint32_seed(
        [
            PROTOCOL_NAMESPACE,
            environment_seed,
            policy_seed,
            replan_index,
            INITIAL_NOISE_STREAM,
        ]
    )


def reverse_noise_seed(*, environment_seed: int, replan_index: int) -> int:
    return _uint32_seed(
        [
            PROTOCOL_NAMESPACE,
            environment_seed,
            replan_index,
            REVERSE_NOISE_STREAM,
        ]
    )


def fixed_observation_noise_seed(*, candidate_ordinal: int, sample_ordinal: int) -> int:
    return initial_noise_seed(
        environment_seed=candidate_ordinal,
        policy_seed=sample_ordinal,
        replan_index=0,
    )


def fixed_observation_reverse_seed(*, candidate_ordinal: int) -> int:
    return reverse_noise_seed(environment_seed=candidate_ordinal, replan_index=0)


def make_generator(*, seed: int, device: torch.device | str) -> torch.Generator:
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator


def generator_state_sha256(generator: torch.Generator) -> str:
    state = generator.get_state().detach().contiguous().cpu()
    return sha256_bytes(state.numpy().tobytes())


@dataclass(frozen=True)
class ExplicitNoise:
    seed: int
    tensor: torch.Tensor
    tensor_sha256: str


def make_initial_noise(
    *,
    seed: int,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    horizon: int = HORIZON,
    action_dim: int = ACTION_DIM,
) -> ExplicitNoise:
    if batch_size <= 0 or horizon <= 0 or action_dim <= 0:
        raise ValueError("Noise dimensions must be positive")
    generator = make_generator(seed=seed, device=device)
    tensor = torch.randn(
        (batch_size, horizon, action_dim),
        device=torch.device(device),
        dtype=dtype,
        generator=generator,
    )
    return ExplicitNoise(seed=seed, tensor=tensor, tensor_sha256=sha256_tensor(tensor))
