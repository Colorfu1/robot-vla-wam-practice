"""Checkpoint loading, scheduler replacement, and controlled Diffusion sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.processor.migrate_policy_normalization import extract_normalization_stats
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from safetensors import safe_open
from torch import Tensor

from robot_practice.policies.bc_chunk.configuration_bc_chunk import BCChunkConfig  # noqa: F401
from robot_practice.policies.mode_bc_chunk.configuration_mode_bc_chunk import (  # noqa: F401
    ModeBCChunkConfig,
)
from robot_practice.policies.mode_diffusion.configuration_mode_diffusion import (  # noqa: F401
    ModeDiffusionConfig,
)

from .noise import generator_state_sha256, make_generator
from .provenance import state_dict_hashes

Scheduler = DDPMScheduler | DDIMScheduler
_LOGGER = logging.getLogger(__name__)
_LEGACY_NORMALIZATION_PREFIXES = (
    "normalize_inputs.buffer_",
    "unnormalize_outputs.buffer_",
    "normalize_targets.buffer_",
    "normalize.",
    "unnormalize.",
    "input_normalizer.",
    "output_normalizer.",
    "normalalize_inputs.",
    "unnormalize_outputs.",
    "normalize_targets.",
    "unnormalize_targets.",
)


@dataclass(frozen=True)
class LoadedPolicy:
    source: str
    config: PreTrainedConfig
    policy: torch.nn.Module
    preprocessor: Any
    postprocessor: Any


@dataclass(frozen=True)
class ControlledSample:
    actions: Tensor
    full_horizon: Tensor
    reverse_seed: int
    reverse_generator_state_sha256: str
    scheduler: str
    num_inference_steps: int


def _saved_processor_directory(
    policy_path: str | Path,
    *,
    revision: str | None,
    local_files_only: bool,
) -> Path | None:
    source = Path(policy_path).expanduser()
    filenames = (
        f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
    )
    if source.is_dir():
        return source.resolve() if all((source / name).is_file() for name in filenames) else None

    try:
        downloaded = [
            Path(
                hf_hub_download(
                    repo_id=str(policy_path),
                    filename=filename,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )
            for filename in filenames
        ]
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return None
    parents = {path.parent for path in downloaded}
    if len(parents) != 1:
        raise RuntimeError("Preprocessor and postprocessor resolved to different revisions")
    return parents.pop()


def load_legacy_normalization_stats(
    policy_path: str | Path,
    *,
    revision: str | None,
    local_files_only: bool,
) -> dict[str, dict[str, Tensor]]:
    """Recover processor statistics embedded by legacy LeRobot checkpoints."""

    source = Path(policy_path).expanduser()
    if source.is_dir():
        model_path = source / "model.safetensors"
        if not model_path.is_file():
            raise FileNotFoundError(f"Legacy model weights are missing: {model_path}")
    else:
        model_path = Path(
            hf_hub_download(
                repo_id=str(policy_path),
                filename="model.safetensors",
                revision=revision,
                local_files_only=local_files_only,
            )
        )

    normalization_state = {}
    with safe_open(model_path, framework="pt", device="cpu") as tensors:
        tensor_keys = tensors.keys()
        for key in tensor_keys:
            if key.startswith(_LEGACY_NORMALIZATION_PREFIXES):
                normalization_state[key] = tensors.get_tensor(key)
    stats = extract_normalization_stats(normalization_state)
    if not stats:
        raise FileNotFoundError(
            f"No processor JSON or legacy normalization statistics found for {policy_path}"
        )
    return stats


def load_policy_with_processors(
    policy_path: str | Path,
    *,
    revision: str | None = None,
    device: str | None = None,
    local_files_only: bool = False,
) -> LoadedPolicy:
    source = str(policy_path)
    config = PreTrainedConfig.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
    )
    if device is not None:
        config.device = device
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        source,
        config=config,
        revision=revision,
        local_files_only=local_files_only,
    )
    processor_directory = _saved_processor_directory(
        policy_path,
        revision=revision,
        local_files_only=local_files_only,
    )
    if processor_directory is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(processor_directory),
        )
    else:
        stats = load_legacy_normalization_stats(
            policy_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        _LOGGER.warning(
            "Processor JSON files are absent for %s; recovered legacy normalization "
            "statistics from model.safetensors",
            source,
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            dataset_stats=stats,
        )
    policy.eval()
    return LoadedPolicy(
        source=source,
        config=config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def build_scheduler(config: DiffusionConfig, scheduler_name: str) -> Scheduler:
    scheduler_name = scheduler_name.upper()
    kwargs = {
        "num_train_timesteps": config.num_train_timesteps,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "beta_schedule": config.beta_schedule,
        "clip_sample": config.clip_sample,
        "clip_sample_range": config.clip_sample_range,
        "prediction_type": config.prediction_type,
    }
    if scheduler_name == "DDPM":
        return DDPMScheduler(**kwargs)
    if scheduler_name == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"scheduler_name must be DDPM or DDIM, got {scheduler_name!r}")


def replace_scheduler(
    policy: DiffusionPolicy,
    *,
    scheduler_name: str,
    num_inference_steps: int,
) -> None:
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    before = state_dict_hashes(policy.state_dict())
    policy.diffusion.noise_scheduler = build_scheduler(policy.config, scheduler_name)
    policy.diffusion.num_inference_steps = num_inference_steps
    after = state_dict_hashes(policy.state_dict())
    if before != after:
        raise RuntimeError("Scheduler replacement changed model parameter hashes")


def controlled_sample(
    policy: DiffusionPolicy,
    frozen_batch: dict[str, Tensor],
    *,
    initial_noise: Tensor,
    reverse_seed: int,
) -> ControlledSample:
    if not isinstance(policy.config, DiffusionConfig):
        raise TypeError("controlled_sample requires a DiffusionConfig")
    if OBS_STATE not in frozen_batch:
        raise KeyError(f"Missing required key {OBS_STATE!r}")
    batch_size, n_obs_steps = frozen_batch[OBS_STATE].shape[:2]
    expected_shape = (
        batch_size,
        policy.config.horizon,
        policy.config.action_feature.shape[0],
    )
    if tuple(initial_noise.shape) != expected_shape:
        raise ValueError(
            f"initial_noise shape={tuple(initial_noise.shape)}, expected {expected_shape}"
        )
    if n_obs_steps != policy.config.n_obs_steps:
        raise ValueError(
            f"observation history has {n_obs_steps} steps, expected {policy.config.n_obs_steps}"
        )
    device = get_device_from_parameters(policy)
    dtype = get_dtype_from_parameters(policy)
    if initial_noise.device != device:
        raise ValueError(f"initial_noise device={initial_noise.device}, expected {device}")
    if initial_noise.dtype != dtype:
        raise ValueError(f"initial_noise dtype={initial_noise.dtype}, expected {dtype}")

    reverse_generator = make_generator(seed=reverse_seed, device=device)
    generator_hash = generator_state_sha256(reverse_generator)
    global_condition = policy.diffusion._prepare_global_conditioning(frozen_batch)
    full_horizon = policy.diffusion.conditional_sample(
        batch_size,
        global_cond=global_condition,
        generator=reverse_generator,
        noise=initial_noise,
    )
    start = policy.config.n_obs_steps - 1
    end = start + policy.config.n_action_steps
    scheduler = policy.diffusion.noise_scheduler
    scheduler_name = "DDPM" if isinstance(scheduler, DDPMScheduler) else "DDIM"
    return ControlledSample(
        actions=full_horizon[:, start:end],
        full_horizon=full_horizon,
        reverse_seed=reverse_seed,
        reverse_generator_state_sha256=generator_hash,
        scheduler=scheduler_name,
        num_inference_steps=policy.diffusion.num_inference_steps,
    )
