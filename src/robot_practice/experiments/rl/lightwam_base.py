"""Light-WAM base-policy adapter for the robomimic pixel contract.

The existing DPPO implementation consumes a ``VisionDiffusionMLP`` through
the upstream ``DiffusionModel`` API.  Light-WAM is a different policy family:
it consumes an RGB frame, a proprioceptive vector and text/context, then uses
Wan latent/video features and a continuous-flow action decoder.  This module
keeps that boundary explicit instead of pretending the two diffusion APIs are
interchangeable.

Only the base-policy path lives here.  DPPO/PPO and the local residual actor do
not import this module and remain unchanged until a separate policy-gradient
adapter is designed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_ACTION_DIM = 7
DEFAULT_ACTION_HORIZON = 4
DEFAULT_VIDEO_FRAMES = 5
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CONTEXT_LEN = 128
DEFAULT_CONTEXT_DIM = 4096


def _as_pair(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two integers, got {value}")
    pair = (int(value[0]), int(value[1]))
    if any(item <= 0 for item in pair):
        raise ValueError(f"{name} entries must be positive, got {pair}")
    return pair


def _normalise_lightwam_image(rgb: torch.Tensor) -> torch.Tensor:
    """Convert DPPO-style RGB values to Light-WAM's VAE ``[-1, 1]`` range."""

    image = rgb.float()
    if not torch.isfinite(image).all():
        raise ValueError("RGB observation contains non-finite values")
    if image.numel() == 0:
        raise ValueError("RGB observation is empty")
    minimum = float(image.detach().amin().item())
    maximum = float(image.detach().amax().item())
    if minimum >= -1.01 and maximum <= 1.01:
        # Accept an already-normalized Light-WAM image as well as a [0, 1]
        # image.  DPPO's actual wrapper emits [0, 255], so this branch is not
        # ambiguous for the current route.
        if minimum < -0.01:
            return image.clamp(-1.0, 1.0)
        return (image * 2.0 - 1.0).clamp(-1.0, 1.0)
    if minimum >= -1.0e-3 and maximum <= 255.0 + 1.0e-3:
        return (image / 255.0 * 2.0 - 1.0).clamp(-1.0, 1.0)
    raise ValueError(
        "RGB observation is neither [0, 1], [-1, 1], nor [0, 255]: "
        f"min={minimum:.5g} max={maximum:.5g}"
    )


def _context_for_sample(
    value: Any,
    index: int,
    batch_size: int,
    *,
    name: str,
    ndim_without_batch: int = 2,
) -> Any:
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == ndim_without_batch:
        return tensor
    if tensor.ndim == ndim_without_batch + 1:
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension must be {batch_size}, got {tuple(tensor.shape)}"
            )
        return tensor[index]
    if ndim_without_batch == 1:
        expected = "[L] or [B,L]"
    else:
        expected = "[L,D] or [B,L,D]"
    raise ValueError(f"{name} must be {expected}, got {tuple(tensor.shape)}")


def _prompt_for_sample(value: Any, index: int, batch_size: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        if len(value) != batch_size:
            raise ValueError(f"prompt batch must have {batch_size} entries, got {len(value)}")
        return str(value[index])
    raise TypeError(f"prompt must be a string or a sequence of strings, got {type(value)}")


def add_lightwam_source(vendor_root: str | Path) -> Path:
    """Expose the pinned official Light-WAM checkout without installing it."""

    root = Path(vendor_root).expanduser().resolve()
    source = root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Light-WAM source directory is missing: {source}")
    source_string = str(source)
    if source_string in sys.path:
        sys.path.remove(source_string)
    sys.path.insert(0, source_string)
    return root


class LightWAMBaseAdapter(nn.Module):
    """Map the current ``{rgb, state}`` observation to official Light-WAM.

    The official ``LightWAM.infer_action`` API intentionally supports one
    sample at a time.  The adapter loops over a batch, which is correct and
    keeps the integration small; batching can be optimized later without
    changing the observation contract.

    The returned action is the normalized 7D action chunk expected by the
    current robomimic wrapper.  No controller-unit conversion occurs here.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        action_dim: int = DEFAULT_ACTION_DIM,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        prompt: str | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        num_inference_steps: int = 10,
        seed: int | None = 0,
        tiled: bool = False,
        clip_actions: bool = True,
    ) -> None:
        super().__init__()
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be positive")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.model = model
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.image_size = _as_pair(image_size, name="image_size")
        if any(item % 16 != 0 for item in self.image_size):
            raise ValueError(
                "Light-WAM input image dimensions must be multiples of 16, "
                f"got {self.image_size}"
            )
        self.prompt = prompt
        self.context = context
        self.context_mask = context_mask
        self.num_inference_steps = int(num_inference_steps)
        self.seed = None if seed is None else int(seed)
        self.tiled = bool(tiled)
        self.clip_actions = bool(clip_actions)

    @property
    def model_device(self) -> torch.device:
        configured = getattr(self.model, "device", None)
        if configured is not None:
            return torch.device(configured)
        for parameter in self.model.parameters():
            return parameter.device
        return torch.device("cpu")

    def _prepare_observation(
        self, cond: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unexpected = sorted(set(cond) - {"rgb", "state", "context", "context_mask", "prompt"})
        if unexpected:
            raise ValueError(f"unexpected observations passed to Light-WAM base: {unexpected}")
        if "rgb" not in cond or "state" not in cond:
            raise KeyError("Light-WAM base requires both `rgb` and `state` observations")

        rgb = cond["rgb"] if isinstance(cond["rgb"], torch.Tensor) else torch.as_tensor(cond["rgb"])
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.ndim == 5:
            rgb = rgb[:, -1]
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(
                "Light-WAM base currently consumes exactly one RGB camera; expected "
                f"[B,3,H,W] or [B,T,3,H,W], got {tuple(rgb.shape)}"
            )
        rgb = _normalise_lightwam_image(rgb)
        if tuple(rgb.shape[-2:]) != self.image_size:
            rgb = F.interpolate(rgb, size=self.image_size, mode="bilinear", align_corners=False)

        state = (
            cond["state"]
            if isinstance(cond["state"], torch.Tensor)
            else torch.as_tensor(cond["state"])
        )
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(
                f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}"
            )
        if state.shape[0] != rgb.shape[0]:
            raise ValueError(
                f"rgb/state batch mismatch: {rgb.shape[0]} vs {state.shape[0]}"
            )
        if not torch.isfinite(state.float()).all():
            raise ValueError("state observation contains non-finite values")
        return rgb, state.float()

    def _sample_context(self, cond: Mapping[str, Any], index: int, batch_size: int):
        context_value = cond.get("context", self.context)
        mask_value = cond.get("context_mask", self.context_mask)
        if context_value is None and mask_value is None:
            return None, None
        if context_value is None or mask_value is None:
            raise ValueError("Light-WAM requires both `context` and `context_mask`")
        context = _context_for_sample(context_value, index, batch_size, name="context")
        context_mask = _context_for_sample(
            mask_value,
            index,
            batch_size,
            name="context_mask",
            ndim_without_batch=1,
        )
        if context.ndim != 2 or context_mask.ndim != 1:
            raise ValueError(
                "per-sample context must be [L,D] and context_mask [L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != context_mask.shape[0]:
            raise ValueError("context and context_mask sequence lengths differ")
        return context, context_mask

    @torch.no_grad()
    def predict_action(
        self,
        cond: Mapping[str, Any],
        *,
        deterministic: bool = True,
        seed: int | Sequence[int] | None = None,
    ) -> torch.Tensor:
        rgb, state = self._prepare_observation(cond)
        batch_size = int(rgb.shape[0])
        prompt_value = cond.get("prompt", self.prompt)
        model_device = self.model_device
        outputs: list[torch.Tensor] = []

        for index in range(batch_size):
            prompt = _prompt_for_sample(prompt_value, index, batch_size)
            context, context_mask = self._sample_context(cond, index, batch_size)
            # A cached context is the offline replacement for prompt encoding.
            # The official Light-WAM API rejects prompt and context together, so
            # the explicit per-sample/cache context takes precedence over the
            # convenience prompt configured on the adapter.
            if context is not None:
                prompt = None
            if prompt is None and context is None:
                raise ValueError(
                    "Light-WAM requires a task prompt or precomputed context; "
                    "the current DPPO observation does not contain language"
                )

            if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
                if len(seed) != batch_size:
                    raise ValueError(f"seed sequence must have {batch_size} entries")
                sample_seed = None if seed[index] is None else int(seed[index])
            elif seed is not None:
                sample_seed = int(seed) + index
            elif deterministic:
                sample_seed = None if self.seed is None else self.seed + index
            else:
                sample_seed = None

            result = self.model.infer_action(
                prompt=prompt,
                input_image=rgb[index].to(device=model_device),
                action_horizon=self.action_horizon,
                proprio=state[index].to(device=model_device),
                context=None if context is None else context.to(device=model_device),
                context_mask=(
                    None
                    if context_mask is None
                    else context_mask.to(device=model_device, dtype=torch.bool)
                ),
                num_inference_steps=self.num_inference_steps,
                seed=sample_seed,
                tiled=self.tiled,
            )
            action = result["action"] if isinstance(result, Mapping) else result
            action = action if isinstance(action, torch.Tensor) else torch.as_tensor(action)
            if action.ndim == 3 and action.shape[0] == 1:
                action = action[0]
            expected = (self.action_horizon, self.action_dim)
            if tuple(action.shape) != expected:
                raise ValueError(
                    f"Light-WAM returned {tuple(action.shape)}, expected {expected}"
                )
            action = action.to(device=rgb.device, dtype=torch.float32)
            if not torch.isfinite(action).all():
                raise ValueError("Light-WAM returned non-finite actions")
            if self.clip_actions:
                action = action.clamp(-1.0, 1.0)
            outputs.append(action)
        return torch.stack(outputs, dim=0)

    @torch.no_grad()
    def forward(
        self, cond: Mapping[str, Any], deterministic: bool = True, **kwargs
    ) -> torch.Tensor:
        """Compatibility spelling for inference-only callers.

        This deliberately does not implement the DPPO ``network(x, t, cond)``
        contract.  It only returns an action chunk and is not differentiable.
        """

        return self.predict_action(cond, deterministic=deterministic, **kwargs)


class DummyLightWAMCore(nn.Module):
    """Tiny ``infer_action``-compatible core used for data-free adapter smoke."""

    def __init__(self, action_dim: int = DEFAULT_ACTION_DIM, device: str = "cpu") -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.device = torch.device(device)

    @torch.no_grad()
    def infer_action(self, *, prompt, input_image, action_horizon, proprio, **kwargs):
        if prompt is None and kwargs.get("context") is None:
            raise ValueError("dummy Light-WAM core requires prompt or context")
        image_signal = input_image.float().mean().to(device=self.device)
        state = proprio.float().reshape(-1)
        signal = torch.zeros(self.action_dim, device=self.device)
        signal[: min(self.action_dim, state.numel())] = state[: self.action_dim]
        signal[0] = signal[0] + image_signal * 0.01
        action = torch.tanh(signal).unsqueeze(0).repeat(int(action_horizon), 1)
        return {"action": action.cpu()}


def load_context_cache(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a Light-WAM ``context/context_mask`` cache without model code."""

    cache_path = Path(path).expanduser()
    if not cache_path.is_file():
        raise FileNotFoundError(f"context cache is missing: {cache_path}")
    payload = torch.load(str(cache_path), map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        context, context_mask = payload, torch.ones(payload.shape[0], dtype=torch.bool)
    elif isinstance(payload, Mapping):
        context = payload.get("context", payload.get("prompt_emb"))
        context_mask = payload.get("context_mask", payload.get("mask"))
        if context is None or context_mask is None:
            raise ValueError(f"context cache must contain context and mask: {cache_path}")
    else:
        raise TypeError(f"unsupported context cache payload: {type(payload)}")
    context = torch.as_tensor(context, dtype=torch.float32)
    context_mask = torch.as_tensor(context_mask, dtype=torch.bool)
    if context.ndim != 2 or context_mask.ndim != 1:
        raise ValueError(
            "context cache must be [L,D]/[L], got "
            f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
        )
    if context.shape[0] != context_mask.shape[0]:
        raise ValueError("context cache sequence lengths differ")
    return context, context_mask


def _load_model_config(path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config = OmegaConf.load(str(Path(path).expanduser()))
    if "model" in config and "_target_" not in config:
        config = config.model
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Light-WAM model config must resolve to a mapping, got {type(payload)}")
    payload.pop("model_dtype", None)
    payload.pop("checkpoint", None)
    payload.pop("_target_", None)
    return payload


def _checkpoint_looks_like_dppo(path: Path) -> bool:
    """Detect DPPO keys without materializing a multi-GB checkpoint on CPU."""

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:  # pragma: no cover - supported by the pinned runtime
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    else:
        with FakeTensorMode():
            payload = torch.load(str(path), map_location="cpu", weights_only=True)
    return isinstance(payload, Mapping) and "model" in payload and "mot" not in payload


def _validate_lightwam_checkpoint(
    value: str | Path,
    *,
    allow_training_state_directory: bool = False,
) -> Path:
    checkpoint_path = Path(value).expanduser().resolve()
    if allow_training_state_directory and checkpoint_path.is_dir():
        return checkpoint_path
    if not checkpoint_path.is_file():
        expected = "file or training-state directory" if allow_training_state_directory else "file"
        raise FileNotFoundError(
            f"Light-WAM checkpoint {expected} is missing: {checkpoint_path}"
        )
    if _checkpoint_looks_like_dppo(checkpoint_path):
        raise ValueError(
            "The supplied checkpoint is a DPPO checkpoint (contains `model` but not `mot`); "
            "DPPO state_4000.pt cannot be loaded into Light-WAM."
        )
    return checkpoint_path


def load_official_lightwam(
    *,
    vendor_root: str | Path,
    model_config: str | Path,
    local_model_root: str | Path | None,
    device: str = "cuda",
    model_dtype: torch.dtype = torch.bfloat16,
    checkpoint: str | Path | None = None,
    allow_download: bool = False,
) -> nn.Module:
    """Construct the official Light-WAM model with an explicit offline gate."""

    checkpoint_path = (
        None if checkpoint is None else _validate_lightwam_checkpoint(checkpoint)
    )
    add_lightwam_source(vendor_root)
    from lightwam.runtime import create_lightwam

    previous_model_root = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
    previous_skip = os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD")
    try:
        if local_model_root is not None:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(
                Path(local_model_root).expanduser().resolve()
            )
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "false" if allow_download else "true"
        kwargs = _load_model_config(model_config)
        kwargs["device"] = device
        kwargs["model_dtype"] = model_dtype
        model = create_lightwam(**kwargs)
    finally:
        if previous_model_root is None:
            os.environ.pop("DIFFSYNTH_MODEL_BASE_PATH", None)
        else:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = previous_model_root
        if previous_skip is None:
            os.environ.pop("DIFFSYNTH_SKIP_DOWNLOAD", None)
        else:
            os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = previous_skip

    if checkpoint_path is not None:
        model.load_checkpoint(str(checkpoint_path))
    return model


def prepare_prompt_context_cache(
    *,
    vendor_root: str | Path,
    local_model_root: str | Path,
    model_id: str,
    tokenizer_model_id: str,
    prompt: str,
    output_path: str | Path,
    context_len: int = DEFAULT_CONTEXT_LEN,
    device: str = "cuda",
    model_dtype: torch.dtype = torch.bfloat16,
    allow_download: bool = False,
    overwrite: bool = False,
) -> Path:
    """Encode one task prompt with the official Light-WAM text encoder.

    The training model can omit the 11 GB text encoder and consume this small
    cache instead. Downloads stay opt-in so a missing local encoder is an
    explicit error rather than an accidental multi-GB transfer.
    """

    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite context cache: {output}; pass --overwrite to replace it"
        )
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}")

    add_lightwam_source(vendor_root)
    model_root = Path(local_model_root).expanduser().resolve()
    model_dir = model_root / model_id
    text_matches = sorted(model_dir.glob("models_t5_umt5-xxl-enc-bf16.*"))
    tokenizer_dir = model_root / tokenizer_model_id / "google" / "umt5-xxl"
    tokenizer_matches = sorted(tokenizer_dir.glob("*"))
    if not allow_download and (not text_matches or not tokenizer_matches):
        missing = []
        if not text_matches:
            missing.append(f"text encoder under {model_dir}")
        if not tokenizer_matches:
            missing.append(f"tokenizer under {tokenizer_dir}")
        raise FileNotFoundError(
            "Light-WAM prompt context assets are missing (downloads are disabled): "
            + "; ".join(missing)
        )

    from lightwam.models.wan22.helpers.loader import (
        _load_registered_model,
        _resolve_configs_for_backbone,
    )
    from lightwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    previous_environment = _set_offline_model_environment(
        model_root,
        allow_download=allow_download,
    )
    try:
        _, text_config, _, tokenizer_config = _resolve_configs_for_backbone(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            video_backbone_type="wan2_1_t2v",
            redirect_common_files=False,
        )
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=model_dtype,
            device=device,
        ).eval()
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(context_len),
            clean="whitespace",
        )
        with torch.inference_mode():
            input_ids, input_mask = tokenizer(
                prompt,
                return_mask=True,
                add_special_tokens=True,
            )
            input_ids = input_ids.to(device=device)
            input_mask = input_mask.to(device=device, dtype=torch.bool)
            context = text_encoder(input_ids, input_mask)
    finally:
        _restore_model_environment(previous_environment)

    context = context[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    context_mask = input_mask[0].detach().to(device="cpu", dtype=torch.bool).contiguous()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"context": context, "mask": context_mask}, str(output))
    print(
        f"LIGHTWAM_CONTEXT_PREPARE=PASS output={output} shape={tuple(context.shape)}",
        flush=True,
    )
    return output


class RobomimicLightWAMWindowDataset(torch.utils.data.Dataset):
    """Turn the current robomimic NPZ into short Light-WAM training windows.

    The current NPZ stores one RGB frame and one state per control transition.
    For the smallest valid Light-WAM training sample we use five consecutive
    frames and four consecutive actions.  This is a temporal adapter only; it
    does not alter the underlying observations or actions.
    """

    def __init__(
        self,
        data_path: str,
        action_horizon: int = DEFAULT_ACTION_HORIZON,
        video_frames: int = DEFAULT_VIDEO_FRAMES,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        context_path: str | None = None,
        prompt: str = "insert the square nut into the square peg",
        normalized: bool = True,
    ) -> None:
        self.data_path = Path(data_path).expanduser().resolve()
        if not self.data_path.is_file():
            raise FileNotFoundError(f"robomimic NPZ is missing: {self.data_path}")
        self.action_horizon = int(action_horizon)
        self.video_frames = int(video_frames)
        if self.action_horizon <= 0 or self.video_frames <= 1:
            raise ValueError("action_horizon must be positive and video_frames must exceed 1")
        if self.action_horizon % (self.video_frames - 1) != 0:
            raise ValueError(
                "Light-WAM requires action_horizon divisible by video transitions: "
                f"{self.action_horizon} vs {self.video_frames - 1}"
            )
        self.image_size = _as_pair(image_size, name="image_size")
        if any(item % 16 != 0 for item in self.image_size):
            raise ValueError(f"image_size must be divisible by 16, got {self.image_size}")
        self.prompt = str(prompt)
        with np.load(self.data_path, allow_pickle=False) as data:
            required = {"states", "actions", "images", "traj_lengths"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"NPZ is missing required arrays: {missing}")
            self.states = np.asarray(data["states"], dtype=np.float32)
            self.actions = np.asarray(data["actions"], dtype=np.float32)
            self.images = np.asarray(data["images"])
            self.traj_lengths = np.asarray(data["traj_lengths"], dtype=np.int64)
        if self.states.ndim != 2 or self.states.shape[1] <= 0:
            raise ValueError(f"states must be [N,D], got {self.states.shape}")
        if self.actions.ndim != 2 or self.actions.shape[1] <= 0:
            raise ValueError(f"actions must be [N,A], got {self.actions.shape}")
        if self.images.ndim != 4 or self.images.shape[1] != 3:
            raise ValueError(f"images must be [N,3,H,W], got {self.images.shape}")
        if not (len(self.states) == len(self.actions) == len(self.images)):
            raise ValueError("states/actions/images transition counts differ")
        if int(self.traj_lengths.sum()) != len(self.states):
            raise ValueError("traj_lengths does not sum to the transition count")
        if normalized:
            for name, array in (("states", self.states), ("actions", self.actions)):
                if not np.isfinite(array).all():
                    raise ValueError(f"{name} contains non-finite values")
                if float(array.min()) < -1.001 or float(array.max()) > 1.001:
                    raise ValueError(
                        f"{name} is not normalized to [-1,1]; pass normalized=False only "
                        "after adding an explicit normalizer"
                    )
        self.context: torch.Tensor | None = None
        self.context_mask: torch.Tensor | None = None
        if context_path is not None:
            self.context, self.context_mask = load_context_cache(context_path)

        windows: list[tuple[int, int]] = []
        offset = 0
        for length in self.traj_lengths.tolist():
            length = int(length)
            max_start = min(
                length - self.action_horizon,
                length - self.video_frames,
            )
            windows.extend((offset + start, length) for start in range(max(max_start + 1, 0)))
            offset += length
        self.windows = windows
        if not self.windows:
            raise ValueError("dataset contains no complete Light-WAM temporal windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start, _ = self.windows[index]
        image_array = self.images[start : start + self.video_frames]
        images = torch.from_numpy(np.ascontiguousarray(image_array)).float() / 255.0
        images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        video = (images * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous()
        state = torch.from_numpy(
            np.ascontiguousarray(self.states[start : start + self.video_frames])
        )
        action = torch.from_numpy(
            np.ascontiguousarray(self.actions[start : start + self.action_horizon])
        )
        sample: dict[str, Any] = {
            "video": video,
            "proprio": state,
            "action": action,
            "prompt": self.prompt,
            "image_is_pad": torch.zeros(self.video_frames, dtype=torch.bool),
            "action_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
        }
        if self.context is None or self.context_mask is None:
            raise ValueError(
                "formal Light-WAM training requires --context-path; use --dummy for a "
                "data-free contract smoke"
            )
        sample["context"] = self.context.clone()
        sample["context_mask"] = self.context_mask.clone()
        return sample


def build_training_config(
    *,
    vendor_root: str | Path,
    model_config: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    context_path: str | Path,
    prompt: str,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    video_frames: int = DEFAULT_VIDEO_FRAMES,
    batch_size: int = 1,
    num_workers: int = 0,
    learning_rate: float = 1.0e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.05,
    warmup_steps: int | None = None,
    weight_decay: float = 0.0,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    num_epochs: int = 1,
    max_steps: int | None = None,
    stop_after_steps: int | None = None,
    save_every: int = 100,
    eval_every: int = 0,
    log_every: int = 1,
    mixed_precision: str = "bf16",
    seed: int = 42,
    resume: str | Path | None = None,
    benchmark_enabled: bool = False,
    benchmark_warmup_steps: int = 10,
    benchmark_measure_steps: int = 50,
) -> Any:
    """Compose the official ``Wan22Trainer`` config with one local dataset."""

    from omegaconf import OmegaConf

    vendor = Path(vendor_root).expanduser().resolve()
    train_template = vendor / "configs" / "train.yaml"
    if not train_template.is_file():
        raise FileNotFoundError(f"Light-WAM train config is missing: {train_template}")
    cfg = OmegaConf.load(str(train_template))
    cfg.model = OmegaConf.load(str(Path(model_config).expanduser().resolve()))
    cfg.data = OmegaConf.create(
        {
            "train": {
                "_target_": (
                    "robot_practice.experiments.rl.lightwam_base."
                    "RobomimicLightWAMWindowDataset"
                ),
                "data_path": str(Path(data_path).expanduser().resolve()),
                "action_horizon": int(action_horizon),
                "video_frames": int(video_frames),
                "image_size": list(_as_pair(image_size, name="image_size")),
                "context_path": str(Path(context_path).expanduser().resolve()),
                "prompt": str(prompt),
                "normalized": True,
            },
            "val": None,
        }
    )
    cfg.output_dir = str(Path(output_dir).expanduser().resolve())
    cfg.batch_size = int(batch_size)
    cfg.num_workers = int(num_workers)
    cfg.learning_rate = float(learning_rate)
    cfg.lr_scheduler_type = str(lr_scheduler_type)
    cfg.warmup_ratio = float(warmup_ratio)
    cfg.warmup_steps = None if warmup_steps is None else int(warmup_steps)
    cfg.weight_decay = float(weight_decay)
    cfg.gradient_accumulation_steps = int(gradient_accumulation_steps)
    cfg.max_grad_norm = float(max_grad_norm)
    cfg.num_epochs = int(num_epochs)
    cfg.max_steps = None if max_steps is None else int(max_steps)
    cfg.stop_after_steps = (
        None if stop_after_steps is None else int(stop_after_steps)
    )
    if cfg.stop_after_steps is not None:
        if cfg.stop_after_steps <= 0:
            raise ValueError("stop_after_steps must be positive when provided")
        if cfg.max_steps is None:
            raise ValueError("stop_after_steps requires max_steps")
        if cfg.stop_after_steps > cfg.max_steps:
            raise ValueError("stop_after_steps cannot exceed max_steps")
    cfg.save_every = int(save_every)
    cfg.eval_every = int(eval_every)
    cfg.log_every = int(log_every)
    cfg.mixed_precision = str(mixed_precision)
    cfg.seed = int(seed)
    cfg.resume = None if resume in (None, "") else str(Path(resume).expanduser().resolve())
    cfg.benchmark.enabled = bool(benchmark_enabled)
    cfg.benchmark.warmup_steps = int(benchmark_warmup_steps)
    cfg.benchmark.measure_steps = int(benchmark_measure_steps)
    cfg.benchmark.output_filename = "training_speed_benchmark.json"
    cfg.benchmark.description = (
        "Light-WAM robomimic Square real-training smoke"
        if benchmark_enabled
        else None
    )
    if benchmark_enabled:
        # The upstream benchmark returns before its checkpoint branch.  These
        # explicit zeros document that smoke never writes a multi-GB checkpoint.
        cfg.save_every = 0
        cfg.eval_every = 0
    cfg.wandb.enabled = False
    OmegaConf.resolve(cfg)
    return cfg


def _model_asset_report(model_config: Path, model_root: Path | None) -> dict[str, Any]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(_load_model_config(model_config))
    model_id = str(cfg.get("model_id", ""))
    backbone_type = str(cfg.get("video_backbone_type", "wan2_2_ti2v"))
    if model_root is None:
        return {
            "model_id": model_id,
            "video_backbone_type": backbone_type,
            "model_root": None,
            "missing": [],
            "download_required": None,
        }
    root = model_root.expanduser().resolve() / model_id
    if backbone_type == "wan2_1_t2v":
        patterns = ("diffusion_pytorch_model*.safetensors", "Wan2.1_VAE.pth")
    else:
        patterns = ("diffusion_pytorch_model*.safetensors", "Wan2.2_VAE.pth")
    found = {pattern: sorted(str(path) for path in root.glob(pattern)) for pattern in patterns}
    missing = [pattern for pattern, paths in found.items() if not paths]
    return {
        "model_id": model_id,
        "video_backbone_type": backbone_type,
        "model_root": str(root),
        "files": found,
        "missing": missing,
        "download_required": bool(missing),
    }


def _context_asset_report(path_value: str | Path | None) -> dict[str, Any]:
    if path_value in (None, ""):
        return {
            "path": None,
            "present": False,
            "valid": False,
            "error": None,
        }
    path = Path(path_value).expanduser().resolve()
    report: dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "valid": False,
        "error": None,
    }
    if not report["present"]:
        return report
    try:
        context, context_mask = load_context_cache(path)
        if context.shape[1] != DEFAULT_CONTEXT_DIM:
            raise ValueError(
                f"context embedding dim must be {DEFAULT_CONTEXT_DIM}, got {context.shape[1]}"
            )
        if context.shape[0] != DEFAULT_CONTEXT_LEN:
            raise ValueError(
                f"context length must be {DEFAULT_CONTEXT_LEN}, got {context.shape[0]}"
            )
        report["valid"] = True
        report["context_shape"] = list(context.shape)
        report["context_mask_shape"] = list(context_mask.shape)
        report["active_tokens"] = int(context_mask.sum().item())
    except (EOFError, OSError, RuntimeError, TypeError, ValueError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


class _Tee:
    """Mirror a training process stream to the terminal and a persistent log."""

    def __init__(self, original, log):
        self.original = original
        self.log = log

    def write(self, value: str) -> int:
        self.original.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.original.flush()
        self.log.flush()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_smoke_timing_summary(
    benchmark: Mapping[str, Any],
    *,
    target_optimizer_steps: int,
    end_to_end_elapsed_sec: float,
    dppo_reference: Mapping[str, Any] | None = None,
    training_alignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn the upstream measured window into a full-run timing estimate."""

    steps_per_sec = float(benchmark.get("steps_per_sec", 0.0))
    if not np.isfinite(steps_per_sec) or steps_per_sec <= 0:
        raise ValueError(f"benchmark steps_per_sec must be positive, got {steps_per_sec}")
    target_steps = int(target_optimizer_steps)
    if target_steps <= 0:
        raise ValueError(f"target_optimizer_steps must be positive, got {target_steps}")
    warmup_steps = int(benchmark.get("warmup_steps", 0))
    measure_steps = int(benchmark.get("measure_steps", 0))
    measured_elapsed_sec = float(benchmark.get("elapsed_sec", 0.0))
    compute_sec = float(target_steps / steps_per_sec)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "route": "lightwam_base_robomimic_square",
        "result_label": "real_training_timing_smoke",
        "step_unit": "optimizer_step",
        "short_run": {
            "warmup_steps": warmup_steps,
            "measure_steps": measure_steps,
            "total_optimizer_steps": warmup_steps + measure_steps,
            "measured_elapsed_sec": measured_elapsed_sec,
            "end_to_end_elapsed_sec": float(end_to_end_elapsed_sec),
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": float(benchmark.get("samples_per_sec", 0.0)),
            "effective_global_batch_size": int(
                benchmark.get("effective_global_batch_size", 0)
            ),
            "peak_gpu_mem_allocated_gb": float(
                benchmark.get("peak_gpu_mem_allocated_gb", 0.0)
            ),
            "peak_gpu_mem_reserved_gb": float(
                benchmark.get("peak_gpu_mem_reserved_gb", 0.0)
            ),
        },
        "full_training_estimate": {
            "target_optimizer_steps": target_steps,
            "compute_seconds": compute_sec,
            "compute_hours": compute_sec / 3600.0,
            "compute_days": compute_sec / 86400.0,
            "basis": "measured optimizer-step throughput after warmup",
            "excludes": [
                "model and optimizer initialization",
                "checkpoint writes",
                "evaluation",
                "cluster scheduling and storage contention",
            ],
        },
        "upstream_benchmark": dict(benchmark),
    }
    if dppo_reference is not None:
        payload["dppo_reference"] = dict(dppo_reference)
    if training_alignment is not None:
        payload["training_alignment"] = dict(training_alignment)
    return payload


def _load_adapter_config(path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config = OmegaConf.load(str(Path(path).expanduser().resolve()))
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Light-WAM adapter config must resolve to a mapping, got {type(payload)}")
    return payload


def _nested_config_value(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return default if value is None else value


def _path_or_default(value: Any, default: str) -> Path:
    return Path(default if value in (None, "") else str(value)).expanduser().resolve()


def _set_offline_model_environment(
    model_root: Path, *, allow_download: bool
) -> tuple[str | None, str | None]:
    previous_root = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
    previous_skip = os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD")
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_root)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "false" if allow_download else "true"
    return previous_root, previous_skip


def _restore_model_environment(previous: tuple[str | None, str | None]) -> None:
    previous_root, previous_skip = previous
    if previous_root is None:
        os.environ.pop("DIFFSYNTH_MODEL_BASE_PATH", None)
    else:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = previous_root
    if previous_skip is None:
        os.environ.pop("DIFFSYNTH_SKIP_DOWNLOAD", None)
    else:
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = previous_skip


def _is_distributed_main_process() -> bool:
    rank_value = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank_value) == 0
    except ValueError:
        return rank_value in ("", "0")


def _parse_int_csv(value: str | Sequence[int]) -> list[int]:
    """Parse a comma-separated seed list while rejecting duplicates."""

    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item) for item in value]
    if not items:
        raise ValueError("at least one evaluation seed is required")
    seeds = [int(item) for item in items]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"evaluation seeds must be unique, got {seeds}")
    return seeds


def _eval_log(path: Path, message: str) -> None:
    """Write one observable eval progress line to stdout and ``run.log``."""

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{timestamp}] {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


class _StreamingArrayDiagnostics:
    """Accumulate compact tensor diagnostics without retaining trajectories."""

    def __init__(self, *, feature_dim: int | None = None, saturation_limit: float = 0.999):
        if feature_dim is not None and int(feature_dim) <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = None if feature_dim is None else int(feature_dim)
        self.saturation_limit = float(saturation_limit)
        self.elements = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.total = 0.0
        self.total_squared = 0.0
        self.saturated = 0
        self.feature_rows = 0
        self.feature_minimum: np.ndarray | None = None
        self.feature_maximum: np.ndarray | None = None
        self.feature_total: np.ndarray | None = None
        self.feature_total_squared: np.ndarray | None = None

    def update(self, value: Any) -> None:
        array = np.asarray(value, dtype=np.float64)
        if array.size == 0:
            raise ValueError("cannot diagnose an empty array")
        if not np.isfinite(array).all():
            raise ValueError("diagnostic array contains non-finite values")
        flattened = array.reshape(-1)
        self.elements += int(flattened.size)
        self.minimum = min(self.minimum, float(flattened.min()))
        self.maximum = max(self.maximum, float(flattened.max()))
        self.total += float(flattened.sum(dtype=np.float64))
        self.total_squared += float(np.square(flattened).sum(dtype=np.float64))
        self.saturated += int(np.count_nonzero(np.abs(flattened) >= self.saturation_limit))

        if self.feature_dim is None:
            return
        if array.shape[-1] != self.feature_dim:
            raise ValueError(
                f"diagnostic last dimension must be {self.feature_dim}, got {array.shape}"
            )
        features = array.reshape(-1, self.feature_dim)
        feature_minimum = features.min(axis=0)
        feature_maximum = features.max(axis=0)
        feature_total = features.sum(axis=0, dtype=np.float64)
        feature_total_squared = np.square(features).sum(axis=0, dtype=np.float64)
        if self.feature_minimum is None:
            self.feature_minimum = feature_minimum
            self.feature_maximum = feature_maximum
            self.feature_total = feature_total
            self.feature_total_squared = feature_total_squared
        else:
            self.feature_minimum = np.minimum(self.feature_minimum, feature_minimum)
            self.feature_maximum = np.maximum(self.feature_maximum, feature_maximum)
            self.feature_total += feature_total
            self.feature_total_squared += feature_total_squared
        self.feature_rows += int(features.shape[0])

    def to_dict(self, *, saturation_limit: float | None = None) -> dict[str, Any]:
        if self.elements <= 0:
            raise ValueError("no diagnostic samples were recorded")
        if saturation_limit is not None and not np.isclose(
            float(saturation_limit), self.saturation_limit
        ):
            raise ValueError(
                "saturation_limit must be fixed before streaming values; "
                f"expected {self.saturation_limit}, got {saturation_limit}"
            )
        mean = self.total / self.elements
        variance = max(self.total_squared / self.elements - mean * mean, 0.0)
        payload: dict[str, Any] = {
            "elements": self.elements,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "saturation_limit": self.saturation_limit,
            "saturation_fraction": self.saturated / self.elements,
        }
        if self.feature_dim is not None:
            assert self.feature_minimum is not None
            assert self.feature_maximum is not None
            assert self.feature_total is not None
            assert self.feature_total_squared is not None
            feature_mean = self.feature_total / self.feature_rows
            feature_variance = np.maximum(
                self.feature_total_squared / self.feature_rows - np.square(feature_mean),
                0.0,
            )
            payload.update(
                {
                    "shape_last_dim": self.feature_dim,
                    "feature_rows": self.feature_rows,
                    "per_dimension_min": self.feature_minimum.tolist(),
                    "per_dimension_max": self.feature_maximum.tolist(),
                    "per_dimension_mean": feature_mean.tolist(),
                    "per_dimension_std": np.sqrt(feature_variance).tolist(),
                }
            )
        return payload


def _as_numpy_observation(value: Any) -> dict[str, np.ndarray]:
    """Normalize the vector-env observation container without changing values."""

    if isinstance(value, list):
        if not value or not isinstance(value[0], Mapping):
            raise TypeError("vector environment returned an invalid observation list")
        value = {
            key: np.stack([np.asarray(item[key]) for item in value], axis=0)
            for key in value[0]
        }
    if not isinstance(value, Mapping):
        raise TypeError(f"vector environment observation must be a mapping, got {type(value)}")
    required = {"rgb", "state"}
    missing = sorted(required - set(value))
    if missing:
        raise KeyError(f"vector environment observation is missing {missing}")
    return {
        "rgb": np.asarray(value["rgb"], dtype=np.float32),
        "state": np.asarray(value["state"], dtype=np.float32),
    }


def _build_dppo_square_env(
    *,
    dppo_root: Path,
    normalization_path: Path,
    num_envs: int,
    max_episode_steps: int,
    action_horizon: int,
):
    """Construct the exact official DPPO Square image-wrapper stack.

    This function runs only in the DPPO worker process.  Keeping it out of the
    Light-WAM interpreter avoids mixing the two pinned Conda environments.
    """

    dppo_source = str(dppo_root)
    if dppo_source in sys.path:
        sys.path.remove(dppo_source)
    sys.path.insert(0, dppo_source)

    from env.gym_utils import make_async
    from omegaconf import OmegaConf

    shape_meta = {
        "obs": {
            "rgb": {"shape": [3, 96, 96]},
            "state": {"shape": [9]},
        },
        "action": {"shape": [DEFAULT_ACTION_DIM]},
    }
    wrappers = OmegaConf.create(
        {
            "robomimic_image": {
                "normalization_path": str(normalization_path),
                "low_dim_keys": [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                ],
                "image_keys": ["agentview_image"],
                "shape_meta": shape_meta,
            },
            "multi_step": {
                "n_obs_steps": 1,
                "n_action_steps": int(action_horizon),
                "max_episode_steps": int(max_episode_steps),
                "reset_within_step": True,
            },
        }
    )
    return make_async(
        id="square",
        num_envs=int(num_envs),
        asynchronous=True,
        wrappers=wrappers,
        robomimic_env_cfg_path=str(
            dppo_root / "cfg/robomimic/env_meta/square-img.json"
        ),
        use_image_obs=True,
        render_offscreen=True,
        reward_shaping=False,
        shape_meta=shape_meta,
        max_episode_steps=int(max_episode_steps),
        obs_steps=1,
        act_steps=int(action_horizon),
        action_dim=DEFAULT_ACTION_DIM,
    )


def run_dppo_square_env_worker(arguments: argparse.Namespace) -> int:
    """Serve the robomimic environment to the Light-WAM eval process."""

    from multiprocessing.connection import Listener

    socket_path = Path(arguments.control_socket)
    dppo_root = Path(arguments.dppo_root).expanduser().resolve()
    normalization_path = Path(arguments.normalization_path).expanduser().resolve()
    if not (dppo_root / "cfg/robomimic/env_meta/square-img.json").is_file():
        raise FileNotFoundError(f"DPPO Square env metadata is missing under {dppo_root}")
    if not normalization_path.is_file():
        raise FileNotFoundError(f"Square normalization file is missing: {normalization_path}")

    socket_path.unlink(missing_ok=True)
    listener = Listener(str(socket_path), family="AF_UNIX")
    connection = None
    vector_env = None
    try:
        connection = listener.accept()
        try:
            vector_env = _build_dppo_square_env(
                dppo_root=dppo_root,
                normalization_path=normalization_path,
                num_envs=int(arguments.num_envs),
                max_episode_steps=int(arguments.max_episode_steps),
                action_horizon=int(arguments.action_horizon),
            )
            environment_seeds = [
                int(arguments.worker_seed) + index
                for index in range(int(arguments.num_envs))
            ]
            vector_env.seed(environment_seeds)
            connection.send(
                {
                    "ok": True,
                    "event": "ready",
                    "environment_seeds": environment_seeds,
                }
            )
        except BaseException as exc:
            connection.send(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            raise

        while True:
            request = connection.recv()
            command = request.get("command")
            try:
                if command == "reset":
                    options = [{} for _ in range(int(arguments.num_envs))]
                    observation = vector_env.reset_arg(options_list=options)
                    connection.send(
                        {"ok": True, "observation": _as_numpy_observation(observation)}
                    )
                elif command == "step":
                    action = np.asarray(request["action"], dtype=np.float32)
                    expected = (
                        int(arguments.num_envs),
                        int(arguments.action_horizon),
                        DEFAULT_ACTION_DIM,
                    )
                    if tuple(action.shape) != expected:
                        raise ValueError(
                            f"action shape must be {expected}, got {tuple(action.shape)}"
                        )
                    observation, reward, terminated, truncated, _info = vector_env.step(action)
                    connection.send(
                        {
                            "ok": True,
                            "observation": _as_numpy_observation(observation),
                            "reward": np.asarray(reward, dtype=np.float64),
                            "terminated": np.asarray(terminated, dtype=bool),
                            "truncated": np.asarray(truncated, dtype=bool),
                        }
                    )
                elif command == "close":
                    connection.send({"ok": True, "event": "closed"})
                    break
                else:
                    raise ValueError(f"unsupported environment-worker command: {command!r}")
            except BaseException as exc:
                connection.send(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                raise
    except EOFError:
        print("LIGHTWAM_SQUARE_ENV_WORKER client disconnected", flush=True)
    finally:
        if vector_env is not None:
            vector_env.close()
        if connection is not None:
            connection.close()
        listener.close()
        socket_path.unlink(missing_ok=True)
    return 0


def _resolve_dppo_runtime(
    *, dppo_python_value: str | None, dppo_root_value: str | None
) -> tuple[Path, Path]:
    python_candidates = [
        dppo_python_value,
        os.environ.get("DPPO_PYTHON"),
        "/opt/conda/envs/dppo/bin/python",
    ]
    dppo_python = next(
        (
            Path(value).expanduser().resolve()
            for value in python_candidates
            if value not in (None, "") and Path(value).expanduser().is_file()
        ),
        None,
    )
    if dppo_python is None:
        if importlib.util.find_spec("robomimic") is not None:
            dppo_python = Path(sys.executable).resolve()
        else:
            raise FileNotFoundError(
                "DPPO Python is unavailable; expected the `dppo` Conda environment "
                "or an explicit --dppo-python"
            )

    root_candidates = [
        dppo_root_value,
        os.environ.get("DPPO_ROOT"),
        "/opt/robot-practice/dppo",
        "third_party/dppo/dppo",
    ]
    dppo_root = next(
        (
            Path(value).expanduser().resolve()
            for value in root_candidates
            if value not in (None, "")
            and (Path(value).expanduser() / "cfg/robomimic/env_meta/square-img.json").is_file()
        ),
        None,
    )
    if dppo_root is None:
        raise FileNotFoundError(
            "DPPO source is unavailable; expected $DPPO_ROOT, /opt/robot-practice/dppo, "
            "or third_party/dppo/dppo"
        )
    return dppo_python, dppo_root


class _DPPOSquareEnvClient:
    """Lifecycle wrapper for the isolated DPPO environment process."""

    def __init__(
        self,
        *,
        dppo_python: Path,
        dppo_root: Path,
        normalization_path: Path,
        num_envs: int,
        max_episode_steps: int,
        action_horizon: int,
        policy_seed: int,
        worker_log: Path,
    ) -> None:
        self.dppo_python = dppo_python
        self.dppo_root = dppo_root
        self.normalization_path = normalization_path
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.action_horizon = int(action_horizon)
        self.policy_seed = int(policy_seed)
        self.worker_log = worker_log
        self.socket_path = Path(tempfile.gettempdir()) / (
            f"lightwam-square-{os.getpid()}-{uuid.uuid4().hex[:10]}.sock"
        )
        self.process: subprocess.Popen[str] | None = None
        self.connection = None
        self.log_handle = None

    def __enter__(self):
        from multiprocessing.connection import Client

        try:
            self.worker_log.parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = self.worker_log.open("a", encoding="utf-8")
            source_root = Path(__file__).resolve().parents[4]
            environment = os.environ.copy()
            source_path = str(source_root / "src")
            environment["PYTHONPATH"] = source_path + (
                f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""
            )
            environment["PYTHONNOUSERSITE"] = "1"
            command = [
                str(self.dppo_python),
                "-m",
                "robot_practice.experiments.rl.lightwam_base",
                "--phase=env-worker",
                f"--control-socket={self.socket_path}",
                f"--dppo-root={self.dppo_root}",
                f"--normalization-path={self.normalization_path}",
                f"--num-envs={self.num_envs}",
                f"--max-episode-steps={self.max_episode_steps}",
                f"--action-horizon={self.action_horizon}",
                f"--worker-seed={self.policy_seed}",
            ]
            self.process = subprocess.Popen(
                command,
                cwd=source_root,
                env=environment,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + 180.0
            while not self.socket_path.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"DPPO environment worker exited with status {self.process.returncode}; "
                        f"see {self.worker_log}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"DPPO environment worker did not open its socket; see {self.worker_log}"
                    )
                time.sleep(0.1)
            self.connection = Client(str(self.socket_path), family="AF_UNIX")
            reply = self.connection.recv()
            self._check(reply)
            self.environment_seeds = [int(value) for value in reply["environment_seeds"]]
            return self
        except BaseException:
            self._cleanup(terminate=True)
            raise

    @staticmethod
    def _check(reply: Mapping[str, Any]) -> None:
        if not bool(reply.get("ok")):
            raise RuntimeError(
                f"DPPO environment worker failed: {reply.get('error')}\n"
                f"{reply.get('traceback', '')}"
            )

    def reset(self) -> dict[str, np.ndarray]:
        self.connection.send({"command": "reset"})
        reply = self.connection.recv()
        self._check(reply)
        return reply["observation"]

    def step(self, action: np.ndarray):
        self.connection.send({"command": "step", "action": action})
        reply = self.connection.recv()
        self._check(reply)
        return (
            reply["observation"],
            reply["reward"],
            reply["terminated"],
            reply["truncated"],
        )

    def _cleanup(self, *, terminate: bool) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            if terminate and self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=10 if terminate else 30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        self.socket_path.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            if (
                exc_type is None
                and self.connection is not None
                and self.process is not None
                and self.process.poll() is None
            ):
                self.connection.send({"command": "close"})
                reply = self.connection.recv()
                self._check(reply)
        finally:
            self._cleanup(terminate=exc_type is not None)


def _summarize_official_square_rollout(
    reward_trajs: np.ndarray,
    firsts_trajs: np.ndarray,
    *,
    act_steps: int,
    success_threshold: float,
) -> list[dict[str, Any]]:
    """Apply DPPO's official episode splitting and Square success criterion."""

    rewards = np.asarray(reward_trajs, dtype=np.float64)
    firsts = np.asarray(firsts_trajs, dtype=bool)
    if rewards.ndim != 2:
        raise ValueError(f"reward_trajs must be [steps,envs], got {rewards.shape}")
    if firsts.shape != (rewards.shape[0] + 1, rewards.shape[1]):
        raise ValueError(
            "firsts_trajs must be [steps+1,envs], got "
            f"{firsts.shape} for rewards {rewards.shape}"
        )
    rows: list[dict[str, Any]] = []
    for env_index in range(rewards.shape[1]):
        boundaries = np.flatnonzero(firsts[:, env_index])
        for boundary_index in range(len(boundaries) - 1):
            start = int(boundaries[boundary_index])
            end = int(boundaries[boundary_index + 1])
            if end - start <= 1:
                continue
            trajectory = rewards[start:end, env_index]
            episode_reward = float(np.sum(trajectory))
            best_reward = float(np.max(trajectory) / int(act_steps))
            rows.append(
                {
                    "env_index": env_index,
                    "start_policy_step": start,
                    "end_policy_step": end - 1,
                    "policy_steps": end - start,
                    "episode_reward": episode_reward,
                    "best_reward": best_reward,
                    "success": bool(best_reward >= float(success_threshold)),
                }
            )
    return rows


def run_dummy(output_dir: Path, *, device: str = "cpu") -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    core = DummyLightWAMCore(device=device)
    adapter = LightWAMBaseAdapter(
        core,
        action_horizon=DEFAULT_ACTION_HORIZON,
        image_size=(32, 32),
        prompt="dummy task",
        seed=0,
    )
    cond = {
        "rgb": torch.zeros(2, 1, 3, 24, 28),
        "state": torch.zeros(2, 1, 9),
    }
    action = adapter(cond)
    payload = {
        "schema_version": 1,
        "route": "lightwam_base_adapter",
        "dummy": True,
        "input": {"rgb": [2, 1, 3, 24, 28], "state": [2, 1, 9]},
        "normalized_image_range": [-1.0, 1.0],
        "output_action_shape": list(action.shape),
        "action_range": [float(action.min()), float(action.max())],
        "official_dppo_connected": False,
        "residual_rl_connected": False,
    }
    _json_dump(output_dir / "dummy.json", payload)
    print("LIGHTWAM_BASE_DUMMY=PASS", flush=True)
    return 0


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "no":
        return torch.float32
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(
        f"Unsupported mixed precision `{mixed_precision}`; expected no, fp16, or bf16"
    )


def run_model_smoke(
    output_dir: Path,
    *,
    vendor_root: Path,
    model_config: Path,
    local_model_root: Path,
    checkpoint: str | Path | None,
    device: str,
    model_dtype: torch.dtype,
    image_size: Sequence[int],
    action_horizon: int,
    num_inference_steps: int,
    allow_download: bool,
) -> int:
    """Instantiate the official model and execute one action-only forward pass.

    This is intentionally opt-in because the Wan backbone is large.  It uses a
    synthetic context and image, so it does not require the dataset or text
    encoder, and the offline environment gate prevents accidental downloads.
    """

    context = torch.zeros(DEFAULT_CONTEXT_LEN, DEFAULT_CONTEXT_DIM, dtype=torch.float32)
    context_mask = torch.ones(DEFAULT_CONTEXT_LEN, dtype=torch.bool)
    model = load_official_lightwam(
        vendor_root=vendor_root,
        model_config=model_config,
        local_model_root=local_model_root,
        device=device,
        model_dtype=model_dtype,
        checkpoint=checkpoint,
        allow_download=allow_download,
    ).eval()
    adapter = LightWAMBaseAdapter(
        model,
        action_horizon=action_horizon,
        action_dim=DEFAULT_ACTION_DIM,
        image_size=image_size,
        context=context,
        context_mask=context_mask,
        num_inference_steps=num_inference_steps,
        seed=0,
    )
    condition = {
        "rgb": torch.zeros(1, 3, *tuple(_as_pair(image_size, name="image_size"))),
        "state": torch.zeros(1, 9),
    }
    action = adapter(condition)
    uses_state_fusion = bool(
        getattr(model, "uses_state_fusion_action_expert", lambda: False)()
    )
    payload = {
        "schema_version": 1,
        "route": "lightwam_base_model_smoke",
        "model_class": model.__class__.__name__,
        "uses_state_fusion_action_expert": uses_state_fusion,
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "input_image_shape": list(condition["rgb"].shape),
        "state_shape": list(condition["state"].shape),
        "output_action_shape": list(action.shape),
        "output_action_finite": bool(torch.isfinite(action).all()),
        "data_or_model_downloaded": False,
    }
    _json_dump(output_dir / "model_smoke.json", payload)
    print("LIGHTWAM_BASE_MODEL_SMOKE=PASS", flush=True)
    return 0


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _completed_eval_seed(seed_dir: Path, *, expected_episodes: int) -> dict[str, Any] | None:
    metrics_path = seed_dir / "metrics.json"
    episodes_path = seed_dir / "episodes.jsonl"
    if not metrics_path.is_file() or not episodes_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if int(metrics.get("episodes", -1)) != int(expected_episodes):
        return None
    if len(episode_rows) != int(expected_episodes):
        return None
    return metrics


def _checkpoint_step(path: Path) -> int | None:
    name = path.stem
    marker = "step_"
    if marker not in name:
        return None
    value = name.rsplit(marker, maxsplit=1)[-1]
    return int(value) if value.isdigit() else None


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _checkpoint_training_run_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Read the saved run budget next to ``checkpoints/weights/*.pt``."""

    config_path = checkpoint_path.parent.parent.parent / "config.yaml"
    result: dict[str, Any] = {
        "config": _display_path(config_path),
        "config_present": config_path.is_file(),
        "configured_max_steps": None,
        "configured_num_epochs": None,
    }
    if not config_path.is_file():
        return result
    from omegaconf import OmegaConf

    config = OmegaConf.load(str(config_path))
    max_steps = config.get("max_steps", None)
    num_epochs = config.get("num_epochs", None)
    result["configured_max_steps"] = None if max_steps is None else int(max_steps)
    result["configured_num_epochs"] = None if num_epochs is None else int(num_epochs)
    return result


def _eval_seed_output_dir(
    output_dir: Path, *, seed: int, all_seeds: Sequence[int]
) -> Path:
    """Avoid a redundant ``seed_N/seed_N`` layer in multi-GPU workers."""

    if len(all_seeds) == 1 and output_dir.name == f"seed_{int(seed)}":
        return output_dir
    return output_dir / f"seed_{int(seed)}"


def _merge_lightwam_square_eval(
    *,
    output_dir: Path,
    seeds: Sequence[int],
    episodes_per_seed: int,
    checkpoint_path: Path,
    max_episode_steps: int,
    action_horizon: int,
    inference_steps: int,
    training_target_steps: int | None,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = _eval_seed_output_dir(output_dir, seed=int(seed), all_seeds=seeds)
        metrics = _completed_eval_seed(seed_dir, expected_episodes=episodes_per_seed)
        if metrics is None:
            raise FileNotFoundError(f"evaluation seed is incomplete: {seed_dir}")
        if "policy_steps_total" not in metrics:
            nested = metrics.get("per_seed", [])
            if len(nested) != 1 or int(nested[0].get("seed", -1)) != int(seed):
                raise ValueError(f"invalid one-seed evaluation summary: {seed_dir / 'metrics.json'}")
            metrics = dict(nested[0])
        per_seed.append(metrics)
        all_episode_rows.extend(
            json.loads(line)
            for line in (seed_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    total_episodes = sum(int(row["episodes"]) for row in per_seed)
    total_successes = sum(int(row["successes"]) for row in per_seed)
    total_policy_steps = sum(int(row["policy_steps_total"]) for row in per_seed)
    total_action_chunks = sum(int(row["action_chunks_total"]) for row in per_seed)
    total_inference_seconds = sum(float(row["inference_seconds"]) for row in per_seed)
    total_elapsed_seconds = sum(float(row["elapsed_seconds"]) for row in per_seed)
    uses_state_fusion = all(
        bool(row.get("uses_state_fusion_action_expert", False)) for row in per_seed
    )
    checkpoint_step = _checkpoint_step(checkpoint_path)
    checkpoint_training_run = _checkpoint_training_run_metadata(checkpoint_path)
    configured_run_steps = checkpoint_training_run["configured_max_steps"]
    formal_protocol = bool(
        int(episodes_per_seed) == 50
        and int(max_episode_steps) == 400
        and int(action_horizon) == 4
        and [int(seed) for seed in seeds] == [1, 2, 3]
    )
    short_horizon = int(max_episode_steps) < 400
    if formal_protocol:
        evaluation_mode = "formal_evaluation"
    elif short_horizon:
        evaluation_mode = "pipeline_smoke"
    else:
        evaluation_mode = "nonformal_evaluation"
    success_rate_interpretable = not short_horizon
    success_rate_note = (
        "Pipeline-only smoke: the episode horizon is shorter than the 400-step Square "
        "protocol, so success/failure is not a model-quality result."
        if short_horizon
        else "Closed-loop Square outcome; statistical confidence depends on episode count."
    )
    training_budget_complete = bool(
        checkpoint_step is not None
        and training_target_steps is not None
        and checkpoint_step >= int(training_target_steps)
    )
    checkpoint_reached_run_budget = bool(
        checkpoint_step is not None
        and configured_run_steps is not None
        and checkpoint_step >= int(configured_run_steps)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "result_label": "Light-WAM robomimic Square closed-loop evaluation",
        "phase": "lightwam-base-eval",
        "evaluation_mode": evaluation_mode,
        "task_suite": "robomimic",
        "task": "square",
        "requested_env_id": "NutAssemblySquare",
        "dataset_size": 26814,
        "training_dataset_windows": 26814,
        "evaluation_dataset_size": None,
        "observation_keys": ["rgb", "state"],
        "observation_shapes": {"rgb": [1, 3, 96, 96], "state": [1, 9]},
        "action_space": "normalized Box[-1,1]^7",
        "action_dim": DEFAULT_ACTION_DIM,
        "action_horizon": int(action_horizon),
        "control_frequency_hz": 20,
        "max_episode_steps": int(max_episode_steps),
        "eval_episodes_per_seed": int(episodes_per_seed),
        "seeds": [int(seed) for seed in seeds],
        "episodes": total_episodes,
        "successes": total_successes,
        "failures": total_episodes - total_successes,
        "success_rate": total_successes / total_episodes,
        "closed_loop_success_rate": total_successes / total_episodes,
        "success_rate_interpretable": success_rate_interpretable,
        "success_rate_note": success_rate_note,
        "avg_episode_reward": float(
            np.mean([float(row["episode_reward"]) for row in all_episode_rows])
        ),
        "avg_best_reward": float(
            np.mean([float(row["best_reward"]) for row in all_episode_rows])
        ),
        "train_loss": None,
        "open_loop_action_error": None,
        "latency_ms_per_vector_step": 1000.0 * total_inference_seconds / total_policy_steps,
        "latency_ms_per_action_chunk": 1000.0 * total_inference_seconds / total_action_chunks,
        "inference_seconds": total_inference_seconds,
        "elapsed_seconds": total_elapsed_seconds,
        "failure_cases": total_episodes - total_successes,
        "checkpoint": _display_path(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_training_run": checkpoint_training_run,
        "checkpoint_reached_training_run_budget": checkpoint_reached_run_budget,
        "training_target_steps": training_target_steps,
        "formal_training_target_steps": training_target_steps,
        "training_budget_complete": training_budget_complete,
        "training_budget_note": (
            f"The checkpoint reached its {configured_run_steps:,}-step run budget but not "
            f"the {int(training_target_steps):,}-step formal training target."
            if checkpoint_reached_run_budget
            and training_target_steps is not None
            and not training_budget_complete
            else None
        ),
        "formal_eval_protocol": formal_protocol,
        "formal_result": bool(formal_protocol and training_budget_complete),
        "official_protocol": {
            "eval_config": "cfg/robomimic/eval/square/eval_diffusion_mlp_img.yaml",
            "environment": "DPPO make_async + RobomimicImageWrapper + MultiStep",
            "reward_shaping": False,
            "success_rule": "max(chunk_reward) / act_steps >= 1",
        },
        "lightwam_inference_steps": int(inference_steps),
        "uses_state_fusion_action_expert": uses_state_fusion,
        "inference_steps_effective": not uses_state_fusion,
        "data_or_model_downloaded": False,
        "per_seed": per_seed,
        "action_diagnostics_by_seed": {
            str(row["seed"]): row["action_diagnostics"]
            for row in per_seed
            if "action_diagnostics" in row
        },
        "observation_diagnostics_by_seed": {
            str(row["seed"]): row["observation_diagnostics"]
            for row in per_seed
            if "observation_diagnostics" in row
        },
    }
    _json_dump(output_dir / "metrics.json", payload)
    _write_jsonl(output_dir / "episodes.jsonl", all_episode_rows)
    return payload


def run_lightwam_square_eval(
    output_dir: Path,
    *,
    vendor_root: Path,
    model_config: Path,
    local_model_root: Path,
    checkpoint: str | Path | None,
    context_path: str | Path | None,
    device: str,
    model_dtype: torch.dtype,
    image_size: Sequence[int],
    action_horizon: int,
    num_inference_steps: int,
    allow_download: bool,
    seeds: Sequence[int],
    episodes_per_seed: int,
    max_episode_steps: int,
    success_threshold: float,
    normalization_path: Path,
    dppo_python_value: str | None,
    dppo_root_value: str | None,
    log_every: int,
    resume: bool,
    training_target_steps: int | None,
) -> int:
    """Evaluate a Light-WAM checkpoint with DPPO's official Square contract."""

    if checkpoint in (None, ""):
        raise ValueError("Light-WAM Square eval requires --checkpoint")
    checkpoint_path = _validate_lightwam_checkpoint(checkpoint)
    if context_path in (None, ""):
        raise ValueError("Light-WAM Square eval requires the cached Square text context")
    context, context_mask = load_context_cache(context_path)
    if int(episodes_per_seed) <= 0:
        raise ValueError("episodes_per_seed must be positive")
    if int(max_episode_steps) <= 0 or int(max_episode_steps) % int(action_horizon):
        raise ValueError("max_episode_steps must be positive and divisible by action_horizon")
    if int(log_every) <= 0:
        raise ValueError("eval log_every must be positive")
    if not normalization_path.is_file():
        raise FileNotFoundError(f"Square normalization file is missing: {normalization_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    complete_by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        seed_dir = _eval_seed_output_dir(output_dir, seed=int(seed), all_seeds=seeds)
        completed = _completed_eval_seed(seed_dir, expected_episodes=episodes_per_seed)
        if completed is not None:
            if not resume:
                raise FileExistsError(
                    f"completed eval already exists at {seed_dir}; pass --resume to reuse it"
                )
            complete_by_seed[int(seed)] = completed
            _eval_log(
                run_log,
                f"STATUS phase=eval seed={seed} state=skip completed={episodes_per_seed} "
                f"total={episodes_per_seed}",
            )

    pending_seeds = [int(seed) for seed in seeds if int(seed) not in complete_by_seed]
    if pending_seeds:
        dppo_python, dppo_root = _resolve_dppo_runtime(
            dppo_python_value=dppo_python_value,
            dppo_root_value=dppo_root_value,
        )
        _eval_log(
            run_log,
            "START phase=eval "
            f"seeds={','.join(str(seed) for seed in seeds)} "
            f"episodes_per_seed={episodes_per_seed} max_episode_steps={max_episode_steps} "
            f"checkpoint={_display_path(checkpoint_path)}",
        )
        model = load_official_lightwam(
            vendor_root=vendor_root,
            model_config=model_config,
            local_model_root=local_model_root,
            device=device,
            model_dtype=model_dtype,
            checkpoint=checkpoint_path,
            allow_download=allow_download,
        ).eval()
        adapter = LightWAMBaseAdapter(
            model,
            action_horizon=action_horizon,
            action_dim=DEFAULT_ACTION_DIM,
            image_size=image_size,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            seed=None,
        ).eval()
        uses_state_fusion = bool(
            getattr(model, "uses_state_fusion_action_expert", lambda: False)()
        )

        eval_steps = int(max_episode_steps) // int(action_horizon)
        for seed_position, policy_seed in enumerate(pending_seeds, start=1):
            random.seed(policy_seed)
            np.random.seed(policy_seed)
            torch.manual_seed(policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(policy_seed)
            seed_dir = _eval_seed_output_dir(
                output_dir,
                seed=policy_seed,
                all_seeds=seeds,
            )
            seed_dir.mkdir(parents=True, exist_ok=True)
            seed_start = time.perf_counter()
            inference_seconds = 0.0
            reward_trajs = np.zeros((eval_steps, episodes_per_seed), dtype=np.float64)
            firsts_trajs = np.zeros((eval_steps + 1, episodes_per_seed), dtype=bool)
            firsts_trajs[0] = True
            action_diagnostics = _StreamingArrayDiagnostics(feature_dim=DEFAULT_ACTION_DIM)
            rgb_diagnostics = _StreamingArrayDiagnostics(saturation_limit=255.0)
            state_diagnostics = _StreamingArrayDiagnostics(feature_dim=9)
            worker_log = seed_dir / "env_worker.log"
            _eval_log(
                run_log,
                f"START phase=eval seed={policy_seed} gpu_device={device} "
                f"envs={episodes_per_seed} policy_steps={eval_steps}",
            )
            with _DPPOSquareEnvClient(
                dppo_python=dppo_python,
                dppo_root=dppo_root,
                normalization_path=normalization_path,
                num_envs=episodes_per_seed,
                max_episode_steps=max_episode_steps,
                action_horizon=action_horizon,
                policy_seed=policy_seed,
                worker_log=worker_log,
            ) as environment:
                observation = environment.reset()
                rgb_diagnostics.update(observation["rgb"])
                state_diagnostics.update(observation["state"])
                for policy_step in range(eval_steps):
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    infer_start = time.perf_counter()
                    action = adapter.predict_action(
                        observation,
                        deterministic=True,
                        seed=environment.environment_seeds,
                    )
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    inference_seconds += time.perf_counter() - infer_start
                    action_numpy = action.detach().cpu().numpy().astype(np.float32, copy=False)
                    action_diagnostics.update(action_numpy)
                    observation, reward, terminated, truncated = environment.step(action_numpy)
                    rgb_diagnostics.update(observation["rgb"])
                    state_diagnostics.update(observation["state"])
                    reward_trajs[policy_step] = reward
                    firsts_trajs[policy_step + 1] = np.logical_or(terminated, truncated)
                    completed_steps = policy_step + 1
                    if completed_steps % log_every == 0 or completed_steps == eval_steps:
                        elapsed = time.perf_counter() - seed_start
                        eta = elapsed / completed_steps * (eval_steps - completed_steps)
                        completed_episodes = int(
                            np.count_nonzero(firsts_trajs[1 : policy_step + 2])
                        )
                        _eval_log(
                            run_log,
                            f"STATUS phase=eval seed={policy_seed} "
                            f"completed_steps={completed_steps} total_steps={eval_steps} "
                            f"completed_episodes={completed_episodes} "
                            f"total_episodes={episodes_per_seed} eta_seconds={eta:.1f}",
                        )

            episode_rows = _summarize_official_square_rollout(
                reward_trajs,
                firsts_trajs,
                act_steps=action_horizon,
                success_threshold=success_threshold,
            )
            if len(episode_rows) != int(episodes_per_seed):
                raise RuntimeError(
                    f"DPPO protocol completed {len(episode_rows)} episodes; "
                    f"expected {episodes_per_seed}. See {worker_log}"
                )
            for row in episode_rows:
                env_index = int(row["env_index"])
                row.update(
                    {
                        "policy_seed": policy_seed,
                        "environment_seed": policy_seed + env_index,
                    }
                )
            elapsed_seconds = time.perf_counter() - seed_start
            successes = sum(bool(row["success"]) for row in episode_rows)
            action_summary = action_diagnostics.to_dict()
            seed_metrics = {
                "schema_version": 1,
                "seed": policy_seed,
                "environment_seeds": [
                    policy_seed + index for index in range(episodes_per_seed)
                ],
                "episodes": int(episodes_per_seed),
                "successes": int(successes),
                "failures": int(episodes_per_seed) - int(successes),
                "success_rate": float(successes / episodes_per_seed),
                "avg_episode_reward": float(
                    np.mean([float(row["episode_reward"]) for row in episode_rows])
                ),
                "avg_best_reward": float(
                    np.mean([float(row["best_reward"]) for row in episode_rows])
                ),
                "policy_steps_total": eval_steps,
                "action_chunks_total": eval_steps * int(episodes_per_seed),
                "inference_seconds": inference_seconds,
                "latency_ms_per_vector_step": 1000.0 * inference_seconds / eval_steps,
                "latency_ms_per_action_chunk": (
                    1000.0 * inference_seconds / (eval_steps * int(episodes_per_seed))
                ),
                "elapsed_seconds": elapsed_seconds,
                "uses_state_fusion_action_expert": uses_state_fusion,
                "action_diagnostics": action_summary,
                "observation_diagnostics": {
                    "rgb": rgb_diagnostics.to_dict(),
                    "state": state_diagnostics.to_dict(),
                },
            }
            _write_jsonl(seed_dir / "episodes.jsonl", episode_rows)
            _json_dump(seed_dir / "metrics.json", seed_metrics)
            _eval_log(
                run_log,
                f"DONE phase=eval seed={policy_seed} completed={episodes_per_seed} "
                f"total={episodes_per_seed} success_rate={seed_metrics['success_rate']:.4f} "
                f"action_min={action_summary['min']:.4f} "
                f"action_max={action_summary['max']:.4f} "
                f"action_saturation={action_summary['saturation_fraction']:.4f} "
                f"seed_progress={seed_position}/{len(pending_seeds)}",
            )

        del adapter
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = _merge_lightwam_square_eval(
        output_dir=output_dir,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        checkpoint_path=checkpoint_path,
        max_episode_steps=max_episode_steps,
        action_horizon=action_horizon,
        inference_steps=num_inference_steps,
        training_target_steps=training_target_steps,
    )
    _eval_log(
        run_log,
        f"DONE phase=eval completed={metrics['episodes']} total={metrics['episodes']} "
        f"success_rate={metrics['success_rate']:.4f} metrics={output_dir / 'metrics.json'}",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "check",
            "smoke",
            "train",
            "eval",
            "eval-merge",
            "preflight",
            "dummy",
            "model-smoke",
            "prepare-context",
            "env-worker",
        ),
        required=True,
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/rl/lightwam_base_robomimic_square.yaml",
    )
    parser.add_argument("--output-root")
    parser.add_argument("--vendor-root")
    parser.add_argument("--model-config")
    parser.add_argument("--local-model-root")
    parser.add_argument("--data-path")
    parser.add_argument("--context-path")
    parser.add_argument("--context-output")
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt")
    parser.add_argument("--device")
    parser.add_argument("--image-size", nargs=2, type=int)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--video-frames", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-scheduler-type")
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--benchmark-warmup-steps", type=int)
    parser.add_argument("--benchmark-measure-steps", type=int)
    parser.add_argument("--eval-seeds")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--eval-log-every", type=int)
    parser.add_argument("--normalization-path")
    parser.add_argument("--dppo-root")
    parser.add_argument("--dppo-python")
    parser.add_argument("--resume", action="store_true")
    # Internal arguments used only by the isolated DPPO environment worker.
    parser.add_argument("--control-socket")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> int:
    if arguments.phase == "env-worker":
        required = {
            "control_socket": arguments.control_socket,
            "dppo_root": arguments.dppo_root,
            "normalization_path": arguments.normalization_path,
            "num_envs": arguments.num_envs,
            "max_episode_steps": arguments.max_episode_steps,
            "action_horizon": arguments.action_horizon,
            "worker_seed": arguments.worker_seed,
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if missing:
            raise ValueError(f"env-worker is missing required arguments: {missing}")
        return run_dppo_square_env_worker(arguments)

    if arguments.phase == "dummy":
        device = arguments.device or "cpu"
        output_dir = _path_or_default(
            arguments.output_root,
            "outputs/lightwam_base_robomimic_square",
        )
        return run_dummy(output_dir, device=device)

    adapter_config_path = Path(arguments.config).expanduser().resolve()
    adapter_config = _load_adapter_config(adapter_config_path)
    lightwam_config = _nested_config_value(adapter_config, "lightwam", default={})
    training_config = _nested_config_value(adapter_config, "training", default={})
    evaluation_config = _nested_config_value(adapter_config, "evaluation", default={})
    dppo_reference = _nested_config_value(adapter_config, "dppo_reference", default={})
    training_alignment = _nested_config_value(
        adapter_config,
        "lightwam_alignment",
        default={},
    )
    smoke_benchmark_config = _nested_config_value(
        training_config,
        "smoke_benchmark",
        default={},
    )

    default_output_by_phase = {
        "check": "outputs/lightwam_base_robomimic_square/check",
        "smoke": "outputs/lightwam_base_robomimic_square/smoke",
        "model-smoke": "outputs/lightwam_base_robomimic_square/smoke",
        "preflight": "outputs/lightwam_base_robomimic_square/preflight",
        "prepare-context": "outputs/lightwam_base_robomimic_square/context_prepare",
        "train": "outputs/lightwam_base_robomimic_square/train",
        "eval": "outputs/lightwam_base_robomimic_square/eval",
        "eval-merge": "outputs/lightwam_base_robomimic_square/eval",
    }
    output_dir = _path_or_default(
        arguments.output_root,
        default_output_by_phase.get(
            arguments.phase,
            "outputs/lightwam_base_robomimic_square",
        ),
    )
    vendor_root = _path_or_default(
        arguments.vendor_root,
        _nested_config_value(
            lightwam_config,
            "vendor_root",
            default="third_party/lightwam/Light-WAM",
        ),
    )
    model_config = _path_or_default(
        arguments.model_config,
        _nested_config_value(
            lightwam_config,
            "model_config",
            default="configs/experiments/rl/lightwam_base_robomimic_square_model.yaml",
        ),
    )
    local_model_root = _path_or_default(
        arguments.local_model_root,
        _nested_config_value(
            lightwam_config,
            "local_model_root",
            default="pretrained/w0_fastwam/models",
        ),
    )
    data_path = _path_or_default(
        arguments.data_path,
        _nested_config_value(
            adapter_config,
            "data",
            "npz",
            default="datasets/raw/dppo/robomimic/square-img/train.npz",
        ),
    )
    context_path_value = (
        arguments.context_path
        if arguments.context_path not in (None, "")
        else _nested_config_value(lightwam_config, "context", default=None)
    )
    checkpoint_value = (
        arguments.checkpoint
        if arguments.checkpoint not in (None, "")
        else _nested_config_value(lightwam_config, "checkpoint", default=None)
    )
    prompt = str(
        arguments.prompt
        if arguments.prompt not in (None, "")
        else _nested_config_value(
            lightwam_config,
            "prompt",
            default="insert the square nut into the square peg",
        )
    )
    image_size = tuple(
        arguments.image_size
        if arguments.image_size is not None
        else _nested_config_value(lightwam_config, "image_size", default=list(DEFAULT_IMAGE_SIZE))
    )
    action_horizon = int(
        arguments.action_horizon
        if arguments.action_horizon is not None
        else _nested_config_value(
            lightwam_config,
            "action_horizon",
            default=DEFAULT_ACTION_HORIZON,
        )
    )
    video_frames = int(
        arguments.video_frames
        if arguments.video_frames is not None
        else _nested_config_value(lightwam_config, "video_frames", default=DEFAULT_VIDEO_FRAMES)
    )
    batch_size = int(
        arguments.batch_size
        if arguments.batch_size is not None
        else _nested_config_value(training_config, "batch_size", default=1)
    )
    num_workers = int(
        arguments.num_workers
        if arguments.num_workers is not None
        else _nested_config_value(training_config, "num_workers", default=0)
    )
    learning_rate = float(
        arguments.learning_rate
        if arguments.learning_rate is not None
        else _nested_config_value(training_config, "learning_rate", default=1.0e-4)
    )
    lr_scheduler_type = str(
        arguments.lr_scheduler_type
        if arguments.lr_scheduler_type is not None
        else _nested_config_value(training_config, "lr_scheduler_type", default="cosine")
    )
    warmup_ratio = float(
        arguments.warmup_ratio
        if arguments.warmup_ratio is not None
        else _nested_config_value(training_config, "warmup_ratio", default=0.05)
    )
    warmup_steps_value = (
        arguments.warmup_steps
        if arguments.warmup_steps is not None
        else _nested_config_value(training_config, "warmup_steps", default=None)
    )
    warmup_steps = None if warmup_steps_value is None else int(warmup_steps_value)
    weight_decay = float(
        arguments.weight_decay
        if arguments.weight_decay is not None
        else _nested_config_value(training_config, "weight_decay", default=0.0)
    )
    gradient_accumulation_steps = int(
        arguments.gradient_accumulation_steps
        if arguments.gradient_accumulation_steps is not None
        else _nested_config_value(
            training_config,
            "gradient_accumulation_steps",
            default=1,
        )
    )
    max_grad_norm = float(
        arguments.max_grad_norm
        if arguments.max_grad_norm is not None
        else _nested_config_value(training_config, "max_grad_norm", default=1.0)
    )
    num_epochs = int(
        arguments.num_epochs
        if arguments.num_epochs is not None
        else _nested_config_value(training_config, "num_epochs", default=1)
    )
    max_steps = (
        arguments.max_steps
        if arguments.max_steps is not None
        else _nested_config_value(training_config, "max_steps", default=None)
    )
    stop_after_steps = (
        arguments.stop_after_steps
        if arguments.stop_after_steps is not None
        else _nested_config_value(training_config, "stop_after_steps", default=None)
    )
    save_every = int(
        arguments.save_every
        if arguments.save_every is not None
        else _nested_config_value(training_config, "save_every", default=100)
    )
    eval_every = int(
        arguments.eval_every
        if arguments.eval_every is not None
        else _nested_config_value(training_config, "eval_every", default=0)
    )
    log_every = int(
        arguments.log_every
        if arguments.log_every is not None
        else _nested_config_value(training_config, "log_every", default=1)
    )
    mixed_precision = str(
        arguments.mixed_precision
        if arguments.mixed_precision is not None
        else _nested_config_value(training_config, "mixed_precision", default="bf16")
    )
    seed = int(
        arguments.seed
        if arguments.seed is not None
        else _nested_config_value(training_config, "seed", default=42)
    )
    benchmark_warmup_steps = int(
        arguments.benchmark_warmup_steps
        if arguments.benchmark_warmup_steps is not None
        else _nested_config_value(smoke_benchmark_config, "warmup_steps", default=10)
    )
    benchmark_measure_steps = int(
        arguments.benchmark_measure_steps
        if arguments.benchmark_measure_steps is not None
        else _nested_config_value(smoke_benchmark_config, "measure_steps", default=50)
    )
    allow_download = bool(
        arguments.allow_download
        or _nested_config_value(lightwam_config, "allow_download", default=False)
    )
    device = str(arguments.device or "cuda")
    num_inference_steps = int(
        _nested_config_value(lightwam_config, "inference_steps", default=10)
    )

    eval_seeds = _parse_int_csv(
        arguments.eval_seeds
        if arguments.eval_seeds not in (None, "")
        else _nested_config_value(evaluation_config, "seeds", default=[1, 2, 3])
    )
    eval_episodes = int(
        arguments.episodes
        if arguments.episodes is not None
        else _nested_config_value(evaluation_config, "episodes_per_seed", default=50)
    )
    eval_max_episode_steps = int(
        arguments.max_episode_steps
        if arguments.max_episode_steps is not None
        else _nested_config_value(evaluation_config, "max_episode_steps", default=400)
    )
    eval_success_threshold = float(
        arguments.success_threshold
        if arguments.success_threshold is not None
        else _nested_config_value(evaluation_config, "success_threshold", default=1.0)
    )
    eval_log_every = int(
        arguments.eval_log_every
        if arguments.eval_log_every is not None
        else _nested_config_value(evaluation_config, "log_every", default=1)
    )
    normalization_path = _path_or_default(
        arguments.normalization_path,
        _nested_config_value(
            evaluation_config,
            "normalization_path",
            default="datasets/raw/dppo/robomimic/square-img/normalization.npz",
        ),
    )
    dppo_root_value = (
        arguments.dppo_root
        if arguments.dppo_root not in (None, "")
        else _nested_config_value(evaluation_config, "dppo_root", default=None)
    )
    dppo_python_value = (
        arguments.dppo_python
        if arguments.dppo_python not in (None, "")
        else _nested_config_value(evaluation_config, "dppo_python", default=None)
    )

    if arguments.phase in ("train", "smoke") and checkpoint_value not in (None, ""):
        _validate_lightwam_checkpoint(
            checkpoint_value,
            allow_training_state_directory=True,
        )
    if (
        arguments.phase == "train"
        and not arguments.dry_run
        and output_dir.is_dir()
        and any(output_dir.iterdir())
        and checkpoint_value in (None, "")
    ):
        raise FileExistsError(
            f"refusing to reuse non-empty Light-WAM training output: {output_dir}; "
            "choose a new --output-root or pass --checkpoint to resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_config.is_file():
        raise FileNotFoundError(f"Light-WAM model config is missing: {model_config}")
    if not vendor_root.is_dir():
        raise FileNotFoundError(f"Light-WAM vendor checkout is missing: {vendor_root}")

    if arguments.phase == "prepare-context":
        model_payload = _load_model_config(model_config)
        context_output = _path_or_default(
            arguments.context_output,
            (
                str(context_path_value)
                if context_path_value not in (None, "")
                else str(output_dir / "context.pt")
            ),
        )
        model_dtype = _mixed_precision_to_dtype(
            _nested_config_value(lightwam_config, "model_dtype", default="bf16")
        )
        return_value = prepare_prompt_context_cache(
            vendor_root=vendor_root,
            local_model_root=local_model_root,
            model_id=str(model_payload.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B")),
            tokenizer_model_id=str(
                model_payload.get(
                    "tokenizer_model_id",
                    model_payload.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
                )
            ),
            prompt=prompt,
            output_path=context_output,
            context_len=int(arguments.context_len),
            device=device,
            model_dtype=model_dtype,
            allow_download=allow_download,
            overwrite=bool(arguments.overwrite),
        )
        return 0 if return_value.is_file() else 1

    if arguments.phase == "check":
        runtime_errors: list[str] = []
        package_versions: dict[str, str] = {}
        for package_name in ("accelerate", "deepspeed", "omegaconf"):
            try:
                package_versions[package_name] = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                runtime_errors.append(f"missing Python package: {package_name}")

        cuda_required = device.startswith("cuda")
        cuda_available = bool(torch.cuda.is_available())
        if cuda_required and not cuda_available:
            runtime_errors.append("CUDA is unavailable in the selected container")

        accelerate_config = vendor_root / "scripts/accelerate_configs/accelerate_zero1_ds.yaml"
        deepspeed_config = vendor_root / "scripts/ds_configs/ds_zero1_config.json"
        if not accelerate_config.is_file():
            runtime_errors.append(f"missing Accelerate config: {accelerate_config}")
        if not deepspeed_config.is_file():
            runtime_errors.append(f"missing DeepSpeed config: {deepspeed_config}")

        model_assets = _model_asset_report(model_config, local_model_root)
        context_prepare_error = None
        if context_path_value in (None, ""):
            context_prepare_error = "Light-WAM context path is not configured"
        else:
            context_candidate = Path(context_path_value).expanduser().resolve()
            if (
                not context_candidate.is_file()
                and not runtime_errors
                and not model_assets["missing"]
                and data_path.is_file()
            ):
                model_payload = _load_model_config(model_config)
                try:
                    prepare_prompt_context_cache(
                        vendor_root=vendor_root,
                        local_model_root=local_model_root,
                        model_id=str(
                            model_payload.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B")
                        ),
                        tokenizer_model_id=str(
                            model_payload.get(
                                "tokenizer_model_id",
                                model_payload.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
                            )
                        ),
                        prompt=prompt,
                        output_path=context_candidate,
                        context_len=DEFAULT_CONTEXT_LEN,
                        device=device,
                        model_dtype=_mixed_precision_to_dtype(
                            _nested_config_value(
                                lightwam_config,
                                "model_dtype",
                                default="bf16",
                            )
                        ),
                        allow_download=allow_download,
                        overwrite=False,
                    )
                except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    context_prepare_error = f"{type(exc).__name__}: {exc}"

        context_report = _context_asset_report(context_path_value)
        asset_errors = []
        if context_prepare_error is not None:
            asset_errors.append(context_prepare_error)
        if context_report["error"] is not None:
            asset_errors.append(context_report["error"])
        report = {
            "schema_version": 1,
            "route": "lightwam_base_robomimic_square",
            "runtime": {
                "python": sys.executable,
                "torch": torch.__version__,
                "packages": package_versions,
                "cuda_required": cuda_required,
                "cuda_available": cuda_available,
                "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
                "accelerate_config": str(accelerate_config),
                "deepspeed_config": str(deepspeed_config),
                "errors": runtime_errors,
            },
            "vendor_root": str(vendor_root),
            "model_config": str(model_config),
            "model_assets": model_assets,
            "data_path": str(data_path),
            "data_present": data_path.is_file(),
            "context": context_report,
            "training_plan": {
                "optimizer_steps": None if max_steps is None else int(max_steps),
                "stop_after_steps": (
                    None if stop_after_steps is None else int(stop_after_steps)
                ),
                "num_epochs": num_epochs,
                "batch_size_per_process": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "lr_scheduler_type": lr_scheduler_type,
                "warmup_ratio": warmup_ratio,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "save_every": save_every,
                "mixed_precision": mixed_precision,
            },
            "dppo_reference": dict(dppo_reference),
            "lightwam_alignment": dict(training_alignment),
            "dppo_connected": False,
            "residual_rl_connected": False,
        }
        report["architecture_contract_ok"] = not model_assets["missing"]
        report["runtime_ok"] = not runtime_errors
        report["asset_errors"] = asset_errors
        report["training_ready"] = bool(
            report["runtime_ok"]
            and report["architecture_contract_ok"]
            and report["data_present"]
            and context_report["valid"]
            and not asset_errors
        )
        report["contract_ok"] = report["training_ready"]
        _json_dump(output_dir / "gates.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(
            "LIGHTWAM_BASE_CHECK=" + ("PASS" if report["training_ready"] else "FAIL"),
            flush=True,
        )
        return 0 if report["training_ready"] else 1

    if arguments.phase == "preflight":
        context_report = _context_asset_report(context_path_value)
        report = {
            "schema_version": 1,
            "route": "lightwam_base_robomimic_square",
            "vendor_root": str(vendor_root),
            "model_config": str(model_config),
            "model_assets": _model_asset_report(
                model_config,
                local_model_root,
            ),
            "data_path": str(data_path),
            "data_present": data_path.is_file(),
            "context": context_report,
            "context_path": context_report["path"],
            "context_present": context_report["present"],
            "context_valid": context_report["valid"],
            "checkpoint_path": (
                None
                if checkpoint_value in (None, "")
                else str(Path(checkpoint_value).expanduser().resolve())
            ),
            "dppo_connected": False,
            "residual_rl_connected": False,
        }
        report["architecture_contract_ok"] = not report["model_assets"]["missing"]
        report["training_ready"] = bool(
            report["architecture_contract_ok"]
            and report["data_present"]
            and report["context_valid"]
        )
        report["missing_training_assets"] = [
            name
            for name, present in (
                ("data", report["data_present"]),
                ("context", report["context_valid"]),
            )
            if not present
        ]
        report["asset_errors"] = (
            [] if context_report["error"] is None else [context_report["error"]]
        )
        report["contract_ok"] = report["architecture_contract_ok"]
        _json_dump(output_dir / "gates.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report["contract_ok"] else 1

    if arguments.phase == "model-smoke":
        model_dtype = _mixed_precision_to_dtype(
            _nested_config_value(
                lightwam_config,
                "model_dtype",
                default="bf16",
            )
        )
        return run_model_smoke(
            output_dir,
            vendor_root=vendor_root,
            model_config=model_config,
            local_model_root=local_model_root,
            checkpoint=checkpoint_value,
            device=device,
            model_dtype=model_dtype,
            image_size=image_size,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            allow_download=allow_download,
        )

    if arguments.phase == "eval":
        model_dtype = _mixed_precision_to_dtype(
            _nested_config_value(lightwam_config, "model_dtype", default="bf16")
        )
        return run_lightwam_square_eval(
            output_dir,
            vendor_root=vendor_root,
            model_config=model_config,
            local_model_root=local_model_root,
            checkpoint=checkpoint_value,
            context_path=context_path_value,
            device=device,
            model_dtype=model_dtype,
            image_size=image_size,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            allow_download=allow_download,
            seeds=eval_seeds,
            episodes_per_seed=eval_episodes,
            max_episode_steps=eval_max_episode_steps,
            success_threshold=eval_success_threshold,
            normalization_path=normalization_path,
            dppo_python_value=(
                None if dppo_python_value in (None, "") else str(dppo_python_value)
            ),
            dppo_root_value=(
                None if dppo_root_value in (None, "") else str(dppo_root_value)
            ),
            log_every=eval_log_every,
            resume=bool(arguments.resume),
            training_target_steps=(None if max_steps is None else int(max_steps)),
        )

    if arguments.phase == "eval-merge":
        if checkpoint_value in (None, ""):
            raise ValueError("Light-WAM Square eval merge requires --checkpoint")
        checkpoint_path = _validate_lightwam_checkpoint(checkpoint_value)
        metrics = _merge_lightwam_square_eval(
            output_dir=output_dir,
            seeds=eval_seeds,
            episodes_per_seed=eval_episodes,
            checkpoint_path=checkpoint_path,
            max_episode_steps=eval_max_episode_steps,
            action_horizon=action_horizon,
            inference_steps=num_inference_steps,
            training_target_steps=(None if max_steps is None else int(max_steps)),
        )
        _eval_log(
            output_dir / "run.log",
            f"DONE phase=eval-merge completed={metrics['episodes']} "
            f"total={metrics['episodes']} success_rate={metrics['success_rate']:.4f}",
        )
        return 0

    if context_path_value in (None, ""):
        raise ValueError(
            "formal Light-WAM training requires --context-path. "
            "Generate a local text context cache first; no online text encoder is enabled."
        )
    cfg = build_training_config(
        vendor_root=vendor_root,
        model_config=model_config,
        data_path=data_path,
        output_dir=output_dir,
        context_path=context_path_value,
        prompt=prompt,
        image_size=image_size,
        action_horizon=action_horizon,
        video_frames=video_frames,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        max_steps=max_steps,
        stop_after_steps=stop_after_steps,
        save_every=save_every,
        eval_every=eval_every,
        log_every=log_every,
        mixed_precision=mixed_precision,
        seed=seed,
        resume=checkpoint_value,
        benchmark_enabled=arguments.phase == "smoke",
        benchmark_warmup_steps=benchmark_warmup_steps,
        benchmark_measure_steps=benchmark_measure_steps,
    )
    if arguments.dry_run:
        from omegaconf import OmegaConf

        OmegaConf.save(cfg, str(output_dir / "config.yaml"), resolve=True)
        _json_dump(
            output_dir / "config_resolved.json",
            OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"DRY_RUN config={output_dir / 'config.yaml'}", flush=True)
        return 0

    add_lightwam_source(vendor_root)
    from lightwam.runtime import run_training

    environment = _set_offline_model_environment(local_model_root, allow_download=allow_download)
    run_log_path = output_dir / "run.log"
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with run_log_path.open("a", encoding="utf-8") as log:
            tee_out = _Tee(sys.stdout, log)
            tee_err = _Tee(sys.stderr, log)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                reported_total = (
                    benchmark_warmup_steps + benchmark_measure_steps
                    if arguments.phase == "smoke"
                    else (
                        stop_after_steps
                        if stop_after_steps is not None
                        else max_steps if max_steps is not None else "computed"
                    )
                )
                print(f"STATUS completed=0 total={reported_total}", flush=True)
                training_started = time.perf_counter()
                run_training(cfg)
                end_to_end_elapsed_sec = time.perf_counter() - training_started
                if arguments.phase == "smoke" and _is_distributed_main_process():
                    benchmark_path = output_dir / "training_speed_benchmark.json"
                    if not benchmark_path.is_file():
                        raise FileNotFoundError(
                            f"Light-WAM benchmark result is missing: {benchmark_path}"
                        )
                    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
                    summary = _build_smoke_timing_summary(
                        benchmark_payload,
                        target_optimizer_steps=int(max_steps),
                        end_to_end_elapsed_sec=end_to_end_elapsed_sec,
                        dppo_reference=dppo_reference,
                        training_alignment=training_alignment,
                    )
                    _json_dump(output_dir / "smoke_summary.json", summary)
                    estimate = summary["full_training_estimate"]
                    short_run = summary["short_run"]
                    print(
                        "LIGHTWAM_BASE_SMOKE=PASS "
                        f"steps_per_sec={short_run['steps_per_sec']:.4f} "
                        f"target_steps={estimate['target_optimizer_steps']} "
                        f"estimated_hours={estimate['compute_hours']:.2f}",
                        flush=True,
                    )
                    print(f"DONE completed={reported_total} total={reported_total}", flush=True)
                elif arguments.phase != "smoke":
                    print(
                        f"DONE lightwam_base_training output={output_dir}",
                        flush=True,
                    )
    finally:
        _restore_model_environment(environment)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
