"""Small adapters between official Light-WAM and ManiSkill2 PegInsertionSide.

The vendored Light-WAM repository remains the source of truth for model
construction, checkpoint loading, normalization, and action inference.  This
module only translates the native ManiSkill2 observation into the input shape
that a separately trained PegInsertionSide Light-WAM checkpoint expects.

The initial Peg contract is deliberately explicit:

* ``obs_mode=rgbd`` with the base and hand RGB cameras only;
* horizontal camera concatenation, matching the official two-camera dataset;
* non-privileged proprioception ``agent.qpos + agent.qvel + agent.base_pose +
  extra.tcp_pose`` (32 values);
* native ``pd_ee_delta_pose`` actions (7 values).

No task/object pose or success signal is exposed to the policy.  The adapter
supports the channel names emitted by both the pinned Policy Decorator
ManiSkill2 checkout (``Color``) and the normalized ``rgb`` form used by newer
wrappers.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


PEG_ENV_ID = "PegInsertionSide-v2"
PEG_CONTROL_MODE = "pd_ee_delta_pose"
PEG_OBS_MODE = "rgbd"
PEG_CAMERA_NAMES = ("base_camera", "hand_camera")
PEG_ACTION_DIM = 7
PEG_PROPRIO_DIM = 32
PEG_DEFAULT_ACTION_HORIZON = 16
PEG_DEFAULT_REPLAN_STEPS = 4
PEG_DEFAULT_VIDEO_SIZE = (224, 448)
PEG_DEFAULT_TASK_INSTRUCTION = "insert the peg into the hole"
LIGHTWAM_DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
LIGHTWAM_PROMPT_PREFIX = (
    "A video recorded from a robot's point of view executing the following instruction:"
)


@dataclass(frozen=True)
class PegLightWAMContract:
    """The adapter contract persisted into every residual-RL run."""

    env_id: str = PEG_ENV_ID
    obs_mode: str = PEG_OBS_MODE
    control_mode: str = PEG_CONTROL_MODE
    camera_names: tuple[str, str] = PEG_CAMERA_NAMES
    proprio_dim: int = PEG_PROPRIO_DIM
    action_dim: int = PEG_ACTION_DIM
    action_horizon: int = PEG_DEFAULT_ACTION_HORIZON
    replan_steps: int = PEG_DEFAULT_REPLAN_STEPS
    video_height: int = PEG_DEFAULT_VIDEO_SIZE[0]
    video_width: int = PEG_DEFAULT_VIDEO_SIZE[1]
    concat_multi_camera: str = "horizontal"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["camera_names"] = list(self.camera_names)
        return payload


@dataclass(frozen=True)
class PegLightWAMObservation:
    """One policy observation after the ManiSkill adapter has run."""

    image: torch.Tensor
    proprio: np.ndarray

    def validate(self, contract: PegLightWAMContract) -> None:
        expected_image = (3, contract.video_height, contract.video_width)
        if tuple(self.image.shape) != expected_image:
            raise ValueError(
                f"Light-WAM image must have shape {expected_image}, got {tuple(self.image.shape)}"
            )
        if self.image.dtype != torch.float32:
            raise ValueError(f"Light-WAM image must be float32, got {self.image.dtype}")
        if self.proprio.shape != (contract.proprio_dim,):
            raise ValueError(
                f"Light-WAM proprio must have shape ({contract.proprio_dim},), "
                f"got {self.proprio.shape}"
            )
        if not bool(torch.isfinite(self.image).all()) or not np.isfinite(self.proprio).all():
            raise ValueError("Light-WAM observation contains non-finite values")


def add_lightwam_source(vendor_root: str | Path) -> Path:
    """Make the pinned vendor ``src`` importable without installing it globally."""

    source = Path(vendor_root).expanduser().resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(
            f"Light-WAM source directory is missing: {source}. "
            "Synchronize the pinned third_party/lightwam/Light-WAM checkout first."
        )
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return source


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        # Avoid Torch's NumPy bridge: the local verification environment may
        # pair a Torch build compiled against NumPy 1.x with NumPy 2.x.
        try:
            return value.detach().cpu().numpy()
        except (RuntimeError, TypeError):
            return np.asarray(value.detach().cpu().tolist())
    return np.asarray(value)


def _to_torch(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    array = np.asarray(value)
    try:
        return torch.as_tensor(array, dtype=dtype)
    except (RuntimeError, TypeError):
        return torch.tensor(array.tolist(), dtype=dtype)


def _mapping_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"none of {keys!r} is present; available keys={list(mapping.keys())!r}")


def _single_vector(value: Any, *, name: str, expected_dim: int) -> np.ndarray:
    array = np.asarray(_to_numpy(value), dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.shape[0] != expected_dim:
        raise ValueError(
            f"PegInsertionSide {name} must have shape ({expected_dim},) for one env, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"PegInsertionSide {name} contains non-finite values")
    return np.ascontiguousarray(array)


def _camera_rgb(value: Any, *, camera_name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        value = _mapping_value(value, "rgb", "Color", "color")
    array = np.asarray(_to_numpy(value))
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"{camera_name} RGB must be HWC or CHW with 3 channels, got {array.shape}"
        )
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = np.rint(array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def extract_peg_camera_rgb(
    observation: Mapping[str, Any],
    *,
    contract: PegLightWAMContract | None = None,
) -> dict[str, np.ndarray]:
    """Return the two native RGB camera frames used by the Peg policy.

    This intentionally preserves the simulator camera resolution and layout.
    The Light-WAM model path below performs its own resize/normalization, while
    rollout visualization needs the unnormalized RGB frames for a faithful
    recording.  Keeping this helper next to the policy adapter also makes the
    recorder independent of private observation-key assumptions.
    """

    contract = contract or PegLightWAMContract()
    sensor_data = observation.get("sensor_data", observation.get("image"))
    if not isinstance(sensor_data, Mapping):
        raise KeyError("PegInsertionSide observation is missing sensor_data/image")
    frames: dict[str, np.ndarray] = {}
    for camera_name in contract.camera_names:
        if camera_name not in sensor_data:
            raise KeyError(
                f"PegInsertionSide observation is missing camera {camera_name!r}; "
                f"available={list(sensor_data.keys())!r}"
            )
        frames[camera_name] = _camera_rgb(sensor_data[camera_name], camera_name=camera_name)
    shapes = {tuple(frame.shape) for frame in frames.values()}
    if len(shapes) != 1:
        raise ValueError(f"Peg camera RGB shapes must match, got {sorted(shapes)!r}")
    return frames


def _resize_smallest_side(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Match official ``ResizeSmallestSideAspectPreserving`` semantics."""

    source_height, source_width = image.shape[-2:]
    scale = max(height / source_height, width / source_width)
    target_height = int(source_height * scale + 0.5)
    target_width = int(source_width * scale + 0.5)
    try:
        import torchvision.transforms.functional as transforms_f

        return transforms_f.resize(
            image,
            size=[target_height, target_width],
            interpolation=transforms_f.InterpolationMode.BICUBIC,
            antialias=True,
        )
    except (ImportError, ModuleNotFoundError, RuntimeError):
        # The full Light-WAM environment contains torchvision.  Keeping this
        # small fallback makes the data-free contract smoke usable in the
        # repository's lightweight development environment as well.
        import torch.nn.functional as functional

        try:
            return functional.interpolate(
                image.unsqueeze(0),
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        except TypeError:  # older Torch without interpolate(antialias=...)
            return functional.interpolate(
                image.unsqueeze(0),
                size=(target_height, target_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)


def _resize_camera_square(image: torch.Tensor, size: int) -> torch.Tensor:
    """Match the official per-camera ``torchvision.Resize([size, size])``."""

    try:
        import torchvision.transforms.functional as transforms_f

        return transforms_f.resize(
            image,
            size=[size, size],
            interpolation=transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )
    except (ImportError, ModuleNotFoundError, RuntimeError):
        import torch.nn.functional as functional

        try:
            return functional.interpolate(
                image.unsqueeze(0),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        except TypeError:  # older Torch without interpolate(antialias=...)
            return functional.interpolate(
                image.unsqueeze(0),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)


def _camera_tensor(value: Any, *, camera_name: str) -> torch.Tensor:
    array = _camera_rgb(value, camera_name=camera_name)
    tensor = _to_torch(array, dtype=torch.uint8).permute(2, 0, 1).contiguous()
    return tensor.to(dtype=torch.float32).div(255.0)


def preprocess_composite_rgb(
    composite_rgb: np.ndarray,
    *,
    video_size: tuple[int, int] = PEG_DEFAULT_VIDEO_SIZE,
) -> torch.Tensor:
    """Apply the same resize/crop/[-1, 1] normalization as official Light-WAM."""

    image = np.asarray(composite_rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"composite RGB must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)
    tensor = _to_torch(np.ascontiguousarray(image), dtype=torch.uint8).permute(2, 0, 1).contiguous()
    tensor = tensor.to(dtype=torch.float32).div(255.0)
    height, width = (int(video_size[0]), int(video_size[1]))
    tensor = _resize_smallest_side(tensor, height, width)

    top = max(0, (tensor.shape[-2] - height) // 2)
    left = max(0, (tensor.shape[-1] - width) // 2)
    tensor = tensor[..., top : top + height, left : left + width]
    if tuple(tensor.shape[-2:]) != (height, width):
        raise RuntimeError(
            f"center crop produced {tuple(tensor.shape[-2:])}, expected {(height, width)}"
        )
    # Official dataset_utils.Normalize(mean=0.5, std=0.5) broadcasts this way.
    tensor = (tensor - 0.5) / 0.5
    return tensor.contiguous()


def extract_peg_proprio(observation: Mapping[str, Any]) -> np.ndarray:
    """Extract only the 32D non-privileged PegInsertionSide proprioception."""

    try:
        agent = observation["agent"]
        extra = observation["extra"]
        values = (
            _single_vector(agent["qpos"], name="agent.qpos", expected_dim=9),
            _single_vector(agent["qvel"], name="agent.qvel", expected_dim=9),
            _single_vector(agent["base_pose"], name="agent.base_pose", expected_dim=7),
            _single_vector(extra["tcp_pose"], name="extra.tcp_pose", expected_dim=7),
        )
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "PegInsertionSide rgbd observation must contain agent.qpos/qvel/base_pose "
            "and extra.tcp_pose"
        ) from exc
    return np.ascontiguousarray(np.concatenate(values, axis=-1), dtype=np.float32)


def build_peg_lightwam_observation(
    observation: Mapping[str, Any],
    *,
    contract: PegLightWAMContract | None = None,
) -> PegLightWAMObservation:
    """Convert one native ManiSkill2 rgbd observation to official Light-WAM input."""

    contract = contract or PegLightWAMContract()
    try:
        # ManiSkill2 0.5.x normally exposes ``sensor_data``.  A few upstream
        # wrappers rename the same nested mapping to ``image``; accept both so
        # this adapter does not impose a new observation wrapper on the env.
        sensor_data = observation.get("sensor_data", observation.get("image"))
        if sensor_data is None:
            raise KeyError("sensor_data/image")
        base_rgb = _camera_rgb(sensor_data[contract.camera_names[0]], camera_name="base_camera")
        hand_rgb = _camera_rgb(sensor_data[contract.camera_names[1]], camera_name="hand_camera")
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "PegInsertionSide rgbd observation must contain "
            "sensor_data/image.base_camera and sensor_data/image.hand_camera RGB values"
        ) from exc
    height = int(contract.video_height)
    base_tensor = _resize_camera_square(
        _camera_tensor(base_rgb, camera_name="base_camera"), height
    )
    hand_tensor = _resize_camera_square(
        _camera_tensor(hand_rgb, camera_name="hand_camera"), height
    )
    composite = torch.cat((base_tensor, hand_tensor), dim=-1)
    composite = _resize_smallest_side(
        composite, int(contract.video_height), int(contract.video_width)
    )
    crop_top = max(0, (composite.shape[-2] - int(contract.video_height)) // 2)
    crop_left = max(0, (composite.shape[-1] - int(contract.video_width)) // 2)
    composite = composite[
        ...,
        crop_top : crop_top + int(contract.video_height),
        crop_left : crop_left + int(contract.video_width),
    ]
    if tuple(composite.shape[-2:]) != (contract.video_height, contract.video_width):
        raise RuntimeError(
            "camera preprocessing produced an unexpected composite shape: "
            f"got {tuple(composite.shape[-2:])}, "
            f"expected {(contract.video_height, contract.video_width)}"
        )
    composite = (composite - 0.5) / 0.5
    processed = PegLightWAMObservation(
        image=composite.contiguous(),
        proprio=extract_peg_proprio(observation),
    )
    processed.validate(contract)
    return processed


def _resolve_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if relative_to is not None:
        candidate = relative_to / path
        if candidate.is_file():
            return candidate.resolve()
    return path.resolve()


class LightWAMStateActionNormalizer:
    """Use the official Light-WAM ``LinearNormalizer`` for online actions."""

    def __init__(
        self,
        *,
        stats_path: str | Path,
        shape_meta: Mapping[str, Any],
        state_key: str,
        action_key: str,
        use_stepwise_action_norm: bool = False,
        norm_default_mode: str = "min/max",
        norm_exception_mode: Mapping[str, Any] | None = None,
        vendor_root: str | Path,
    ) -> None:
        add_lightwam_source(vendor_root)
        from lightwam.datasets.lerobot.utils.normalizer import (
            LinearNormalizer,
            load_dataset_stats_from_json,
        )

        stats_file = _resolve_path(stats_path)
        if not stats_file.is_file():
            raise FileNotFoundError(
                f"Light-WAM dataset stats do not exist: {stats_file}. "
                "Use the stats generated by the same Peg Light-WAM training run."
            )
        stats = load_dataset_stats_from_json(str(stats_file))
        shape = {
            "state": [dict(shape_meta["state"][0])],
            "action": [dict(shape_meta["action"][0])],
        }
        state_dim = int(shape["state"][0]["shape"])
        action_dim = int(shape["action"][0]["shape"])
        if state_dim != PEG_PROPRIO_DIM or action_dim != PEG_ACTION_DIM:
            raise ValueError(
                "Peg Light-WAM stats/config must describe state=32 and action=7, "
                f"got state={state_dim}, action={action_dim}"
            )
        for group, key in (("state", state_key), ("action", action_key)):
            if group not in stats or key not in stats[group]:
                raise KeyError(
                    f"stats file {stats_file} is missing {group}[{key!r}]; "
                    f"available={list(stats.get(group, {}).keys())!r}"
                )
        self.stats_path = stats_file
        self.state_key = state_key
        self.action_key = action_key
        self._normalizer = LinearNormalizer(
            shape_meta=shape,
            use_stepwise_action_norm=bool(use_stepwise_action_norm),
            default_mode=norm_default_mode,
            exception_mode=None if norm_exception_mode is None else dict(norm_exception_mode),
            stats=stats,
        )

    @property
    def state_dim(self) -> int:
        return PEG_PROPRIO_DIM

    @property
    def action_dim(self) -> int:
        return PEG_ACTION_DIM

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        values = _to_torch(state, dtype=torch.float32)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[-1] != self.state_dim:
            raise ValueError(f"state must be [B,{self.state_dim}], got {tuple(values.shape)}")
        normalizer = self._normalizer.normalizers["state"][self.state_key]
        return normalizer.forward(values)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        values = _to_torch(action, dtype=torch.float32)
        if values.ndim == 2 and values.shape[-1] == self.action_dim:
            pass
        elif values.ndim == 3 and values.shape[-1] == self.action_dim and values.shape[0] == 1:
            values = values[0]
        else:
            raise ValueError(
                f"action must be [T,{self.action_dim}] or [1,T,{self.action_dim}], "
                f"got {tuple(values.shape)}"
            )
        normalizer = self._normalizer.normalizers["action"][self.action_key]
        return normalizer.backward(values)

    def action_std(self) -> np.ndarray:
        stats = self._normalizer.normalizers["action"][self.action_key].stats
        std = stats.get("std")
        if std is None:
            raise ValueError(
                "Light-WAM action stats do not contain global_std; residual scaling "
                "requires action statistics from the same training run."
            )
        values = np.asarray(_to_numpy(std), dtype=np.float32)
        if values.shape != (self.action_dim,) or not np.isfinite(values).all():
            raise ValueError(
                f"action global_std must have shape ({self.action_dim},), got {values.shape}"
            )
        return np.maximum(values, 1e-8)


def _load_context_file(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    context_path = _resolve_path(path)
    if not context_path.is_file():
        raise FileNotFoundError(f"Light-WAM context file does not exist: {context_path}")
    payload = torch.load(str(context_path), map_location="cpu")
    if isinstance(payload, Mapping):
        context = payload.get("context")
        context_mask = payload.get("context_mask", payload.get("mask"))
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        context, context_mask = payload
    else:
        raise ValueError(
            "context file must contain a mapping with context/context_mask (or mask), "
            "or a (context, mask) pair"
        )
    if context is None or context_mask is None:
        raise ValueError(f"context file is missing context or context_mask: {context_path}")
    context_tensor = _to_torch(context, dtype=torch.float32)
    mask_tensor = _to_torch(context_mask, dtype=torch.bool)
    if context_tensor.ndim == 3 and context_tensor.shape[0] == 1:
        context_tensor = context_tensor[0]
    if mask_tensor.ndim == 2 and mask_tensor.shape[0] == 1:
        mask_tensor = mask_tensor[0]
    if context_tensor.ndim != 2 or mask_tensor.ndim != 1:
        raise ValueError(
            f"context must be [L,D] and context_mask [L], got "
            f"{tuple(context_tensor.shape)} and {tuple(mask_tensor.shape)}"
        )
    if context_tensor.shape[0] != mask_tensor.shape[0]:
        raise ValueError("context and context_mask sequence lengths differ")
    context_tensor = context_tensor.clone()
    context_tensor[~mask_tensor] = 0.0
    # Match the official dataset path, which masks padded embedding positions
    # in the values and sends an all-ones attention mask to the model.
    return context_tensor.contiguous(), torch.ones_like(mask_tensor)


def _official_prompt(prompt_or_task: str) -> str:
    """Return the exact prompt string used by the official dataset cache."""

    value = str(prompt_or_task).strip()
    if value.startswith(LIGHTWAM_PROMPT_PREFIX):
        return value
    return LIGHTWAM_DEFAULT_PROMPT.format(task=value)


def resolve_text_embedding_cache(
    cache_dir: str | Path,
    *,
    prompt_or_task: str,
    context_len: int = 128,
    text_embedding_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
) -> Path:
    """Resolve one official ``precompute_text_embeds.py`` cache file."""

    import re

    prompt = _official_prompt(prompt_or_task)
    model_base = str(text_embedding_model_id).split("/")[-1]
    cache_id = re.sub(r"[^a-z0-9]+", "", model_base.lower()) or "textenc"
    filename = (
        f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"
        f".t5_len{int(context_len)}.{cache_id}.pt"
    )
    path = Path(cache_dir).expanduser().resolve() / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"official Light-WAM text cache is missing: {path}. "
            "Run the pinned precompute_text_embeds.py only after its model download "
            "has been approved, or synchronize the prepared cache artifact."
        )
    return path


def _resolve_config_payload(config_path: str | Path) -> tuple[Any, Any, Path]:
    from omegaconf import OmegaConf

    path = _resolve_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Light-WAM config does not exist: {path}")
    payload = OmegaConf.load(str(path))
    try:
        OmegaConf.resolve(payload)
    except Exception as exc:
        raise ValueError(
            f"could not resolve {path}; pass the resolved config.yaml written by official "
            "Light-WAM training, not configs/model/lightwam.yaml by itself"
        ) from exc
    if "model" in payload and "_target_" not in payload:
        model_cfg = payload.model
        data_cfg = payload.get("data")
    else:
        model_cfg = payload
        data_cfg = None
    if "_target_" not in model_cfg:
        raise ValueError(f"Light-WAM model config has no _target_: {path}")
    return model_cfg, data_cfg, path


class LightWAMBasePolicy:
    """Frozen official Light-WAM policy with the Peg observation adapter."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        config_path: str | Path,
        stats_path: str | Path | None = None,
        vendor_root: str | Path = "third_party/lightwam/Light-WAM",
        device: str | torch.device = "cuda",
        model_dtype: str = "auto",
        action_horizon: int = PEG_DEFAULT_ACTION_HORIZON,
        num_inference_steps: int = 10,
        prompt: str | None = None,
        context_path: str | Path | None = None,
        text_embedding_cache_dir: str | Path | None = None,
        state_key: str = "default",
        action_key: str = "default",
    ) -> None:
        from hydra.utils import instantiate

        self.vendor_root = Path(vendor_root).expanduser().resolve()
        add_lightwam_source(self.vendor_root)
        model_cfg, data_cfg, resolved_config_path = _resolve_config_payload(config_path)
        self.config_path = resolved_config_path
        self.checkpoint_path = _resolve_path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Light-WAM checkpoint does not exist: {self.checkpoint_path}")

        self.device = torch.device(device)
        if model_dtype == "auto":
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        elif model_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif model_dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError("model_dtype must be auto, bfloat16, or float32")

        self.model = instantiate(model_cfg, model_dtype=dtype, device=str(self.device))
        self.model.load_checkpoint(str(self.checkpoint_path))
        self.model.eval()
        self.model.requires_grad_(False)

        self.contract = PegLightWAMContract(action_horizon=int(action_horizon))
        actual_action_dim = int(getattr(self.model.action_expert, "action_dim", -1))
        actual_proprio_dim = int(getattr(self.model, "proprio_dim", -1))
        if (
            actual_action_dim != self.contract.action_dim
            or actual_proprio_dim != self.contract.proprio_dim
        ):
            raise ValueError(
                "Light-WAM checkpoint/config contract mismatch: expected action=7/proprio=32, "
                f"got action={actual_action_dim}/proprio={actual_proprio_dim}. "
                "A RoboTwin checkpoint or S1 Policy Decorator best.pt cannot be used here."
            )
        if not hasattr(self.model, "infer_action"):
            raise TypeError("official Light-WAM model does not expose infer_action()")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.num_inference_steps = int(num_inference_steps)

        if data_cfg is None or data_cfg.get("train") is None:
            raise ValueError(
                "Light-WAM config must be a full resolved run config with data.train; "
                "the model-only config does not describe Peg normalization/image settings"
            )
        train_cfg = data_cfg.train
        shape_meta = train_cfg.shape_meta
        if int(train_cfg.processor.proprio_output_dim) != self.contract.proprio_dim:
            raise ValueError(
                "data.train.processor.proprio_output_dim must be 32 for PegInsertionSide"
            )
        if int(train_cfg.processor.action_output_dim) != self.contract.action_dim:
            raise ValueError(
                "data.train.processor.action_output_dim must be 7 for PegInsertionSide"
            )
        configured_action_horizon = int(train_cfg.num_frames) - 1
        if configured_action_horizon != self.contract.action_horizon:
            raise ValueError(
                "Light-WAM checkpoint/config action horizon does not match the Peg contract: "
                f"data.train.num_frames-1={configured_action_horizon}, "
                f"requested={self.contract.action_horizon}"
            )
        if int(train_cfg.action_video_freq_ratio) <= 0:
            raise ValueError("data.train.action_video_freq_ratio must be positive")
        configured_size = tuple(int(value) for value in train_cfg.video_size)
        if len(configured_size) != 2 or any(value <= 0 for value in configured_size):
            raise ValueError(
                "data.train.video_size must contain two positive values [height, width]"
            )
        self.contract = PegLightWAMContract(
            action_horizon=int(action_horizon),
            video_height=configured_size[0],
            video_width=configured_size[1],
            concat_multi_camera=str(train_cfg.concat_multi_camera),
        )
        if self.contract.concat_multi_camera != "horizontal":
            raise ValueError(
                "Peg Light-WAM adapter currently requires data.train.concat_multi_camera=horizontal"
            )
        image_keys = tuple(str(item["key"]) for item in shape_meta.images)
        if image_keys != PEG_CAMERA_NAMES:
            raise ValueError(
                "Peg Light-WAM adapter requires shape_meta images in the order "
                f"{PEG_CAMERA_NAMES}, got {image_keys}"
            )
        if len(shape_meta.state) != 1 or len(shape_meta.action) != 1:
            raise ValueError(
                "Peg Light-WAM adapter requires exactly one merged state key and one action key"
            )
        if int(shape_meta.state[0]["shape"]) != self.contract.proprio_dim:
            raise ValueError("data.train.shape_meta.state must describe a 32D state")
        if int(shape_meta.action[0]["shape"]) != self.contract.action_dim:
            raise ValueError("data.train.shape_meta.action must describe a 7D action")
        if train_cfg.processor.get("action_state_transforms") not in (None, "null"):
            raise ValueError(
                "Peg Light-WAM adapter requires processor.action_state_transforms=null; "
                "task-specific transforms are not silently applied online"
            )
        stats_value = stats_path
        if stats_value is None:
            stats_value = train_cfg.get("pretrained_norm_stats")
        if stats_value in (None, "", "null"):
            raise ValueError(
                "Light-WAM stats are required for formal inference; pass --stats or set "
                "data.train.pretrained_norm_stats in the resolved config"
            )
        stats_path_resolved = _resolve_path(stats_value, relative_to=self.config_path.parent)
        self.normalizer = LightWAMStateActionNormalizer(
            stats_path=stats_path_resolved,
            shape_meta=shape_meta,
            state_key=state_key,
            action_key=action_key,
            use_stepwise_action_norm=bool(train_cfg.processor.use_stepwise_action_norm),
            norm_default_mode=str(train_cfg.processor.norm_default_mode),
            norm_exception_mode=train_cfg.processor.get("norm_exception_mode"),
            vendor_root=self.vendor_root,
        )
        self.action_std = self.normalizer.action_std()
        self.prompt = None if prompt is None else _official_prompt(prompt)
        self.context = None
        self.context_mask = None
        resolved_context_path = context_path
        if resolved_context_path is None and text_embedding_cache_dir is not None:
            resolved_context_path = resolve_text_embedding_cache(
                text_embedding_cache_dir,
                prompt_or_task=prompt or PEG_DEFAULT_TASK_INSTRUCTION,
                context_len=int(train_cfg.get("context_len", 128)),
                text_embedding_model_id=str(
                    train_cfg.get(
                        "text_embedding_model_id", "Wan-AI/Wan2.1-T2V-1.3B"
                    )
                ),
            )
        if resolved_context_path is not None:
            self.context, self.context_mask = _load_context_file(resolved_context_path)
            self.prompt = None
        if self.context is None and self.prompt is None:
            raise ValueError(
                "formal Light-WAM inference needs either --prompt with a loaded text encoder "
                "or --context-path/--text-embedding-cache-dir containing the official cache"
            )
        if self.context is None and getattr(self.model, "text_encoder", None) is None:
            raise ValueError(
                "this Light-WAM config has no loaded text encoder; provide --context-path "
                "or --text-embedding-cache-dir from the official text-embedding cache"
            )

    @property
    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in self.model.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(str(parameter.dtype).encode("ascii"))
            digest.update(repr(tuple(parameter.shape)).encode("ascii"))
            digest.update(bytes(parameter.detach().cpu().contiguous().untyped_storage()))
        return digest.hexdigest()

    def normalize_proprio(self, observation: PegLightWAMObservation) -> np.ndarray:
        values = self.normalizer.normalize_state(
            _to_torch(np.asarray(observation.proprio, dtype=np.float32), dtype=torch.float32)
        )[0]
        result = np.asarray(values.detach().cpu().tolist(), dtype=np.float32)
        if result.shape != (self.contract.proprio_dim,) or not np.isfinite(result).all():
            raise ValueError(
                f"normalized Light-WAM proprio must be finite with shape "
                f"({self.contract.proprio_dim},), got {result.shape}"
            )
        return result

    def predict(
        self,
        observation: Mapping[str, Any] | PegLightWAMObservation,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        if isinstance(observation, PegLightWAMObservation):
            processed = observation
        else:
            processed = build_peg_lightwam_observation(observation, contract=self.contract)
        processed.validate(self.contract)
        image = processed.image.unsqueeze(0).to(device=self.device, dtype=self.model.torch_dtype)
        state = _to_torch(processed.proprio, dtype=torch.float32).to(
            device="cpu", dtype=torch.float32
        )
        state = self.normalizer.normalize_state(state)
        if self.context is None:
            infer_kwargs: dict[str, Any] = {
                "prompt": self.prompt,
                "context": None,
                "context_mask": None,
            }
        else:
            infer_kwargs = {
                "prompt": None,
                "context": self.context,
                "context_mask": self.context_mask,
            }
        with torch.no_grad():
            result = self.model.infer_action(
                input_image=image,
                action_horizon=self.contract.action_horizon,
                proprio=state[0],
                num_inference_steps=self.num_inference_steps,
                seed=None if seed is None else int(seed),
                **infer_kwargs,
            )
        if not isinstance(result, Mapping) or "action" not in result:
            raise RuntimeError("official Light-WAM infer_action() did not return an action")
        action = self.normalizer.denormalize_action(result["action"])
        action_np = np.asarray(action.detach().cpu().tolist(), dtype=np.float32)
        expected = (self.contract.action_horizon, self.contract.action_dim)
        if action_np.shape != expected or not np.isfinite(action_np).all():
            raise ValueError(
                f"Light-WAM action must be finite with shape {expected}, got {action_np.shape}"
            )
        return np.ascontiguousarray(action_np)
