"""The M2 Action-only Flow policy primitive.

The module intentionally contains no future/world target, privileged state, or
environment code.  The same trunk can be trained against the 4D
``pd_ee_delta_pos`` official RGB protocol or the 7D ``pd_ee_delta_pose`` M1-W
protocol.  The M2-W runner still applies its own frozen 7D contract; allowing
4D here keeps the model/checkpoint primitive reusable without weakening that
contract.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

BASE_IMAGE_KEY = "observation.images.base_camera"
HAND_IMAGE_KEY = "observation.images.hand_camera"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEYS = (BASE_IMAGE_KEY, HAND_IMAGE_KEY)


@dataclass(frozen=True)
class ActionFlowConfig:
    """Architecture and sampler values shared by the M2 base and M2-W."""

    state_dim: int = 18
    action_dim: int = 7
    n_obs_steps: int = 2
    horizon: int = 16
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    spatial_softmax_keypoints: int = 32
    image_size: tuple[int, int] = (128, 128)
    flow_solver: str = "euler"
    integration_steps: int = 10
    seed: int = 1000

    def __post_init__(self) -> None:
        if self.state_dim != 18:
            raise ValueError("M2 Action-only Flow requires state_dim=18 (qpos+qvel)")
        if self.action_dim not in (4, 7):
            raise ValueError("M2 Action-only Flow supports action_dim=4 or action_dim=7")
        if self.n_obs_steps != 2 or self.horizon != 16:
            raise ValueError("M2 Action-only Flow requires n_obs_steps=2 and horizon=16")
        if self.d_model != 256 or self.n_layers != 6 or self.n_heads != 8:
            raise ValueError(
                "M2 Action-only Flow transformer must be d_model=256, layers=6, heads=8"
            )
        if self.mlp_ratio != 4 or self.dropout != 0.1:
            raise ValueError("M2 Action-only Flow transformer must use mlp_ratio=4 and dropout=0.1")
        if self.spatial_softmax_keypoints != 32:
            raise ValueError("M2 Action-only Flow requires 32 SpatialSoftmax keypoints")
        if self.image_size != (128, 128):
            raise ValueError("M2 Action-only Flow RGB input must be 128x128")
        if self.flow_solver != "euler" or self.integration_steps != 10:
            raise ValueError("M2 Action-only Flow sampler is fixed to Euler with 10 integration steps")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ActionFlowConfig:
        image_size = tuple(int(item) for item in value.get("image_size", (128, 128)))
        return cls(
            state_dim=int(value.get("state_dim", 18)),
            action_dim=int(value.get("action_dim", 7)),
            n_obs_steps=int(value.get("n_obs_steps", 2)),
            horizon=int(value.get("horizon", 16)),
            d_model=int(value.get("d_model", 256)),
            n_layers=int(value.get("n_layers", value.get("num_layers", 6))),
            n_heads=int(value.get("n_heads", 8)),
            mlp_ratio=int(value.get("mlp_ratio", 4)),
            dropout=float(value.get("dropout", 0.1)),
            spatial_softmax_keypoints=int(value.get("spatial_softmax_keypoints", 32)),
            image_size=image_size,
            flow_solver=str(value.get("flow_solver", "euler")),
            integration_steps=int(value.get("integration_steps", 10)),
            seed=int(value.get("seed", 1000)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["image_size"] = list(self.image_size)
        return result


ActionOnlyFlowConfig = ActionFlowConfig


@dataclass
class NormalizationStats:
    """Train-only channel/state/action normalization statistics."""

    image_mean: Tensor
    image_std: Tensor
    state_mean: Tensor
    state_std: Tensor
    action_mean: Tensor
    action_std: Tensor

    def __post_init__(self) -> None:
        self.image_mean = torch.as_tensor(self.image_mean, dtype=torch.float32).reshape(3)
        self.image_std = torch.as_tensor(self.image_std, dtype=torch.float32).reshape(3)
        self.state_mean = torch.as_tensor(self.state_mean, dtype=torch.float32).reshape(18)
        self.state_std = torch.as_tensor(self.state_std, dtype=torch.float32).reshape(18)
        self.action_mean = torch.as_tensor(self.action_mean, dtype=torch.float32).reshape(-1)
        self.action_std = torch.as_tensor(self.action_std, dtype=torch.float32).reshape(-1)
        if self.action_mean.numel() not in (4, 7) or self.action_std.numel() != self.action_mean.numel():
            raise ValueError("Action normalization must contain matching 4D or 7D statistics")
        for name in ("image_std", "state_std", "action_std"):
            value = getattr(self, name)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value.clamp_min(1e-6))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NormalizationStats:
        return cls(
            image_mean=value["image_mean"],
            image_std=value["image_std"],
            state_mean=value["state_mean"],
            state_std=value["state_std"],
            action_mean=value["action_mean"],
            action_std=value["action_std"],
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "image_mean": self.image_mean.tolist(),
            "image_std": self.image_std.tolist(),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    def to(self, device: torch.device | str) -> NormalizationStats:
        return NormalizationStats(
            image_mean=self.image_mean.to(device),
            image_std=self.image_std.to(device),
            state_mean=self.state_mean.to(device),
            state_std=self.state_std.to(device),
            action_mean=self.action_mean.to(device),
            action_std=self.action_std.to(device),
        )

    def normalize_images(self, images: Tensor) -> Tensor:
        if images.ndim < 3 or images.shape[-3] != 3:
            raise ValueError(f"Expected channel-first RGB tensor, got {tuple(images.shape)}")
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        return (images - mean.reshape((1,) * (images.ndim - 3) + (3, 1, 1))) / std.reshape(
            (1,) * (images.ndim - 3) + (3, 1, 1)
        )

    def normalize_state(self, state: Tensor) -> Tensor:
        mean = self.state_mean.to(device=state.device, dtype=state.dtype)
        std = self.state_std.to(device=state.device, dtype=state.dtype)
        return (state - mean) / std

    def normalize_action(self, action: Tensor) -> Tensor:
        mean = self.action_mean.to(device=action.device, dtype=action.dtype)
        std = self.action_std.to(device=action.device, dtype=action.dtype)
        return (action - mean) / std

    def denormalize_action(self, action: Tensor) -> tuple[Tensor, dict[str, int]]:
        mean = self.action_mean.to(device=action.device, dtype=action.dtype)
        std = self.action_std.to(device=action.device, dtype=action.dtype)
        raw = action * std + mean
        finite = torch.isfinite(raw)
        if not bool(finite.all()):
            raise ValueError("Action denormalization produced a non-finite value")
        clipped = raw.clamp(-1.0, 1.0)
        return clipped, {
            "clipped_elements": int((clipped != raw).sum().item()),
            "total_elements": int(raw.numel()),
        }


def derive_flow_seed(fingerprint: str, chunk_index: int, *, base_seed: int = 1000) -> int:
    """Derive a stable source-noise seed without using a simulator seed as identity."""

    if not fingerprint or chunk_index < 0:
        raise ValueError("fingerprint must be non-empty and chunk_index must be non-negative")
    payload = f"m2-w-action-only-flow/{base_seed}/{fingerprint}/{chunk_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _group_count(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


def _replace_batch_norm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(_group_count(child.num_features), child.num_features)
            setattr(module, name, replacement)
        else:
            _replace_batch_norm(child)


class SpatialSoftmax(nn.Module):
    """Differentiable 2-D keypoint extraction from a convolutional feature map."""

    def __init__(self, input_channels: int, num_keypoints: int) -> None:
        super().__init__()
        self.heatmap = nn.Conv2d(input_channels, num_keypoints, kernel_size=1)
        self.num_keypoints = num_keypoints

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"SpatialSoftmax expects [B,C,H,W], got {tuple(features.shape)}")
        batch, _, height, width = features.shape
        logits = self.heatmap(features).reshape(batch, self.num_keypoints, height * width)
        probabilities = logits.softmax(dim=-1)
        y = torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        return (probabilities @ coordinates).reshape(batch, self.num_keypoints * 2)


class SharedResNet18SpatialSoftmax(nn.Module):
    """Uninitialised shared ResNet-18 with BatchNorm replaced by GroupNorm."""

    def __init__(self, num_keypoints: int = 32) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18
        except Exception as exc:  # pragma: no cover - environment-specific import failure
            raise RuntimeError("M2-W requires torchvision's ResNet-18 implementation") from exc
        backbone = resnet18(weights=None)
        _replace_batch_norm(backbone)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.spatial_softmax = SpatialSoftmax(512, num_keypoints)
        self.feature_dim = num_keypoints * 2

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Image encoder expects [B,3,H,W], got {tuple(images.shape)}")
        return self.spatial_softmax(self.backbone(images))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, time: Tensor) -> Tensor:
        time = time.reshape(-1).to(dtype=torch.float32)
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = time[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(
                embedding, (0, self.dimension - embedding.shape[-1])
            )
        return self.projection(embedding.to(dtype=self.projection[0].weight.dtype))


class ActionOnlyFlowModel(nn.Module):
    """Action-only context/action transformer trained with linear Flow Matching."""

    def __init__(self, config: ActionFlowConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, ActionFlowConfig)
            else ActionFlowConfig.from_mapping(config or {})
        )
        cfg = self.config
        self.visual_encoder = SharedResNet18SpatialSoftmax(cfg.spatial_softmax_keypoints)
        self.visual_projection = nn.Linear(self.visual_encoder.feature_dim, cfg.d_model)
        self.state_projection = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.action_projection = nn.Linear(cfg.action_dim, cfg.d_model)
        self.observation_time_embedding = nn.Parameter(torch.zeros(cfg.n_obs_steps, cfg.d_model))
        self.action_position_embedding = nn.Parameter(torch.zeros(cfg.horizon, cfg.d_model))
        self.camera_embedding = nn.Embedding(2, cfg.d_model)
        self.type_embedding = nn.Embedding(4, cfg.d_model)
        self.task_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.time_embedding = SinusoidalTimeEmbedding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.mlp_ratio,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.action_head = nn.Linear(cfg.d_model, cfg.action_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.observation_time_embedding, std=0.02)
        nn.init.normal_(self.action_position_embedding, std=0.02)
        nn.init.normal_(self.task_token, std=0.02)

    def _validate_inputs(self, observations: Mapping[str, Tensor], noisy_actions: Tensor) -> None:
        for key in IMAGE_KEYS:
            if key not in observations:
                raise KeyError(f"Missing required Action-only Flow input {key!r}")
            value = observations[key]
            expected = (self.config.n_obs_steps, 3, *self.config.image_size)
            if value.ndim != 5 or tuple(value.shape[1:]) != expected:
                raise ValueError(f"{key} must have shape [B,{expected}], got {tuple(value.shape)}")
        state = observations.get(STATE_KEY)
        expected_state = (self.config.n_obs_steps, self.config.state_dim)
        if state is None or state.ndim != 3 or tuple(state.shape[1:]) != expected_state:
            raise ValueError(
                f"{STATE_KEY} must have shape [B,{expected_state}], got {None if state is None else tuple(state.shape)}"
            )
        expected_action = (self.config.horizon, self.config.action_dim)
        if noisy_actions.ndim != 3 or tuple(noisy_actions.shape[1:]) != expected_action:
            raise ValueError(
                f"noisy_actions must have shape [B,{expected_action}], got {tuple(noisy_actions.shape)}"
            )
        if observations[BASE_IMAGE_KEY].shape[0] != noisy_actions.shape[0]:
            raise ValueError("Observation and action batch sizes disagree")

    def _context_tokens(self, observations: Mapping[str, Tensor]) -> Tensor:
        state = observations[STATE_KEY]
        batch_size, n_obs = state.shape[:2]
        images = torch.stack((observations[BASE_IMAGE_KEY], observations[HAND_IMAGE_KEY]), dim=2)
        flat_images = images.reshape(batch_size * n_obs * 2, 3, *self.config.image_size)
        visual = self.visual_encoder(flat_images)
        visual = self.visual_projection(visual).reshape(batch_size, n_obs, 2, -1)
        time_embedding = self.observation_time_embedding[:n_obs].reshape(1, n_obs, 1, -1)
        camera_ids = torch.arange(2, device=state.device).reshape(1, 1, 2)
        image_tokens = visual + time_embedding
        image_tokens = image_tokens + self.camera_embedding(camera_ids)
        image_tokens = image_tokens + self.type_embedding.weight[0].reshape(1, 1, 1, -1)
        image_tokens = image_tokens.reshape(batch_size, n_obs * 2, -1)

        state_tokens = self.state_projection(state)
        state_tokens = state_tokens + self.observation_time_embedding[:n_obs].reshape(1, n_obs, -1)
        state_tokens = state_tokens + self.type_embedding.weight[1].reshape(1, 1, -1)
        task_tokens = self.task_token.expand(batch_size, -1, -1)
        task_tokens = task_tokens + self.type_embedding.weight[2].reshape(1, 1, -1)
        return torch.cat((task_tokens, image_tokens, state_tokens), dim=1)

    def forward(
        self, observations: Mapping[str, Tensor], noisy_actions: Tensor, time: Tensor
    ) -> Tensor:
        self._validate_inputs(observations, noisy_actions)
        context = self._context_tokens(observations)
        batch_size = noisy_actions.shape[0]
        action_tokens = self.action_projection(noisy_actions)
        action_tokens = action_tokens + self.action_position_embedding.reshape(
            1, self.config.horizon, -1
        )
        action_tokens = action_tokens + self.type_embedding.weight[3].reshape(1, 1, -1)
        time_token = self.time_embedding(time).reshape(batch_size, 1, -1)
        action_tokens = action_tokens + time_token
        tokens = torch.cat((context, action_tokens), dim=1)
        encoded = self.trunk(tokens)
        return self.action_head(self.output_norm(encoded[:, -self.config.horizon :]))

    def flow_loss(
        self,
        observations: Mapping[str, Tensor],
        clean_actions: Tensor,
        *,
        source_noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        expected = (self.config.horizon, self.config.action_dim)
        if clean_actions.ndim != 3 or tuple(clean_actions.shape[1:]) != expected:
            raise ValueError(
                f"clean_actions must have shape [B,{expected}], got {tuple(clean_actions.shape)}"
            )
        if source_noise is None:
            source_noise = torch.randn_like(clean_actions)
        if source_noise.shape != clean_actions.shape:
            raise ValueError("source_noise and clean_actions must have identical shapes")
        if time is None:
            time = torch.rand(clean_actions.shape[0], device=clean_actions.device)
        time = time.reshape(-1).to(device=clean_actions.device, dtype=clean_actions.dtype)
        if time.shape[0] != clean_actions.shape[0] or bool(((time < 0) | (time > 1)).any()):
            raise ValueError("Flow time must have one value per item in [0,1]")
        noisy = (1.0 - time[:, None, None]) * source_noise + time[:, None, None] * clean_actions
        target = clean_actions - source_noise
        prediction = self(observations, noisy, time)
        return torch.nn.functional.mse_loss(prediction, target)

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Mapping[str, Tensor],
        *,
        fingerprints: Sequence[str] | None = None,
        chunk_index: int = 0,
        source_noise: Tensor | None = None,
        base_seed: int | None = None,
    ) -> Tensor:
        batch_size = observations[STATE_KEY].shape[0]
        shape = (batch_size, self.config.horizon, self.config.action_dim)
        if source_noise is None:
            if fingerprints is None:
                seed = int(self.config.seed if base_seed is None else base_seed) + chunk_index
                generator = torch.Generator(device="cpu").manual_seed(seed)
                source_noise = torch.randn(shape, generator=generator, dtype=torch.float32)
            else:
                if len(fingerprints) != batch_size:
                    raise ValueError("One state fingerprint is required per observation batch item")
                rows = []
                for fingerprint in fingerprints:
                    seed = derive_flow_seed(
                        str(fingerprint),
                        chunk_index,
                        base_seed=self.config.seed if base_seed is None else base_seed,
                    )
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    rows.append(torch.randn(shape[1:], generator=generator, dtype=torch.float32))
                source_noise = torch.stack(rows, dim=0)
        if tuple(source_noise.shape) != shape:
            raise ValueError(
                f"source_noise must have shape {shape}, got {tuple(source_noise.shape)}"
            )
        device = observations[STATE_KEY].device
        state_dtype = next(self.parameters()).dtype
        current = source_noise.to(device=device, dtype=state_dtype)
        step = 1.0 / self.config.integration_steps
        for index in range(self.config.integration_steps):
            time = torch.full(
                (batch_size,),
                index / self.config.integration_steps,
                device=device,
                dtype=state_dtype,
            )
            velocity = self(observations, current, time)
            current = current + step * velocity
        return current


class ActionOnlyFlowPolicy:
    """Inference wrapper with an explicit n_action_steps queue and clip accounting."""

    def __init__(
        self,
        model: ActionOnlyFlowModel,
        stats: NormalizationStats,
        *,
        n_action_steps: int = 4,
    ) -> None:
        if n_action_steps not in (4, 8):
            raise ValueError("M2 Action-only Flow evaluates only n_action_steps=4 or 8")
        self.model = model
        self.stats = stats
        self.n_action_steps = n_action_steps
        self._queue: deque[Tensor] = deque()
        self._chunk_index = 0
        self.clip_elements = 0
        self.total_elements = 0

    @property
    def chunk_index(self) -> int:
        return self._chunk_index

    def reset(self) -> None:
        self._queue.clear()
        self._chunk_index = 0
        self.clip_elements = 0
        self.total_elements = 0

    def normalize_observation(self, observations: Mapping[str, Tensor]) -> dict[str, Tensor]:
        missing = [key for key in (*IMAGE_KEYS, STATE_KEY) if key not in observations]
        if missing:
            raise KeyError(f"Missing Action-only Flow observations: {missing}")
        result = dict(observations)
        result[BASE_IMAGE_KEY] = self.stats.normalize_images(
            observations[BASE_IMAGE_KEY].to(dtype=torch.float32)
        )
        result[HAND_IMAGE_KEY] = self.stats.normalize_images(
            observations[HAND_IMAGE_KEY].to(dtype=torch.float32)
        )
        result[STATE_KEY] = self.stats.normalize_state(
            observations[STATE_KEY].to(dtype=torch.float32)
        )
        return result

    @torch.no_grad()
    def predict_action_chunk(
        self,
        observations: Mapping[str, Tensor],
        *,
        fingerprints: Sequence[str] | None = None,
        source_noise: Tensor | None = None,
    ) -> Tensor:
        normalized = self.normalize_observation(observations)
        return self.model.sample_actions(
            normalized,
            fingerprints=fingerprints,
            chunk_index=self._chunk_index,
            source_noise=source_noise,
        )

    @torch.no_grad()
    def select_action(
        self,
        observations: Mapping[str, Tensor],
        *,
        fingerprints: Sequence[str] | None = None,
        return_normalized: bool = False,
    ) -> Tensor:
        if not self._queue:
            chunk = self.predict_action_chunk(observations, fingerprints=fingerprints)
            self._queue.extend(chunk[:, : self.n_action_steps].unbind(dim=1))
            self._chunk_index += 1
        action = self._queue.popleft()
        if return_normalized:
            return action
        environment_action, counts = self.stats.denormalize_action(action)
        self.clip_elements += counts["clipped_elements"]
        self.total_elements += counts["total_elements"]
        return environment_action


def save_action_flow_checkpoint(
    path: str | Path,
    *,
    model: ActionOnlyFlowModel,
    stats: NormalizationStats,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a self-contained, ordinary file checkpoint; no mutable latest link."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "step": int(step),
        "model_config": model.config.to_dict(),
        "model": model.state_dict(),
        "normalization": stats.to_dict(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    temporary = target.with_name(f".{target.name}.incomplete")
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    temporary.replace(target)


def load_action_flow_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    n_action_steps: int = 4,
) -> tuple[ActionOnlyFlowPolicy, dict[str, Any]]:
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.is_symlink():
        raise FileNotFoundError(f"Action Flow checkpoint must be a regular file: {target}")
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported Action Flow checkpoint: {target}")
    model = ActionOnlyFlowModel(ActionFlowConfig.from_mapping(payload["model_config"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    model.eval()
    stats = NormalizationStats.from_mapping(payload["normalization"]).to(device)
    policy = ActionOnlyFlowPolicy(model, stats, n_action_steps=n_action_steps)
    return policy, dict(payload)


__all__ = [
    "ACTION_KEY",
    "BASE_IMAGE_KEY",
    "HAND_IMAGE_KEY",
    "STATE_KEY",
    "ActionFlowConfig",
    "ActionOnlyFlowConfig",
    "ActionOnlyFlowModel",
    "ActionOnlyFlowPolicy",
    "NormalizationStats",
    "derive_flow_seed",
    "load_action_flow_checkpoint",
    "save_action_flow_checkpoint",
]
