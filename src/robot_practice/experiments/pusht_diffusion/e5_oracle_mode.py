"""Dataset-first-frame reconstruction and oracle-mode E5 rollout helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gym_pusht.envs.pusht import PushTEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.utils import get_device_from_parameters

from robot_practice.policies.mode_bc_chunk.modeling_mode_bc_chunk import ModeBCChunkPolicy
from robot_practice.policies.mode_diffusion.modeling_mode_diffusion import ModeDiffusionPolicy

from .config import DATASET_REPO_ID, DATASET_REVISION
from .e0_geometry import extract_moving_block_pose, moving_block_mask
from .e5_dataset import PrototypeSidecar, load_prototype_sidecar
from .e5_runtime import FixedPrototypePreprocessor
from .rerun_visualization import image_to_hwc_uint8
from .rollout import FULL_DDPM_SAMPLING, RolloutResult, run_rollout

PUSHT_T_GEOMETRIC_CENTROID_LOCAL = np.asarray([0.0, 285.0 / 7.0], dtype=np.float64)
EXTRACTOR_TO_BODY_ANGLE_OFFSET_RAD = math.pi / 2.0
MIN_RECONSTRUCTION_MASK_IOU = 0.75
MAX_RECONSTRUCTION_CENTER_ERROR_PX = 2.0
MAX_RECONSTRUCTION_ANGLE_ERROR_DEG = 5.0
MAX_RECONSTRUCTION_AGENT_ERROR = 1e-6


def wrapped_angle_difference(first: float, second: float) -> float:
    """Return the signed shortest angular difference ``first - second``."""

    return float((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def body_pose_from_geometric_pose(
    center_workspace: Sequence[float],
    extractor_angle_rad: float,
) -> tuple[tuple[float, float], float]:
    """Convert image-derived T geometry to the PushT pymunk body pose."""

    center = np.asarray(center_workspace, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError(f"center_workspace must contain two finite values, got {center_workspace}")
    if not math.isfinite(extractor_angle_rad):
        raise ValueError("extractor_angle_rad must be finite")
    body_angle = wrapped_angle_difference(
        float(extractor_angle_rad) + EXTRACTOR_TO_BODY_ANGLE_OFFSET_RAD,
        0.0,
    )
    cosine = math.cos(body_angle)
    sine = math.sin(body_angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    body_origin = center - rotation @ PUSHT_T_GEOMETRIC_CENTROID_LOCAL
    return (float(body_origin[0]), float(body_origin[1])), body_angle


@dataclass(frozen=True)
class OracleModeEpisode:
    episode_index: int
    split: str
    prototype_id: int
    classifiable: bool
    distance_to_medoid: float | None
    agent_position: tuple[float, float]
    block_center_workspace: tuple[float, float]
    extractor_angle_rad: float
    block_body_origin: tuple[float, float]
    block_body_angle_rad: float
    reconstruction_angle_adjustment_deg: float
    reconstruction_center_adjustment_px: tuple[float, float]
    first_frame: np.ndarray


@dataclass(frozen=True)
class ReconstructionMetric:
    episode_index: int
    split: str
    prototype_id: int
    classifiable: bool
    mask_iou: float
    center_error_px: float
    angle_error_deg: float
    agent_error: float
    angle_adjustment_deg: float
    center_adjustment_x_px: float
    center_adjustment_y_px: float
    passed: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "split": self.split,
            "prototype_id": self.prototype_id,
            "gt_mode_classifiable": self.classifiable,
            "mask_iou": self.mask_iou,
            "center_error_px": self.center_error_px,
            "angle_error_deg": self.angle_error_deg,
            "agent_error": self.agent_error,
            "angle_adjustment_deg": self.angle_adjustment_deg,
            "center_adjustment_x_px": self.center_adjustment_x_px,
            "center_adjustment_y_px": self.center_adjustment_y_px,
            "passed": self.passed,
        }


class DatasetFirstFramePushTEnv(PushTEnv):
    """PushT environment whose initial state is reconstructed from a dataset frame."""

    def __init__(
        self,
        *,
        agent_position: Sequence[float],
        block_body_origin: Sequence[float],
        block_body_angle_rad: float,
        **kwargs: Any,
    ) -> None:
        self._dataset_agent_position = np.asarray(agent_position, dtype=np.float64)
        self._dataset_block_body_origin = np.asarray(block_body_origin, dtype=np.float64)
        self._dataset_block_body_angle_rad = float(block_body_angle_rad)
        if (
            self._dataset_agent_position.shape != (2,)
            or self._dataset_block_body_origin.shape != (2,)
            or not np.isfinite(self._dataset_agent_position).all()
            or not np.isfinite(self._dataset_block_body_origin).all()
            or not math.isfinite(self._dataset_block_body_angle_rad)
        ):
            raise ValueError("Dataset-first-frame reset pose must be finite")
        super().__init__(**kwargs)

    def _set_state(self, state: np.ndarray) -> None:
        del state
        self.agent.position = tuple(self._dataset_agent_position)
        self.agent.velocity = (0.0, 0.0)
        self.agent.angular_velocity = 0.0
        # Pymunk rotates a body around its center of gravity. Set angle before the
        # body origin so the reconstructed geometric center is not displaced.
        self.block.angle = self._dataset_block_body_angle_rad
        self.block.position = tuple(self._dataset_block_body_origin)
        self.block.velocity = (0.0, 0.0)
        self.block.angular_velocity = 0.0
        self.space.reindex_shapes_for_body(self.agent)
        self.space.reindex_shapes_for_body(self.block)
        self.space.step(self.dt)


def make_dataset_first_frame_env(
    episode: OracleModeEpisode,
    *,
    episode_length: int = 300,
    render: bool = True,
) -> gym.Env:
    del render
    environment = DatasetFirstFramePushTEnv(
        agent_position=episode.agent_position,
        block_body_origin=episode.block_body_origin,
        block_body_angle_rad=episode.block_body_angle_rad,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=96,
        observation_height=96,
        visualization_width=384,
        visualization_height=384,
    )
    return gym.wrappers.TimeLimit(environment, max_episode_steps=episode_length)


def _frame_lookup(dataset: LeRobotDataset) -> dict[int, int]:
    dataset._ensure_hf_dataset_loaded()
    result: dict[int, int] = {}
    episode_frames: dict[int, list[tuple[int, int]]] = defaultdict(list)
    episodes = dataset.hf_dataset["episode_index"]
    frames = dataset.hf_dataset["frame_index"]
    for index, (raw_episode, raw_frame) in enumerate(zip(episodes, frames, strict=True)):
        episode = int(raw_episode.item() if hasattr(raw_episode, "item") else raw_episode)
        frame = int(raw_frame.item() if hasattr(raw_frame, "item") else raw_frame)
        episode_frames[episode].append((frame, index))
    for episode, candidates in episode_frames.items():
        result[episode] = min(candidates)[1]
    return result


def load_oracle_mode_episodes(
    *,
    dataset_root: str | Path,
    prototype_manifest: str | Path,
    episode_indices: Sequence[int],
    split_name: str,
) -> tuple[OracleModeEpisode, ...]:
    """Load first frames and full-trajectory oracle IDs for one committed split."""

    requested = tuple(int(value) for value in episode_indices)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("episode_indices must be a non-empty unique sequence")
    sidecar: PrototypeSidecar = load_prototype_sidecar(prototype_manifest)
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=Path(dataset_root).expanduser().resolve(),
        episodes=list(requested),
        revision=DATASET_REVISION,
    )
    lookup = _frame_lookup(dataset)
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"Dataset is missing requested first frames: {missing}")

    episodes: list[OracleModeEpisode] = []
    for episode_index in requested:
        try:
            assignment = sidecar.assignments[episode_index]
        except KeyError as exc:
            raise KeyError(f"Prototype sidecar has no episode {episode_index}") from exc
        if assignment.split != split_name:
            raise ValueError(
                f"Episode {episode_index} sidecar split is {assignment.split!r}, "
                f"expected {split_name!r}"
            )
        if assignment.training_condition_id is not None:
            raise ValueError(
                f"{split_name} episode {episode_index} unexpectedly has a training condition"
            )
        if assignment.prototype_id is None:
            raise ValueError(f"Episode {episode_index} has no oracle prototype ID")
        if not 0 <= assignment.prototype_id < sidecar.selected_k:
            raise ValueError(
                f"Episode {episode_index} oracle prototype {assignment.prototype_id} "
                f"is outside P0..P{sidecar.selected_k - 1}"
            )

        item = dataset[lookup[episode_index]]
        image = image_to_hwc_uint8(item["observation.image"])
        raw_agent = item["observation.state"]
        agent = np.asarray(
            raw_agent.detach().cpu().numpy() if hasattr(raw_agent, "detach") else raw_agent,
            dtype=np.float64,
        ).reshape(-1)
        if agent.shape != (2,) or not np.isfinite(agent).all():
            raise ValueError(f"Episode {episode_index} has invalid first-frame agent state")
        pose = extract_moving_block_pose(image, agent_position=agent)
        if not pose.valid:
            raise ValueError(
                f"Episode {episode_index} first-frame moving-block pose is not reconstructable"
            )
        body_origin, body_angle = body_pose_from_geometric_pose(
            pose.position_workspace,
            pose.angle_rad,
        )
        episode = OracleModeEpisode(
            episode_index=episode_index,
            split=split_name,
            prototype_id=assignment.prototype_id,
            classifiable=assignment.classifiable,
            distance_to_medoid=assignment.distance_to_medoid,
            agent_position=(float(agent[0]), float(agent[1])),
            block_center_workspace=pose.position_workspace,
            extractor_angle_rad=pose.angle_rad,
            block_body_origin=body_origin,
            block_body_angle_rad=body_angle,
            reconstruction_angle_adjustment_deg=0.0,
            reconstruction_center_adjustment_px=(0.0, 0.0),
            first_frame=image,
        )
        episodes.append(refine_reconstruction_pose(episode))
    return tuple(episodes)


def reconstruction_metric(episode: OracleModeEpisode) -> ReconstructionMetric:
    environment = make_dataset_first_frame_env(episode, episode_length=1, render=False)
    try:
        observation, _ = environment.reset(seed=episode.episode_index)
        rebuilt = image_to_hwc_uint8(observation["pixels"])
        rebuilt_agent = np.asarray(observation["agent_pos"], dtype=np.float64)
    finally:
        environment.close()

    source_mask = moving_block_mask(episode.first_frame)
    rebuilt_mask = moving_block_mask(rebuilt)
    union = int(np.logical_or(source_mask, rebuilt_mask).sum())
    intersection = int(np.logical_and(source_mask, rebuilt_mask).sum())
    mask_iou = float(intersection / union) if union else 0.0
    rebuilt_pose = extract_moving_block_pose(rebuilt, agent_position=rebuilt_agent)
    if rebuilt_pose.valid:
        center_error_px = float(
            np.linalg.norm(
                np.asarray(rebuilt_pose.position_image)
                - np.asarray(episode.block_center_workspace) * (96.0 / 512.0)
            )
        )
        angle_error_deg = abs(
            math.degrees(
                wrapped_angle_difference(
                    rebuilt_pose.angle_rad,
                    episode.extractor_angle_rad,
                )
            )
        )
    else:
        center_error_px = math.inf
        angle_error_deg = math.inf
    agent_error = float(
        np.linalg.norm(rebuilt_agent - np.asarray(episode.agent_position, dtype=np.float64))
    )
    passed = bool(
        rebuilt_pose.valid
        and mask_iou >= MIN_RECONSTRUCTION_MASK_IOU
        and center_error_px <= MAX_RECONSTRUCTION_CENTER_ERROR_PX
        and angle_error_deg <= MAX_RECONSTRUCTION_ANGLE_ERROR_DEG
        and agent_error <= MAX_RECONSTRUCTION_AGENT_ERROR
    )
    return ReconstructionMetric(
        episode_index=episode.episode_index,
        split=episode.split,
        prototype_id=episode.prototype_id,
        classifiable=episode.classifiable,
        mask_iou=mask_iou,
        center_error_px=center_error_px,
        angle_error_deg=angle_error_deg,
        agent_error=agent_error,
        angle_adjustment_deg=episode.reconstruction_angle_adjustment_deg,
        center_adjustment_x_px=episode.reconstruction_center_adjustment_px[0],
        center_adjustment_y_px=episode.reconstruction_center_adjustment_px[1],
        passed=passed,
    )


def _adjust_reconstruction_pose(
    episode: OracleModeEpisode,
    *,
    angle_adjustment_deg: float,
    center_adjustment_px: tuple[float, float],
) -> OracleModeEpisode:
    angle = episode.block_body_angle_rad + math.radians(angle_adjustment_deg)
    center = np.asarray(episode.block_center_workspace, dtype=np.float64) + (
        np.asarray(center_adjustment_px, dtype=np.float64) * (512.0 / 96.0)
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    origin = center - rotation @ PUSHT_T_GEOMETRIC_CENTROID_LOCAL
    return replace(
        episode,
        block_body_origin=(float(origin[0]), float(origin[1])),
        block_body_angle_rad=angle,
        reconstruction_angle_adjustment_deg=float(angle_adjustment_deg),
        reconstruction_center_adjustment_px=(
            float(center_adjustment_px[0]),
            float(center_adjustment_px[1]),
        ),
    )


def refine_reconstruction_pose(episode: OracleModeEpisode) -> OracleModeEpisode:
    """Resolve rendering/occlusion ambiguity using the first frame only."""

    initial = reconstruction_metric(episode)
    if initial.passed:
        return episode

    # Codec quantization can bias the image-derived angle by several degrees.
    local_candidates: list[tuple[ReconstructionMetric, OracleModeEpisode]] = []
    for angle_delta in range(-20, 21, 2):
        candidate = _adjust_reconstruction_pose(
            episode,
            angle_adjustment_deg=float(angle_delta),
            center_adjustment_px=(0.0, 0.0),
        )
        metric = reconstruction_metric(candidate)
        if metric.passed:
            local_candidates.append((metric, candidate))
    if local_candidates:
        return max(local_candidates, key=lambda value: value[0].mask_iou)[1]

    # When the agent occludes part of the T, PCA can choose a perpendicular
    # orientation. Probe the other three shape quadrants, then permit at most
    # two observation pixels of center correction, matching the hard gate.
    orientation_candidates: list[tuple[ReconstructionMetric, OracleModeEpisode]] = []
    angle_deltas = {float(base + local) for base in (-90, 90, 180) for local in range(-12, 13, 2)}
    for angle_delta in sorted(angle_deltas):
        candidate = _adjust_reconstruction_pose(
            episode,
            angle_adjustment_deg=angle_delta,
            center_adjustment_px=(0.0, 0.0),
        )
        metric = reconstruction_metric(candidate)
        if metric.center_error_px <= 2.0 and metric.angle_error_deg <= 15.0:
            orientation_candidates.append((metric, candidate))
    orientation_candidates.sort(key=lambda value: value[0].mask_iou, reverse=True)

    refined_candidates: list[tuple[ReconstructionMetric, OracleModeEpisode]] = []
    for _, oriented in orientation_candidates[:6]:
        for horizontal in range(-2, 3):
            for vertical in range(-2, 3):
                candidate = _adjust_reconstruction_pose(
                    episode,
                    angle_adjustment_deg=oriented.reconstruction_angle_adjustment_deg,
                    center_adjustment_px=(float(horizontal), float(vertical)),
                )
                metric = reconstruction_metric(candidate)
                if metric.passed:
                    refined_candidates.append((metric, candidate))
    if refined_candidates:
        return max(refined_candidates, key=lambda value: value[0].mask_iou)[1]
    return episode


def preflight_reconstruction(
    episodes: Sequence[OracleModeEpisode],
) -> tuple[ReconstructionMetric, ...]:
    metrics = tuple(reconstruction_metric(episode) for episode in episodes)
    failures = [metric for metric in metrics if not metric.passed]
    if failures:
        details = ", ".join(
            (
                f"episode={metric.episode_index} IoU={metric.mask_iou:.3f} "
                f"center_px={metric.center_error_px:.3f} "
                f"angle_deg={metric.angle_error_deg:.3f} agent={metric.agent_error:.3g}"
            )
            for metric in failures
        )
        raise RuntimeError(f"Dataset-first-frame reconstruction preflight failed: {details}")
    return metrics


def validate_mode_policy(policy: Any, *, expected_model: str, selected_k: int) -> None:
    expected_type: type[Any]
    if expected_model == "mode_bc_chunk":
        expected_type = ModeBCChunkPolicy
    elif expected_model == "mode_diffusion":
        expected_type = ModeDiffusionPolicy
    else:
        raise ValueError(f"Unsupported E5 model: {expected_model}")
    if not isinstance(policy, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__} for {expected_model}, got {type(policy).__name__}"
        )
    if int(policy.config.num_prototypes) != selected_k:
        raise ValueError(
            f"Checkpoint uses K={policy.config.num_prototypes}, sidecar uses K={selected_k}"
        )


def run_oracle_mode_rollout(
    loaded: Any,
    *,
    episode: OracleModeEpisode,
    commanded_condition_id: int,
    selected_k: int,
    null_prototype_id: int,
    record_frames: bool,
    episode_length: int = 300,
    policy_seed: int = 0,
) -> RolloutResult:
    """Roll out once from a held-out episode state under one explicit mode condition."""

    valid_conditions = {*range(selected_k), null_prototype_id}
    if commanded_condition_id not in valid_conditions:
        raise ValueError(
            f"Commanded condition must be in P0..P{selected_k - 1} or NULL={null_prototype_id}"
        )
    condition_name = (
        "NULL" if commanded_condition_id == null_prototype_id else f"P{commanded_condition_id}"
    )

    device = get_device_from_parameters(loaded.policy)
    preprocessor = FixedPrototypePreprocessor(
        loaded.preprocessor,
        prototype_id=commanded_condition_id,
        num_prototypes=selected_k,
        device=device,
    )

    def environment_factory(*, episode_length: int, render: bool) -> gym.Env:
        return make_dataset_first_frame_env(
            episode,
            episode_length=episode_length,
            render=render,
        )

    return run_rollout(
        policy=loaded.policy,
        preprocessor=preprocessor,
        postprocessor=loaded.postprocessor,
        environment_seed=episode.episode_index,
        policy_seed=policy_seed,
        record_frames=record_frames,
        episode_length=episode_length,
        condition_id=(
            f"{episode.split}_episode_{episode.episode_index:06d}"
            f"_command_{condition_name}_gt_P{episode.prototype_id}"
        ),
        sampling_protocol=FULL_DDPM_SAMPLING,
        environment_factory=environment_factory,
    )


__all__ = [
    "DatasetFirstFramePushTEnv",
    "OracleModeEpisode",
    "ReconstructionMetric",
    "body_pose_from_geometric_pose",
    "load_oracle_mode_episodes",
    "make_dataset_first_frame_env",
    "preflight_reconstruction",
    "reconstruction_metric",
    "run_oracle_mode_rollout",
    "validate_mode_policy",
    "wrapped_angle_difference",
]
