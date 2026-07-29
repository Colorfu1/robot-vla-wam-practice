"""Lightweight, interpretable geometry for the PushT E0 demonstration audit."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

WORKSPACE_SIZE = 512.0
CONTACT_DISTANCE = 20.0
MIN_TRANSLATION_FOR_MODE = 12.0
ROTATION_FIRST_THRESHOLD_DEG = 15.0


@dataclass(frozen=True)
class BlockPose:
    """Image-derived pose of the moving gray-blue T block."""

    valid: bool
    position_workspace: tuple[float, float]
    position_image: tuple[float, float]
    angle_rad: float
    mask_area: int
    confidence: float
    agent_surface_distance: float


def _invalid_pose() -> BlockPose:
    return BlockPose(
        valid=False,
        position_workspace=(math.nan, math.nan),
        position_image=(math.nan, math.nan),
        angle_rad=math.nan,
        mask_area=0,
        confidence=0.0,
        agent_surface_distance=math.nan,
    )


def moving_block_mask(image: np.ndarray) -> np.ndarray:
    """Return a codec-tolerant mask for PushT's moving gray-blue block."""

    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"Expected an HWC RGB image, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        value = np.asarray(np.clip(value, 0.0, 1.0) * 255.0, dtype=np.uint8)
    rgb = value[..., :3].astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (
        (red >= 75)
        & (red <= 210)
        & ((green - red) >= 3)
        & ((green - red) <= 60)
        & ((blue - green) >= 3)
        & ((blue - green) <= 65)
        & ((blue - red) >= 12)
        & ((blue - red) <= 105)
    )
    margin = max(1, round(min(value.shape[:2]) * 4 / 96))
    mask[:margin] = False
    mask[-margin:] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    return mask


def extract_moving_block_pose(
    image: np.ndarray,
    *,
    agent_position: Sequence[float] | np.ndarray,
    workspace_size: float = WORKSPACE_SIZE,
) -> BlockPose:
    """Recover block center/orientation and agent-to-surface distance from one frame."""

    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError(f"Expected an HWC image, got {value.shape}")
    height, width = value.shape[:2]
    labels, _ = ndimage.label(moving_block_mask(value))
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return _invalid_pose()
    counts[0] = 0
    component_label = int(np.argmax(counts))
    area = int(counts[component_label])
    area_scale = height * width / float(96 * 96)
    if area < round(90 * area_scale) or area > round(400 * area_scale):
        return _invalid_pose()

    rows, columns = np.nonzero(labels == component_label)
    points_image = np.column_stack((columns, rows)).astype(np.float64)
    scale = np.asarray([workspace_size / width, workspace_size / height], dtype=np.float64)
    points_workspace = (points_image + 0.5) * scale
    center_image = points_image.mean(axis=0) + 0.5
    center_workspace = points_workspace.mean(axis=0)

    centered = points_workspace - center_workspace
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    perpendicular = np.asarray([-direction[1], direction[0]])
    along = centered @ direction
    across = np.abs(centered @ perpendicular)
    endpoint_cutoff = float(np.quantile(np.abs(along), 0.7))
    positive = across[along >= endpoint_cutoff]
    negative = across[along <= -endpoint_cutoff]
    positive_width = float(positive.mean()) if positive.size else 0.0
    negative_width = float(negative.mean()) if negative.size else 0.0
    if positive_width < negative_width:
        direction = -direction
    angle = float(math.atan2(direction[1], direction[0]))

    agent = np.asarray(agent_position, dtype=np.float64).reshape(-1)
    if agent.shape != (2,) or not np.isfinite(agent).all():
        raise ValueError(f"agent_position must contain two finite values, got {agent_position}")
    surface_distance = float(np.linalg.norm(points_workspace - agent, axis=1).min())
    expected_area = 235.0 * area_scale
    confidence = max(0.0, 1.0 - abs(area - expected_area) / (170.0 * area_scale))

    return BlockPose(
        valid=True,
        position_workspace=(float(center_workspace[0]), float(center_workspace[1])),
        position_image=(float(center_image[0]), float(center_image[1])),
        angle_rad=angle,
        mask_area=area,
        confidence=float(confidence),
        agent_surface_distance=surface_distance,
    )


def _contact_runs(contact: np.ndarray) -> tuple[np.ndarray, int]:
    if contact.ndim != 1:
        raise ValueError("contact must be one-dimensional")
    if contact.size == 0:
        return contact, 0
    smoothed = ndimage.binary_closing(contact, structure=np.ones(5, dtype=bool))
    labels, count = ndimage.label(smoothed)
    if count:
        sizes = np.bincount(labels)
        for label in range(1, count + 1):
            if sizes[label] < 3:
                smoothed[labels == label] = False
        _, count = ndimage.label(smoothed)
    return np.asarray(smoothed, dtype=bool), int(count)


def _approach_side(agent: np.ndarray, block: np.ndarray) -> str:
    offset = agent - block
    if abs(float(offset[0])) >= abs(float(offset[1])):
        return "left" if offset[0] < 0 else "right"
    return "top" if offset[1] < 0 else "bottom"


def _primary_motion(displacement: np.ndarray) -> str:
    if float(np.linalg.norm(displacement)) < MIN_TRANSLATION_FOR_MODE:
        return "static"
    if abs(float(displacement[0])) >= abs(float(displacement[1])):
        return "left" if displacement[0] < 0 else "right"
    return "up" if displacement[1] < 0 else "down"


def classify_demonstration(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one full demonstration without claiming task success."""

    if not records:
        raise ValueError("Cannot classify an empty demonstration")
    valid = np.asarray([bool(record["block_pose_valid"]) for record in records], dtype=bool)
    coverage = float(valid.mean())
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return {
            "geometry_coverage": coverage,
            "first_contact_frame": None,
            "contact_segments": 0,
            "approach_side": "unknown",
            "primary_block_motion": "unknown",
            "net_block_translation": math.nan,
            "net_block_rotation_deg": math.nan,
            "total_block_rotation_deg": math.nan,
            "rotation_first": False,
            "recontact": False,
            "strategy_label": "unclassified",
        }

    agent = np.asarray([record["agent_position"] for record in records], dtype=np.float64)
    block = np.asarray([record["block_position"] for record in records], dtype=np.float64)
    distances = np.asarray(
        [record["agent_block_surface_distance"] for record in records],
        dtype=np.float64,
    )
    raw_contact = valid & np.isfinite(distances) & (distances <= CONTACT_DISTANCE)
    contact, contact_segments = _contact_runs(raw_contact)
    contact_indices = np.flatnonzero(contact)
    first_contact = int(contact_indices[0]) if contact_indices.size else None

    first_valid, last_valid = int(valid_indices[0]), int(valid_indices[-1])
    displacement = block[last_valid] - block[first_valid]
    net_translation = float(np.linalg.norm(displacement))
    approach = (
        _approach_side(agent[first_contact], block[first_contact])
        if first_contact is not None and valid[first_contact]
        else "no_contact"
    )

    valid_angles = np.asarray(
        [float(records[index]["block_angle_rad"]) for index in valid_indices],
        dtype=np.float64,
    )
    unwrapped = np.unwrap(valid_angles)
    net_rotation = float(math.degrees(unwrapped[-1] - unwrapped[0]))
    total_rotation = float(math.degrees(np.abs(np.diff(unwrapped)).sum()))

    positions = block[valid_indices]
    distance_from_start = np.linalg.norm(positions - positions[0], axis=1)
    translated = np.flatnonzero(distance_from_start >= 25.0)
    prefix_end = int(translated[0]) if translated.size else len(valid_indices) - 1
    rotation_before_translation = abs(math.degrees(unwrapped[prefix_end] - unwrapped[0]))
    rotation_first = rotation_before_translation >= ROTATION_FIRST_THRESHOLD_DEG
    recontact = contact_segments >= 2
    motion = _primary_motion(displacement)
    strategy = f"{approach}_{'rotation_first' if rotation_first else 'direct'}"

    return {
        "geometry_coverage": coverage,
        "first_contact_frame": first_contact,
        "contact_segments": contact_segments,
        "approach_side": approach,
        "primary_block_motion": motion,
        "net_block_translation": net_translation,
        "net_block_rotation_deg": net_rotation,
        "total_block_rotation_deg": total_rotation,
        "rotation_first": rotation_first,
        "recontact": recontact,
        "strategy_label": strategy,
    }


def representative_episode_order(
    episode_summaries: Sequence[Mapping[str, Any]],
    *,
    eligible_episodes: Sequence[int],
    count: int,
) -> tuple[int, ...]:
    """Round-robin strategy groups to select diverse, deterministic E1 anchors."""

    if count <= 0:
        raise ValueError("count must be positive")
    eligible = {int(value) for value in eligible_episodes}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in episode_summaries:
        episode = int(summary["episode_index"])
        if episode in eligible:
            groups[str(summary["strategy_label"])].append(summary)
    available = sum(len(values) for values in groups.values())
    if count > available:
        raise ValueError(f"Requested {count} representatives from {available} eligible episodes")
    for values in groups.values():
        values.sort(
            key=lambda item: (
                -float(item["geometry_coverage"]),
                -float(item["max_reward"]),
                int(item["episode_index"]),
            )
        )
    labels = sorted(groups, key=lambda label: (-len(groups[label]), label))
    selected: list[int] = []
    offset = 0
    while len(selected) < count:
        added = False
        for label in labels:
            values = groups[label]
            if offset < len(values):
                selected.append(int(values[offset]["episode_index"]))
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    if len(selected) != count:
        raise RuntimeError("Representative episode selection ended unexpectedly")
    return tuple(selected)


def strategy_groups(
    episode_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministic global-strategy groups without implying conditional modes."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for summary in episode_summaries:
        grouped[str(summary["strategy_label"])].append(int(summary["episode_index"]))
    return [
        {
            "group_id": f"strategy_{ordinal:02d}",
            "group_kind": "global_demonstration_strategy",
            "strategy_label": label,
            "num_episodes": len(grouped[label]),
            "episode_indices": sorted(grouped[label]),
            "strict_conditional_multimodality_proven": False,
        }
        for ordinal, label in enumerate(sorted(grouped))
    ]


__all__ = [
    "CONTACT_DISTANCE",
    "WORKSPACE_SIZE",
    "BlockPose",
    "classify_demonstration",
    "extract_moving_block_pose",
    "moving_block_mask",
    "representative_episode_order",
    "strategy_groups",
]
