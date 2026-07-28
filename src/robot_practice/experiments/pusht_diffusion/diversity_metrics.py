"""E1 action diversity and E2 success-conditioned trajectory metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import PUSHT_WORKSPACE_SIZE


def _as_finite_array(value: np.ndarray | Sequence[float], *, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got shape {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Metric inputs must be finite")
    return result


def pairwise_rms_distance(trajectories: np.ndarray) -> np.ndarray:
    values = _as_finite_array(trajectories, ndim=3)
    count = values.shape[0]
    result = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        delta = values[left + 1 :] - values[left]
        distances = np.sqrt(np.mean(delta**2, axis=(1, 2)))
        result[left, left + 1 :] = distances
        result[left + 1 :, left] = distances
    return result


def pairwise_endpoint_distance(trajectories: np.ndarray) -> np.ndarray:
    values = _as_finite_array(trajectories, ndim=3)
    endpoints = values[:, -1]
    delta = endpoints[:, None, :] - endpoints[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def upper_triangle_mean(distance_matrix: np.ndarray) -> float:
    matrix = _as_finite_array(distance_matrix, ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be square")
    if matrix.shape[0] < 2:
        return 0.0
    indices = np.triu_indices(matrix.shape[0], k=1)
    return float(np.mean(matrix[indices]))


def _pairwise_direction_angle(actions: np.ndarray, agent_position: np.ndarray) -> float:
    relative = actions[:, :5] - agent_position[None, None, :]
    directions = relative.mean(axis=1)
    norms = np.linalg.norm(directions, axis=-1)
    valid = norms > 1e-12
    if np.count_nonzero(valid) < 2:
        return 0.0
    unit = directions[valid] / norms[valid, None]
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    angles = np.arccos(cosine)
    return upper_triangle_mean(angles)


@dataclass(frozen=True)
class ActionDiversitySummary:
    num_action_samples: int
    trajectory_pairwise_rms: float
    endpoint_pairwise_distance: float
    first_five_direction_angle_rad: float
    mean_target_path_length: float
    mean_target_total_variation: float
    in_bounds_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "num_action_samples": self.num_action_samples,
            "trajectory_pairwise_rms": self.trajectory_pairwise_rms,
            "endpoint_pairwise_distance": self.endpoint_pairwise_distance,
            "first_five_direction_angle_rad": self.first_five_direction_angle_rad,
            "mean_target_path_length": self.mean_target_path_length,
            "mean_target_total_variation": self.mean_target_total_variation,
            "in_bounds_ratio": self.in_bounds_ratio,
        }


def summarize_action_diversity(
    actions: np.ndarray,
    *,
    agent_position: np.ndarray,
    lower_bound: float = 0.0,
    upper_bound: float = PUSHT_WORKSPACE_SIZE,
) -> ActionDiversitySummary:
    values = _as_finite_array(actions, ndim=3)
    if values.shape[-1] != 2 or values.shape[1] == 0:
        raise ValueError("Actions must have shape [samples,steps,2]")
    agent = _as_finite_array(agent_position, ndim=1)
    if agent.shape != (2,):
        raise ValueError("agent_position must have shape [2]")
    deltas = np.diff(values, axis=1)
    step_lengths = np.linalg.norm(deltas, axis=-1)
    target_path_length = step_lengths.sum(axis=1)
    target_total_variation = np.abs(deltas).sum(axis=(1, 2))
    in_bounds = (values >= lower_bound) & (values <= upper_bound)
    return ActionDiversitySummary(
        num_action_samples=values.shape[0],
        trajectory_pairwise_rms=upper_triangle_mean(pairwise_rms_distance(values)),
        endpoint_pairwise_distance=upper_triangle_mean(pairwise_endpoint_distance(values)),
        first_five_direction_angle_rad=_pairwise_direction_angle(values, agent),
        mean_target_path_length=float(np.mean(target_path_length)),
        mean_target_total_variation=float(np.mean(target_total_variation)),
        in_bounds_ratio=float(np.mean(np.all(in_bounds, axis=-1))),
    )


def resample_agent_block_trajectory(
    agent_path: np.ndarray,
    block_path: np.ndarray,
    *,
    num_points: int = 100,
) -> np.ndarray:
    agent = _as_finite_array(agent_path, ndim=2)
    block = _as_finite_array(block_path, ndim=2)
    if agent.shape[0] != block.shape[0] or agent.shape[0] == 0:
        raise ValueError("Agent and block paths must have the same nonzero length")
    if agent.shape[1] != 2 or block.shape[1] != 3:
        raise ValueError("Expected agent [steps,2] and block [steps,3]")
    if num_points < 2:
        raise ValueError("num_points must be at least two")

    source_progress = np.linspace(0.0, 1.0, agent.shape[0])
    target_progress = np.linspace(0.0, 1.0, num_points)
    combined = np.empty((num_points, 5), dtype=np.float64)
    for dimension in range(2):
        combined[:, dimension] = np.interp(
            target_progress,
            source_progress,
            agent[:, dimension],
        )
    for dimension in range(2):
        combined[:, 2 + dimension] = np.interp(
            target_progress,
            source_progress,
            block[:, dimension],
        )
    combined[:, 4] = np.interp(
        target_progress,
        source_progress,
        np.unwrap(block[:, 2]),
    )
    return combined


def _wrapped_angle_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(left - right), np.cos(left - right))


def trajectory_mode_distance(left: np.ndarray, right: np.ndarray) -> float:
    first = _as_finite_array(left, ndim=2)
    second = _as_finite_array(right, ndim=2)
    if first.shape != second.shape or first.shape[1] != 5:
        raise ValueError("Resampled trajectories must share shape [points,5]")
    delta = np.empty_like(first)
    delta[:, :4] = (first[:, :4] - second[:, :4]) / PUSHT_WORKSPACE_SIZE
    delta[:, 4] = _wrapped_angle_difference(first[:, 4], second[:, 4]) / np.pi
    return float(np.sqrt(np.mean(delta**2)))


def trajectory_distance_matrix(trajectories: Sequence[np.ndarray]) -> np.ndarray:
    count = len(trajectories)
    matrix = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            distance = trajectory_mode_distance(trajectories[left], trajectories[right])
            matrix[left, right] = distance
            matrix[right, left] = distance
    return matrix


def complete_linkage_mode_labels(
    distance_matrix: np.ndarray,
    *,
    threshold: float = 0.05,
) -> np.ndarray:
    matrix = _as_finite_array(distance_matrix, ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Distance matrix must be square")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if matrix.shape[0] == 1:
        return np.zeros((1,), dtype=np.int64)

    from sklearn.cluster import AgglomerativeClustering

    kwargs = {
        "n_clusters": None,
        "distance_threshold": threshold,
        "linkage": "complete",
    }
    try:
        estimator = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:  # pragma: no cover - compatibility with older scikit-learn
        estimator = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return estimator.fit_predict(matrix).astype(np.int64)
