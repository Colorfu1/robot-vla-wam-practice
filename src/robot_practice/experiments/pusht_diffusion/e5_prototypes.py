"""Fit train-only PushT E5 trajectory prototypes and write sidecar assignments."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import default_output_root, default_split_path, practice_root
from .dataset_split import EpisodeSplit, load_split
from .e0_geometry import (
    CONTACT_DISTANCE,
    MIN_TRANSLATION_FOR_MODE,
    ROTATION_FIRST_THRESHOLD_DEG,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .provenance import sha256_file, write_json_atomic

E5_SCHEMA_VERSION = 1
SKETCH_VERSION = "object-centric-events-v1"
DISTANCE_VERSION = "weighted-components-v1"
ROTATION_ARC_RADIUS = 64.0
TOKEN_COUNT = 8
K_CANDIDATES = (2, 3, 4)
INITIALIZATION_SEEDS = tuple(range(20260730, 20260740))
SUBSAMPLE_SEED = 20260740
SUBSAMPLE_FRACTION = 0.8
SUBSAMPLE_ITERATIONS = 100
MIN_CLUSTER_SIZE = 8
MIN_INITIALIZATION_ARI = 0.8
MIN_SUBSAMPLE_ARI = 0.7
MAX_KMEDOIDS_ITERATIONS = 300

COMPONENT_WEIGHTS = {
    "first_contact": 2.0,
    "approach": 1.5,
    "block_path": 1.0,
    "rotation": 1.0,
    "contact_sequence": 1.5,
    "edge": 0.25,
}


@dataclass(frozen=True)
class EpisodeSketch:
    """One event-aligned object-centric trajectory sketch."""

    episode_index: int
    split: str
    eligible: bool
    exclusion_reason: str | None
    token_frames: tuple[int, ...]
    agent_local: np.ndarray
    block_displacement: np.ndarray
    relative_angle: np.ndarray
    stable_contact: np.ndarray
    normalized_progress: np.ndarray
    contact_runs: int
    initial_observation_features: np.ndarray

    def __post_init__(self) -> None:
        if self.eligible:
            if len(self.token_frames) != TOKEN_COUNT:
                raise ValueError("Eligible sketches require eight token frames")
            expected = {
                "agent_local": (TOKEN_COUNT, 2),
                "block_displacement": (TOKEN_COUNT, 2),
                "relative_angle": (TOKEN_COUNT,),
                "stable_contact": (TOKEN_COUNT,),
                "normalized_progress": (TOKEN_COUNT,),
            }
            for name, shape in expected.items():
                if np.asarray(getattr(self, name)).shape != shape:
                    raise ValueError(f"{name} must have shape {shape}")

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "split": self.split,
            "eligible": self.eligible,
            "exclusion_reason": self.exclusion_reason,
            "token_frames": list(self.token_frames),
            "agent_local": np.asarray(self.agent_local).tolist(),
            "block_displacement": np.asarray(self.block_displacement).tolist(),
            "relative_angle": np.asarray(self.relative_angle).tolist(),
            "stable_contact": np.asarray(self.stable_contact, dtype=bool).tolist(),
            "normalized_progress": np.asarray(self.normalized_progress).tolist(),
            "contact_runs": self.contact_runs,
        }


def episode_sketch_from_record(record: Mapping[str, Any]) -> EpisodeSketch:
    """Restore a persisted eligible or excluded sketch."""

    eligible = bool(record["eligible"])
    return EpisodeSketch(
        episode_index=int(record["episode_index"]),
        split=str(record["split"]),
        eligible=eligible,
        exclusion_reason=record.get("exclusion_reason"),
        token_frames=tuple(int(value) for value in record.get("token_frames", ())),
        agent_local=np.asarray(record.get("agent_local", ()), dtype=np.float64).reshape((-1, 2)),
        block_displacement=np.asarray(
            record.get("block_displacement", ()),
            dtype=np.float64,
        ).reshape((-1, 2)),
        relative_angle=np.asarray(
            record.get("relative_angle", ()),
            dtype=np.float64,
        ),
        stable_contact=np.asarray(
            record.get("stable_contact", ()),
            dtype=bool,
        ),
        normalized_progress=np.asarray(
            record.get("normalized_progress", ()),
            dtype=np.float64,
        ),
        contact_runs=int(record.get("contact_runs", 0)),
        initial_observation_features=np.asarray(
            record.get("initial_observation_features", ()),
            dtype=np.float64,
        ),
    )


@dataclass(frozen=True)
class KMedoidsRun:
    """One deterministic K-medoids fit."""

    labels: np.ndarray
    medoid_indices: tuple[int, ...]
    objective: float
    seed: int
    iterations: int


@dataclass(frozen=True)
class PrototypeCandidate:
    """Full-data and subsample diagnostics for one candidate K."""

    k: int
    labels: np.ndarray
    medoid_indices: tuple[int, ...]
    objective: float
    silhouette: float
    cluster_sizes: tuple[int, ...]
    initialization_ari_median: float
    subsample_ari_median: float
    subsample_ari_q05: float
    subsample_ari_q95: float
    medoid_distance_median: float
    medoid_distance_q05: float
    medoid_distance_q95: float
    passes: bool
    visualization_file: str

    def to_record(
        self,
        *,
        sketches: Sequence[EpisodeSketch],
    ) -> dict[str, Any]:
        return {
            "k": self.k,
            "objective": self.objective,
            "silhouette": self.silhouette,
            "cluster_sizes": list(self.cluster_sizes),
            "medoid_episodes": [sketches[index].episode_index for index in self.medoid_indices],
            "initialization_ari_median": self.initialization_ari_median,
            "subsample_ari_median": self.subsample_ari_median,
            "subsample_ari_q05": self.subsample_ari_q05,
            "subsample_ari_q95": self.subsample_ari_q95,
            "medoid_distance_median": self.medoid_distance_median,
            "medoid_distance_q05": self.medoid_distance_q05,
            "medoid_distance_q95": self.medoid_distance_q95,
            "passes": self.passes,
            "visualization_file": self.visualization_file,
        }


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def _rotate_local(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) @ rotation.T


def _stable_contact_mask(raw_contact: np.ndarray) -> tuple[np.ndarray, int]:
    value = np.asarray(raw_contact, dtype=bool)
    if value.ndim != 1:
        raise ValueError("contact must be one-dimensional")
    if value.size == 0:
        return value, 0
    smoothed = ndimage.binary_closing(value, structure=np.ones(5, dtype=bool))
    labels, count = ndimage.label(smoothed)
    if count:
        sizes = np.bincount(labels)
        for label in range(1, count + 1):
            if sizes[label] < 3:
                smoothed[labels == label] = False
        _, count = ndimage.label(smoothed)
    return np.asarray(smoothed, dtype=bool), int(count)


def _empty_sketch(
    *,
    episode_index: int,
    split: str,
    reason: str,
    initial_features: np.ndarray,
) -> EpisodeSketch:
    return EpisodeSketch(
        episode_index=episode_index,
        split=split,
        eligible=False,
        exclusion_reason=reason,
        token_frames=(),
        agent_local=np.empty((0, 2), dtype=np.float64),
        block_displacement=np.empty((0, 2), dtype=np.float64),
        relative_angle=np.empty((0,), dtype=np.float64),
        stable_contact=np.empty((0,), dtype=bool),
        normalized_progress=np.empty((0,), dtype=np.float64),
        contact_runs=0,
        initial_observation_features=initial_features,
    )


def _initial_observation_features(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = [records[min(index, len(records) - 1)] for index in range(2)]
    features: list[float] = []
    for record in rows:
        agent = np.asarray(record["agent_position"], dtype=np.float64).reshape(2)
        block = np.asarray(record["block_position"], dtype=np.float64).reshape(2)
        angle = float(record["block_angle_rad"])
        features.extend([*agent, *block, math.sin(angle), math.cos(angle)])
    return np.asarray(features, dtype=np.float64)


def build_episode_sketch(records: Sequence[Mapping[str, Any]]) -> EpisodeSketch:
    """Build the pre-registered eight-token sketch for one demonstration."""

    if not records:
        raise ValueError("Cannot build an E5 sketch from an empty episode")
    ordered = sorted(records, key=lambda record: int(record["frame_index"]))
    episode_index = int(ordered[0]["episode_index"])
    split = str(ordered[0]["split"])
    if any(int(record["episode_index"]) != episode_index for record in ordered):
        raise ValueError("Episode records contain multiple episode_index values")
    if any(str(record["split"]) != split for record in ordered):
        raise ValueError("Episode records contain multiple split values")

    initial_features = _initial_observation_features(ordered)
    valid = np.asarray([bool(record["block_pose_valid"]) for record in ordered], dtype=bool)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return _empty_sketch(
            episode_index=episode_index,
            split=split,
            reason="fewer_than_two_valid_block_poses",
            initial_features=initial_features,
        )

    agent = np.asarray([record["agent_position"] for record in ordered], dtype=np.float64)
    block = np.asarray([record["block_position"] for record in ordered], dtype=np.float64)
    angle = np.asarray([record["block_angle_rad"] for record in ordered], dtype=np.float64)
    distance = np.asarray(
        [record["agent_block_surface_distance"] for record in ordered],
        dtype=np.float64,
    )
    raw_contact = valid & np.isfinite(distance) & (distance <= CONTACT_DISTANCE)
    stable_contact, contact_runs = _stable_contact_mask(raw_contact)
    contact_indices = np.flatnonzero(stable_contact)
    if contact_indices.size == 0:
        return _empty_sketch(
            episode_index=episode_index,
            split=split,
            reason="no_stable_contact",
            initial_features=initial_features,
        )

    first_valid = int(valid_indices[0])
    final_valid = int(valid_indices[-1])
    unwrapped_angle = np.unwrap(angle)
    relative_angle = unwrapped_angle - unwrapped_angle[first_valid]
    translation = np.linalg.norm(block - block[first_valid], axis=1)
    significant_motion = valid & (
        (translation >= MIN_TRANSLATION_FOR_MODE)
        | (np.abs(np.degrees(relative_angle)) >= ROTATION_FIRST_THRESHOLD_DEG)
    )
    motion_indices = np.flatnonzero(significant_motion)
    if motion_indices.size == 0:
        return _empty_sketch(
            episode_index=episode_index,
            split=split,
            reason="no_significant_block_motion",
            initial_features=initial_features,
        )

    steps = np.zeros(len(ordered), dtype=np.float64)
    delta_position = np.diff(block, axis=0)
    delta_angle = np.diff(unwrapped_angle)
    pair_valid = valid[1:] & valid[:-1]
    steps[1:] = np.where(
        pair_valid,
        np.sqrt(
            np.square(np.linalg.norm(delta_position, axis=1))
            + np.square(ROTATION_ARC_RADIUS * delta_angle)
        ),
        0.0,
    )
    cumulative = np.cumsum(steps)
    total_motion = float(cumulative[final_valid] - cumulative[first_valid])
    if not math.isfinite(total_motion) or total_motion <= 0.0:
        return _empty_sketch(
            episode_index=episode_index,
            split=split,
            reason="zero_block_motion_arc",
            initial_features=initial_features,
        )

    first_contact = int(contact_indices[0])
    previous_valid = valid_indices[valid_indices < first_contact]
    if previous_valid.size == 0:
        return _empty_sketch(
            episode_index=episode_index,
            split=split,
            reason="no_valid_pre_contact_frame",
            initial_features=initial_features,
        )
    pre_contact = int(previous_valid[-1])
    first_motion = int(motion_indices[0])
    progress_indices: list[int] = []
    for fraction in (0.25, 0.50, 0.75):
        threshold = cumulative[first_valid] + fraction * total_motion
        candidates = np.flatnonzero(valid & (cumulative >= threshold))
        if candidates.size == 0:
            return _empty_sketch(
                episode_index=episode_index,
                split=split,
                reason=f"missing_motion_progress_{fraction:g}",
                initial_features=initial_features,
            )
        progress_indices.append(int(candidates[0]))

    token_indices = (
        first_valid,
        pre_contact,
        first_contact,
        first_motion,
        *progress_indices,
        final_valid,
    )
    if len(token_indices) != TOKEN_COUNT or not valid[list(token_indices)].all():
        return _empty_sketch(
            episode_index=episode_index,
            split=split,
            reason="invalid_token_block_pose",
            initial_features=initial_features,
        )

    agent_local = np.stack(
        [_rotate_local(agent[index] - block[index], angle[index]) for index in token_indices]
    )
    block_displacement = np.stack(
        [
            _rotate_local(block[index] - block[first_valid], angle[first_valid])
            for index in token_indices
        ]
    )
    token_angles = np.asarray(relative_angle[list(token_indices)], dtype=np.float64)
    token_contact = np.asarray(stable_contact[list(token_indices)], dtype=bool)
    denominator = max(1, len(ordered) - 1)
    normalized_progress = np.asarray(token_indices, dtype=np.float64) / denominator
    frame_numbers = tuple(int(ordered[index]["frame_index"]) for index in token_indices)

    arrays = (agent_local, block_displacement, token_angles, normalized_progress)
    if any(not np.isfinite(value).all() for value in arrays):
        return _empty_sketch(
            episode_index=episode_index,
            split=split,
            reason="non_finite_token_feature",
            initial_features=initial_features,
        )
    return EpisodeSketch(
        episode_index=episode_index,
        split=split,
        eligible=True,
        exclusion_reason=None,
        token_frames=frame_numbers,
        agent_local=agent_local,
        block_displacement=block_displacement,
        relative_angle=token_angles,
        stable_contact=token_contact,
        normalized_progress=normalized_progress,
        contact_runs=contact_runs,
        initial_observation_features=initial_features,
    )


def component_distances(
    first: EpisodeSketch,
    second: EpisodeSketch,
) -> dict[str, float]:
    """Return all raw pre-registered trajectory-distance components."""

    if not first.eligible or not second.eligible:
        raise ValueError("Trajectory distance requires two eligible sketches")
    contact_denominator = max(1, first.contact_runs, second.contact_runs)
    return {
        "first_contact": float(np.linalg.norm(first.agent_local[2] - second.agent_local[2])),
        "approach": float(
            np.linalg.norm(first.agent_local[:3] - second.agent_local[:3], axis=1).mean()
        ),
        "block_path": float(
            np.linalg.norm(
                first.block_displacement[3:7] - second.block_displacement[3:7],
                axis=1,
            ).mean()
        ),
        "rotation": float(
            np.abs(_wrap_angle(first.relative_angle[3:7] - second.relative_angle[3:7])).mean()
        ),
        "contact_sequence": float(
            np.not_equal(first.stable_contact, second.stable_contact).mean()
            + abs(first.contact_runs - second.contact_runs) / contact_denominator
        ),
        "edge": float(
            sum(
                np.linalg.norm(first.agent_local[index] - second.agent_local[index])
                + np.linalg.norm(first.block_displacement[index] - second.block_displacement[index])
                for index in (0, 7)
            )
            / 2.0
        ),
    }


def fit_component_scales(sketches: Sequence[EpisodeSketch]) -> dict[str, float]:
    """Fit non-zero median scales using eligible train pairs only."""

    if len(sketches) < 2:
        raise ValueError("At least two eligible train sketches are required")
    values: dict[str, list[float]] = defaultdict(list)
    for left in range(len(sketches)):
        for right in range(left + 1, len(sketches)):
            for name, value in component_distances(sketches[left], sketches[right]).items():
                if value > 0.0 and math.isfinite(value):
                    values[name].append(value)
    scales = {
        name: float(np.median(component_values))
        for name, component_values in values.items()
        if component_values
    }
    if not scales:
        raise ValueError("All trajectory-distance components are degenerate")
    return scales


def trajectory_distance(
    first: EpisodeSketch,
    second: EpisodeSketch,
    *,
    scales: Mapping[str, float],
) -> float:
    raw = component_distances(first, second)
    weighted = 0.0
    weight_sum = 0.0
    for name, scale in scales.items():
        if name not in COMPONENT_WEIGHTS:
            raise KeyError(f"Unknown trajectory-distance component: {name}")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid scale for {name}: {scale}")
        weight = COMPONENT_WEIGHTS[name]
        weighted += weight * raw[name] / scale
        weight_sum += weight
    if weight_sum <= 0.0:
        raise ValueError("Trajectory distance has no active components")
    return float(weighted / weight_sum)


def trajectory_distance_matrix(
    sketches: Sequence[EpisodeSketch],
    *,
    scales: Mapping[str, float],
) -> np.ndarray:
    result = np.zeros((len(sketches), len(sketches)), dtype=np.float64)
    for left in range(len(sketches)):
        for right in range(left + 1, len(sketches)):
            value = trajectory_distance(sketches[left], sketches[right], scales=scales)
            result[left, right] = value
            result[right, left] = value
    return result


def _assign_to_medoids(
    distance_matrix: np.ndarray,
    medoids: Sequence[int],
    *,
    episode_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    ordered = tuple(sorted((int(index) for index in medoids), key=lambda i: episode_ids[i]))
    distances = distance_matrix[:, ordered]
    labels = np.argmin(distances, axis=1).astype(np.int64)
    nearest = distances[np.arange(len(distance_matrix)), labels]
    return labels, nearest, ordered


def _kmedoids_plus_plus(
    distance_matrix: np.ndarray,
    *,
    k: int,
    seed: int,
    episode_ids: Sequence[int],
) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(len(distance_matrix)))]
    while len(medoids) < k:
        nearest = distance_matrix[:, medoids].min(axis=1)
        nearest[medoids] = 0.0
        probabilities = np.square(nearest)
        total = float(probabilities.sum())
        if total <= 0.0:
            remaining = sorted(
                set(range(len(distance_matrix))) - set(medoids),
                key=lambda index: episode_ids[index],
            )
            medoids.append(remaining[0])
        else:
            probabilities /= total
            medoids.append(int(rng.choice(len(distance_matrix), p=probabilities)))
    return tuple(medoids)


def fit_kmedoids(
    distance_matrix: np.ndarray,
    *,
    k: int,
    seed: int,
    episode_ids: Sequence[int],
    max_iterations: int = MAX_KMEDOIDS_ITERATIONS,
) -> KMedoidsRun:
    """Fit one seeded K-medoids run with deterministic ties."""

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    if len(episode_ids) != len(matrix):
        raise ValueError("episode_ids length must match distance_matrix")
    if not 1 < k < len(matrix):
        raise ValueError("k must be in [2, n-1]")
    medoids = _kmedoids_plus_plus(matrix, k=k, seed=seed, episode_ids=episode_ids)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        labels, nearest, ordered = _assign_to_medoids(
            matrix,
            medoids,
            episode_ids=episode_ids,
        )
        updated: list[int] = []
        for label in range(k):
            members = np.flatnonzero(labels == label)
            if members.size == 0:
                available = [index for index in range(len(matrix)) if index not in updated]
                max_distance = max(float(nearest[index]) for index in available)
                candidates = [
                    index
                    for index in available
                    if math.isclose(
                        float(nearest[index]),
                        max_distance,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ]
                updated.append(min(candidates, key=lambda index: episode_ids[index]))
                continue
            costs = matrix[np.ix_(members, members)].sum(axis=1)
            minimum = float(costs.min())
            candidates = [
                int(member)
                for member, cost in zip(members, costs, strict=True)
                if math.isclose(float(cost), minimum, rel_tol=0.0, abs_tol=1e-12)
            ]
            updated.append(min(candidates, key=lambda index: episode_ids[index]))
        new_medoids = tuple(updated)
        if set(new_medoids) == set(ordered):
            medoids = ordered
            break
        medoids = new_medoids
    else:
        raise RuntimeError(f"K-medoids did not converge within {max_iterations} iterations")

    # Alternating medoid updates can stop at a partition-local optimum. Complete the
    # standard PAM SWAP phase so the stability audit measures the data rather than a
    # weak optimizer. The vectorized objective keeps this tractable for 165 episodes.
    for swap_iteration in range(1, max_iterations + 1):
        _, nearest, ordered = _assign_to_medoids(
            matrix,
            medoids,
            episode_ids=episode_ids,
        )
        current_objective = float(nearest.sum())
        medoid_set = set(ordered)
        non_medoids = np.asarray(
            [index for index in range(len(matrix)) if index not in medoid_set],
            dtype=np.int64,
        )
        best_objective = current_objective
        best_medoids = ordered
        for position in range(k):
            retained = ordered[:position] + ordered[position + 1 :]
            if retained:
                retained_distance = matrix[:, retained].min(axis=1)
            else:
                retained_distance = np.full(len(matrix), np.inf, dtype=np.float64)
            objectives = np.minimum(
                retained_distance[:, None],
                matrix[:, non_medoids],
            ).sum(axis=0)
            minimum = float(objectives.min())
            candidate_offsets = np.flatnonzero(
                np.isclose(objectives, minimum, rtol=0.0, atol=1e-12)
            )
            for offset in candidate_offsets:
                candidate = int(non_medoids[int(offset)])
                proposed = tuple(
                    sorted(
                        retained + (candidate,),
                        key=lambda index: episode_ids[index],
                    )
                )
                if minimum < best_objective - 1e-12 or (
                    math.isclose(
                        minimum,
                        best_objective,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and tuple(episode_ids[index] for index in proposed)
                    < tuple(episode_ids[index] for index in best_medoids)
                ):
                    best_objective = minimum
                    best_medoids = proposed
        if best_objective >= current_objective - 1e-12:
            break
        medoids = best_medoids
    else:
        raise RuntimeError(f"PAM SWAP did not converge within {max_iterations} iterations")

    labels, nearest, ordered = _assign_to_medoids(
        matrix,
        medoids,
        episode_ids=episode_ids,
    )
    return KMedoidsRun(
        labels=labels,
        medoid_indices=ordered,
        objective=float(nearest.sum()),
        seed=seed,
        iterations=iterations + swap_iteration,
    )


def _best_run(runs: Sequence[KMedoidsRun], episode_ids: Sequence[int]) -> KMedoidsRun:
    if not runs:
        raise ValueError("At least one K-medoids run is required")
    best_objective = min(run.objective for run in runs)
    candidates = [
        run
        for run in runs
        if math.isclose(run.objective, best_objective, rel_tol=0.0, abs_tol=1e-12)
    ]
    return min(
        candidates,
        key=lambda run: tuple(sorted(episode_ids[index] for index in run.medoid_indices)),
    )


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.05)),
        float(np.quantile(array, 0.95)),
    )


def evaluate_candidate(
    distance_matrix: np.ndarray,
    *,
    sketches: Sequence[EpisodeSketch],
    k: int,
    subsample_iterations: int = SUBSAMPLE_ITERATIONS,
) -> PrototypeCandidate:
    """Evaluate one K under the frozen initialization and subsample protocol."""

    episode_ids = [sketch.episode_index for sketch in sketches]
    runs = [
        fit_kmedoids(
            distance_matrix,
            k=k,
            seed=seed,
            episode_ids=episode_ids,
        )
        for seed in INITIALIZATION_SEEDS
    ]
    best = _best_run(runs, episode_ids)
    initialization_aris = [
        adjusted_rand_score(runs[left].labels, runs[right].labels)
        for left in range(len(runs))
        for right in range(left + 1, len(runs))
    ]
    initialization_ari = float(np.median(initialization_aris))
    cluster_sizes = tuple(int(np.count_nonzero(best.labels == label)) for label in range(k))
    silhouette = float(silhouette_score(distance_matrix, best.labels, metric="precomputed"))

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    sample_size = max(k + 1, math.floor(len(sketches) * SUBSAMPLE_FRACTION))
    subsample_aris: list[float] = []
    medoid_distances: list[float] = []
    for _ in range(subsample_iterations):
        selected = np.sort(rng.choice(len(sketches), size=sample_size, replace=False))
        submatrix = distance_matrix[np.ix_(selected, selected)]
        sub_episode_ids = [episode_ids[index] for index in selected]
        sub_runs = [
            fit_kmedoids(
                submatrix,
                k=k,
                seed=seed,
                episode_ids=sub_episode_ids,
            )
            for seed in INITIALIZATION_SEEDS
        ]
        sub_best = _best_run(sub_runs, sub_episode_ids)
        subsample_aris.append(float(adjusted_rand_score(best.labels[selected], sub_best.labels)))
        global_sub_medoids = [int(selected[index]) for index in sub_best.medoid_indices]
        matching_cost = distance_matrix[np.ix_(best.medoid_indices, global_sub_medoids)]
        rows, columns = linear_sum_assignment(matching_cost)
        medoid_distances.extend(
            float(matching_cost[row, column]) for row, column in zip(rows, columns, strict=True)
        )

    subsample_median, subsample_q05, subsample_q95 = _quantiles(subsample_aris)
    medoid_median, medoid_q05, medoid_q95 = _quantiles(medoid_distances)
    passes = (
        min(cluster_sizes) >= MIN_CLUSTER_SIZE
        and initialization_ari >= MIN_INITIALIZATION_ARI
        and subsample_median >= MIN_SUBSAMPLE_ARI
    )
    return PrototypeCandidate(
        k=k,
        labels=best.labels,
        medoid_indices=best.medoid_indices,
        objective=best.objective,
        silhouette=silhouette,
        cluster_sizes=cluster_sizes,
        initialization_ari_median=initialization_ari,
        subsample_ari_median=subsample_median,
        subsample_ari_q05=subsample_q05,
        subsample_ari_q95=subsample_q95,
        medoid_distance_median=medoid_median,
        medoid_distance_q05=medoid_q05,
        medoid_distance_q95=medoid_q95,
        passes=passes,
        visualization_file=f"visualizations/k_{k:02d}_prototypes.png",
    )


def _leakage_audit(
    sketches: Sequence[EpisodeSketch],
    labels: np.ndarray,
) -> dict[str, Any]:
    features = np.stack([sketch.initial_observation_features for sketch in sketches])
    values = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(values)
    folds = min(5, int(counts.min()))
    if folds < 2:
        raise ValueError("Leakage audit requires at least two examples per prototype")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260730)
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=20260730),
    )
    forest = RandomForestClassifier(
        n_estimators=200,
        random_state=20260730,
        class_weight="balanced",
    )
    logistic_predictions = cross_val_predict(logistic, features, values, cv=splitter)
    forest_predictions = cross_val_predict(forest, features, values, cv=splitter)
    majority_label = int(np.flatnonzero(counts == counts.max())[0])
    majority_predictions = np.full(values.shape, majority_label, dtype=np.int64)

    def metrics(predictions: np.ndarray) -> dict[str, Any]:
        return {
            "accuracy": float(accuracy_score(values, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(values, predictions)),
            "confusion_matrix": confusion_matrix(
                values,
                predictions,
                labels=list(range(len(counts))),
            ).tolist(),
        }

    return {
        "num_folds": folds,
        "num_samples": len(values),
        "class_counts": counts.tolist(),
        "feature_description": (
            "first two agent positions plus image-derived block positions and sin/cos angles"
        ),
        "majority": metrics(majority_predictions),
        "logistic_regression": metrics(logistic_predictions),
        "random_forest": metrics(forest_predictions),
    }


def _plot_candidate(
    path: Path,
    *,
    candidate: PrototypeCandidate,
    sketches: Sequence[EpisodeSketch],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, candidate.k))
    for prototype_id, medoid_index in enumerate(candidate.medoid_indices):
        sketch = sketches[medoid_index]
        axes[0].plot(
            sketch.agent_local[:, 0],
            sketch.agent_local[:, 1],
            marker="o",
            color=colors[prototype_id],
            label=f"P{prototype_id} · ep {sketch.episode_index}",
        )
        axes[1].plot(
            sketch.block_displacement[:, 0],
            sketch.block_displacement[:, 1],
            marker="o",
            color=colors[prototype_id],
        )
    axes[0].set_title("Agent path relative to moving T")
    axes[1].set_title("T displacement in initial-T frame")
    for axis in axes:
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.invert_yaxis()
    axes[0].legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _read_trajectory_records(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"E0 trajectories are missing: {path}")
    rows = pq.read_table(path).to_pylist()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_index"])].append(row)
    if not grouped:
        raise ValueError(f"E0 trajectories are empty: {path}")
    return dict(grouped)


def _split_name_by_episode(split: EpisodeSplit) -> dict[int, str]:
    result = {episode: "train" for episode in split.train}
    result.update({episode: "development" for episode in split.development})
    result.update({episode: "audit" for episode in split.audit})
    return result


def _cluster_radii(
    distance_matrix: np.ndarray,
    candidate: PrototypeCandidate,
) -> dict[int, float]:
    result = {}
    for prototype_id, medoid_index in enumerate(candidate.medoid_indices):
        members = np.flatnonzero(candidate.labels == prototype_id)
        values = distance_matrix[members, medoid_index]
        result[prototype_id] = float(np.quantile(values, 0.95, method="higher"))
    return result


def fit_e5_prototypes(
    *,
    trajectories_path: str | Path,
    output_dir: str | Path,
    split_file: str | Path | None = None,
    num_prototypes: int | None = None,
    selection_rationale: str | None = None,
    overwrite: bool = False,
    subsample_iterations: int = SUBSAMPLE_ITERATIONS,
) -> Path:
    """Fit E5 prototypes and write a trainable complete sidecar manifest."""

    if num_prototypes is not None and num_prototypes not in K_CANDIDATES:
        raise ValueError(f"num_prototypes must be one of {K_CANDIDATES}")
    if subsample_iterations <= 0:
        raise ValueError("subsample_iterations must be positive")
    trajectories = Path(trajectories_path).expanduser().resolve()
    split = load_split(split_file)
    grouped = _read_trajectory_records(trajectories)
    expected = set(range(split.num_episodes))
    missing = sorted(expected - set(grouped))
    if missing:
        raise ValueError(f"E0 trajectories are missing episodes: {missing}")
    split_names = _split_name_by_episode(split)

    sketches: list[EpisodeSketch] = []
    frame_counts: dict[int, int] = {}
    for episode in range(split.num_episodes):
        records = grouped[episode]
        frame_counts[episode] = len(records)
        recorded_split = {str(record["split"]) for record in records}
        if recorded_split != {split_names[episode]}:
            raise ValueError(
                f"Episode {episode} split mismatch: records={recorded_split}, "
                f"manifest={split_names[episode]!r}"
            )
        sketches.append(build_episode_sketch(records))

    train_sketches = [sketches[episode] for episode in split.train if sketches[episode].eligible]
    if len(train_sketches) <= max(K_CANDIDATES):
        raise ValueError("Too few eligible train demonstrations for E5 prototype fitting")
    scales = fit_component_scales(train_sketches)
    train_distance = trajectory_distance_matrix(train_sketches, scales=scales)

    candidates = [
        evaluate_candidate(
            train_distance,
            sketches=train_sketches,
            k=k,
            subsample_iterations=subsample_iterations,
        )
        for k in K_CANDIDATES
    ]
    passing = [candidate for candidate in candidates if candidate.passes]
    if not passing:
        raise RuntimeError("No K=2/3/4 prototype candidate passed the pre-registered gates")
    recommended = min(passing, key=lambda candidate: candidate.k)
    if num_prototypes is None:
        selected = recommended
        selection_kind = "automatic_smallest_passing_k"
        rationale = "smallest K satisfying all pre-registered gates"
    else:
        selected = next(candidate for candidate in candidates if candidate.k == num_prototypes)
        if not selected.passes:
            raise ValueError(f"Explicit K={num_prototypes} did not pass the pre-registered gates")
        if selected.k != recommended.k:
            if not selection_rationale or not selection_rationale.strip():
                raise ValueError(
                    "Overriding the recommended K requires a non-empty selection_rationale"
                )
            selection_kind = "human_override"
            rationale = selection_rationale.strip()
        else:
            selection_kind = "explicit_recommended_k"
            rationale = (
                selection_rationale.strip()
                if selection_rationale
                else ("explicitly selected the automatically recommended K")
            )

    train_episode_to_local = {
        sketch.episode_index: index for index, sketch in enumerate(train_sketches)
    }
    selected_medoids = [train_sketches[index] for index in selected.medoid_indices]
    radii = _cluster_radii(train_distance, selected)

    assignments: list[dict[str, Any]] = []
    conditional_frames = 0
    always_null_frames = 0
    for sketch in sketches:
        if not sketch.eligible:
            assignments.append(
                {
                    "episode_index": sketch.episode_index,
                    "split": sketch.split,
                    "prototype_id": None,
                    "training_condition_id": selected.k if sketch.split == "train" else None,
                    "distance_to_medoid": None,
                    "classifiable": False,
                    "eligible": False,
                    "exclusion_reason": sketch.exclusion_reason,
                    "num_frames": frame_counts[sketch.episode_index],
                }
            )
            if sketch.split == "train":
                always_null_frames += frame_counts[sketch.episode_index]
            continue
        if sketch.episode_index in train_episode_to_local:
            local_index = train_episode_to_local[sketch.episode_index]
            prototype_id = int(selected.labels[local_index])
            medoid_index = selected.medoid_indices[prototype_id]
            distance = float(train_distance[local_index, medoid_index])
        else:
            distances = [
                trajectory_distance(sketch, medoid, scales=scales) for medoid in selected_medoids
            ]
            prototype_id = int(np.argmin(distances))
            distance = float(distances[prototype_id])
        assignments.append(
            {
                "episode_index": sketch.episode_index,
                "split": sketch.split,
                "prototype_id": prototype_id,
                "training_condition_id": (prototype_id if sketch.split == "train" else None),
                "distance_to_medoid": distance,
                "classifiable": distance <= radii[prototype_id],
                "eligible": True,
                "exclusion_reason": None,
                "num_frames": frame_counts[sketch.episode_index],
            }
        )
        if sketch.split == "train":
            conditional_frames += frame_counts[sketch.episode_index]

    candidate_records = [candidate.to_record(sketches=train_sketches) for candidate in candidates]
    leakage = {
        str(candidate.k): _leakage_audit(train_sketches, candidate.labels)
        for candidate in candidates
    }
    destination = Path(output_dir).expanduser().resolve()
    try:
        source_trajectories = str(trajectories.relative_to(practice_root()))
    except ValueError:
        source_trajectories = trajectories.name
    with AtomicOutputDirectory(destination, overwrite=overwrite) as temporary:
        for candidate in candidates:
            _plot_candidate(
                temporary / candidate.visualization_file,
                candidate=candidate,
                sketches=train_sketches,
            )
        write_records_parquet(temporary / "episode_assignments.parquet", assignments)
        write_json_atomic(
            temporary / "prototype_candidates.json",
            {
                "schema_version": E5_SCHEMA_VERSION,
                "candidates": candidate_records,
            },
        )
        write_json_atomic(
            temporary / "leakage_audit.json",
            {
                "schema_version": E5_SCHEMA_VERSION,
                "candidate_results": leakage,
            },
        )
        write_json_atomic(
            temporary / "prototype_sketches.json",
            {
                "schema_version": E5_SCHEMA_VERSION,
                "prototypes": [
                    {
                        "prototype_id": prototype_id,
                        "medoid_episode": sketch.episode_index,
                        "rejection_radius": radii[prototype_id],
                        "sketch": sketch.to_record(),
                    }
                    for prototype_id, sketch in enumerate(selected_medoids)
                ],
            },
        )
        artifact_hashes = {
            name: sha256_file(temporary / name)
            for name in (
                "episode_assignments.parquet",
                "prototype_candidates.json",
                "leakage_audit.json",
                "prototype_sketches.json",
            )
        }
        manifest = {
            "schema_version": E5_SCHEMA_VERSION,
            "status": "complete",
            "label": "Offline",
            "experiment": "E5 prototype fitting",
            "source_trajectories": source_trajectories,
            "source_trajectories_sha256": sha256_file(trajectories),
            "split_manifest_sha256": split.manifest_sha256,
            "sketch_version": SKETCH_VERSION,
            "distance_version": DISTANCE_VERSION,
            "component_weights": COMPONENT_WEIGHTS,
            "component_scales": scales,
            "candidate_k": list(K_CANDIDATES),
            "selected_k": selected.k,
            "null_prototype_id": selected.k,
            "mode_embedding_dim": 32,
            "condition_dropout_probability": 0.20,
            "selection": {
                "kind": selection_kind,
                "recommended_k": recommended.k,
                "rationale": rationale,
            },
            "canonical_prototypes": [
                {
                    "prototype_id": prototype_id,
                    "medoid_episode": sketch.episode_index,
                    "cluster_size": selected.cluster_sizes[prototype_id],
                    "rejection_radius": radii[prototype_id],
                }
                for prototype_id, sketch in enumerate(selected_medoids)
            ],
            "training_accounting": {
                "eligible_conditional_frames": conditional_frames,
                "ineligible_always_null_frames": always_null_frames,
                "expected_condition_dropout_frames": (conditional_frames * 0.20),
            },
            "initialization_seeds": list(INITIALIZATION_SEEDS),
            "subsample_seed": SUBSAMPLE_SEED,
            "subsample_fraction": SUBSAMPLE_FRACTION,
            "subsample_iterations": subsample_iterations,
            "artifact_hashes": artifact_hashes,
        }
        write_json_atomic(temporary / "prototype_manifest.json", manifest)
        write_metrics(
            temporary,
            {
                "selected_k": selected.k,
                "eligible_train_episodes": len(train_sketches),
                "ineligible_train_episodes": len(split.train) - len(train_sketches),
                "minimum_cluster_size": min(selected.cluster_sizes),
                "initialization_ari_median": selected.initialization_ari_median,
                "subsample_ari_median": selected.subsample_ari_median,
                "silhouette": selected.silhouette,
            },
        )
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=default_output_root() / "data_inspection" / "trajectories.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "e5_mode_conditioned" / "prototypes",
    )
    parser.add_argument("--split-file", type=Path, default=default_split_path())
    parser.add_argument("--num-prototypes", type=int, choices=K_CANDIDATES)
    parser.add_argument("--selection-rationale")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output = fit_e5_prototypes(
        trajectories_path=args.trajectories,
        output_dir=args.output_dir,
        split_file=args.split_file,
        num_prototypes=args.num_prototypes,
        selection_rationale=args.selection_rationale,
        overwrite=args.overwrite,
    )
    print(f"E5 prototype artifacts: {output}")


if __name__ == "__main__":
    main()


__all__ = [
    "COMPONENT_WEIGHTS",
    "DISTANCE_VERSION",
    "INITIALIZATION_SEEDS",
    "SKETCH_VERSION",
    "EpisodeSketch",
    "KMedoidsRun",
    "PrototypeCandidate",
    "build_episode_sketch",
    "component_distances",
    "episode_sketch_from_record",
    "evaluate_candidate",
    "fit_component_scales",
    "fit_e5_prototypes",
    "fit_kmedoids",
    "trajectory_distance",
    "trajectory_distance_matrix",
]
