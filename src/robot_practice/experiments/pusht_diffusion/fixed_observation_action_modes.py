"""Explore fixed-observation Diffusion action modes from completed E1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .config import default_output_root, practice_root
from .diversity_metrics import pairwise_rms_distance
from .output import AtomicOutputDirectory, write_records_parquet
from .provenance import (
    collect_runtime_provenance,
    sha256_file,
    write_json_atomic,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_SEED = 20260729
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_CANDIDATES = 10
EXPECTED_SAMPLES = 20
EXPECTED_STEPS = 8
EXPECTED_BC_CALLS = 3
BC_TOLERANCE = 1e-6
DEGENERATE_VARIANCE = 1e-12
GMM_COMPONENTS = (1, 2, 3, 4)
GMM_INITIALIZATIONS = 20
GMM_REGULARIZATION = 1e-6
BIC_IMPROVEMENT_THRESHOLD = 10.0
MINIMUM_MODE_SIZE = 3
SUBSAMPLE_ITERATIONS = 200
SUBSAMPLE_FRACTION = 0.8
SUBSAMPLE_PASS_FRACTION = 0.7
SILHOUETTE_THRESHOLD = 0.25
ARI_THRESHOLD = 0.5
LOW_DENSITY_QUANTILE = 0.95
NEGATIVE_CONTROL_FAILURE_FRACTION = 0.2

INITIAL_NOISE_ONLY = "initial_noise_only"
FULL_SAMPLING_SEED = "full_sampling_seed"
BC_DETERMINISTIC = "bc_deterministic"
PROTOCOLS = (INITIAL_NOISE_ONLY, FULL_SAMPLING_SEED)


@dataclass(frozen=True)
class CandidateActions:
    candidate_id: str
    episode_index: int
    frame_index: int
    initial_noise_only: np.ndarray
    full_sampling_seed: np.ndarray
    bc_action: np.ndarray


@dataclass(frozen=True)
class ModeAnalysis:
    candidate_id: str
    protocol: str
    degenerate_unimodal: bool
    pca_components: int
    selected_k: int
    bics: tuple[float, ...]
    bic_improvement: float
    minimum_gmm_cluster_size: int
    subsample_multimodal_fraction: float
    minimum_distance_cluster_size: int
    silhouette: float
    adjusted_rand_index: float
    multimodal: bool
    gmm_labels: np.ndarray
    distance_labels: np.ndarray
    pca_coordinates: np.ndarray
    pca_mean: np.ndarray
    pca_axes: np.ndarray
    distance_matrix: np.ndarray
    threshold_values: np.ndarray
    threshold_cluster_counts: np.ndarray


@dataclass(frozen=True)
class BCAveragingSignature:
    bc_to_dp_mean: float
    bc_to_nearest_mode_medoid: float
    bc_to_nearest_sample: float
    dp_leave_one_out_nn_q95: float
    bc_nn_density_percentile: float
    mean_like: bool
    low_density: bool
    averaging_consistent_signature: bool


@dataclass(frozen=True)
class _PCAProjection:
    coordinates: np.ndarray
    mean: np.ndarray
    axes: np.ndarray


@dataclass(frozen=True)
class _GMMSelection:
    selected_k: int
    bics: tuple[float, ...]
    bic_improvement: float
    labels: np.ndarray


def reconstruct_action_chunks(
    rows: pd.DataFrame,
    *,
    expected_samples: int,
    expected_steps: int,
) -> np.ndarray:
    """Reconstruct canonical sample-major action chunks from row-wise E1 records."""

    required = {
        "sample_ordinal",
        "step",
        "normalized_action_x",
        "normalized_action_y",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Action rows are missing required columns: {missing}")
    expected_ordinals = list(range(expected_samples))
    observed_ordinals = sorted(int(value) for value in rows["sample_ordinal"].unique())
    if observed_ordinals != expected_ordinals:
        raise ValueError(f"sample_ordinal must be {expected_ordinals}, got {observed_ordinals}")
    if rows.duplicated(["sample_ordinal", "step"]).any():
        raise ValueError("Every sample step must appear exactly once")
    if len(rows) != expected_samples * expected_steps:
        raise ValueError(
            f"Expected {expected_samples * expected_steps} action rows, got {len(rows)}"
        )

    result = np.empty((expected_samples, expected_steps, 2), dtype=np.float64)
    for sample_ordinal in expected_ordinals:
        sample = rows[rows["sample_ordinal"] == sample_ordinal].sort_values("step")
        observed_steps = [int(value) for value in sample["step"]]
        expected_step_values = list(range(expected_steps))
        if observed_steps != expected_step_values:
            raise ValueError(
                "Every sample step must appear exactly once; "
                f"sample {sample_ordinal} has {observed_steps}"
            )
        result[sample_ordinal, :, 0] = sample["normalized_action_x"].to_numpy(dtype=np.float64)
        result[sample_ordinal, :, 1] = sample["normalized_action_y"].to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Action chunks must contain only finite values")
    return result


def _seed(candidate_id: str, *, stream: int, ordinal: int = 0) -> int:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    words = [
        int.from_bytes(digest[offset : offset + 4], byteorder="little")
        for offset in range(0, 16, 4)
    ]
    sequence = np.random.SeedSequence([ANALYSIS_SEED, stream, ordinal, *words])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _project_pca(flattened: np.ndarray) -> _PCAProjection | None:
    total_variance = float(np.var(flattened, axis=0, ddof=0).sum())
    if total_variance <= DEGENERATE_VARIANCE:
        return None
    centered = flattened - np.mean(flattened, axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered))
    maximum_components = min(3, flattened.shape[0] - 1, rank)
    if maximum_components < 1:
        return None
    estimator = PCA(n_components=maximum_components, svd_solver="full")
    coordinates = estimator.fit_transform(flattened)
    cumulative = np.cumsum(estimator.explained_variance_ratio_)
    retained = min(maximum_components, int(np.searchsorted(cumulative, 0.95) + 1))
    return _PCAProjection(
        coordinates=np.asarray(coordinates[:, :retained], dtype=np.float64),
        mean=np.asarray(estimator.mean_, dtype=np.float64),
        axes=np.asarray(estimator.components_[:retained], dtype=np.float64),
    )


def _select_gmm(coordinates: np.ndarray, *, random_state: int) -> _GMMSelection:
    bics = []
    models = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for components in GMM_COMPONENTS:
            estimator = GaussianMixture(
                n_components=components,
                covariance_type="diag",
                reg_covar=GMM_REGULARIZATION,
                n_init=GMM_INITIALIZATIONS,
                random_state=random_state,
            )
            estimator.fit(coordinates)
            bics.append(float(estimator.bic(coordinates)))
            models.append(estimator)
    best_index = int(np.argmin(bics))
    selected = models[best_index]
    selected_k = GMM_COMPONENTS[best_index]
    return _GMMSelection(
        selected_k=selected_k,
        bics=tuple(bics),
        bic_improvement=float(bics[0] - bics[best_index]),
        labels=selected.predict(coordinates).astype(np.int64),
    )


def _subsample_multimodal_fraction(
    flattened: np.ndarray,
    *,
    candidate_id: str,
    protocol: str,
    iterations: int,
) -> float:
    if iterations <= 0:
        raise ValueError("subsample_iterations must be positive")
    protocol_stream = 100 if protocol == INITIAL_NOISE_ONLY else 200
    generator = np.random.default_rng(_seed(candidate_id, stream=protocol_stream))
    sample_count = max(2, round(flattened.shape[0] * SUBSAMPLE_FRACTION))
    passed = 0
    for ordinal in range(iterations):
        indices = np.sort(generator.choice(flattened.shape[0], sample_count, replace=False))
        projection = _project_pca(flattened[indices])
        if projection is None:
            continue
        selection = _select_gmm(
            projection.coordinates,
            random_state=_seed(
                candidate_id,
                stream=protocol_stream + 1,
                ordinal=ordinal,
            ),
        )
        if selection.selected_k >= 2 and selection.bic_improvement >= BIC_IMPROVEMENT_THRESHOLD:
            passed += 1
    return float(passed / iterations)


def _distance_cluster(distance_matrix: np.ndarray, *, clusters: int) -> np.ndarray:
    if clusters == 1:
        return np.zeros(distance_matrix.shape[0], dtype=np.int64)
    kwargs: dict[str, Any] = {
        "n_clusters": clusters,
        "linkage": "complete",
    }
    try:
        estimator = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:  # pragma: no cover - compatibility with older scikit-learn
        estimator = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return estimator.fit_predict(distance_matrix).astype(np.int64)


def _minimum_cluster_size(labels: np.ndarray) -> int:
    return int(min(np.count_nonzero(labels == label) for label in np.unique(labels)))


def _threshold_curve(distance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if distance_matrix.shape[0] < 2:
        return np.asarray([0.0]), np.asarray([1], dtype=np.int64)
    hierarchy = linkage(squareform(distance_matrix, checks=True), method="complete")
    positive = np.unique(hierarchy[:, 2][hierarchy[:, 2] > 0.0])
    if positive.size == 0:
        return np.asarray([0.0]), np.asarray([1], dtype=np.int64)
    thresholds = np.concatenate(
        [
            np.asarray([0.0]),
            positive,
            np.asarray([np.nextafter(float(positive[-1]), np.inf)]),
        ]
    )
    counts = np.asarray(
        [
            len(np.unique(fcluster(hierarchy, threshold, criterion="distance")))
            for threshold in thresholds
        ],
        dtype=np.int64,
    )
    return thresholds, counts


def _degenerate_analysis(
    actions: np.ndarray,
    *,
    candidate_id: str,
    protocol: str,
) -> ModeAnalysis:
    count = actions.shape[0]
    return ModeAnalysis(
        candidate_id=candidate_id,
        protocol=protocol,
        degenerate_unimodal=True,
        pca_components=0,
        selected_k=1,
        bics=(0.0, 0.0, 0.0, 0.0),
        bic_improvement=0.0,
        minimum_gmm_cluster_size=count,
        subsample_multimodal_fraction=0.0,
        minimum_distance_cluster_size=count,
        silhouette=0.0,
        adjusted_rand_index=1.0,
        multimodal=False,
        gmm_labels=np.zeros(count, dtype=np.int64),
        distance_labels=np.zeros(count, dtype=np.int64),
        pca_coordinates=np.zeros((count, 1), dtype=np.float64),
        pca_mean=np.mean(actions.reshape(count, -1), axis=0),
        pca_axes=np.zeros((0, actions.shape[1] * actions.shape[2]), dtype=np.float64),
        distance_matrix=pairwise_rms_distance(actions),
        threshold_values=np.asarray([0.0]),
        threshold_cluster_counts=np.asarray([1], dtype=np.int64),
    )


def analyze_action_samples(
    actions: np.ndarray,
    *,
    candidate_id: str,
    protocol: str,
    subsample_iterations: int = SUBSAMPLE_ITERATIONS,
) -> ModeAnalysis:
    """Apply the pre-registered hybrid exploratory mode decision."""

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (EXPECTED_STEPS, 2):
        raise ValueError(
            f"actions must have shape [samples,{EXPECTED_STEPS},2], got {values.shape}"
        )
    if values.shape[0] < max(GMM_COMPONENTS):
        raise ValueError("actions must contain at least four samples")
    if not np.isfinite(values).all():
        raise ValueError("actions must contain only finite values")
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")

    flattened = values.reshape(values.shape[0], -1)
    projection = _project_pca(flattened)
    if projection is None:
        return _degenerate_analysis(
            values,
            candidate_id=candidate_id,
            protocol=protocol,
        )

    selection = _select_gmm(
        projection.coordinates,
        random_state=_seed(
            candidate_id,
            stream=10 if protocol == INITIAL_NOISE_ONLY else 20,
        ),
    )
    gmm_minimum = _minimum_cluster_size(selection.labels)
    subsample_fraction = _subsample_multimodal_fraction(
        flattened,
        candidate_id=candidate_id,
        protocol=protocol,
        iterations=subsample_iterations,
    )
    distances = pairwise_rms_distance(values)
    distance_labels = _distance_cluster(distances, clusters=selection.selected_k)
    distance_minimum = _minimum_cluster_size(distance_labels)
    if selection.selected_k == 1:
        silhouette = 0.0
    else:
        silhouette = float(silhouette_score(distances, distance_labels, metric="precomputed"))
    adjusted_rand = float(adjusted_rand_score(selection.labels, distance_labels))
    thresholds, cluster_counts = _threshold_curve(distances)
    multimodal = bool(
        selection.selected_k >= 2
        and selection.bic_improvement >= BIC_IMPROVEMENT_THRESHOLD
        and gmm_minimum >= MINIMUM_MODE_SIZE
        and subsample_fraction >= SUBSAMPLE_PASS_FRACTION
        and distance_minimum >= MINIMUM_MODE_SIZE
        and silhouette >= SILHOUETTE_THRESHOLD
        and adjusted_rand >= ARI_THRESHOLD
    )
    return ModeAnalysis(
        candidate_id=candidate_id,
        protocol=protocol,
        degenerate_unimodal=False,
        pca_components=projection.coordinates.shape[1],
        selected_k=selection.selected_k,
        bics=selection.bics,
        bic_improvement=selection.bic_improvement,
        minimum_gmm_cluster_size=gmm_minimum,
        subsample_multimodal_fraction=subsample_fraction,
        minimum_distance_cluster_size=distance_minimum,
        silhouette=silhouette,
        adjusted_rand_index=adjusted_rand,
        multimodal=multimodal,
        gmm_labels=selection.labels,
        distance_labels=distance_labels,
        pca_coordinates=projection.coordinates,
        pca_mean=projection.mean,
        pca_axes=projection.axes,
        distance_matrix=distances,
        threshold_values=thresholds,
        threshold_cluster_counts=cluster_counts,
    )


def _rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _mode_medoids(actions: np.ndarray, result: ModeAnalysis) -> list[np.ndarray]:
    medoids = []
    for label in sorted(int(value) for value in np.unique(result.distance_labels)):
        indices = np.flatnonzero(result.distance_labels == label)
        within = result.distance_matrix[np.ix_(indices, indices)]
        medoid_index = indices[int(np.argmin(np.sum(within, axis=1)))]
        medoids.append(actions[medoid_index])
    return medoids


def bc_averaging_signature(
    dp_actions: np.ndarray,
    bc_action: np.ndarray,
    result: ModeAnalysis,
) -> BCAveragingSignature:
    """Measure whether BC is mean-like and low-density relative to E1b samples."""

    actions = np.asarray(dp_actions, dtype=np.float64)
    bc = np.asarray(bc_action, dtype=np.float64)
    if actions.ndim != 3 or bc.shape != actions.shape[1:]:
        raise ValueError("BC and DP action chunks have incompatible shapes")
    dp_mean = np.mean(actions, axis=0)
    medoids = _mode_medoids(actions, result)
    bc_to_mean = _rms_distance(bc, dp_mean)
    bc_to_medoids = [_rms_distance(bc, medoid) for medoid in medoids]
    bc_to_nearest_medoid = float(min(bc_to_medoids))
    bc_to_samples = np.asarray([_rms_distance(bc, sample) for sample in actions])
    bc_to_nearest_sample = float(np.min(bc_to_samples))
    leave_one_out = result.distance_matrix.copy()
    np.fill_diagonal(leave_one_out, np.inf)
    leave_one_out_nearest = np.min(leave_one_out, axis=1)
    q95 = float(np.quantile(leave_one_out_nearest, LOW_DENSITY_QUANTILE))
    density_percentile = float(np.mean(leave_one_out_nearest <= bc_to_nearest_sample))
    mean_like = bc_to_mean < bc_to_nearest_medoid
    low_density = bc_to_nearest_sample > q95
    return BCAveragingSignature(
        bc_to_dp_mean=bc_to_mean,
        bc_to_nearest_mode_medoid=bc_to_nearest_medoid,
        bc_to_nearest_sample=bc_to_nearest_sample,
        dp_leave_one_out_nn_q95=q95,
        bc_nn_density_percentile=density_percentile,
        mean_like=mean_like,
        low_density=low_density,
        averaging_consistent_signature=bool(result.multimodal and mean_like and low_density),
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _single_int(frame: pd.DataFrame, column: str, *, candidate_id: str) -> int:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{candidate_id} has inconsistent {column}: {values}")
    return int(values[0])


def _validate_metadata(
    action_rows: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    for model, expected in manifest["checkpoint_ids"].items():
        observed = action_rows[action_rows["model"] == model]["checkpoint_id"].unique().tolist()
        if observed != [expected]:
            raise ValueError(f"Checkpoint metadata for {model} does not match manifest: {observed}")
    revisions = action_rows["dataset_revision"].drop_duplicates().tolist()
    if revisions != [manifest["dataset_revision"]]:
        raise ValueError(f"Dataset revision does not match manifest: {revisions}")


def load_e1_candidates(input_dir: str | Path) -> tuple[list[CandidateActions], dict[str, Any]]:
    """Load and strictly validate the completed formal E1 artifact set."""

    root = Path(input_dir).expanduser().resolve()
    required = {
        name: root / name
        for name in (
            "action_chunks.parquet",
            "diversity_metrics.parquet",
            "metrics.json",
            "run_manifest.json",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required E1 artifacts: {missing}")

    manifest = _load_json(required["run_manifest.json"])
    metrics = _load_json(required["metrics.json"])
    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("E1 manifest schema_version must be 2")
    if metrics.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("E1 metrics schema_version must be 2")
    if manifest.get("experiment") != "E1":
        raise ValueError("Input manifest is not an E1 experiment")
    if manifest.get("num_action_samples_per_protocol") != EXPECTED_SAMPLES:
        raise ValueError("Formal E1 must contain 20 action samples per protocol")
    if metrics.get("num_candidates") != EXPECTED_CANDIDATES:
        raise ValueError("Formal E1 must contain 10 candidates")
    if float(metrics.get("bc_repeat_max_error", np.inf)) > BC_TOLERANCE:
        raise ValueError("BC repeat error exceeds the registered tolerance")

    action_rows = pd.read_parquet(required["action_chunks.parquet"])
    diversity_rows = pd.read_parquet(required["diversity_metrics.parquet"])
    if {int(value) for value in action_rows["schema_version"].unique()} != {
        EXPECTED_SCHEMA_VERSION
    }:
        raise ValueError("Action rows must use E1 schema version 2")
    if len(diversity_rows) != EXPECTED_CANDIDATES * len(PROTOCOLS):
        raise ValueError("E1 diversity metrics must contain two rows per candidate")
    _validate_metadata(action_rows, manifest)

    candidate_keys = manifest.get("candidate_keys")
    if not isinstance(candidate_keys, list) or len(candidate_keys) != EXPECTED_CANDIDATES:
        raise ValueError("E1 manifest must contain exactly 10 candidate keys")
    observed_ids = {str(value) for value in action_rows["candidate_id"].unique()}
    expected_ids = {str(item["candidate_id"]) for item in candidate_keys}
    if observed_ids != expected_ids:
        raise ValueError("Action-row candidates do not match the E1 manifest")

    candidates = []
    for key in candidate_keys:
        candidate_id = str(key["candidate_id"])
        rows = action_rows[action_rows["candidate_id"] == candidate_id]
        episode_index = _single_int(rows, "episode_index", candidate_id=candidate_id)
        frame_index = _single_int(rows, "frame_index", candidate_id=candidate_id)
        if episode_index != int(key["episode_index"]) or frame_index != int(key["frame_index"]):
            raise ValueError(f"Candidate key metadata changed for {candidate_id}")
        protocol_actions = {}
        for protocol in PROTOCOLS:
            protocol_rows = rows[
                (rows["model"] == "diffusion") & (rows["sampling_protocol"] == protocol)
            ]
            protocol_actions[protocol] = reconstruct_action_chunks(
                protocol_rows,
                expected_samples=EXPECTED_SAMPLES,
                expected_steps=EXPECTED_STEPS,
            )
        bc_rows = rows[
            (rows["model"] == "bc_chunk") & (rows["sampling_protocol"] == BC_DETERMINISTIC)
        ]
        bc_calls = reconstruct_action_chunks(
            bc_rows,
            expected_samples=EXPECTED_BC_CALLS,
            expected_steps=EXPECTED_STEPS,
        )
        if float(np.max(np.abs(bc_calls - bc_calls[0]))) > BC_TOLERANCE:
            raise ValueError(f"BC calls are not deterministic for {candidate_id}")
        candidates.append(
            CandidateActions(
                candidate_id=candidate_id,
                episode_index=episode_index,
                frame_index=frame_index,
                initial_noise_only=protocol_actions[INITIAL_NOISE_ONLY],
                full_sampling_seed=protocol_actions[FULL_SAMPLING_SEED],
                bc_action=bc_calls[0],
            )
        )
    return candidates, manifest


def _project_markers(
    result: ModeAnalysis,
    *,
    dp_mean: np.ndarray,
    bc_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if result.pca_axes.shape[0] == 0:
        return np.zeros(1), np.zeros(1)
    dp_flat = dp_mean.reshape(-1)
    bc_flat = bc_action.reshape(-1)
    return (
        (dp_flat - result.pca_mean) @ result.pca_axes.T,
        (bc_flat - result.pca_mean) @ result.pca_axes.T,
    )


def _xy(value: np.ndarray) -> tuple[float, float]:
    return float(value[0]), float(value[1]) if value.shape[0] > 1 else 0.0


def _write_candidate_figure(
    path: Path,
    *,
    candidate: CandidateActions,
    result: ModeAnalysis,
    signature: BCAveragingSignature,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = result.pca_coordinates[:, 0]
    y = result.pca_coordinates[:, 1] if result.pca_coordinates.shape[1] > 1 else np.zeros_like(x)
    axes[0].scatter(
        x,
        y,
        c=result.distance_labels,
        cmap="tab10",
        edgecolors="black",
        linewidths=0.4,
        label="DP samples",
    )
    for label in sorted(int(value) for value in np.unique(result.distance_labels)):
        indices = np.flatnonzero(result.distance_labels == label)
        within = result.distance_matrix[np.ix_(indices, indices)]
        medoid_index = indices[int(np.argmin(np.sum(within, axis=1)))]
        axes[0].scatter(
            x[medoid_index],
            y[medoid_index],
            marker="X",
            s=120,
            color="black",
        )
    dp_mean_projection, bc_projection = _project_markers(
        result,
        dp_mean=np.mean(candidate.full_sampling_seed, axis=0),
        bc_action=candidate.bc_action,
    )
    axes[0].scatter(*_xy(dp_mean_projection), marker="P", s=140, color="orange", label="DP mean")
    axes[0].scatter(*_xy(bc_projection), marker="*", s=180, color="red", label="BC")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].set_title(
        f"E1b modes: K={result.selected_k}, pass={result.multimodal}\n"
        f"BC averaging signature={signature.averaging_consistent_signature}"
    )
    axes[0].legend(loc="best")

    axes[1].step(
        result.threshold_values,
        result.threshold_cluster_counts,
        where="post",
        color="tab:blue",
    )
    axes[1].set_xlabel("Complete-linkage distance threshold")
    axes[1].set_ylabel("Cluster count")
    axes[1].set_title("Exact threshold sensitivity")
    axes[1].grid(alpha=0.25)
    figure.suptitle(candidate.candidate_id)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _analysis_summary_row(
    candidate: CandidateActions,
    result: ModeAnalysis,
    signature: BCAveragingSignature | None,
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "episode_index": candidate.episode_index,
        "frame_index": candidate.frame_index,
        "sampling_protocol": result.protocol,
        "degenerate_unimodal": result.degenerate_unimodal,
        "pca_components": result.pca_components,
        "selected_k": result.selected_k,
        "bics_json": json.dumps(result.bics),
        "bic_improvement": result.bic_improvement,
        "minimum_gmm_cluster_size": result.minimum_gmm_cluster_size,
        "subsample_multimodal_fraction": result.subsample_multimodal_fraction,
        "minimum_distance_cluster_size": result.minimum_distance_cluster_size,
        "silhouette": result.silhouette,
        "adjusted_rand_index": result.adjusted_rand_index,
        "multimodal": result.multimodal,
        "gmm_cluster_sizes_json": json.dumps(
            [
                int(np.count_nonzero(result.gmm_labels == label))
                for label in sorted(np.unique(result.gmm_labels))
            ]
        ),
        "distance_cluster_sizes_json": json.dumps(
            [
                int(np.count_nonzero(result.distance_labels == label))
                for label in sorted(np.unique(result.distance_labels))
            ]
        ),
        "threshold_values_json": json.dumps(result.threshold_values.tolist()),
        "threshold_cluster_counts_json": json.dumps(result.threshold_cluster_counts.tolist()),
        "bc_to_dp_mean": None if signature is None else signature.bc_to_dp_mean,
        "bc_to_nearest_mode_medoid": (
            None if signature is None else signature.bc_to_nearest_mode_medoid
        ),
        "bc_to_nearest_sample": (None if signature is None else signature.bc_to_nearest_sample),
        "dp_leave_one_out_nn_q95": (
            None if signature is None else signature.dp_leave_one_out_nn_q95
        ),
        "bc_nn_density_percentile": (
            None if signature is None else signature.bc_nn_density_percentile
        ),
        "bc_mean_like": None if signature is None else signature.mean_like,
        "bc_low_density": None if signature is None else signature.low_density,
        "averaging_consistent_signature": (
            None if signature is None else signature.averaging_consistent_signature
        ),
    }


def _assignment_rows(
    candidate: CandidateActions,
    result: ModeAnalysis,
) -> list[dict[str, Any]]:
    rows = []
    for sample_ordinal in range(result.gmm_labels.shape[0]):
        coordinates = result.pca_coordinates[sample_ordinal]
        rows.append(
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "candidate_id": candidate.candidate_id,
                "episode_index": candidate.episode_index,
                "frame_index": candidate.frame_index,
                "sampling_protocol": result.protocol,
                "sample_ordinal": sample_ordinal,
                "gmm_label": int(result.gmm_labels[sample_ordinal]),
                "distance_cluster_label": int(result.distance_labels[sample_ordinal]),
                "pca_1": float(coordinates[0]),
                "pca_2": (float(coordinates[1]) if coordinates.shape[0] > 1 else 0.0),
                "pca_3": (float(coordinates[2]) if coordinates.shape[0] > 2 else 0.0),
            }
        )
    return rows


def _write_metrics_markdown(path: Path, metrics: dict[str, Any]) -> None:
    if metrics["negative_control_failed"]:
        conclusion = (
            "The E1a negative control failed; do not make an aggregate action-multimodality claim."
        )
    elif metrics["e1b_multimodal_candidate_count"] > 0:
        conclusion = (
            "Existing E1 samples provide exploratory fixed-observation action-mode evidence."
        )
    else:
        conclusion = (
            "Existing E1 samples do not provide exploratory fixed-observation action-mode "
            "evidence under the pre-registered gates."
        )
    lines = [
        "# Fixed-observation Action-mode Analysis",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "This result is exploratory and does not prove that BC failure is caused by MSE averaging.",
        "",
        "## Evidence",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {rendered} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_fixed_observation_action_modes(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    subsample_iterations: int = SUBSAMPLE_ITERATIONS,
) -> Path:
    """Run the read-only exploratory analyzer on one completed formal E1 output."""

    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root == input_root:
        raise ValueError("output_dir must differ from input_dir")
    candidates, input_manifest = load_e1_candidates(input_root)
    required_names = (
        "action_chunks.parquet",
        "diversity_metrics.parquet",
        "metrics.json",
        "run_manifest.json",
    )
    input_hashes = {name: sha256_file(input_root / name) for name in required_names}

    with AtomicOutputDirectory(output_root, overwrite=overwrite) as temporary:
        (temporary / "figures").mkdir()
        summary_rows = []
        assignment_rows = []
        protocol_results: dict[tuple[str, str], ModeAnalysis] = {}
        signatures: dict[str, BCAveragingSignature] = {}
        for index, candidate in enumerate(candidates, start=1):
            print(
                f"[{index}/{len(candidates)}] analyzing {candidate.candidate_id}",
                flush=True,
            )
            for protocol, actions in (
                (INITIAL_NOISE_ONLY, candidate.initial_noise_only),
                (FULL_SAMPLING_SEED, candidate.full_sampling_seed),
            ):
                result = analyze_action_samples(
                    actions,
                    candidate_id=candidate.candidate_id,
                    protocol=protocol,
                    subsample_iterations=subsample_iterations,
                )
                protocol_results[(candidate.candidate_id, protocol)] = result
                assignment_rows.extend(_assignment_rows(candidate, result))
            e1b = protocol_results[(candidate.candidate_id, FULL_SAMPLING_SEED)]
            signature = bc_averaging_signature(
                candidate.full_sampling_seed,
                candidate.bc_action,
                e1b,
            )
            signatures[candidate.candidate_id] = signature
            for protocol in PROTOCOLS:
                result = protocol_results[(candidate.candidate_id, protocol)]
                summary_rows.append(
                    _analysis_summary_row(
                        candidate,
                        result,
                        signature if protocol == FULL_SAMPLING_SEED else None,
                    )
                )
            _write_candidate_figure(
                temporary / "figures" / f"{candidate.candidate_id}.png",
                candidate=candidate,
                result=e1b,
                signature=signature,
            )

        e1a_count = sum(
            protocol_results[(candidate.candidate_id, INITIAL_NOISE_ONLY)].multimodal
            for candidate in candidates
        )
        e1b_count = sum(
            protocol_results[(candidate.candidate_id, FULL_SAMPLING_SEED)].multimodal
            for candidate in candidates
        )
        signature_count = sum(
            signature.averaging_consistent_signature for signature in signatures.values()
        )
        negative_control_failed = e1a_count / len(candidates) > NEGATIVE_CONTROL_FAILURE_FRACTION
        metrics = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "exploratory_only": True,
            "num_candidates": len(candidates),
            "num_action_samples_per_protocol": EXPECTED_SAMPLES,
            "e1a_multimodal_candidate_count": int(e1a_count),
            "e1a_multimodal_candidate_fraction": float(e1a_count / len(candidates)),
            "e1b_multimodal_candidate_count": int(e1b_count),
            "e1b_multimodal_candidate_fraction": float(e1b_count / len(candidates)),
            "negative_control_failed": bool(negative_control_failed),
            "aggregate_action_multimodality_evidence": bool(
                e1b_count > 0 and not negative_control_failed
            ),
            "bc_averaging_signature_count": int(signature_count),
            "bc_averaging_signature_fraction": float(signature_count / len(candidates)),
            "subsample_iterations": int(subsample_iterations),
        }
        write_records_parquet(temporary / "candidate_summary.parquet", summary_rows)
        write_records_parquet(temporary / "mode_assignments.parquet", assignment_rows)
        write_json_atomic(temporary / "metrics.json", metrics)
        _write_metrics_markdown(temporary / "metrics.md", metrics)
        write_json_atomic(
            temporary / "analysis_manifest.json",
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "experiment": "E1 fixed-observation action-mode analysis",
                "exploratory_only": True,
                "input_artifact_hashes": input_hashes,
                "input_identity": {
                    "experiment_schema": input_manifest["experiment_schema"],
                    "dataset_revision": input_manifest["dataset_revision"],
                    "checkpoint_ids": input_manifest["checkpoint_ids"],
                    "candidate_ids": [candidate.candidate_id for candidate in candidates],
                },
                "analysis": {
                    "seed": ANALYSIS_SEED,
                    "gmm_components": list(GMM_COMPONENTS),
                    "gmm_initializations": GMM_INITIALIZATIONS,
                    "gmm_regularization": GMM_REGULARIZATION,
                    "bic_improvement_threshold": BIC_IMPROVEMENT_THRESHOLD,
                    "minimum_mode_size": MINIMUM_MODE_SIZE,
                    "subsample_iterations": int(subsample_iterations),
                    "subsample_fraction": SUBSAMPLE_FRACTION,
                    "subsample_pass_fraction": SUBSAMPLE_PASS_FRACTION,
                    "silhouette_threshold": SILHOUETTE_THRESHOLD,
                    "adjusted_rand_index_threshold": ARI_THRESHOLD,
                    "low_density_quantile": LOW_DENSITY_QUANTILE,
                    "negative_control_failure_fraction": (NEGATIVE_CONTROL_FAILURE_FRACTION),
                },
                "runtime": collect_runtime_provenance(practice_root()),
            },
        )
    return output_root


def _build_parser() -> argparse.ArgumentParser:
    default_input = default_output_root() / "pilot" / "e1_fixed_observation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_input / "action_mode_analysis",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output = analyze_fixed_observation_action_modes(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
