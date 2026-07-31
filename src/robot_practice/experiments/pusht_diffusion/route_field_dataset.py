"""Validated route-field assignments and a read-only LeRobot dataset wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from robot_practice.policies.route_field_conditioning import ROUTE_FIELD

from .config import load_json_object
from .provenance import sha256_file
from .route_field_data import (
    ROUTE_FIELD_CHANNELS,
    ROUTE_FIELD_GRID_SIZE,
    ROUTE_FIELD_SCHEMA_VERSION,
    route_to_field,
)


@dataclass(frozen=True)
class RouteFieldAssignment:
    episode_index: int
    split: str
    raw_route_cells: tuple[int, ...]
    route_cells: tuple[int, ...]
    family_id: int | None
    training_condition_id: int | None
    eligible: bool
    exclusion_reason: str | None
    backtrack_removed_cells: int
    loop_removed_cells: int
    num_frames: int


@dataclass(frozen=True)
class RouteFieldSidecar:
    root: Path
    manifest_path: Path
    fields: torch.Tensor
    assignments: Mapping[int, RouteFieldAssignment]
    manifest: Mapping[str, Any]

    @property
    def num_families(self) -> int:
        return int(self.fields.shape[0])

    def training_episodes(self, requested: Sequence[int]) -> tuple[int, ...]:
        return tuple(
            int(episode)
            for episode in requested
            if self.assignments[int(episode)].training_condition_id is not None
        )

    def field_for_episode(self, episode_index: int) -> torch.Tensor:
        try:
            assignment = self.assignments[int(episode_index)]
        except KeyError as exc:
            raise KeyError(f"Route-field sidecar has no episode {episode_index}") from exc
        family_id = assignment.training_condition_id
        if family_id is None:
            raise ValueError(
                f"Episode {episode_index} has no training route field; "
                "only eligible train episodes may be sampled"
            )
        return self.fields[family_id]


def load_route_field_sidecar(path: str | Path) -> RouteFieldSidecar:
    """Load and validate one complete route-field sidecar."""

    source = Path(path).expanduser().resolve()
    manifest_path = (
        source if source.name == "route_manifest.json" else source / "route_manifest.json"
    )
    manifest = load_json_object(manifest_path)
    if (
        manifest.get("schema_version") != ROUTE_FIELD_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"Incomplete route-field manifest: {manifest_path}")
    grid = manifest.get("grid")
    field_spec = manifest.get("field")
    if not isinstance(grid, dict) or int(grid.get("size", -1)) != ROUTE_FIELD_GRID_SIZE:
        raise ValueError(
            f"{manifest_path}: expected a {ROUTE_FIELD_GRID_SIZE}x{ROUTE_FIELD_GRID_SIZE} grid"
        )
    expected_shape = [
        len(ROUTE_FIELD_CHANNELS),
        ROUTE_FIELD_GRID_SIZE,
        ROUTE_FIELD_GRID_SIZE,
    ]
    if (
        not isinstance(field_spec, dict)
        or field_spec.get("channels") != list(ROUTE_FIELD_CHANNELS)
        or field_spec.get("shape") != expected_shape
    ):
        raise ValueError(f"{manifest_path}: route-field channel contract does not match")

    root = manifest_path.parent
    assignments_path = root / "episode_assignments.parquet"
    fields_path = root / "route_fields.npz"
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise TypeError(f"{manifest_path}: artifact_hashes must be an object")
    for artifact in (assignments_path, fields_path):
        if not artifact.is_file():
            raise FileNotFoundError(f"Route-field artifact is missing: {artifact}")
        if artifact_hashes.get(artifact.name) != sha256_file(artifact):
            raise ValueError(f"Route-field artifact hash mismatch: {artifact}")

    with np.load(fields_path, allow_pickle=False) as archive:
        family_ids = np.asarray(archive["family_ids"], dtype=np.int64)
        values = np.asarray(archive["fields"], dtype=np.float32)
    expected_ids = np.arange(values.shape[0], dtype=np.int64)
    if not np.array_equal(family_ids, expected_ids):
        raise ValueError("Route-field family IDs must be contiguous and zero-based")
    if values.ndim != 4 or tuple(values.shape[1:]) != tuple(expected_shape):
        raise ValueError(f"Invalid route-field tensor shape: {values.shape}")
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("Route fields must be non-empty and finite")
    families = manifest.get("canonical_families")
    if not isinstance(families, list) or len(families) != values.shape[0]:
        raise ValueError("Canonical family metadata does not match route_fields.npz")
    for family_id, family in enumerate(families):
        if int(family["family_id"]) != family_id:
            raise ValueError("Canonical family IDs must be contiguous and sorted")
        expected_field = route_to_field(family["route_cells"])
        if not np.array_equal(values[family_id], expected_field):
            raise ValueError(f"Canonical route field does not match family {family_id}")

    assignments: dict[int, RouteFieldAssignment] = {}
    for row in pq.read_table(assignments_path).to_pylist():
        episode = int(row["episode_index"])
        if episode in assignments:
            raise ValueError(f"Duplicate route-field assignment for episode {episode}")
        family = row.get("family_id")
        training_condition = row.get("training_condition_id")
        assignment = RouteFieldAssignment(
            episode_index=episode,
            split=str(row["split"]),
            raw_route_cells=tuple(int(cell) for cell in (row.get("raw_route_cells") or ())),
            route_cells=tuple(int(cell) for cell in (row.get("route_cells") or ())),
            family_id=None if family is None else int(family),
            training_condition_id=(None if training_condition is None else int(training_condition)),
            eligible=bool(row["eligible"]),
            exclusion_reason=(
                None if row.get("exclusion_reason") is None else str(row["exclusion_reason"])
            ),
            backtrack_removed_cells=int(row.get("backtrack_removed_cells", 0)),
            loop_removed_cells=int(row.get("loop_removed_cells", 0)),
            num_frames=int(row["num_frames"]),
        )
        if not assignment.raw_route_cells or not assignment.route_cells:
            raise ValueError(f"Episode {episode} has an empty RF7 route")
        if (
            len(assignment.raw_route_cells) - len(assignment.route_cells)
            != assignment.backtrack_removed_cells + assignment.loop_removed_cells
        ):
            raise ValueError(f"Episode {episode} route simplification accounting mismatch")
        if assignment.training_condition_id is not None:
            if assignment.split != "train":
                raise ValueError(f"Non-train episode {episode} has a training condition")
            if not 0 <= assignment.training_condition_id < values.shape[0]:
                raise ValueError(f"Episode {episode} has an invalid route family")
            if assignment.training_condition_id != assignment.family_id:
                raise ValueError(f"Episode {episode} condition and family IDs disagree")
        if assignment.eligible != (assignment.family_id is not None):
            raise ValueError(f"Episode {episode} eligibility and family ID disagree")
        assignments[episode] = assignment
    if not assignments:
        raise ValueError(f"Route-field assignment table is empty: {assignments_path}")
    training = manifest.get("training")
    if not isinstance(training, dict):
        raise TypeError(f"{manifest_path}: training must be an object")
    manifest_training_episodes = tuple(int(value) for value in training.get("episode_ids", ()))
    assignment_training_episodes = tuple(
        episode
        for episode in sorted(assignments)
        if assignments[episode].training_condition_id is not None
    )
    if tuple(sorted(manifest_training_episodes)) != assignment_training_episodes:
        raise ValueError("Manifest training episode IDs do not match the assignment table")

    return RouteFieldSidecar(
        root=root,
        manifest_path=manifest_path,
        fields=torch.from_numpy(values),
        assignments=assignments,
        manifest=manifest,
    )


def _episode_index(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


class RouteFieldConditionedDataset(Dataset):
    """Attach the canonical per-episode field without mutating LeRobot data."""

    def __init__(self, dataset: Dataset, sidecar: RouteFieldSidecar) -> None:
        self.dataset = dataset
        self.sidecar = sidecar
        episodes = tuple(int(value) for value in dataset.episodes)
        missing = sorted(set(episodes) - set(sidecar.assignments))
        if missing:
            raise ValueError(f"Route-field sidecar is missing selected episodes: {missing}")
        invalid = [
            episode
            for episode in episodes
            if sidecar.assignments[episode].training_condition_id is None
        ]
        if invalid:
            raise ValueError(f"Selected episodes have no training route field: {invalid}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.dataset[index]
        if "episode_index" not in original:
            raise KeyError("Wrapped LeRobot sample has no episode_index")
        sample = dict(original)
        episode = _episode_index(original["episode_index"])
        sample[ROUTE_FIELD] = self.sidecar.field_for_episode(episode).clone()
        return sample

    def __getattr__(self, name: str) -> Any:
        if name in {"dataset", "sidecar"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


def wrap_dataset_with_route_fields(
    dataset: Dataset,
    *,
    route_manifest: str | Path,
) -> RouteFieldConditionedDataset:
    return RouteFieldConditionedDataset(dataset, load_route_field_sidecar(route_manifest))


__all__ = [
    "RouteFieldAssignment",
    "RouteFieldConditionedDataset",
    "RouteFieldSidecar",
    "load_route_field_sidecar",
    "wrap_dataset_with_route_fields",
]
