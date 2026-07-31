"""Sidecar prototype assignments and a read-only LeRobot dataset wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from robot_practice.policies.mode_conditioning import PROTOTYPE_ID

from .config import load_json_object
from .provenance import sha256_file


@dataclass(frozen=True)
class PrototypeAssignment:
    episode_index: int
    split: str
    prototype_id: int | None
    training_condition_id: int | None
    distance_to_medoid: float | None
    classifiable: bool
    eligible: bool
    exclusion_reason: str | None
    num_frames: int


@dataclass(frozen=True)
class PrototypeSidecar:
    root: Path
    manifest_path: Path
    selected_k: int
    null_prototype_id: int
    assignments: Mapping[int, PrototypeAssignment]
    manifest: Mapping[str, Any]

    def condition_for_episode(self, episode_index: int) -> int:
        try:
            assignment = self.assignments[int(episode_index)]
        except KeyError as exc:
            raise KeyError(f"Prototype sidecar has no episode {episode_index}") from exc
        condition = assignment.training_condition_id
        if condition is None:
            raise ValueError(
                f"Episode {episode_index} has no training condition; "
                "only train-split episodes may be sampled"
            )
        if not 0 <= condition <= self.null_prototype_id:
            raise ValueError(f"Episode {episode_index} has invalid condition ID {condition}")
        return condition

    def accounting_for_episodes(self, episodes: list[int] | tuple[int, ...]) -> dict[str, int]:
        conditional_frames = 0
        ineligible_null_frames = 0
        for episode in episodes:
            assignment = self.assignments[int(episode)]
            if assignment.training_condition_id == self.null_prototype_id:
                ineligible_null_frames += assignment.num_frames
            else:
                conditional_frames += assignment.num_frames
        return {
            "eligible_conditional_frames": conditional_frames,
            "ineligible_always_null_frames": ineligible_null_frames,
            "total_frames": conditional_frames + ineligible_null_frames,
        }


def load_prototype_sidecar(path: str | Path) -> PrototypeSidecar:
    """Load and validate one complete E5 prototype sidecar."""

    source = Path(path).expanduser().resolve()
    manifest_path = (
        source if source.name == "prototype_manifest.json" else (source / "prototype_manifest.json")
    )
    manifest = load_json_object(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError(f"Incomplete E5 prototype manifest: {manifest_path}")
    selected_k = int(manifest["selected_k"])
    null_id = int(manifest["null_prototype_id"])
    if selected_k not in {2, 3, 4} or null_id != selected_k:
        raise ValueError(
            f"Invalid E5 K/NULL IDs in {manifest_path}: K={selected_k}, NULL={null_id}"
        )
    root = manifest_path.parent
    assignments_path = root / "episode_assignments.parquet"
    if not assignments_path.is_file():
        raise FileNotFoundError(f"E5 assignments are missing: {assignments_path}")
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise TypeError(f"{manifest_path}: artifact_hashes must be an object")
    expected_hash = artifact_hashes.get("episode_assignments.parquet")
    actual_hash = sha256_file(assignments_path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"E5 assignment hash mismatch: expected {expected_hash!r}, got {actual_hash!r}"
        )

    assignments: dict[int, PrototypeAssignment] = {}
    for row in pq.read_table(assignments_path).to_pylist():
        episode = int(row["episode_index"])
        if episode in assignments:
            raise ValueError(f"Duplicate E5 assignment for episode {episode}")
        prototype = row["prototype_id"]
        training_condition = row["training_condition_id"]
        assignment = PrototypeAssignment(
            episode_index=episode,
            split=str(row.get("split", "unknown")),
            prototype_id=None if prototype is None else int(prototype),
            training_condition_id=(None if training_condition is None else int(training_condition)),
            distance_to_medoid=(
                None if row.get("distance_to_medoid") is None else float(row["distance_to_medoid"])
            ),
            classifiable=bool(row.get("classifiable", row["eligible"])),
            eligible=bool(row["eligible"]),
            exclusion_reason=(
                None if row.get("exclusion_reason") is None else str(row["exclusion_reason"])
            ),
            num_frames=int(row["num_frames"]),
        )
        if assignment.eligible and assignment.prototype_id is None:
            raise ValueError(f"Eligible episode {episode} has no prototype ID")
        if (
            assignment.training_condition_id is not None
            and not 0 <= assignment.training_condition_id <= null_id
        ):
            raise ValueError(
                f"Episode {episode} has invalid training condition "
                f"{assignment.training_condition_id}"
            )
        assignments[episode] = assignment
    if not assignments:
        raise ValueError(f"E5 assignment table is empty: {assignments_path}")
    return PrototypeSidecar(
        root=root,
        manifest_path=manifest_path,
        selected_k=selected_k,
        null_prototype_id=null_id,
        assignments=assignments,
        manifest=manifest,
    )


def _episode_index(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


class PrototypeConditionedDataset(Dataset):
    """Add a categorical condition without changing the wrapped dataset."""

    def __init__(self, dataset: Dataset, sidecar: PrototypeSidecar) -> None:
        self.dataset = dataset
        self.sidecar = sidecar
        episodes = tuple(int(value) for value in dataset.episodes)
        missing = sorted(set(episodes) - set(sidecar.assignments))
        if missing:
            raise ValueError(f"Prototype sidecar is missing selected episodes: {missing}")
        for episode in episodes:
            sidecar.condition_for_episode(episode)
        self.accounting = sidecar.accounting_for_episodes(episodes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        original = self.dataset[index]
        if "episode_index" not in original:
            raise KeyError("Wrapped LeRobot sample has no episode_index")
        episode = _episode_index(original["episode_index"])
        sample = dict(original)
        sample[PROTOTYPE_ID] = torch.tensor(
            self.sidecar.condition_for_episode(episode),
            dtype=torch.long,
        )
        return sample

    def __getattr__(self, name: str) -> Any:
        if name in {"dataset", "sidecar", "accounting"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


def wrap_dataset_with_prototypes(
    dataset: Dataset,
    *,
    prototype_manifest: str | Path,
) -> PrototypeConditionedDataset:
    return PrototypeConditionedDataset(
        dataset,
        load_prototype_sidecar(prototype_manifest),
    )


__all__ = [
    "PrototypeAssignment",
    "PrototypeConditionedDataset",
    "PrototypeSidecar",
    "load_prototype_sidecar",
    "wrap_dataset_with_prototypes",
]
