"""Load and validate the immutable PushT episode split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_SPLIT_SEED,
    default_split_path,
    load_json_object,
)

EXPECTED_SPLIT_SIZES = {
    "train": 165,
    "development": 20,
    "audit": 21,
    "smoke_train": 8,
}


def _integer_list(payload: dict[str, Any], key: str) -> tuple[int, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw
    ):
        raise ValueError(f"{key} must be an array of integers")
    return tuple(raw)


@dataclass(frozen=True)
class EpisodeSplit:
    source_path: Path
    schema_version: int
    dataset_repo_id: str
    dataset_revision: str
    metadata_info_sha256: str
    num_episodes: int
    split_seed: int
    train: tuple[int, ...]
    development: tuple[int, ...]
    audit: tuple[int, ...]
    smoke_train: tuple[int, ...]
    development_environment_seeds: tuple[int, ...]
    final_environment_seeds: tuple[int, ...]
    e2_environment_seeds: tuple[int, ...]
    manifest_sha256: str

    def episodes_for_mode(self, mode: str) -> tuple[int, ...]:
        if mode == "smoke":
            return self.smoke_train
        if mode == "pilot":
            return self.train
        if mode == "development":
            return self.development
        if mode == "audit":
            return self.audit
        raise ValueError("mode must be one of smoke, pilot, development, audit")


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_unique_range(name: str, values: tuple[int, ...], num_episodes: int) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate episode IDs")
    invalid = [value for value in values if not 0 <= value < num_episodes]
    if invalid:
        raise ValueError(f"{name} contains out-of-range episode IDs: {invalid}")


def _expected_permutation(num_episodes: int, split_seed: int) -> tuple[int, ...]:
    episode_ids = np.arange(num_episodes, dtype=np.int64)
    return tuple(
        int(value) for value in np.random.default_rng(split_seed).permutation(episode_ids).tolist()
    )


def load_split(path: str | Path | None = None) -> EpisodeSplit:
    resolved = (Path(path) if path is not None else default_split_path()).expanduser().resolve()
    payload = load_json_object(resolved)

    schema_version = payload.get("schema_version")
    num_episodes = payload.get("num_episodes")
    split_seed = payload.get("split_seed")
    if schema_version != 1:
        raise ValueError(f"Unsupported split schema_version: {schema_version!r}")
    if num_episodes != 206:
        raise ValueError(f"Expected 206 PushT episodes, got {num_episodes!r}")
    if split_seed != DEFAULT_SPLIT_SEED:
        raise ValueError(f"Expected split_seed={DEFAULT_SPLIT_SEED}, got {split_seed!r}")
    if payload.get("dataset_repo_id") != DATASET_REPO_ID:
        raise ValueError(f"Unexpected dataset_repo_id: {payload.get('dataset_repo_id')!r}")
    if payload.get("dataset_revision") != DATASET_REVISION:
        raise ValueError(f"Unexpected dataset_revision: {payload.get('dataset_revision')!r}")
    metadata_info_sha256 = payload.get("metadata_info_sha256")
    if (
        not isinstance(metadata_info_sha256, str)
        or len(metadata_info_sha256) != 64
        or any(character not in "0123456789abcdef" for character in metadata_info_sha256)
    ):
        raise ValueError("metadata_info_sha256 must be a lowercase SHA256 digest")
    if payload.get("numpy_bit_generator") != "PCG64":
        raise ValueError("The committed split must use NumPy PCG64")

    train = _integer_list(payload, "train")
    development = _integer_list(payload, "development")
    audit = _integer_list(payload, "audit")
    smoke_train = _integer_list(payload, "smoke_train")
    dev_seeds = _integer_list(payload, "development_environment_seeds")
    final_seeds = _integer_list(payload, "final_environment_seeds")
    e2_seeds = _integer_list(payload, "e2_environment_seeds")

    for name, expected_size in EXPECTED_SPLIT_SIZES.items():
        values = {
            "train": train,
            "development": development,
            "audit": audit,
            "smoke_train": smoke_train,
        }[name]
        if len(values) != expected_size:
            raise ValueError(f"{name} must contain {expected_size} episodes, got {len(values)}")
        _validate_unique_range(name, values, num_episodes)

    train_set, dev_set, audit_set = set(train), set(development), set(audit)
    if train_set & dev_set or train_set & audit_set or dev_set & audit_set:
        raise ValueError("train, development, and audit episode sets must be disjoint")
    if train_set | dev_set | audit_set != set(range(num_episodes)):
        raise ValueError("train, development, and audit must cover exactly episode IDs 0..205")
    if smoke_train != train[: len(smoke_train)]:
        raise ValueError("smoke_train must be the first eight committed train episodes")

    expected = _expected_permutation(num_episodes, split_seed)
    if train + development + audit != expected:
        raise ValueError("Committed split does not match the documented PCG64 permutation")
    if dev_seeds != tuple(range(1000, 1020)):
        raise ValueError("development_environment_seeds must be exactly 1000..1019")
    if final_seeds != tuple(range(2000, 2050)):
        raise ValueError("final_environment_seeds must be exactly 2000..2049")
    if e2_seeds != final_seeds[:10]:
        raise ValueError("e2_environment_seeds must be the first ten final seeds")

    return EpisodeSplit(
        source_path=resolved,
        schema_version=schema_version,
        dataset_repo_id=DATASET_REPO_ID,
        dataset_revision=DATASET_REVISION,
        metadata_info_sha256=metadata_info_sha256,
        num_episodes=num_episodes,
        split_seed=split_seed,
        train=train,
        development=development,
        audit=audit,
        smoke_train=smoke_train,
        development_environment_seeds=dev_seeds,
        final_environment_seeds=final_seeds,
        e2_environment_seeds=e2_seeds,
        manifest_sha256=canonical_json_sha256(payload),
    )


def validate_dataset_metadata(split: EpisodeSplit, metadata: Any) -> None:
    repo_id = getattr(metadata, "repo_id", None)
    if repo_id is not None and repo_id != split.dataset_repo_id:
        raise ValueError(f"Dataset repo_id={repo_id!r}, expected {split.dataset_repo_id!r}")
    episodes = getattr(metadata, "episodes", None)
    if episodes is None:
        raise ValueError("Dataset metadata does not expose episodes")
    if len(episodes) != split.num_episodes:
        raise ValueError(
            f"Dataset metadata has {len(episodes)} episodes, expected {split.num_episodes}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=default_split_path())
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "development", "audit"),
        default="pilot",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    split = load_split(args.split_file)
    episodes = split.episodes_for_mode(args.mode)
    if args.json:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "episodes": list(episodes),
                    "manifest_sha256": split.manifest_sha256,
                },
                sort_keys=True,
            )
        )
    else:
        print(f"PushT split valid: {split.source_path}")
        print(f"{args.mode} episodes ({len(episodes)}): {list(episodes)}")
        print(f"manifest sha256: {split.manifest_sha256}")


if __name__ == "__main__":
    main()
