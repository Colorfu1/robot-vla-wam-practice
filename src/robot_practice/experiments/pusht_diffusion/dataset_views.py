"""Create ignored PushT dataset views with split-specific normalization statistics."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_nested_dataset, unflatten_dict, write_stats

from .config import DATASET_REPO_ID, DATASET_REVISION, default_dataset_root, default_output_root
from .dataset_split import EpisodeSplit, load_split
from .provenance import sha256_file, write_json_atomic


def _episode_stats(root: Path, selected_episodes: Sequence[int]) -> list[dict[str, Any]]:
    nested = load_nested_dataset(root / "meta" / "episodes")
    index_to_row = {int(nested[index]["episode_index"]): index for index in range(len(nested))}
    missing = [episode for episode in selected_episodes if episode not in index_to_row]
    if missing:
        raise ValueError(f"Episode-level statistics are missing for episodes: {missing}")

    result: list[dict[str, Any]] = []
    for episode in selected_episodes:
        row = nested[index_to_row[episode]]
        flattened: dict[str, np.ndarray] = {}
        for key, value in row.items():
            if key.startswith("stats/"):
                flattened[key] = np.atleast_1d(np.asarray(value))
        if not flattened:
            raise ValueError(f"Episode {episode} does not contain statistics")
        result.append(unflatten_dict(flattened)["stats"])
    return result


def aggregate_selected_stats(
    source_root: str | Path,
    episodes: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    if not episodes:
        raise ValueError("At least one episode is required to aggregate statistics")
    return aggregate_stats(_episode_stats(Path(source_root), episodes))


def _safe_symlink(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required dataset directory is missing: {source}")
    destination.symlink_to(source.resolve(), target_is_directory=True)


def create_dataset_view(
    *,
    source_root: str | Path,
    view_root: str | Path,
    episodes: Sequence[int],
    split: EpisodeSplit,
    mode: str,
    overwrite: bool = False,
) -> Path:
    source = Path(source_root).expanduser().resolve()
    destination = Path(view_root).expanduser().resolve()
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"PushT metadata is missing at {source}")
    actual_info_hash = sha256_file(source / "meta" / "info.json")
    if actual_info_hash != split.metadata_info_sha256:
        raise ValueError(
            "Local PushT meta/info.json does not match the committed dataset revision: "
            f"{actual_info_hash}"
        )
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset view already exists: {destination}")
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.incomplete.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        shutil.copytree(source / "meta", temporary / "meta")
        _safe_symlink(source / "data", temporary / "data")
        if (source / "videos").exists():
            _safe_symlink(source / "videos", temporary / "videos")
        stats = aggregate_selected_stats(source, episodes)
        write_stats(stats, temporary)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_revision": DATASET_REVISION,
            "metadata_info_sha256": actual_info_hash,
            "split_manifest_sha256": split.manifest_sha256,
            "mode": mode,
            "episodes": list(episodes),
            "num_episodes": len(episodes),
            "stats_sha256": sha256_file(temporary / "meta" / "stats.json"),
        }
        write_json_atomic(temporary / "view_manifest.json", manifest)
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def ensure_source_dataset(
    *,
    source_root: str | Path,
    episodes: Sequence[int],
) -> Path:
    source = Path(source_root).expanduser().resolve()
    LeRobotDataset(
        DATASET_REPO_ID,
        root=source,
        episodes=list(episodes),
        revision=DATASET_REVISION,
    )
    return source


def prepare_views(
    *,
    source_root: str | Path,
    views_root: str | Path,
    split: EpisodeSplit,
    download: bool,
    overwrite: bool,
) -> dict[str, Path]:
    source = Path(source_root).expanduser().resolve()
    if download:
        ensure_source_dataset(source_root=source, episodes=range(split.num_episodes))
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"PushT dataset is not present at {source}; rerun with --download")
    root = Path(views_root).expanduser().resolve()
    return {
        mode: create_dataset_view(
            source_root=source,
            view_root=root / mode,
            episodes=split.episodes_for_mode(mode),
            split=split,
            mode=mode,
            overwrite=overwrite,
        )
        for mode in ("smoke", "pilot")
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--views-root",
        type=Path,
        default=default_output_root() / "dataset_views",
    )
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    split = load_split(args.split_file)
    views = prepare_views(
        source_root=args.source_root,
        views_root=args.views_root,
        split=split,
        download=args.download,
        overwrite=args.overwrite,
    )
    for mode, path in views.items():
        print(f"{mode} dataset view: {path}")


if __name__ == "__main__":
    main()
