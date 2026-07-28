"""Compute exact PushT action-padding exposure under the Pilot sampler."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from lerobot.datasets.utils import load_nested_dataset

from .config import HORIZON, default_output_root
from .dataset_split import load_split
from .provenance import write_json_atomic

ACTION_INDICES = tuple(range(-1, 15))
DROP_N_LAST_FRAMES = 7


def padding_counts_for_episode(length: int) -> np.ndarray:
    if length <= DROP_N_LAST_FRAMES:
        raise ValueError(f"Episode is too short: {length}")
    return np.asarray(
        [
            sum(not 0 <= frame + delta < length for delta in ACTION_INDICES)
            for frame in range(length - DROP_N_LAST_FRAMES)
        ],
        dtype=np.int64,
    )


def compute_padding_stats(
    *,
    dataset_root: Path,
    episodes: Sequence[int],
) -> dict[str, int | float | list[int]]:
    metadata = load_nested_dataset(dataset_root / "meta" / "episodes")
    lengths = {
        int(metadata[index]["episode_index"]): int(metadata[index]["length"])
        for index in range(len(metadata))
    }
    missing = [episode for episode in episodes if episode not in lengths]
    if missing:
        raise ValueError(f"Episode metadata is missing for: {missing}")
    padding_counts = []
    for episode in episodes:
        length = lengths[episode]
        padding_counts.extend(padding_counts_for_episode(length).tolist())
    values = np.asarray(padding_counts, dtype=np.int64)
    if values.size == 0:
        raise ValueError("No sampled frames were available")
    return {
        "episodes": list(episodes),
        "num_episodes": len(episodes),
        "num_samples": int(values.size),
        "action_horizon": HORIZON,
        "drop_n_last_frames": DROP_N_LAST_FRAMES,
        "samples_with_padding_ratio": float(np.mean(values > 0)),
        "overall_padding_position_ratio": float(np.sum(values) / (values.size * HORIZON)),
        "padding_count_p50": float(np.percentile(values, 50)),
        "padding_count_p95": float(np.percentile(values, 95)),
        "maximum_padding_count": int(np.max(values)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot"),
        required=True,
    )
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    split = load_split(args.split_file)
    output = args.output
    if output is None:
        output = default_output_root() / "dataset_views" / args.mode / "padding_stats.json"
    payload = compute_padding_stats(
        dataset_root=args.dataset_root.expanduser().resolve(),
        episodes=split.episodes_for_mode(args.mode),
    )
    destination = write_json_atomic(output.expanduser().resolve(), payload)
    print(f"Padding statistics: {destination}")


if __name__ == "__main__":
    main()
