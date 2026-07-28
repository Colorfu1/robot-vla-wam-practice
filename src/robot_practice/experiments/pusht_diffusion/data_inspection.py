"""E0-lite demonstration inspection and held-out candidate selection."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    default_dataset_root,
    default_output_root,
    load_named_config,
)
from .dataset_split import load_split
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .provenance import collect_runtime_provenance, write_json_atomic
from .rerun_visualization import image_to_hwc_uint8, record_demonstrations

_E0_CONFIG = load_named_config("e0_lite.json")


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


def _episode_frames(dataset: LeRobotDataset) -> dict[int, list[int]]:
    dataset._ensure_hf_dataset_loaded()
    result: dict[int, list[int]] = defaultdict(list)
    for local_index, raw_episode in enumerate(dataset.hf_dataset["episode_index"]):
        episode = int(raw_episode.item() if hasattr(raw_episode, "item") else raw_episode)
        result[episode].append(local_index)
    return dict(result)


def _outcome(records: Sequence[dict[str, Any]]) -> tuple[bool, str]:
    if any(record["success"] for record in records):
        return True, "next.success"
    if any(record["done"] for record in records):
        return True, "next.done"
    if max((record["reward"] for record in records), default=0.0) >= 0.95:
        return True, "max_reward>=0.95"
    return False, "no_positive_signal"


def inspect_demonstrations(
    *,
    dataset_root: Path,
    output_dir: Path,
    split_file: Path | None,
    num_demonstrations: int,
    num_candidates: int,
    overwrite: bool,
) -> Path:
    split = load_split(split_file)
    if not 1 <= num_demonstrations <= len(split.train):
        raise ValueError(f"num_demonstrations must be in [1,{len(split.train)}]")
    if not 1 <= num_candidates <= len(split.audit):
        raise ValueError("num_candidates must fit inside the audit split")

    demonstration_episodes = tuple(split.train[:num_demonstrations])
    candidate_episodes = tuple(split.audit[:num_candidates])
    selected = tuple(dict.fromkeys(demonstration_episodes + candidate_episodes))
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=dataset_root,
        episodes=list(selected),
        revision=DATASET_REVISION,
    )
    if not dataset.meta.camera_keys:
        raise ValueError("PushT dataset does not contain a camera feature")
    camera_key = dataset.meta.camera_keys[0]
    episode_indices = _episode_frames(dataset)

    frame_records: list[dict[str, Any]] = []
    rerun_frames: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    for episode in demonstration_episodes:
        records: list[dict[str, Any]] = []
        for local_index in episode_indices[episode]:
            item = dataset[local_index]
            record = {
                "episode_index": episode,
                "frame_index": int(_scalar(item["frame_index"])),
                "agent_position": np.asarray(item[OBS_STATE]).reshape(-1).tolist(),
                "action_target": np.asarray(item[ACTION]).reshape(-1).tolist(),
                "reward": _scalar(item.get("next.reward", item.get("reward"))),
                "done": bool(_scalar(item.get("next.done"), 0.0)),
                "success": bool(_scalar(item.get("next.success"), 0.0)),
            }
            records.append(record)
            frame_records.append(record)
            rerun_frames.append(
                {
                    **record,
                    "image": image_to_hwc_uint8(item[camera_key]),
                }
            )
        success, outcome_rule = _outcome(records)
        episode_summaries.append(
            {
                "episode_index": episode,
                "num_frames": len(records),
                "success": success,
                "outcome_rule": outcome_rule,
                "max_reward": max((record["reward"] for record in records), default=0.0),
            }
        )

    candidates: list[dict[str, Any]] = []
    for ordinal, episode in enumerate(candidate_episodes):
        indices = episode_indices[episode]
        if len(indices) < 10:
            raise ValueError(f"Audit episode {episode} is too short for an E1 candidate")
        frame_offset = min(max(1, len(indices) // 2), len(indices) - 9)
        previous = dataset[indices[frame_offset - 1]]
        current = dataset[indices[frame_offset]]
        future = [dataset[index] for index in indices[frame_offset : frame_offset + 8]]
        candidate_id = f"audit_ep{episode:03d}_frame{frame_offset:03d}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_ordinal": ordinal,
                "episode_index": episode,
                "frame_index": frame_offset,
                "camera_key": camera_key,
                "agent_position": np.asarray(current[OBS_STATE]).reshape(-1).tolist(),
                "future_actions": [
                    np.asarray(item[ACTION]).reshape(-1).tolist() for item in future
                ],
                "demonstration_outcome": _outcome(
                    [
                        {
                            "reward": _scalar(item.get("next.reward", item.get("reward"))),
                            "done": bool(_scalar(item.get("next.done"), 0.0)),
                            "success": bool(_scalar(item.get("next.success"), 0.0)),
                        }
                        for item in (dataset[index] for index in indices)
                    ]
                )[0],
                "previous_image_file": f"candidates/{candidate_id}_previous.npy",
                "current_image_file": f"candidates/{candidate_id}_current.npy",
                "future_images_file": f"candidates/{candidate_id}_future.npy",
            }
        )

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        (temporary / "candidates").mkdir()
        for candidate, episode in zip(candidates, candidate_episodes, strict=True):
            indices = episode_indices[episode]
            offset = int(candidate["frame_index"])
            previous = dataset[indices[offset - 1]]
            current = dataset[indices[offset]]
            np.save(
                temporary / str(candidate["previous_image_file"]),
                np.asarray(previous[camera_key]),
            )
            np.save(
                temporary / str(candidate["current_image_file"]),
                np.asarray(current[camera_key]),
            )
            future_images = np.stack(
                [np.asarray(dataset[index][camera_key]) for index in indices[offset : offset + 8]]
            )
            np.save(
                temporary / str(candidate["future_images_file"]),
                future_images,
            )
        write_records_parquet(temporary / "trajectories.parquet", frame_records)
        write_json_atomic(
            temporary / "candidates.json",
            {
                "schema_version": 1,
                "status": "qualitative_candidates",
                "strict_conditional_multimodality_proven": False,
                "candidates": candidates,
            },
        )
        write_json_atomic(temporary / "episode_summaries.json", episode_summaries)
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "label": "Offline",
                "experiment": "E0-lite",
                "dataset_repo_id": DATASET_REPO_ID,
                "dataset_revision": DATASET_REVISION,
                "split_manifest_sha256": split.manifest_sha256,
                "demonstration_episodes": list(demonstration_episodes),
                "candidate_episodes": list(candidate_episodes),
                "runtime": collect_runtime_provenance(Path(__file__).resolve().parents[4]),
            },
        )
        write_metrics(
            temporary,
            {
                "num_demonstrations": len(demonstration_episodes),
                "num_candidates": len(candidates),
                "num_recorded_frames": len(frame_records),
                "strict_conditional_multimodality_proven": False,
            },
            comments={
                "num_demonstrations": "Demonstration episodes inspected qualitatively",
                "num_candidates": "Held-out two-frame histories exported for E1",
                "num_recorded_frames": "Dataset frames included in the trajectory table",
                "strict_conditional_multimodality_proven": (
                    "E0-lite does not statistically prove conditional multimodality"
                ),
            },
        )
        record_demonstrations(
            output_path=temporary / "recording.rrd",
            frames=rerun_frames,
        )
        inspection_lines = [
            "# PushT E0-lite Inspection",
            "",
            "- The task may admit multiple solutions.",
            "- The selected demonstrations allow qualitative trajectory comparison.",
            "- Strict conditional multimodality has not been statistically proven.",
            "- Block paths are omitted because the dataset does not record block pose directly.",
        ]
        (temporary / "inspection.md").write_text(
            "\n".join(inspection_lines) + "\n",
            encoding="utf-8",
        )
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split-file", type=Path)
    parser.add_argument(
        "--num-demonstrations",
        type=int,
        default=int(_E0_CONFIG["num_demonstrations"]),
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=int(_E0_CONFIG["num_candidates"]),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "data_inspection",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = inspect_demonstrations(
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        split_file=args.split_file,
        num_demonstrations=args.num_demonstrations,
        num_candidates=args.num_candidates,
        overwrite=args.overwrite,
    )
    print(f"E0-lite output: {result}")


if __name__ == "__main__":
    main()
