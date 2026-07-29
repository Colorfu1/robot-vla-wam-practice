"""E0-lite demonstration inspection and held-out candidate selection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from .e0_geometry import (
    CONTACT_DISTANCE,
    classify_demonstration,
    extract_moving_block_pose,
    representative_episode_order,
    strategy_groups,
)
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .provenance import collect_runtime_provenance, write_json_atomic
from .rerun_visualization import (
    image_to_hwc_uint8,
    merge_rerun_recordings,
    record_demonstration_episode,
)

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


def _read_episode(
    *,
    dataset: LeRobotDataset,
    local_indices: Sequence[int],
    camera_key: str,
    episode: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    rerun_frames: list[dict[str, Any]] = []
    for local_index in local_indices:
        item = dataset[local_index]
        image = image_to_hwc_uint8(item[camera_key])
        agent_position = np.asarray(item[OBS_STATE]).reshape(-1)
        block = extract_moving_block_pose(
            image,
            agent_position=agent_position,
        )
        record = {
            "episode_index": episode,
            "split": split_name,
            "frame_index": int(_scalar(item["frame_index"])),
            "agent_position": agent_position.tolist(),
            "action_target": np.asarray(item[ACTION]).reshape(-1).tolist(),
            "block_pose_valid": block.valid,
            "block_position": list(block.position_workspace),
            "block_position_image": list(block.position_image),
            "block_angle_rad": block.angle_rad,
            "block_mask_area": block.mask_area,
            "block_pose_confidence": block.confidence,
            "agent_block_surface_distance": block.agent_surface_distance,
            "contact": bool(block.valid and block.agent_surface_distance <= CONTACT_DISTANCE),
            "reward": _scalar(item.get("next.reward", item.get("reward"))),
            "done": bool(_scalar(item.get("next.done"), 0.0)),
            "success": bool(_scalar(item.get("next.success"), 0.0)),
        }
        records.append(record)
        rerun_frames.append({**record, "image": image})
    return records, rerun_frames


def _outcome(records: Sequence[dict[str, Any]]) -> tuple[bool | None, str]:
    if any(record["success"] for record in records):
        return True, "next.success"
    if any(record["done"] for record in records):
        return None, "terminal_observed_without_success_label"
    return None, "no_terminal_or_success_label"


def _summarize_episode(
    records: Sequence[dict[str, Any]],
    *,
    episode: int,
    split_name: str,
) -> dict[str, Any]:
    success_signal, outcome_label = _outcome(records)
    return {
        "episode_index": episode,
        "split": split_name,
        "num_frames": len(records),
        "success_signal": success_signal,
        "outcome_label": outcome_label,
        "terminal_observed": any(record["done"] for record in records),
        "max_reward": max((record["reward"] for record in records), default=0.0),
        **classify_demonstration(records),
    }


def _inspection_episode_order(split: Any, count: int) -> tuple[int, ...]:
    ordered = split.train + split.development + split.audit
    if not 1 <= count <= split.num_episodes:
        raise ValueError(f"num_demonstrations must be in [1,{split.num_episodes}]")
    if count == split.num_episodes:
        return tuple(range(split.num_episodes))
    return ordered[:count]


def _episode_split_names(split: Any) -> dict[int, str]:
    result = {episode: "train" for episode in split.train}
    result.update({episode: "development" for episode in split.development})
    result.update({episode: "audit" for episode in split.audit})
    return result


def _anchor_frame(summary: dict[str, Any]) -> tuple[int, str]:
    num_frames = int(summary["num_frames"])
    first_contact = summary["first_contact_frame"]
    if first_contact is None:
        frame = num_frames // 2
        reason = "episode midpoint because no contact was detected"
    else:
        frame = max(1, int(first_contact) - 1)
        reason = "frame immediately before first detected contact"
    return min(frame, num_frames - 8), reason


def _inspection_markdown(
    *,
    num_demonstrations: int,
    num_frames: int,
    geometry_coverage: float,
    groups: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# PushT E0 Full-Demonstration Audit",
        "",
        "## Measured overview",
        "",
        f"- Complete demonstrations inspected: {num_demonstrations}.",
        f"- Dataset frames inspected: {num_frames}.",
        f"- Moving-block geometry coverage: {geometry_coverage:.2%}.",
        f"- Global strategy groups: {len(groups)}.",
        f"- Representative E1 candidates: {len(candidates)}.",
        "",
        "## Global strategy groups",
        "",
        "| Strategy | Episodes | Representative episode IDs |",
        "|---|---:|---|",
    ]
    for group in sorted(groups, key=lambda item: (-int(item["num_episodes"]), item["strategy_label"])):
        representative = ", ".join(str(value) for value in group["episode_indices"][:8])
        lines.append(
            f"| `{group['strategy_label']}` | {group['num_episodes']} | {representative} |"
        )
    lines.extend(
        [
            "",
            "## Selected E1 candidates",
            "",
            "Each row is one fixed two-frame observation history selected from an audit demo.",
            "",
            "| Candidate | Demo episode | Current frame | Strategy | Selection reason |",
            "|---|---:|---:|---|---|",
        ]
    )
    for candidate in candidates:
        lines.append(
            "| "
            f"`{candidate['candidate_id']}` | "
            f"{candidate['episode_index']} | "
            f"{candidate['frame_index']} | "
            f"`{candidate['strategy_label']}` | "
            f"{candidate['selection_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- These labels summarize complete demonstration geometry.",
            "- Different global strategies do not by themselves prove conditional multimodality.",
            "- `next.done` is recorded only as a terminal marker and is not treated as success.",
            "- Strict similar-condition/different-future matching remains an optional E0+ analysis.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    if not 1 <= num_candidates <= len(split.audit):
        raise ValueError("num_candidates must fit inside the audit split")

    demonstration_episodes = _inspection_episode_order(split, num_demonstrations)
    selected = tuple(dict.fromkeys(demonstration_episodes + split.audit))
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
    split_names = _episode_split_names(split)

    frame_records: list[dict[str, Any]] = []
    frame_records_by_episode: dict[int, list[dict[str, Any]]] = {}
    episode_summaries: list[dict[str, Any]] = []

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        demonstrations_dir = temporary / "demonstrations"
        demonstrations_dir.mkdir()
        recording_paths: list[Path] = []
        for episode in demonstration_episodes:
            records, rerun_frames = _read_episode(
                dataset=dataset,
                local_indices=episode_indices[episode],
                camera_key=camera_key,
                episode=episode,
                split_name=split_names[episode],
            )
            frame_records.extend(records)
            summary = _summarize_episode(
                records,
                episode=episode,
                split_name=split_names[episode],
            )
            episode_summaries.append(summary)
            frame_records_by_episode[episode] = records
            recording_paths.append(
                record_demonstration_episode(
                    output_path=demonstrations_dir / f"episode_{episode:03d}.rrd",
                    frames=rerun_frames,
                    summary=summary,
                )
            )

        merge_rerun_recordings(
            input_paths=recording_paths,
            output_path=temporary / "recording.rrd",
        )

        selection_summaries = list(episode_summaries)
        inspected = {int(summary["episode_index"]) for summary in selection_summaries}
        for episode in split.audit:
            if episode in inspected:
                continue
            records, _ = _read_episode(
                dataset=dataset,
                local_indices=episode_indices[episode],
                camera_key=camera_key,
                episode=episode,
                split_name="audit",
            )
            frame_records_by_episode[episode] = records
            selection_summaries.append(
                _summarize_episode(
                    records,
                    episode=episode,
                    split_name="audit",
                )
            )
        candidate_episodes = representative_episode_order(
            selection_summaries,
            eligible_episodes=split.audit,
            count=num_candidates,
        )
        summary_by_episode = {
            int(summary["episode_index"]): summary for summary in selection_summaries
        }
        candidates: list[dict[str, Any]] = []
        for ordinal, episode in enumerate(candidate_episodes):
            indices = episode_indices[episode]
            summary = summary_by_episode[episode]
            frame_offset, selection_reason = _anchor_frame(summary)
            if frame_offset < 1 or frame_offset + 8 > len(indices):
                raise RuntimeError(f"Episode {episode} cannot provide the E1 history window")
            previous = dataset[indices[frame_offset - 1]]
            current = dataset[indices[frame_offset]]
            future = [dataset[index] for index in indices[frame_offset : frame_offset + 8]]
            actual_frame = int(_scalar(current["frame_index"]))
            candidate_id = f"audit_ep{episode:03d}_frame{actual_frame:03d}"
            current_geometry = frame_records_by_episode[episode][frame_offset]
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_ordinal": ordinal,
                    "candidate_group": str(summary["strategy_label"]),
                    "episode_index": episode,
                    "frame_index": actual_frame,
                    "source_split": "audit",
                    "selection_reason": selection_reason,
                    "strategy_label": str(summary["strategy_label"]),
                    "camera_key": camera_key,
                    "previous_state": np.asarray(previous[OBS_STATE]).reshape(-1).tolist(),
                    "current_state": np.asarray(current[OBS_STATE]).reshape(-1).tolist(),
                    "agent_position": np.asarray(current[OBS_STATE]).reshape(-1).tolist(),
                    "block_position": current_geometry["block_position"],
                    "block_angle_rad": current_geometry["block_angle_rad"],
                    "future_actions": [
                        np.asarray(item[ACTION]).reshape(-1).tolist() for item in future
                    ],
                    "demonstration_outcome": summary["success_signal"],
                    "outcome_label": str(summary["outcome_label"]),
                    "previous_image_file": f"candidates/{candidate_id}_previous.npy",
                    "current_image_file": f"candidates/{candidate_id}_current.npy",
                    "future_images_file": f"candidates/{candidate_id}_future.npy",
                }
            )

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
        write_records_parquet(
            temporary / "demonstration_features.parquet",
            episode_summaries,
        )
        groups = strategy_groups(episode_summaries)
        selected_by_strategy: dict[str, list[str]] = defaultdict(list)
        for candidate in candidates:
            selected_by_strategy[str(candidate["strategy_label"])].append(
                str(candidate["candidate_id"])
            )
        for group in groups:
            group["selected_candidate_ids"] = selected_by_strategy.get(
                str(group["strategy_label"]),
                [],
            )
        write_json_atomic(
            temporary / "candidates.json",
            {
                "schema_version": 2,
                "status": "representative_global_strategy_candidates",
                "strict_conditional_multimodality_proven": False,
                "candidates": candidates,
            },
        )
        write_json_atomic(
            temporary / "candidate_groups.json",
            {
                "schema_version": 1,
                "group_kind": "global_demonstration_strategy",
                "strict_conditional_multimodality_proven": False,
                "groups": groups,
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
                "candidate_selection": (
                    "round-robin global strategy groups within the committed audit split"
                ),
                "runtime": collect_runtime_provenance(Path(__file__).resolve().parents[4]),
            },
        )
        valid_geometry_frames = sum(
            int(record["block_pose_valid"]) for record in frame_records
        )
        geometry_coverage = valid_geometry_frames / len(frame_records)
        outcome_counts = Counter(
            str(summary["outcome_label"]) for summary in episode_summaries
        )
        rewards = np.asarray(
            [float(summary["max_reward"]) for summary in episode_summaries],
            dtype=np.float64,
        )
        write_metrics(
            temporary,
            {
                "num_demonstrations": len(demonstration_episodes),
                "num_candidates": len(candidates),
                "num_recorded_frames": len(frame_records),
                "num_strategy_groups": len(groups),
                "block_geometry_coverage": geometry_coverage,
                "num_positive_success_labels": sum(
                    summary["success_signal"] is True for summary in episode_summaries
                ),
                "num_terminal_only_episodes": outcome_counts[
                    "terminal_observed_without_success_label"
                ],
                "mean_episode_max_reward": float(rewards.mean()),
                "median_episode_max_reward": float(np.median(rewards)),
                "strict_conditional_multimodality_proven": False,
            },
            comments={
                "num_demonstrations": "Complete demonstration episodes inspected",
                "num_candidates": "Representative audit histories exported for E1",
                "num_recorded_frames": "Dataset frames included in the trajectory table",
                "num_strategy_groups": "Automatically classified global trajectory groups",
                "block_geometry_coverage": "Fraction of frames with valid moving-block geometry",
                "num_positive_success_labels": "Episodes with an explicit next.success signal",
                "num_terminal_only_episodes": (
                    "Episodes ending without an explicit positive success label"
                ),
                "mean_episode_max_reward": "Mean of per-episode maximum dense reward",
                "median_episode_max_reward": "Median of per-episode maximum dense reward",
                "strict_conditional_multimodality_proven": (
                    "Global strategy labels do not prove conditional multimodality"
                ),
            },
        )
        (temporary / "inspection.md").write_text(
            _inspection_markdown(
                num_demonstrations=len(demonstration_episodes),
                num_frames=len(frame_records),
                geometry_coverage=geometry_coverage,
                groups=groups,
                candidates=candidates,
            ),
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
