from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rerun.dataframe as rdf

from robot_practice.experiments.pusht_diffusion.data_inspection import _inspection_markdown
from robot_practice.experiments.pusht_diffusion.e0_geometry import (
    classify_demonstration,
    extract_moving_block_pose,
    representative_episode_order,
    strategy_groups,
)
from robot_practice.experiments.pusht_diffusion.rerun_visualization import (
    merge_rerun_recordings,
    record_demonstration_episode,
)


def _synthetic_t_image() -> np.ndarray:
    image = np.full((96, 96, 3), 248, dtype=np.uint8)
    image[10:22, 12:34] = (130, 225, 140)  # fixed green target
    image[35:43, 35:61] = (145, 165, 185)  # moving T head
    image[42:68, 45:51] = (145, 165, 185)  # moving T stem
    image[72:77, 46:51] = (55, 95, 225)  # blue agent
    return image


def _angle_error(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2 * math.pi) - math.pi)


def test_moving_block_pose_recovers_t_center_and_head_direction() -> None:
    pose = extract_moving_block_pose(
        _synthetic_t_image(),
        agent_position=(48 / 96 * 512, 74 / 96 * 512),
    )

    assert pose.valid
    assert 250 <= pose.mask_area <= 400
    assert np.allclose(pose.position_image, (48.0, 47.8), atol=2.0)
    assert _angle_error(pose.angle_rad, -math.pi / 2) < math.radians(8)
    assert pose.agent_surface_distance > 0
    assert 0 < pose.confidence <= 1


def test_moving_block_pose_fails_explicitly_when_component_is_missing() -> None:
    pose = extract_moving_block_pose(
        np.full((96, 96, 3), 255, dtype=np.uint8),
        agent_position=(100.0, 100.0),
    )

    assert not pose.valid
    assert pose.mask_area == 0
    assert math.isnan(pose.angle_rad)


def _episode_records(*, recontact: bool = False) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    contact_frames = {3, 4, 5, 6, 7}
    if recontact:
        contact_frames |= {10, 11}
    for frame in range(14):
        block_x = 200.0 + frame * 5.0
        records.append(
            {
                "agent_position": [block_x - 30.0, 200.0],
                "block_position": [block_x, 200.0],
                "block_pose_valid": True,
                "block_angle_rad": 0.0,
                "agent_block_surface_distance": 10.0 if frame in contact_frames else 60.0,
            }
        )
    return records


def test_complete_demonstration_classification_is_interpretable() -> None:
    summary = classify_demonstration(_episode_records())

    assert summary["geometry_coverage"] == 1.0
    assert summary["first_contact_frame"] == 3
    assert summary["contact_segments"] == 1
    assert summary["approach_side"] == "left"
    assert summary["primary_block_motion"] == "right"
    assert not summary["rotation_first"]
    assert summary["strategy_label"] == "left_direct"


def test_strategy_selection_round_robins_across_modes() -> None:
    summaries = [
        {
            "episode_index": 1,
            "strategy_label": "left_direct",
            "geometry_coverage": 1.0,
            "max_reward": 0.8,
        },
        {
            "episode_index": 2,
            "strategy_label": "left_direct",
            "geometry_coverage": 0.9,
            "max_reward": 0.9,
        },
        {
            "episode_index": 3,
            "strategy_label": "right_rotation_first",
            "geometry_coverage": 1.0,
            "max_reward": 0.7,
        },
    ]

    selected = representative_episode_order(
        summaries,
        eligible_episodes=(1, 2, 3),
        count=3,
    )
    groups = strategy_groups(summaries)

    assert selected == (1, 3, 2)
    assert {group["strategy_label"] for group in groups} == {
        "left_direct",
        "right_rotation_first",
    }
    assert all(not group["strict_conditional_multimodality_proven"] for group in groups)


def test_inspection_markdown_lists_selected_candidate_sources() -> None:
    markdown = _inspection_markdown(
        num_demonstrations=206,
        num_frames=25_650,
        geometry_coverage=1.0,
        groups=[
            {
                "strategy_label": "right_rotation_first",
                "num_episodes": 1,
                "episode_indices": [111],
            }
        ],
        candidates=[
            {
                "candidate_id": "audit_ep111_frame031",
                "episode_index": 111,
                "frame_index": 31,
                "strategy_label": "right_rotation_first",
                "selection_reason": "frame immediately before first detected contact",
            }
        ],
    )

    assert "## Selected E1 candidates" in markdown
    assert "| `audit_ep111_frame031` | 111 | 31 | `right_rotation_first` |" in markdown
    assert "fixed two-frame observation history selected from an audit demo" in markdown


def test_demonstration_archive_exposes_one_recording_per_episode(tmp_path: Path) -> None:
    recordings = []
    for episode in (1, 2):
        frames = [
            {
                "episode_index": episode,
                "frame_index": frame,
                "agent_position": [100.0 + frame, 200.0],
                "action_target": [110.0 + frame, 210.0],
                "block_pose_valid": True,
                "block_position_image": [40.0 + frame, 45.0],
                "block_angle_rad": 0.25,
                "agent_block_surface_distance": 10.0,
                "reward": frame / 10,
                "image": np.full((96, 96, 3), 255, dtype=np.uint8),
            }
            for frame in range(3)
        ]
        summary = {
            "episode_index": episode,
            "split": "train",
            "strategy_label": "left_direct",
            "max_reward": 0.2,
            "outcome_label": "terminal_observed_without_success_label",
            "approach_side": "left",
            "primary_block_motion": "right",
            "rotation_first": False,
            "recontact": False,
            "geometry_coverage": 1.0,
        }
        recordings.append(
            record_demonstration_episode(
                output_path=tmp_path / f"episode_{episode:03d}.rrd",
                frames=frames,
                summary=summary,
            )
        )

    archive_path = merge_rerun_recordings(
        input_paths=recordings,
        output_path=tmp_path / "recording.rrd",
    )
    archive = rdf.load_archive(archive_path)

    assert archive.num_recordings() == 2
    assert sorted(str(recording.recording_id()) for recording in archive.all_recordings()) == [
        "episode_001",
        "episode_002",
    ]
