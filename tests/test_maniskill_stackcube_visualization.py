from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import rerun.dataframe as rdf

from robot_practice.experiments.maniskill.stackcube_visualization import (
    load_stackcube_episode,
    select_train_episode_ids,
    write_train_visualizations,
)
from robot_practice.rerun_launcher import OFFLINE_APPLICATION_ID


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    d_core_h5 = tmp_path / "trajectory.h5"
    d_core_json = tmp_path / "trajectory.json"
    split_manifest = tmp_path / "split_manifest.json"
    train = list(range(787))
    development = list(range(787, 887))
    test = list(range(887, 987))
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "StackCube-v1",
                "counts": {"train": 787, "development": 100, "test": 100},
                "train": train,
                "development": development,
                "test": test,
            }
        ),
        encoding="utf-8",
    )
    metadata = []
    with h5py.File(d_core_h5, "w") as handle:
        for episode_id in (0, 1):
            group = handle.create_group(f"traj_{episode_id}")
            action_steps = 2
            frame_count = action_steps + 1
            group.create_dataset(
                "actions",
                data=np.full((action_steps, 7), episode_id, dtype=np.float32),
            )
            obs = group.create_group("obs")
            agent = obs.create_group("agent")
            agent.create_dataset("qpos", data=np.zeros((frame_count, 9), dtype=np.float32))
            agent.create_dataset("qvel", data=np.zeros((frame_count, 9), dtype=np.float32))
            sensor_data = obs.create_group("sensor_data")
            for camera_name in ("base_camera", "hand_camera"):
                camera = sensor_data.create_group(camera_name)
                camera.create_dataset(
                    "rgb",
                    data=np.full(
                        (frame_count, 128, 128, 3),
                        episode_id,
                        dtype=np.uint8,
                    ),
                )
            actors = group.create_group("env_states").create_group("actors")
            for actor_name, offset in (("cubeA", 0.0), ("cubeB", 0.1)):
                state = np.zeros((frame_count, 13), dtype=np.float32)
                state[:, 0] = offset + np.arange(frame_count, dtype=np.float32)
                state[:, 2] = 0.02
                actors.create_dataset(actor_name, data=state)
            metadata.append(
                {
                    "episode_id": episode_id,
                    "source_episode_id": episode_id,
                    "route": "baseline",
                    "success": True,
                    "first_success_step": 2,
                }
            )
    d_core_json.write_text(json.dumps({"episodes": metadata}), encoding="utf-8")
    return d_core_h5, d_core_json, split_manifest


def test_select_train_episode_ids_is_seeded_and_train_only() -> None:
    values_a = select_train_episode_ids(range(20), count=5, seed=123)
    values_b = select_train_episode_ids(range(20), count=5, seed=123)
    assert values_a == values_b
    assert len(values_a) == 5
    assert all(value in range(20) for value in values_a)


def test_load_stackcube_episode_aligns_two_cameras_and_actions(tmp_path: Path) -> None:
    d_core_h5, _d_core_json, _split_manifest = _make_fixture(tmp_path)
    with h5py.File(d_core_h5, "r") as handle:
        episode = load_stackcube_episode(
            handle["traj_0"],
            episode_id=0,
            metadata={"route": "baseline", "success": True},
            split="train",
        )
    assert episode.images_base.shape == (3, 128, 128, 3)
    assert episode.images_hand.shape == (3, 128, 128, 3)
    assert episode.qpos.shape == (3, 9)
    assert episode.actions.shape == (2, 7)
    assert episode.cube_a.shape == (3, 3)
    assert episode.cube_b.shape == (3, 3)


def test_write_train_visualizations_creates_rerun_recordings_and_preview(tmp_path: Path) -> None:
    d_core_h5, d_core_json, split_manifest = _make_fixture(tmp_path)
    output = tmp_path / "visualization"
    manifest = write_train_visualizations(
        d_core_h5=d_core_h5,
        d_core_json=d_core_json,
        split_manifest=split_manifest,
        output_dir=output,
        episode_ids=[0, 1],
        seed=123,
    )
    assert manifest["selection"]["selected_episode_ids"] == [0, 1]
    assert (output / "preview" / "contact_sheet.png").is_file()
    recordings = sorted((output / "episodes").glob("*.rrd"))
    assert len(recordings) == 2
    for recording in recordings:
        assert recording.stat().st_size > 0
        recording_ids = {
            str(item.application_id()) for item in rdf.load_archive(recording).all_recordings()
        }
        assert recording_ids == {OFFLINE_APPLICATION_ID}
