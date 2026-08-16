from __future__ import annotations

import json

import h5py
import numpy as np
from torch.utils.data import DataLoader

from robot_practice.experiments.maniskill.official_stackcube_18d4_adapter import (
    M1W_PROJECTED_FORMAT,
    OFFICIAL_FORMAT,
    StackCube18D4DDataset,
    prepare_ablation,
)


def _write_h5(path, action_dim: int) -> None:
    with h5py.File(path, "w") as handle:
        for episode_id in (0, 1):
            length = 5
            trajectory = handle.create_group(f"traj_{episode_id}")
            trajectory.create_dataset("actions", data=np.arange(length * action_dim, dtype=np.float32).reshape(length, action_dim))
            obs = trajectory.create_group("obs")
            agent = obs.create_group("agent")
            agent.create_dataset("qpos", data=np.zeros((length + 1, 9), dtype=np.float32))
            agent.create_dataset("qvel", data=np.zeros((length + 1, 9), dtype=np.float32))
            sensor_data = obs.create_group("sensor_data")
            for camera in ("base_camera", "hand_camera"):
                camera_group = sensor_data.create_group(camera)
                camera_group.create_dataset(
                    "rgb", data=np.zeros((length + 1, 128, 128, 3), dtype=np.uint8)
                )


def test_m1w_projection_selects_translation_and_gripper(tmp_path):
    h5_path = tmp_path / "m1w.h5"
    ids_path = tmp_path / "ids.json"
    _write_h5(h5_path, action_dim=7)
    ids_path.write_text("[0]\n", encoding="utf-8")

    dataset = StackCube18D4DDataset(
        h5_path, ids_path, obs_horizon=2, pred_horizon=4, dataset_format=M1W_PROJECTED_FORMAT
    )
    sample = dataset[1]

    np.testing.assert_array_equal(
        sample["actions"][0].numpy(), np.array([0.0, 1.0, 2.0, 6.0], dtype=np.float32)
    )
    assert tuple(sample["observations"]["state"].shape) == (2, 18)
    assert tuple(sample["observations"]["rgb"].shape) == (2, 6, 128, 128)


def test_native_dataset_can_read_from_multiple_workers(tmp_path):
    h5_path = tmp_path / "official.h5"
    ids_path = tmp_path / "ids.json"
    _write_h5(h5_path, action_dim=4)
    ids_path.write_text("[0, 1]\n", encoding="utf-8")

    dataset = StackCube18D4DDataset(
        h5_path, ids_path, obs_horizon=2, pred_horizon=4, dataset_format=OFFICIAL_FORMAT
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=2)))

    assert tuple(batch["observations"]["state"].shape) == (2, 2, 18)
    assert tuple(batch["observations"]["rgb"].shape) == (2, 2, 6, 128, 128)
    assert tuple(batch["actions"].shape) == (2, 4, 4)


def test_prepare_sidecar_records_native_vs_projected_contract(tmp_path):
    official_h5 = tmp_path / "official.h5"
    m1w_h5 = tmp_path / "m1w.h5"
    _write_h5(official_h5, action_dim=4)
    _write_h5(m1w_h5, action_dim=7)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": [1], "development": [0], "test": []}), encoding="utf-8")

    official = prepare_ablation(
        dataset_format=OFFICIAL_FORMAT,
        h5_path=official_h5,
        output_dir=tmp_path / "official",
        num_demos=1,
    )
    projected = prepare_ablation(
        dataset_format=M1W_PROJECTED_FORMAT,
        h5_path=m1w_h5,
        output_dir=tmp_path / "m1w",
        split_manifest=split,
        num_demos=1,
    )

    assert official["source_action_dim"] == 4
    assert official["native_action_warning"] is False
    assert projected["source_action_dim"] == 7
    assert projected["native_action_warning"] is True
