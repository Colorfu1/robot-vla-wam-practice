import json
import pickle

import numpy as np
import pytest

from robot_practice.experiments.rl.fastwam_dual_data import (
    PROTOCOL,
    DualViewDataset,
    executed_actions,
    window_starts,
)


def test_temporal_boundaries():
    assert window_starts([32, 33, 34]).tolist() == [32, 65, 66]
    assert PROTOCOL["video_stride"] * (PROTOCOL["video_frames"] - 1) == 32


def test_execute_first_four_not_all_predictions():
    actions = np.arange(32 * 7).reshape(32, 7)
    np.testing.assert_array_equal(executed_actions(actions), actions[:4])
    with pytest.raises(ValueError):
        executed_actions(np.zeros((4, 7)))


def test_dataset_stride_spatial_layout_and_spawn(tmp_path):
    n = 34
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", "protocol": PROTOCOL}))
    np.save(tmp_path / "traj_lengths.npy", [n])
    np.save(tmp_path / "states.npy", np.repeat(np.arange(n)[:, None], 9, axis=1).astype(np.float32))
    np.save(tmp_path / "actions.npy", np.repeat(np.arange(n)[:, None], 7, axis=1).astype(np.float32))
    images = np.lib.format.open_memmap(tmp_path / "images.npy", mode="w+", dtype=np.uint8,
                                      shape=(n, 3, 224, 448))
    for frame in range(n):
        images[frame, :, :, :224] = frame
        images[frame, :, :, 224:] = frame + 100
    images.flush()
    dataset = DualViewDataset(tmp_path)
    sample = dataset[1]
    assert sample["video"].shape == (3, 9, 224, 448)
    assert sample["action"].shape == (32, 7)
    np.testing.assert_array_equal(sample["proprio"][:, 0], np.arange(1, 34, 4))
    np.testing.assert_array_equal(sample["action"][:, 0], np.arange(1, 33))
    np.testing.assert_allclose(sample["video"][0, :, 0, 0], np.arange(1, 34, 4) / 127.5 - 1, atol=1e-7)
    np.testing.assert_allclose(sample["video"][0, :, 0, 224], (np.arange(1, 34, 4)+100) / 127.5 - 1, atol=1e-7)
    serialized = pickle.dumps(dataset)
    assert len(serialized) < 10000
    restored = pickle.loads(serialized)
    assert restored._arrays is None
    np.testing.assert_array_equal(restored[1]["action"], sample["action"])
