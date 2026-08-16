from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from robot_practice.experiments.maniskill.rgb_materialization_validation import _compare_h5


def _write_episode(path: Path, *, target_id: int, source_id: int, rgb_delta: int = 0) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group(f"traj_{target_id}")
        env_states = group.create_group("env_states")
        env_states.create_group("actors").create_dataset(
            "cube", data=np.asarray([[0.1] * 13, [0.2] * 13], dtype=np.float32)
        )
        group.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float32))
        group.create_dataset("success", data=np.asarray([True], dtype=bool))
        obs = group.create_group("obs")
        agent = obs.create_group("agent")
        agent.create_dataset("qpos", data=np.zeros((2, 9), dtype=np.float32))
        agent.create_dataset("qvel", data=np.zeros((2, 9), dtype=np.float32))
        sensors = obs.create_group("sensor_data")
        for camera in ("base_camera", "hand_camera"):
            rgb = np.zeros((2, 128, 128, 3), dtype=np.uint8)
            if rgb_delta:
                rgb[0, 0, 0, 0] = rgb_delta
            sensors.create_group(camera).create_dataset("rgb", data=rgb)
    path.with_suffix(".json").write_text(
        json.dumps({"episodes": [{"episode_id": target_id, "source_episode_id": source_id}]}),
        encoding="utf-8",
    )


def test_rgb_materialization_comparison_uses_fingerprint_and_render_tolerance(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct.h5"
    materialized = tmp_path / "materialized.h5"
    _write_episode(direct, target_id=4, source_id=17)
    _write_episode(materialized, target_id=4, source_id=17, rgb_delta=1)

    result = _compare_h5(direct, materialized, [17])

    assert result["identity"]["exact_match"]
    assert result["payload_exact_episode_count"] == 1
    assert result["gate_pass"]
    assert result["rgb"]["hand_camera"]["mismatched_pixels"] == 1


def test_rgb_materialization_comparison_rejects_large_render_difference(tmp_path: Path) -> None:
    direct = tmp_path / "direct.h5"
    materialized = tmp_path / "materialized.h5"
    _write_episode(direct, target_id=4, source_id=17)
    _write_episode(materialized, target_id=4, source_id=17, rgb_delta=255)
    with h5py.File(materialized, "a") as handle:
        for camera in ("base_camera", "hand_camera"):
            handle[f"traj_4/obs/sensor_data/{camera}/rgb"][...] = 255

    result = _compare_h5(direct, materialized, [17])

    assert not result["gate_pass"]
