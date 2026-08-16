from __future__ import annotations

import numpy as np

from robot_practice.experiments.maniskill.stackcube_actual_ee_retarget_probe import (
    build_parser,
    interpolate_pose_path,
    quaternion_slerp,
)


def test_slerp_preserves_endpoints_and_unit_norm() -> None:
    left = np.asarray([1.0, 0.0, 0.0, 0.0])
    right = np.asarray([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(quaternion_slerp(left, right, 0.0), left)
    np.testing.assert_allclose(quaternion_slerp(left, right, 1.0), right)
    assert np.isclose(np.linalg.norm(quaternion_slerp(left, right, 0.5)), 1.0)


def test_pose_resampling_keeps_length_and_endpoints() -> None:
    path = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = interpolate_pose_path(path, 5)
    assert result.shape == (5, 7)
    np.testing.assert_allclose(result[0, :3], path[0, :3])
    np.testing.assert_allclose(result[-1, :3], path[-1, :3])
    np.testing.assert_allclose(result[0, 3:], path[0, 3:])
    np.testing.assert_allclose(result[-1, 3:], path[-1, 3:])
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:], axis=1), 1.0, atol=1e-6)


def test_probe_parser_accepts_plug_charger() -> None:
    parsed = build_parser().parse_args(
        [
            "--task=PlugCharger-v1",
            "--source-h5=source.h5",
            "--output-dir=out",
            "--episode-ids=1,3",
        ]
    )
    assert parsed.task == "PlugCharger-v1"
