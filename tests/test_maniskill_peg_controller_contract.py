import numpy as np

from robot_practice.experiments.maniskill.peg_controller_contract import (
    convert_ms2_action_to_ms3,
    decode_ms2_ee_action,
    encode_ms3_ee_action,
    pose_delta,
    rotation_geodesic,
)


def test_ms2_body_translation_is_rotated_into_ms3_root_frame() -> None:
    current_quaternion = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    result = decode_ms2_ee_action(
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        current_quaternion,
    )
    np.testing.assert_allclose(result["delta_position_ee_m"], [0.01, 0.0, 0.0])
    np.testing.assert_allclose(result["delta_position_root_m"], [0.0, 0.01, 0.0], atol=1e-8)


def test_ms2_right_rotation_is_conjugated_for_ms3_left_rotation() -> None:
    current_quaternion = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    result = convert_ms2_action_to_ms3(
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        current_quaternion,
    )
    np.testing.assert_allclose(result["decoded_ms2"]["delta_rotation_root_rad"], [-0.02, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(result["encoded_ms3"]["action"][:6], [0.0, 0.0, 0.0, 0.2, 0.0, 0.0], atol=1e-8)


def test_ms3_negative_rotation_bound_is_encoded_explicitly() -> None:
    result = encode_ms3_ee_action([0.01, 0.0, 0.0], [0.0, 0.02, 0.0], 0.0)
    np.testing.assert_allclose(result["action"], [0.1, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0])


def test_pose_delta_and_rotation_geodesic() -> None:
    before = [1.0, 0.0, 0.0, 0.0]
    after = [np.cos(0.01), 0.0, np.sin(0.01), 0.0]
    delta = pose_delta([0, 0, 0], before, [0.01, 0, 0], after)
    np.testing.assert_allclose(delta["position_root_m"], [0.01, 0, 0])
    assert abs(rotation_geodesic(before, after) - 0.02) < 1e-8
