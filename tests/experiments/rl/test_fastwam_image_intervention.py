import numpy as np
import pytest

from robot_practice.experiments.rl.fastwam_image_intervention import (
    substitute_rgb,
    verify_prefix,
)


def test_replace_only_rgb_without_mutating_original():
    obs = {"rgb": np.zeros((3, 4, 4), dtype=np.uint8), "state": np.arange(9.)}
    image = np.ones_like(obs["rgb"])
    replaced = substitute_rgb(obs, image)
    np.testing.assert_array_equal(replaced["state"], obs["state"])
    assert replaced["rgb"].sum() == 48
    assert obs["rgb"].sum() == 0
    replaced["state"][0] = 100
    assert obs["state"][0] == 0


def test_prefix_gate_checks_hidden_physics_not_only_images():
    reference = {k: np.zeros((9, 3)) for k in
                 ("sim_states", "integration_states", "rgb", "proprio")}
    replay = {k: v[:5].copy() for k, v in reference.items()}
    assert max(verify_prefix(reference, replay, 4).values()) == 0
    replay["integration_states"][2, 1] = 0.1
    with pytest.raises(ValueError, match="restoration failed"):
        verify_prefix(reference, replay, 4)


def test_replacement_rejects_wrong_shape():
    with pytest.raises(ValueError):
        substitute_rgb({"rgb": np.zeros((3, 4, 4))}, np.zeros((4, 4, 3)))


def test_uint8_demo_into_float32_live_pixels_preserves_values():
    obs = {"rgb": np.full((3, 96, 96), 255, dtype=np.float32), "state": np.arange(9.)}
    demo = np.full((3, 96, 96), 253, dtype=np.uint8)
    replaced = substitute_rgb(obs, demo)
    assert replaced["rgb"].dtype == np.float32
    np.testing.assert_array_equal(replaced["rgb"], demo)
    np.testing.assert_array_equal(replaced["state"], obs["state"])


@pytest.mark.parametrize("bad", [float("nan"), -1., 256.])
def test_reject_invalid_pixel_values(bad):
    obs = {"rgb": np.full((3, 4, 4), 255, dtype=np.float32)}
    with pytest.raises(ValueError):
        substitute_rgb(obs, np.full((3, 4, 4), bad, dtype=np.float32))
