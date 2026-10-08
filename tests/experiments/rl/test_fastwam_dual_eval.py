import pytest

from robot_practice.experiments.rl.fastwam_dual_eval import (
    episode_selection,
    validate_rendering_info,
)


def test_episode_selection_defaults_to_full_seed_range():
    assert episode_selection(None, 4) == [0, 1, 2, 3]
    assert episode_selection("3,1", 4) == [3, 1]


@pytest.mark.parametrize("value,total", [("", 4), ("0,0", 4), ("4", 4), ("-1", 4)])
def test_episode_selection_rejects_bad_shard(value, total):
    with pytest.raises(ValueError):
        episode_selection(value, total)


def test_egl_renderer_verification_is_class_based():
    info = {"backend": "egl", "context_class": "EGLGLContext",
            "egl_device_id": "6", "cuda_visible_devices": None}
    validate_rendering_info(info, "egl", 6)
    for mutation in ({"context_class": "OSMesaGLContext"},
                     {"egl_device_id": "0"}, {"cuda_visible_devices": "0"}):
        with pytest.raises(ValueError):
            validate_rendering_info(dict(info, **mutation), "egl", 6)


def test_osmesa_renderer_verification():
    validate_rendering_info({"backend": "osmesa", "context_class": "OSMesaGLContext"},
                            "osmesa", None)
