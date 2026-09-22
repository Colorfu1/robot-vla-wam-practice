import pytest

from robot_practice.experiments.rl.fastwam_midstate_eval import checked_frames


def test_midstate_frame_validation():
    assert checked_frames("101,85,85,77", 180) == [77, 85, 101]
    with pytest.raises(ValueError):
        checked_frames("179", 180)
    with pytest.raises(ValueError):
        checked_frames("-1", 180)
