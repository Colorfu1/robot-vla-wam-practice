import csv

import numpy as np
from PIL import Image

from robot_practice.experiments.rl.fastwam_dual_sampling_audit import (
    close_onsets,
    export_event,
    frame_coverage,
    sample_windows,
)


def test_coverage_matches_actual_sliding_windows():
    for length in (10, 32, 33, 70, 127):
        inputs, future = frame_coverage(length)
        expected_i = np.zeros(length, dtype=int)
        expected_f = np.zeros(length, dtype=int)
        for start in range(max(0, length - 32)):
            expected_i[start] += 1
            expected_f[start + np.arange(1, 9) * 4] += 1
        np.testing.assert_array_equal(inputs, expected_i)
        np.testing.assert_array_equal(future, expected_f)
    inputs, future = frame_coverage(127)
    assert np.all(inputs + future > 0)
    assert inputs[-32:].sum() == 0


def test_close_onsets_and_phases():
    actions = np.zeros((100, 7))
    actions[:, 6] = -1
    actions[40:60, 6] = 1
    actions[80:, 6] = 1
    assert close_onsets(actions) == [40, 80]
    windows = sample_windows(100, 40)
    assert [w["start"] for w in windows] == [36, 37, 38, 39]
    assert {w["phase_mod4"] for w in windows} == {0, 1, 2, 3}
    for w in windows:
        assert len(w["frames"]) == 9
        assert np.all(np.diff(w["frames"]) == 4)
    assert sample_windows(100, 99) == []


def test_export_keeps_exact_native_pixels_and_does_not_label_grasp(tmp_path):
    images = np.zeros((12, 3, 224, 448), dtype=np.uint8)
    images[6, :, :, 224:] = 123
    states = np.zeros((12, 9))
    actions = np.zeros((12, 7))
    actions[6:, 6] = 1
    event = export_event(tmp_path, 0, 6, images, states, actions, 0, 1, 2)
    directory = (tmp_path / event["page"]).parent
    restored = np.asarray(Image.open(directory / "frame_0006.png"))
    np.testing.assert_array_equal(restored, images[6].transpose(1, 2, 0))
    assert event["physical_contact_frame"] is None
    assert event["object_lift_frame"] is None
    with (directory / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["frame"]) for r in rows] == [5, 6, 7, 8]
    assert (directory / "continuous_000.png").is_file()
