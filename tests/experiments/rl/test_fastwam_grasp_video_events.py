import numpy as np

from robot_practice.experiments.rl.fastwam_grasp_video_events import export_event, find_events
from robot_practice.experiments.rl.fastwam_rollout_video import VideoWriter, ffmpeg_binary


def test_select_closure_and_subsequent_upward_motion():
    raw = np.zeros((150, 9))
    raw[:, 7], raw[:, 8] = 0.04, -0.04
    raw[11:, 7], raw[11:, 8] = 0.015, -0.015
    raw[14:, 2] = 0.03
    actions = np.zeros((150, 4, 7))
    actions[:, :, -1] = 1
    low, high = np.zeros(9), np.ones(9)
    normalized = 2 * raw / (high-low+1e-6) - 1
    result = find_events(normalized, actions, low, high)
    assert [r["chunk"] for r in result["events"]] == [10, 13]
    assert result["has_closing_above_5mm"]
    assert result["has_upward_candidate"]
    np.testing.assert_allclose(result["events"][0]["closing_mm"], 50)


def test_stationary_case_is_not_labeled_grasp():
    result = find_events(np.zeros((150, 9)), np.zeros((150, 4, 7)),
                         np.zeros(9), np.ones(9))
    assert not result["has_closing_above_5mm"]
    assert not result["has_upward_candidate"]
    assert result["events"][0]["kind"] == "no_event_reference"


def test_export_real_event_clip(tmp_path):
    from PIL import Image

    binary = ffmpeg_binary()
    source = tmp_path / "source"
    source.mkdir()
    for name in ("comparison_5hz", "forecast_strip_5hz"):
        writer = VideoWriter(binary, source / f"{name}.mp4", (224, 272), 5)
        for i in range(8):
            writer.append(Image.new("RGB", (224, 272), (i*25, 0, 0)))
        writer.close()
    output = tmp_path / "event"
    export_event(binary, source, output, 3)
    assert len(list(output.glob("*.mp4"))) == 2
    for p in output.glob("*.png"):
        with Image.open(p) as image:
            assert image.size == (448, 544)
            assert abs(image.getpixel((0, 0))[0] - 75) < 5
