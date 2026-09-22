import numpy as np
from PIL import Image

from robot_practice.experiments.rl.fastwam_failure_preview import (
    select_windows,
    sheet,
    write_viewer,
)


def test_selection_is_based_on_real_states_and_commands():
    gt = np.zeros((40, 9))
    actions = np.zeros((40, 7))
    actions[:20, 6] = -1
    actions[20:, 6] = 1
    actual = np.ones((61, 9))
    actual[32, :3] = 0
    r = select_windows(gt, actions, actual, -np.ones(9), np.ones(9))
    assert r["demo_anchor"] == 20
    assert r["rollout_anchor"] == 32
    assert all(t % 4 == 0 and t + 4 < len(actual) for t in r["rollout_frames"])
    assert not r["alignment_verified"]


def test_pred_gt_and_input_are_distinct_rows(tmp_path):
    first = Image.new("RGB", (224, 224), "blue")
    predictions = [Image.new("RGB", (224, 224), "red")] * 5
    truth = [Image.new("RGB", (224, 224), "green")] * 5
    path = tmp_path / "comparison.png"
    sheet("rollout", 32, first, predictions, truth, path)
    image = Image.open(path)
    assert image.getpixel((10, 40)) == (0, 0, 255)
    assert image.getpixel((234, 294)) == (255, 0, 0)
    assert image.getpixel((234, 548)) == (0, 128, 0)
    assert image.getpixel((234, 802)) == (0, 0, 255)


def test_viewer_preserves_every_input_frame(tmp_path):
    cases = {f"demo_step_{t:04d}": {"source": "demo", "control_step": t}
             for t in range(80, 121)}
    write_viewer(tmp_path, cases)
    content = (tmp_path / "index.html").read_text()
    for t in range(80, 121):
        assert f'"step": {t}' in content
    assert "Event beyond forecast horizon" in content
    assert "not a pre-event prediction" in content
