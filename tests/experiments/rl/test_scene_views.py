from types import SimpleNamespace

import numpy as np
from PIL import Image

from robot_practice.experiments.rl import grasp_probe
from robot_practice.experiments.rl.scene_views import SceneViews, select_wrist


def test_wrist_requires_unique_real_camera():
    assert select_wrist(["agentview", "robot0_eye_in_hand"]) == "robot0_eye_in_hand"
    assert select_wrist(["agentview"]) is None
    assert select_wrist(["robot0_eye_in_hand", "robot1_eye_in_hand"]) is None


def test_views_are_same_state_diagnostics_not_policy_observations(tmp_path, monkeypatch):
    views = SceneViews.__new__(SceneViews)
    views.env = SimpleNamespace(env=object(), is_success=lambda: {"task": False})
    views.directory = tmp_path
    views.cameras = ["agentview", "robot0_eye_in_hand"]
    views.wrist = views.cameras[1]
    views.rows = []
    views.images = tmp_path / "views"
    views.images.mkdir()
    calls = []

    def render(camera, size):
        calls.append((camera, size))
        return np.full((size, size, 3), 7, dtype=np.uint8)

    views.render = render
    monkeypatch.setattr(grasp_probe, "snapshot", lambda env: {
        "nuts": {"SquareNut": {"position": [0, 0, 0.8]}}, "gripper_qpos": [0.04, -0.04],
        "contacts_endpoint_only": [],
    })
    obs = {"rgb": np.full((3, 96, 96), 7, dtype=np.uint8), "state": np.zeros(9)}
    views.append(obs)
    assert set(obs) == {"rgb", "state"}
    assert obs["rgb"].shape == (3, 96, 96)
    assert calls == [("agentview", 96), ("agentview", 224), ("robot0_eye_in_hand", 224)]
    assert Image.open(views.images / "0000_strip.png").size == (896, 252)
    summary = views.finish()
    assert summary["fresh96_vs_policy_mae_max"] == 0
    assert (tmp_path / "views.html").is_file()
    assert views.rows[0]["square_nut_delta_height_m"] == 0
