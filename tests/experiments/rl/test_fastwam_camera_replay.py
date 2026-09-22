import numpy as np

from robot_practice.experiments.rl.fastwam_camera_replay import prefix_trace
from robot_practice.experiments.rl.scene_capture import compare_traces


def test_prefix_keeps_final_observation_but_not_next_action():
    trace = {k: np.arange(9) for k in ("sim_states", "integration_states", "rgb", "proprio")}
    trace.update({k: np.arange(8) for k in ("actions", "rewards", "successes")})
    cropped = prefix_trace(trace, 4)
    assert len(cropped["rgb"]) == 5
    assert len(cropped["actions"]) == 4
    assert compare_traces(cropped, cropped)["passed"]
    changed = {k: v.copy() for k, v in cropped.items()}
    changed["rgb"][-1] += 1
    assert not compare_traces(cropped, changed)["passed"]
