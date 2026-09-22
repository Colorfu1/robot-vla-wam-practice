import numpy as np

from robot_practice.experiments.rl.scene_capture import compare_traces


def trace():
    return {key: np.zeros((3, 2)) for key in (
        "sim_states", "integration_states", "proprio", "rgb", "rewards", "successes"
    )}


def test_prefix_gate_checks_intermediate_states_not_just_endpoint():
    a, b = trace(), trace()
    assert compare_traces(a, b)["passed"]
    b["integration_states"][1, 0] = 0.01
    assert not compare_traces(a, b)["passed"]


def test_prefix_gate_rejects_nan_and_shape_mismatch():
    a, b = trace(), trace()
    b["rgb"][0, 0] = np.nan
    assert not compare_traces(a, b)["passed"]
    b = trace()
    b["proprio"] = b["proprio"][:-1]
    assert not compare_traces(a, b)["passed"]
