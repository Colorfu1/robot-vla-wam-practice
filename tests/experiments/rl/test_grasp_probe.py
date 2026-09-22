import numpy as np

from robot_practice.experiments.rl import grasp_probe


def test_passive_hook_preserves_actions_results_and_step_count(monkeypatch, tmp_path):
    events = []
    actions = []
    result = object()

    class Env:
        def step(self, action):
            actions.append(action)
            return result

    monkeypatch.setattr(grasp_probe, "snapshot", lambda env: {"steps": len(actions)})
    monkeypatch.setattr(grasp_probe, "record", lambda *args, **kwargs: events.append(kwargs))
    grasp_probe.install(tmp_path, 0, Env)
    env = Env()
    action = np.arange(7, dtype=float)
    for _ in range(5):
        assert env.step(action) is result
    assert all(a is action for a in actions)
    np.testing.assert_array_equal(action, np.arange(7))
    assert events[0]["before"] == {"steps": 0}
    assert events[0]["after"] == {"steps": 1}
    assert [(e["chunk"], e["substep"]) for e in events] == [
        (0, 0), (0, 1), (0, 2), (0, 3), (1, 0)
    ]
