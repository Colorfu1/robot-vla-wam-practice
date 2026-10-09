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


def test_square_snapshot_reads_physics_without_forward_or_control():
    from types import SimpleNamespace as NS

    data = NS(body_xpos=np.array([[1., 2., 3.]]), site_xpos=np.array([[4., 5., 6.]]),
              qpos=np.array([.02, -.02]))
    robot = NS(eef_site_id=0, _ref_gripper_joint_pos_indexes=[0, 1], gripper=object())
    nut = NS(name="SquareNut", contact_geoms=["nut"])

    def check(gripper, geoms):
        assert gripper is robot.gripper and geoms == ["nut"]
        return True

    env = NS(nuts=[nut], robots=[robot], sim=NS(data=data), obj_body_id={"SquareNut": 0},
             _check_grasp=check)
    result = grasp_probe.square_snapshot(env)
    assert result["bilateral_contact"]
    assert result["nut_xyz"] == [1, 2, 3]
    result["finger_q"][0] = 99
    assert data.qpos[0] == .02
