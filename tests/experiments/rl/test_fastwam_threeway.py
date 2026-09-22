import json

import numpy as np
import pytest

from robot_practice.experiments.rl.fastwam_threeway import intervention_sources, threeway_viewer


def make_run(root):
    directory = root / "frame_0085" / "policy_continuation"
    directory.mkdir(parents=True)
    np.savez(directory / "control_trace.npz", rgb=np.full((9, 3, 96, 96), 17),
             proprio=np.full((9, 9), 18), actions=np.full((8, 7), 19))
    data = {"status": "complete", "trajectory": 16, "source_metrics_sha256": "abc",
            "cases": {"85": {"eligible": True,
                              "demo_continuation": {"restore_gate": {"passed": True},
                                                    "task_success_any_step": True},
                              "policy_continuation": {"restore_gate": {"passed": True},
                                                      "task_success_any_step": False,
                                                      "artifact_directory":
                                                          "frame_0085/policy_continuation"}},
                      "101": {"eligible": False, "policy_skipped_reason": "demo failed"}}}
    (root / "metrics.json").write_text(json.dumps(data))


def test_intervention_uses_actual_policy_trace_not_demo(tmp_path):
    make_run(tmp_path)
    sources, details, excluded = intervention_sources(tmp_path, "abc", 16, [0, 1, 4])
    assert len(sources) == 1
    assert sources[0][0] == "intervention_0085"
    assert np.all(sources[0][1]["rgb"] == 17)
    assert np.all(sources[0][1]["actions"] == 19)
    assert details["intervention_0085"]["restored_demo_frame"] == 85
    assert excluded == {"101": "demo failed"}


def test_reject_mismatched_identity_and_unrecorded_future(tmp_path):
    make_run(tmp_path)
    for digest, trajectory, frames in [("wrong", 16, [0]), ("abc", 7, [0]), ("abc", 16, [5])]:
        with pytest.raises(ValueError):
            intervention_sources(tmp_path, digest, trajectory, frames)


def test_viewer_has_three_independent_groups(tmp_path):
    cases = {name: {"source": name, "control_step": step}
             for name, step in [("demo", 100), ("rollout", 276), ("intervention_0085", 0)]}
    threeway_viewer(tmp_path, cases, {}, {"101": "excluded"})
    page = (tmp_path / "index.html").read_text()
    for group in ("demo", "rollout", "intervention"):
        assert f'id="{group}"' in page
    assert '"control_step": 0' in page
    assert '"control_step": 276' in page
    assert "GT该轨迹自己的真实未来" in page
