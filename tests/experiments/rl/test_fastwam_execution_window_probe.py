import json
from pathlib import Path

import numpy as np
import pytest

from robot_practice.experiments.rl.fastwam_execution_window_probe import (
    KEYS,
    action_slices,
    compare_interval,
    handoff_jobs,
    handoff_stage_gate,
    read_handoff,
    splice,
    stage_summary,
    validate_handoff_pairs,
)


def trace(n):
    return {**{k: np.arange(n + 1)[:, None].astype(float) for k in KEYS},
            **{k: np.arange(n)[:, None].astype(float)
               for k in ("actions", "rewards", "successes")}}


def test_splice_has_one_branch_state_and_no_missing_action():
    original = trace(12)
    intervention = {k: v[4:].copy() for k, v in original.items()}
    restored = splice(original, intervention, 4)
    assert all(np.array_equal(original[k], restored[k]) for k in original)
    assert all(v == 0 for v in compare_interval(original, restored, 0, 12).values())


def test_interval_checks_actions_as_well_as_states():
    original = trace(12)
    partial = {k: v[4:9 if k in KEYS else 8].copy() for k, v in original.items()}
    assert all(v == 0 for v in compare_interval(original, partial, 4, 8).values())
    partial["actions"][0] += .1
    with pytest.raises(ValueError, match="gate failed"):
        compare_interval(original, partial, 4, 8)


@pytest.mark.parametrize("execution", [4, 8, 16, 32])
def test_execution_uses_requested_prefix_and_truncates_only_at_budget(execution):
    raw = np.ones((32, 7)) * 2
    raw[:15, 6] = -1
    full = action_slices(raw, execution, 476)
    assert full.shape == (execution, 7)
    assert (full[:, :6] == 1).all()
    assert bool((full[:, 6] > 0).any()) == (execution >= 16)
    assert action_slices(raw, execution, 4).shape == (4, 7)


def test_nonfinite_prediction_fails():
    with pytest.raises(ValueError, match="Invalid"):
        action_slices(np.full((32, 7), np.nan), 4, 600)


def test_handoff_job_pairs_keep_switch_fixed():
    assert handoff_jobs("380,412", 600) == [(380, 4), (380, 32), (412, 4), (412, 32)]


@pytest.mark.parametrize("steps", ["380,380", "381", "600", "-4", "", "0"])
def test_handoff_rejects_invalid_boundaries(steps):
    with pytest.raises(ValueError):
        handoff_jobs(steps, 600)


def saved_handoff(tmp_path):
    visual = tmp_path / "visual"
    source = tmp_path / "execute32"
    (source / "prefix").mkdir(parents=True)
    (source / "continuation").mkdir()
    reference = trace(68)
    reference["actions"] = np.zeros((68, 7))
    prefix = {k: v[:5 if k in KEYS else 4] for k, v in reference.items()}
    tail = {k: v[4:] for k, v in reference.items()}
    np.savez(source / "prefix/control_trace.npz", **prefix)
    np.savez(source / "continuation/control_trace.npz", **tail)
    np.savez(source / "plans.npz", decision_steps=[4, 36], executed_counts=[32, 32],
             native_actions=np.zeros((2, 32, 7)))
    meta = {"status": "complete", "branch": "original", "execute_steps": 32,
            "trajectory": 0, "source": str(visual), "checkpoint_sha256": "test",
            "first_prediction_max_error": 0, "prefix_gate": {"sim_states": 0},
            "branch_step": 4, "noise_seed": 0}
    (source / "metrics.json").write_text(json.dumps(meta))
    return source, visual, reference


def test_handoff_reconstructs_full_policy_history_and_plan(tmp_path):
    source, visual, reference = saved_handoff(tmp_path)
    restored, plan, hashes, seed = read_handoff(source, visual, 0, 36, 68, "test")
    assert all(np.array_equal(v, restored[k]) for k, v in reference.items())
    assert plan.shape == (32, 7) and seed == 0 and len(hashes) == 4
    assert all(len(h) == 64 for h in hashes.values())


@pytest.mark.parametrize("mismatch", ["boundary", "actions", "step", "checkpoint"])
def test_handoff_rejects_source_mismatches(tmp_path, mismatch):
    source, visual, _ = saved_handoff(tmp_path)
    if mismatch in ("boundary", "actions"):
        path = source / "continuation/control_trace.npz"
        with np.load(path) as z:
            d = {k: z[k].copy() for k in z.files}
        d["sim_states" if mismatch == "boundary" else "actions"][0] += 1
        np.savez(path, **d)
    with pytest.raises(ValueError):
        read_handoff(source, visual, 0, 40 if mismatch == "step" else 36, 68,
                     "changed" if mismatch == "checkpoint" else "test")


def stage_row(height=0.0, contact=False):
    return {"nut_xyz": [0, 0, .83 + height], "eef_xyz": [0, 0, 1],
            "finger_q": [.01, -.01], "bilateral_contact": contact}


def test_handoff_stage_requires_sustained_elevation_and_contacts():
    assert handoff_stage_gate([stage_row()] + [stage_row(.05, True)] * 5)["passed"]
    assert not handoff_stage_gate([stage_row()] + [stage_row(.05)] * 5)["passed"]
    assert not handoff_stage_gate([stage_row()] + [stage_row(0, True)] * 5)["passed"]
    assert not handoff_stage_gate([stage_row()] * 4 + [stage_row(.05, True)])["passed"]


def test_stage_summary_uses_absolute_steps_and_does_not_call_contact_loss_drop():
    rows = [stage_row()] + [stage_row(.05, True)] * 5 + [stage_row()] * 6
    actions = np.ones((6, 7))
    actions[2:, 6] = -1
    result = stage_summary(rows, 5, actions)
    assert result["first_open_command_control_step"] == 7
    assert result["first_5_endpoint_contact_loss_start"] == 6
    assert result["first_return_within_10mm_of_reset_height"] == 6
    assert "drop" not in result
    with pytest.raises(ValueError, match="indexing"):
        stage_summary(rows[:-1], 5, actions)


def pair_rows():
    return {f"handoff_0380_execute{k}": {
        "status": "complete", "handoff_stage_gate": {"passed": True},
        "first_prediction_max_error": 0, "prefix_gate": {"sim_states": 0},
        "branch_rgb_sha256": "rgb", "branch_state_sha256": "state",
        "checkpoint_sha256": "weights", "noise_seed": 0, "handoff_source_sha256": {},
        "execute32_suffix_gate": {"sim_states": 0},
    } for k in (4, 32)}


def test_handoff_pair_requires_same_inputs_and_control_suffix():
    pairs = pair_rows()
    assert validate_handoff_pairs(pairs, "380")["380"]["passed"]
    pairs["handoff_0380_execute4"]["branch_rgb_sha256"] = "other"
    with pytest.raises(ValueError, match="different"):
        validate_handoff_pairs(pairs, "380")
    pairs = pair_rows()
    pairs["handoff_0380_execute32"]["execute32_suffix_gate"]["sim_states"] = 1e-4
    with pytest.raises(ValueError, match="suffix"):
        validate_handoff_pairs(pairs, "380")


@pytest.mark.parametrize("execution", [4, 32])
@pytest.mark.parametrize("stage_ok", [True, False])
def test_handoff_worker_restores_policy_prefix_then_uses_live_inputs(
    tmp_path, monkeypatch, execution, stage_ok
):
    from types import SimpleNamespace as NS

    from robot_practice.experiments.rl import fastwam_execution_window_probe as probe
    from robot_practice.experiments.rl import fastwam_square_eval, lightwam_base
    from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL

    source, visual, reference = saved_handoff(tmp_path)
    scene = visual / "gpu_0/trajectory_0"
    (scene / "original").mkdir(parents=True)
    np.savez(scene / "original/control_trace.npz", **reference)
    checkpoint = tmp_path / "train/checkpoints/step.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    dual = tmp_path / "dual"
    dual.mkdir()
    (dual / "manifest.json").write_text(json.dumps({
        "source_sha256": {"hdf5": "test", "normalization": "test"}}))
    (checkpoint.parent.parent / "metrics.json").write_text(json.dumps({
        "arguments": {"dual_data": str(dual), "npz": "unused", "context": "unused"}}))
    (scene.parent / "metrics.json").write_text(json.dumps({
        "status": "complete", "protocol": PROTOCOL, "checkpoint": str(checkpoint),
        "checkpoint_sha256": "test", "seed": 0}))
    (scene / "metrics.json").write_text("{}")
    norm = tmp_path / "norm.npz"
    np.savez(norm, obs_min=np.full(9, -1), obs_max=np.ones(9))
    output = tmp_path / "out"
    output.mkdir()
    commands, predicted_steps, executed_actions = [], [], []

    class Sim:
        def __init__(self, *args, **kwargs):
            self.t = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def stage(self):
            return stage_row(.05 if self.t >= 28 else 0, stage_ok and self.t >= 28)

        def call(self, command, value):
            commands.append(command)
            if command == "reset":
                self.t = 0
            elif command == "square_grasp_evidence":
                return self.stage()
            elif command == "capture_start":
                self.start, self.path, self.actions = self.t, Path(value), []
                self.path.mkdir()
            elif command == "capture_stop":
                captured = {k: v[self.start:self.t + int(k in KEYS)].copy()
                            for k, v in reference.items()}
                captured["actions"] = np.asarray(self.actions)
                np.savez(self.path / "control_trace.npz", **captured)
            elif command == "dual_observation":
                return {"rgb": np.full((3, 8, 16), self.t, dtype=np.uint8),
                        "state": np.full(9, self.t)}
            elif command == "rendering_info":
                return {"backend": "egl", "context_class": "EGLGLContext",
                        "egl_device_id": "6", "cuda_visible_devices": None}
            elif command == "step_grasp":
                actions, expected_frame = value
                assert expected_frame is None
                rows = []
                for action in actions:
                    self.actions.append(action.copy())
                    executed_actions.append(action.copy())
                    self.t += 1
                    rows.append(self.stage())
                return {"grasp_evidence": rows, "rewards": [0] * len(rows),
                        "successes": [False] * len(rows)}
            else:
                raise AssertionError(f"Unexpected simulator command: {command}")

    def predict(adapter, obs, seed):
        step = int(obs["state"][0])
        assert (obs["rgb"] == step).all() and seed == 0
        predicted_steps.append(step)
        plan = np.zeros((32, 7))
        plan[:, 6] = step > 36
        return plan

    monkeypatch.setattr(probe, "Simulator", Sim)
    monkeypatch.setattr(probe, "predict", predict)
    monkeypatch.setattr(probe, "file_hash", lambda path: "test")
    monkeypatch.setattr(fastwam_square_eval, "load_fastwam", lambda *a: (None, object(), {}))
    monkeypatch.setattr(lightwam_base, "_resolve_dppo_runtime", lambda **kw: ("unused", "unused"))
    result = {}
    args = NS(source=visual, handoff_source=source, trajectory=0, step=36,
              max_steps=68, branch="original", execute_steps=execution, output=output,
              hdf5="unused", normalization=str(norm), egl_device_id=6)
    if not stage_ok:
        with pytest.raises(ValueError, match="stage gate failed"):
            probe.worker(args, result)
        assert not predicted_steps and len(executed_actions) == 36
        assert not result["handoff_stage_gate"]["passed"]
        assert (output / "prefix_stage_evidence.npz").is_file()
        return
    probe.worker(args, result)
    assert result["status"] == "complete"
    assert predicted_steps == list(range(36, 68, execution))
    assert commands.count("reset") == 1
    assert result["handoff_stage_gate"]["passed"]
    assert np.array_equal(np.asarray(executed_actions)[:36], reference["actions"][:36])
    assert (output / "prefix_stage_evidence.npz").is_file()
    if execution == 32:
        assert all(v == 0 for v in result["execute32_suffix_gate"].values())
    else:
        assert np.asarray(executed_actions)[40:, 6].all()
