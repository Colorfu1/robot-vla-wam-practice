from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np

import robot_practice.experiments.rl.policy_decorator_peg_visualization as visualization
from robot_practice.experiments.rl.policy_decorator_peg_visualization import (
    PolicyDecoratorPegRolloutRecorder,
    load_policy_decorator_peg_episode,
)


def test_s1_rollout_recorder_persists_frames_state_and_residual_contract(
    tmp_path: Path,
) -> None:
    recorder = PolicyDecoratorPegRolloutRecorder(
        output_dir=tmp_path / "raw",
        max_episodes=1,
        metadata={"arm": "residual", "policy_seed": 1},
    )
    state = np.arange(50, dtype=np.float32)
    frame = np.full((8, 10, 3), 127, dtype=np.uint8)
    recorder.start_episode(
        episode_id=0,
        reset_seed=1001,
        state=state,
        frame=frame,
        metadata={"requested_env_id": "PegInsertionSide-v2"},
    )
    recorder.record_step(
        state=state + 1,
        frame=frame + 1,
        base_action=np.arange(7, dtype=np.float32),
        residual_unit=np.full(7, 0.5, dtype=np.float32),
        residual_delta=np.full(7, 0.05, dtype=np.float32),
        composed_action=np.full(7, 0.2, dtype=np.float32),
        reward=1.0,
        success=True,
        success_once=True,
        terminated=True,
        truncated=False,
        decision_step=0,
        horizon_index=0,
    )
    recorder.finish_episode(
        success=True,
        first_success_step=1,
        return_total=1.0,
        steps=1,
        decision_steps=1,
    )
    manifest = recorder.finalize()

    assert manifest["episodes"][0]["file"] == "episode_0000.npz"
    episode = load_policy_decorator_peg_episode(tmp_path / "raw", episode_id=0)
    assert episode.frames.shape == (2, 8, 10, 3)
    assert episode.state.shape == (2, 50)
    assert episode.base_action.shape == (1, 7)
    assert episode.residual_unit.shape == (1, 7)
    assert episode.residual_delta.shape == (1, 7)
    assert episode.composed_action.shape == (1, 7)
    assert episode.frame_success_once.tolist() == [False, True]


def test_s1_rollout_recorder_honors_explicit_global_episode_ids(tmp_path: Path) -> None:
    recorder = PolicyDecoratorPegRolloutRecorder(
        output_dir=tmp_path / "raw",
        max_episodes=2,
        episode_ids=(2, 4),
    )

    assert recorder.should_capture(0) is False
    assert recorder.should_capture(2) is True
    assert recorder.should_capture(4) is True


def test_visualization_without_viewer_does_not_import_optional_evaluation_stack(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recorder = PolicyDecoratorPegRolloutRecorder(
        output_dir=tmp_path / "raw",
        max_episodes=1,
    )
    state = np.zeros(50, dtype=np.float32)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    action = np.zeros(7, dtype=np.float32)
    recorder.start_episode(
        episode_id=0,
        reset_seed=1001,
        state=state,
        frame=frame,
        metadata={"requested_env_id": "PegInsertionSide-v2"},
    )
    recorder.record_step(
        state=state,
        frame=frame,
        base_action=action,
        residual_unit=action,
        residual_delta=action,
        composed_action=action,
        reward=0.0,
        success=False,
        success_once=False,
        terminated=True,
        truncated=False,
        decision_step=0,
        horizon_index=0,
    )
    recorder.finish_episode(
        success=False,
        first_success_step=None,
        return_total=0.0,
        steps=1,
        decision_steps=1,
    )
    recorder.finalize()

    def fake_recording_writer(episode, *, output_path: Path, **kwargs):
        del episode, kwargs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-rrd")
        return output_path

    monkeypatch.setattr(
        visualization,
        "write_policy_decorator_peg_episode_recording",
        fake_recording_writer,
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "robot_practice.evaluation.common":
            raise AssertionError("evaluation.common must be lazy-loaded")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = visualization.write_policy_decorator_peg_visualization(
        rollout_dir=tmp_path / "raw",
        output_dir=tmp_path / "rrd",
        episode_ids=(0,),
        write_video=False,
        open_viewer=False,
    )

    assert result["selected_episode_ids"] == [0]
    assert result["episodes"][0]["viewer_opened"] is False
