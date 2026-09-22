from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from robot_practice.experiments.rl import lightwam_square_diagnostic as diag
from robot_practice.experiments.rl.lightwam_base import (
    DummyLightWAMCore,
    LightWAMBaseAdapter,
    RobomimicLightWAMWindowDataset,
)


def test_donors_and_offline(tmp_path):
    images = np.zeros((14, 3, 16, 16), dtype=np.uint8)
    images[7:] = 255
    np.savez(
        tmp_path / "data.npz",
        images=images,
        states=np.zeros((14, 9)),
        actions=np.zeros((14, 7)),
        traj_lengths=[7, 7],
    )
    torch.save(torch.zeros(2, 3), tmp_path / "context.pt")
    dataset = RobomimicLightWAMWindowDataset(
        str(tmp_path / "data.npz"),
        image_size=(32, 32),
        context_path=str(tmp_path / "context.pt"),
    )
    indices = list(range(len(dataset)))
    donors = diag.donor_indices(dataset, indices)
    assert donors == [3, 4, 5, 0, 1, 2]
    adapter = LightWAMBaseAdapter(
        DummyLightWAMCore(),
        image_size=(32, 32),
        context=dataset.context,
        context_mask=dataset.context_mask,
        clip_actions=False,
    )
    result = diag.offline(adapter, dataset, indices, donors, tmp_path)
    assert result["preprocessing_pass_at_1e-5"]
    assert result["prediction_change_mse_raw"] > 0
    saved = np.load(tmp_path / "offline_predictions.npz")
    assert saved["target"].shape == (6, 4, 7)
    assert (tmp_path / "offline_metrics.json").is_file()


def test_single_trajectory_rejected():
    dataset = SimpleNamespace(traj_lengths=[7], video_frames=5, action_horizon=4)
    with pytest.raises(ValueError, match="two"):
        diag.donor_indices(dataset, [0])


def test_rollout_clipping_recording_and_seed_pairing(tmp_path, monkeypatch):
    class FakeEnv:
        def __init__(self, **kwargs):
            self.environment_seeds = [kwargs["policy_seed"] + i for i in range(2)]
            self.step_index = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def reset(self):
            return {"rgb": np.zeros((2, 1, 3, 16, 16)), "state": np.zeros((2, 1, 9))}

        def step(self, actions):
            assert actions.shape == (2, 4, 7)
            assert np.all(actions == 1)
            self.step_index += 1
            return (
                self.reset(),
                np.array([4, 0]),
                np.zeros(2, bool),
                np.full(2, self.step_index == 2),
            )

    class FakeAdapter:
        def predict_action(self, obs, seed):
            assert seed == [10001, 10002]
            return torch.full((2, 4, 7), 2.0)

    monkeypatch.setattr(diag, "_DPPOSquareEnvClient", FakeEnv)
    monkeypatch.setattr(diag, "_resolve_dppo_runtime", lambda **kwargs: (None, None))
    cfg = OmegaConf.create({"evaluation": {"normalization_path": "unused", "success_threshold": 1}})
    args = SimpleNamespace(episodes=2, max_steps=8, seed=10001)
    result = diag.rollout(FakeAdapter(), cfg, args, tmp_path)
    assert result["success_rate"] == 0.5
    assert result["environment_seeds"] == [10001, 10002]
    assert (tmp_path / "env_10001.gif").is_file()
    trace = np.load(tmp_path / "rollout_trace.npz")
    assert trace["states"].shape == (3, 2, 1, 9)
    assert np.all(trace["action_raw"] == 2)
    assert np.all(trace["action_executed"] == 1)
    assert (tmp_path / "forensics/chunk_0000_input.npz").is_file()
    assert (tmp_path / "forensics/chunk_0001_result.npz").is_file()


def test_rollout_failure_restores_opt_in_and_records_memory(tmp_path, monkeypatch):
    import json
    import os

    def fail(*args):
        raise EOFError("simulated failure")

    monkeypatch.setattr(diag, "_rollout", fail)
    monkeypatch.setenv("ROBOT_PRACTICE_ROLLOUT_FORENSICS", "previous")
    with pytest.raises(EOFError):
        diag.rollout(None, None, None, tmp_path)
    assert os.environ["ROBOT_PRACTICE_ROLLOUT_FORENSICS"] == "previous"
    events = [
        json.loads(line)
        for path in (tmp_path / "forensics").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [row["event"] for row in events] == ["rollout_begin", "rollout_failed", "rollout_end"]
    assert "memory" in events[1]
