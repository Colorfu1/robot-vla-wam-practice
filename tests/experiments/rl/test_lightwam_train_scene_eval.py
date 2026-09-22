import numpy as np
import pytest

from robot_practice.experiments.rl.lightwam_train_scene_eval import (
    match_demo,
    normalize,
    policy_action,
    summarize,
)


def test_mapping_uses_full_actions_not_demo_number():
    actions = np.arange(21).reshape(3, 7) / 21
    low, high = np.full(7, -1), np.ones(7)
    demos = {"demo_9": {"actions": actions}, "demo_0": {"actions": actions + 0.1}}
    name, error = match_demo(normalize(actions, low, high), demos, low, high)
    assert name == "demo_9" and error == 0


def test_mapping_rejects_ambiguous_or_missing():
    actions = np.zeros((3, 7))
    low, high = np.full(7, -1), np.ones(7)
    target = normalize(actions, low, high)
    with pytest.raises(ValueError):
        match_demo(target, {}, low, high)
    with pytest.raises(ValueError):
        match_demo(target, {"a": {"actions": actions}, "b": {"actions": actions}}, low, high)


def test_partial_chunk_and_transient_success_are_separate():
    r = summarize([0, 0, 0, 1, 1], [False, False, False, True, True])
    assert r["task_success_any_step"]
    assert not r["official_chunk_success"]
    assert r["first_success_step"] == 4
    assert summarize([1] * 4, [True] * 4)["official_chunk_success"]


def test_live_observation_policy_action_and_clipping():
    import torch

    class Adapter:
        def predict_action(self, cond, seed):
            assert set(cond) == {"rgb", "state"}
            assert cond["rgb"].shape == (1, 1, 3, 96, 96)
            assert cond["state"].shape == (1, 1, 9)
            assert cond["state"][0, 0, 0] == 0.3
            assert seed == 0
            return torch.full((1, 4, 7), 2.0)

    raw, executed = policy_action(
        Adapter(),
        {
            "rgb": np.zeros((3, 96, 96)),
            "state": np.full(9, 0.3),
        },
        0,
    )
    assert np.all(raw == 2) and np.all(executed == 1)


@pytest.mark.parametrize("policy", ["lightwam", "dp", "fastwam"])
def test_paired_driver_uses_live_observation_after_initial_reset(tmp_path, monkeypatch, policy):
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf

    from robot_practice.experiments.rl import lightwam_base as base
    from robot_practice.experiments.rl import lightwam_train_scene_eval as scene

    npz = tmp_path / "train.npz"
    np.savez(npz, actions=np.zeros((8, 7)), traj_lengths=[8])
    config = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "action": {"dim": 7},
                "data": {"npz": str(npz)},
                "evaluation": {"normalization_path": str(npz)},
                "lightwam": {
                    "action_horizon": 4,
                    "context": "unused",
                    "prompt": "square",
                    "image_size": [224, 224],
                    "video_frames": 5,
                    "vendor_root": ".",
                    "model_config": "unused",
                    "local_model_root": ".",
                    "model_dtype": "bf16",
                    "inference_steps": 10,
                },
            }
        ),
        config,
    )
    checkpoint = tmp_path / "step_1.pt"
    checkpoint.touch()
    calls, seen = [], []

    class Sim:
        def __init__(self, *args):
            self.audit = {"mapping": {0: {"demo": "demo_9"}}}
            self.step = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def obs(self):
            return {"rgb": np.zeros((3, 96, 96)), "state": np.full(9, self.step)}

        def call(self, command, value):
            calls.append(command)
            if command == "audit":
                return {
                    "checks": [
                        {
                            "frame": 0,
                            "sim_state_max_abs_error": 0,
                            "proprio_normalized_max_abs_error": 0,
                            "image_mae_0_255": 1,
                        }
                    ]
                }
            if command == "reset":
                self.step = 0
                return self.obs()
            action, _expected = value
            self.step += len(action)
            return {
                "obs": self.obs(),
                "rewards": [1] * len(action),
                "successes": [True] * len(action),
                "sim_state_errors": [],
            }

    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self):
            return self

        def predict_action(self, cond, seed):
            seen.append(float(cond["state"][0, 0, 0]))
            return torch.zeros((1, 4, 7))

    monkeypatch.setattr(scene, "Simulator", Sim)
    monkeypatch.setattr(base, "_resolve_dppo_runtime", lambda **kwargs: (None, None))
    monkeypatch.setattr(
        base, "load_official_lightwam", lambda **kwargs: SimpleNamespace(eval=lambda: None)
    )
    monkeypatch.setattr(base, "LightWAMBaseAdapter", Adapter)
    monkeypatch.setattr(scene, "load_dp", lambda *a: (None, Adapter()))
    monkeypatch.setattr(
        base,
        "RobomimicLightWAMWindowDataset",
        lambda *a, **k: SimpleNamespace(context=None, context_mask=None),
    )
    args = SimpleNamespace(
        output=str(tmp_path / "out"),
        config=str(config),
        checkpoints=str(checkpoint),
        replay_only=False,
        npz=str(npz),
        hdf5=str(npz),
        normalization=str(npz),
        trajectory_ids=[0],
        seed=0,
        max_steps=8,
        state_tolerance=1e-3,
        image_mae_tolerance=10,
        policy=policy,
        dp_vendor="unused",
    )
    if policy == "fastwam":
        from robot_practice.experiments.rl import fastwam_square_eval

        monkeypatch.setattr(fastwam_square_eval, "load_fastwam",
                            lambda *a: (Adapter(), Adapter(), {"strict_checkpoint_load": True}))
    scene.run(args)
    assert seen == [0, 4]
    assert calls == ["audit", "reset", "step", "step", "reset", "step", "step"]


def test_600_step_budget_keeps_400_step_timeout_distinct():
    success = [False] * 498 + [True] * 102
    assert not summarize(success[:400], success[:400])["task_success_any_step"]
    assert summarize(success, success)["first_success_step"] == 499
