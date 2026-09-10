from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from robot_practice.experiments.rl.lightwam_base import (
    DummyLightWAMCore,
    LightWAMBaseAdapter,
    RobomimicLightWAMWindowDataset,
    _build_smoke_timing_summary,
    _checkpoint_looks_like_dppo,
    _context_asset_report,
    _merge_lightwam_square_eval,
    _parse_int_csv,
    _summarize_official_square_rollout,
    _validate_lightwam_checkpoint,
    build_training_config,
    load_context_cache,
)


def test_smoke_timing_summary_estimates_full_optimizer_budget():
    summary = _build_smoke_timing_summary(
        {
            "warmup_steps": 10,
            "measure_steps": 50,
            "elapsed_sec": 25.0,
            "steps_per_sec": 2.0,
            "samples_per_sec": 2.0,
            "effective_global_batch_size": 1,
            "peak_gpu_mem_allocated_gb": 20.0,
            "peak_gpu_mem_reserved_gb": 21.0,
        },
        target_optimizer_steps=420000,
        end_to_end_elapsed_sec=40.0,
        dppo_reference={"optimizer_steps": 424000},
        training_alignment={"optimizer_steps": 420000},
    )
    assert summary["short_run"]["total_optimizer_steps"] == 60
    assert summary["full_training_estimate"]["compute_seconds"] == 210000.0
    assert summary["full_training_estimate"]["compute_hours"] == pytest.approx(58.3333)
    assert summary["dppo_reference"]["optimizer_steps"] == 424000
    assert summary["training_alignment"]["optimizer_steps"] == 420000


def test_smoke_timing_summary_rejects_invalid_throughput():
    with pytest.raises(ValueError, match="steps_per_sec must be positive"):
        _build_smoke_timing_summary(
            {"steps_per_sec": 0.0},
            target_optimizer_steps=424000,
            end_to_end_elapsed_sec=1.0,
        )


def test_formal_lightwam_matches_dppo_epochs_and_effective_global_batch():
    repo_root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        repo_root / "configs/experiments/rl/lightwam_base_robomimic_square.yaml"
    )
    reference = config.dppo_reference
    updates_per_epoch = (
        int(reference.dataset_windows) + int(reference.batch_size) - 1
    ) // int(reference.batch_size)
    assert updates_per_epoch == int(reference.optimizer_steps_per_epoch) == 106
    assert int(reference.epochs) * updates_per_epoch == int(reference.optimizer_steps)
    alignment = config.lightwam_alignment
    assert int(alignment.effective_global_batch_size) == int(reference.batch_size) == 256
    assert int(alignment.optimizer_steps_per_epoch) == 105
    assert int(alignment.epochs) * int(alignment.optimizer_steps_per_epoch) == int(
        alignment.optimizer_steps
    )
    assert int(config.training.num_epochs) == int(reference.epochs) == 4000
    assert int(config.training.max_steps) == int(alignment.optimizer_steps) == 420000
    assert int(config.training.batch_size) == 32
    assert int(config.training.save_every) == 52500


def test_staged_training_keeps_formal_scheduler_horizon(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    cfg = build_training_config(
        vendor_root=repo_root / "third_party/lightwam/Light-WAM",
        model_config=(
            repo_root / "configs/experiments/rl/lightwam_base_robomimic_square_model.yaml"
        ),
        data_path=tmp_path / "train.npz",
        context_path=tmp_path / "context.pt",
        output_dir=tmp_path / "run",
        prompt="insert the square nut into the square peg",
        max_steps=420000,
        stop_after_steps=26350,
    )
    assert int(cfg.max_steps) == 420000
    assert int(cfg.stop_after_steps) == 26350


def test_staged_training_rejects_stop_beyond_scheduler_horizon(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="cannot exceed max_steps"):
        build_training_config(
            vendor_root=repo_root / "third_party/lightwam/Light-WAM",
            model_config=(
                repo_root
                / "configs/experiments/rl/lightwam_base_robomimic_square_model.yaml"
            ),
            data_path=tmp_path / "train.npz",
            context_path=tmp_path / "context.pt",
            output_dir=tmp_path / "run",
            prompt="insert the square nut into the square peg",
            max_steps=100,
            stop_after_steps=101,
        )


def test_lightwam_adapter_accepts_temporal_single_camera_observation():
    adapter = LightWAMBaseAdapter(
        DummyLightWAMCore(device="cpu"),
        image_size=(32, 32),
        prompt="dummy task",
        seed=7,
    )
    action = adapter(
        {
            "rgb": torch.zeros(2, 3, 3, 24, 28),
            "state": torch.zeros(2, 3, 9),
        }
    )
    assert action.shape == (2, 4, 7)
    assert torch.isfinite(action).all()


def test_lightwam_adapter_rejects_multicamera_tensor():
    adapter = LightWAMBaseAdapter(
        DummyLightWAMCore(device="cpu"), image_size=(32, 32), prompt="dummy task"
    )
    with pytest.raises(ValueError, match="exactly one RGB camera"):
        adapter({"rgb": torch.zeros(1, 6, 24, 28), "state": torch.zeros(1, 9)})


def test_cached_context_takes_precedence_over_configured_prompt():
    class SpyCore(DummyLightWAMCore):
        def __init__(self):
            super().__init__(device="cpu")
            self.last_prompt = "unset"

        def infer_action(self, *, prompt, **kwargs):
            self.last_prompt = prompt
            return super().infer_action(prompt=prompt, **kwargs)

    core = SpyCore()
    adapter = LightWAMBaseAdapter(
        core,
        image_size=(32, 32),
        prompt="configured prompt",
        context=torch.zeros(128, 4096),
        context_mask=torch.ones(128, dtype=torch.bool),
    )
    adapter({"rgb": torch.zeros(1, 3, 24, 24), "state": torch.zeros(1, 9)})
    assert core.last_prompt is None


def test_context_cache_tensor_uses_sequence_length_for_default_mask(tmp_path):
    path = tmp_path / "context.pt"
    context = torch.zeros(128, 4096)
    torch.save(context, path)
    loaded_context, loaded_mask = load_context_cache(path)
    assert loaded_context.shape == (128, 4096)
    assert loaded_mask.shape == (128,)
    assert loaded_mask.dtype == torch.bool


def test_context_asset_report_validates_lightwam_embedding_contract(tmp_path):
    path = tmp_path / "context.pt"
    torch.save(
        {
            "context": torch.zeros(128, 4096),
            "mask": torch.ones(128, dtype=torch.bool),
        },
        path,
    )
    report = _context_asset_report(path)
    assert report["present"] is True
    assert report["valid"] is True
    assert report["context_shape"] == [128, 4096]


def test_context_asset_report_rejects_wrong_embedding_dim(tmp_path):
    path = tmp_path / "context.pt"
    torch.save(
        {
            "context": torch.zeros(128, 8),
            "mask": torch.ones(128, dtype=torch.bool),
        },
        path,
    )
    report = _context_asset_report(path)
    assert report["present"] is True
    assert report["valid"] is False
    assert "embedding dim" in report["error"]


def test_robomimic_window_dataset_builds_five_frames_and_four_actions(tmp_path):
    data_path = tmp_path / "train.npz"
    states = np.zeros((8, 9), dtype=np.float32)
    actions = np.zeros((8, 7), dtype=np.float32)
    images = np.zeros((8, 3, 12, 16), dtype=np.uint8)
    images[:, 0] = 255
    np.savez(data_path, states=states, actions=actions, images=images, traj_lengths=np.array([8]))

    context_path = tmp_path / "context.pt"
    torch.save(
        {"context": torch.zeros(128, 4096), "mask": torch.ones(128, dtype=torch.bool)},
        context_path,
    )
    dataset = RobomimicLightWAMWindowDataset(
        str(data_path), image_size=(16, 16), context_path=str(context_path)
    )
    sample = dataset[0]
    assert len(dataset) == 4
    assert sample["video"].shape == (3, 5, 16, 16)
    assert sample["action"].shape == (4, 7)
    assert sample["proprio"].shape == (5, 9)
    assert float(sample["video"].min()) == -1.0
    assert float(sample["video"].max()) == 1.0
    assert sample["context"].shape == (128, 4096)
    assert dataset[len(dataset) - 1]["video"].shape == (3, 5, 16, 16)


def test_robomimic_window_dataset_requires_context_for_formal_sample(tmp_path):
    data_path = tmp_path / "train.npz"
    np.savez(
        data_path,
        states=np.zeros((5, 9), dtype=np.float32),
        actions=np.zeros((5, 7), dtype=np.float32),
        images=np.zeros((5, 3, 16, 16), dtype=np.uint8),
        traj_lengths=np.array([5]),
    )
    dataset = RobomimicLightWAMWindowDataset(str(data_path), image_size=(16, 16))
    with pytest.raises(ValueError, match="requires --context-path"):
        dataset[0]


def test_dppo_checkpoint_is_rejected_by_shape_probe(tmp_path):
    dppo_path = tmp_path / "state_4000.pt"
    torch.save({"model": {"weight": torch.zeros(1)}, "step": 4000}, dppo_path)
    assert _checkpoint_looks_like_dppo(dppo_path)
    with pytest.raises(ValueError, match="DPPO state_4000.pt cannot be loaded"):
        _validate_lightwam_checkpoint(dppo_path)


def test_full_training_state_directory_is_accepted_for_resume(tmp_path):
    state_path = tmp_path / "checkpoints" / "state" / "step_000100"
    state_path.mkdir(parents=True)
    assert (
        _validate_lightwam_checkpoint(
            state_path,
            allow_training_state_directory=True,
        )
        == state_path.resolve()
    )


def test_dummy_cli_payload_is_json_serializable(tmp_path):
    adapter = LightWAMBaseAdapter(
        DummyLightWAMCore(device="cpu"), image_size=(32, 32), prompt="dummy"
    )
    output = adapter({"rgb": torch.zeros(1, 3, 24, 24), "state": torch.zeros(1, 9)})
    payload = {"shape": list(output.shape), "finite": bool(torch.isfinite(output).all())}
    assert json.loads(json.dumps(payload))["shape"] == [1, 4, 7]


def test_eval_seed_parser_accepts_csv_and_rejects_duplicates():
    assert _parse_int_csv("1,2,3") == [1, 2, 3]
    with pytest.raises(ValueError, match="unique"):
        _parse_int_csv("1,2,1")


def test_square_rollout_uses_official_chunk_reward_success_rule():
    rewards = np.zeros((3, 2), dtype=np.float64)
    rewards[1, 0] = 4.0
    rewards[2, 1] = 3.0
    firsts = np.zeros((4, 2), dtype=bool)
    firsts[0] = True
    firsts[3] = True
    rows = _summarize_official_square_rollout(
        rewards,
        firsts,
        act_steps=4,
        success_threshold=1.0,
    )
    assert [row["success"] for row in rows] == [True, False]
    assert [row["best_reward"] for row in rows] == [1.0, 0.75]
    assert [row["policy_steps"] for row in rows] == [3, 3]


def test_eval_merge_marks_4000_step_pilot_as_nonformal(tmp_path):
    output = tmp_path / "eval"
    run_root = tmp_path / "pilot"
    checkpoint = run_root / "checkpoints" / "weights" / "step_004000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (run_root / "config.yaml").write_text("max_steps: 4000\nnum_epochs: 4000\n")
    for seed, success in ((1, True), (2, False), (3, True)):
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        episode = {
            "policy_seed": seed,
            "environment_seed": seed,
            "episode_reward": float(success),
            "best_reward": float(success),
            "success": success,
        }
        (seed_dir / "episodes.jsonl").write_text(json.dumps(episode) + "\n")
        seed_metrics = {
            "seed": seed,
            "episodes": 1,
            "successes": int(success),
            "success_rate": float(success),
            "policy_steps_total": 2,
            "action_chunks_total": 2,
            "inference_seconds": 1.0,
            "elapsed_seconds": 2.0,
        }
        # Seed 1 mimics the one-seed summary directory emitted by the
        # multi-GPU launcher; the others mimic a direct serial eval layout.
        stored_metrics = (
            {"episodes": 1, "per_seed": [seed_metrics]} if seed == 1 else seed_metrics
        )
        (seed_dir / "metrics.json").write_text(json.dumps(stored_metrics))
    one_seed_metrics = _merge_lightwam_square_eval(
        output_dir=output / "seed_1",
        seeds=[1],
        episodes_per_seed=1,
        checkpoint_path=checkpoint,
        max_episode_steps=8,
        action_horizon=4,
        inference_steps=10,
        training_target_steps=420000,
    )
    assert one_seed_metrics["episodes"] == 1
    metrics = _merge_lightwam_square_eval(
        output_dir=output,
        seeds=[1, 2, 3],
        episodes_per_seed=1,
        checkpoint_path=checkpoint,
        max_episode_steps=8,
        action_horizon=4,
        inference_steps=10,
        training_target_steps=420000,
    )
    assert metrics["episodes"] == 3
    assert metrics["success_rate"] == pytest.approx(2 / 3)
    assert metrics["checkpoint_step"] == 4000
    assert metrics["training_budget_complete"] is False
    assert metrics["checkpoint_reached_training_run_budget"] is True
    assert metrics["checkpoint_training_run"]["configured_max_steps"] == 4000
    assert "reached its 4,000-step run budget" in metrics["training_budget_note"]
    assert metrics["formal_eval_protocol"] is False
    assert metrics["formal_result"] is False
    assert metrics["evaluation_mode"] == "pipeline_smoke"
    assert metrics["success_rate_interpretable"] is False
    assert "shorter than the 400-step" in metrics["success_rate_note"]
    assert metrics["dataset_size"] == 26814


def test_streaming_array_diagnostics_reports_action_saturation():
    from robot_practice.experiments.rl.lightwam_base import _StreamingArrayDiagnostics

    stats = _StreamingArrayDiagnostics(feature_dim=2)
    stats.update(np.array([[[1.0, -1.0], [0.0, 0.5]]], dtype=np.float32))
    stats.update(np.array([[[0.25, -0.5]]], dtype=np.float32))
    payload = stats.to_dict(saturation_limit=0.999)
    assert payload["shape_last_dim"] == 2
    assert payload["elements"] == 6
    assert payload["min"] == -1.0
    assert payload["max"] == 1.0
    assert payload["saturation_fraction"] == pytest.approx(2 / 6)
    assert payload["per_dimension_min"] == [0.0, -1.0]
    assert payload["per_dimension_max"] == [1.0, 0.5]
