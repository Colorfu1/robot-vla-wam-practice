from pathlib import Path

import pytest

from robot_practice.experiments.rl.fastwam_square_benchmark import (
    model_kwargs,
    project_times,
    write_result,
)


def test_projection_counts_optimizer_updates_and_samples():
    p = project_times([10, 20], 256, 1000, 420000, 256, 4000)
    assert p["mean_update_seconds"] == 15
    assert p["hours_for_target_sample_volume"] == 1750
    assert p["target_sample_presentations"] == 107520000
    assert p["comparable_update_batch"]
    assert p["time_to_convergence"] is None
    assert p["hours_for_epochs"] == pytest.approx(4000000 / (256 / 15) / 3600)


def test_different_batch_reports_sample_volume_not_same_update_time():
    p = project_times([10], 32, 1000, 100, 256, 1)
    assert not p["comparable_update_batch"]
    assert p["hours_for_target_sample_volume"] == pytest.approx(8000 / 3600)
    with pytest.raises(ValueError):
        project_times([], 32, 1000, 100, 256, 1)


def test_config_resolves_official_task_interpolations(tmp_path):
    directory = tmp_path / "configs/model"
    directory.mkdir(parents=True)
    (directory / "fastwam.yaml").write_text(
        "_target_: fastwam.runtime.create_fastwam\n"
        "proprio_dim: ${data.train.processor.proprio_output_dim}\n"
        "video_dit_config:\n  action_dim: ${data.train.processor.action_output_dim}\n"
        "  use_gradient_checkpointing: ${model.mot_checkpoint_mixed_attn}\n"
        "action_dit_config:\n  action_dim: ${data.train.processor.action_output_dim}\n"
        "  use_gradient_checkpointing: ${model.mot_checkpoint_mixed_attn}\n"
    )
    cfg = model_kwargs(tmp_path, tmp_path / "action.pt", True)
    assert cfg["proprio_dim"] == 9
    assert cfg["video_dit_config"]["action_dim"] == 7
    assert cfg["action_dit_config"]["use_gradient_checkpointing"] is True
    assert cfg["load_text_encoder"] is False
    assert "_target_" not in cfg


def test_local_vendor_config_if_available():
    vendor = Path("third_party/w0_fastwam/FastWAM")
    if not (vendor / "configs/model/fastwam.yaml").exists():
        pytest.skip("Ignored vendor not installed")
    cfg = model_kwargs(vendor, "action.pt", False)
    assert cfg["model_id"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert cfg["video_dit_config"]["use_gradient_checkpointing"] is False


def test_metrics_are_replaced(tmp_path):
    import json

    path = tmp_path / "metrics.json"
    write_result(path, {"status": "running"})
    write_result(path, {"status": "complete"})
    assert json.loads(path.read_text())["status"] == "complete"
    assert not path.with_suffix(".tmp").exists()
