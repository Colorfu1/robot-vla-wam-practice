import os
from pathlib import Path

import pytest
import torch

from robot_practice.experiments.maniskill import m2_base_action_flow as runner
from robot_practice.experiments.maniskill import m2_base_checkpoint_selection as selector
from robot_practice.policies.wam_tiny.action_flow import (
    BASE_IMAGE_KEY,
    HAND_IMAGE_KEY,
    STATE_KEY,
    ActionFlowConfig,
    ActionOnlyFlowModel,
    NormalizationStats,
)

CONFIG = Path("configs/experiments/maniskill/m2_base_official_rgb_4d.json")
NATIVE_4D_CONFIG = Path("configs/experiments/maniskill/m2_base_m1w_native_rgb_4d.json")
NATIVE_4D_100K_CONFIG = Path("configs/experiments/maniskill/m2_base_m1w_native_rgb_4d_100k.json")
M1W_7D_CONFIG = Path("configs/experiments/maniskill/m2_base_m1w_rgb_7d.json")


def test_m2_base_config_supports_official_four_d_contract() -> None:
    result = runner.validate_config(CONFIG, check_inputs=False)
    assert result["action_dim"] == 4
    assert result["state_dim"] == 18
    assert result["world_size"] == 4
    assert result["per_process_batch_size"] == 64
    assert result["split_counts"] == {"train": 100, "development": 100, "test": 790}
    assert result["test_access_allowed"] is False


def test_m2_base_config_supports_m1w_native_four_d_contract() -> None:
    result = runner.validate_config(NATIVE_4D_CONFIG, check_inputs=True)
    assert result["action_dim"] == 4
    assert result["world_size"] == 4
    assert result["per_process_batch_size"] == 64
    assert result["split_counts"] == {"train": 748, "development": 95, "test": 96}
    assert result["test_access_allowed"] is False


def test_m2_base_config_supports_m1w_native_four_d_100k_schedule() -> None:
    config = runner.load_config(NATIVE_4D_100K_CONFIG)
    result = runner.validate_config(config, check_inputs=False)
    assert config["training"]["optimizer_steps"] == 100000
    assert config["training"]["selection_checkpoint_steps"][-1] == 100000
    assert result["world_size"] == 4


def test_m2_base_config_supports_m1w_gpu_seven_d_contract() -> None:
    result = runner.validate_config(M1W_7D_CONFIG, check_inputs=True)
    assert result["action_dim"] == 7
    assert result["backend"] == "physx_cuda"
    assert result["world_size"] == 4
    assert result["per_process_batch_size"] == 64
    assert result["split_counts"] == {"train": 787, "development": 100, "test": 100}
    assert result["test_access_allowed"] is False


def test_markdown_episode_index_and_7d_to_4d_projection_contract(tmp_path: Path) -> None:
    index = tmp_path / "episode_index.md"
    index.write_text(
        "# index\n\n"
        "## train_episode_ids\n\n"
        "```text\n7\n11\n```\n\n"
        "## all_success_at_end_episode_ids\n\n"
        "```text\n7\n11\n19\n```\n",
        encoding="utf-8",
    )
    assert runner._read_markdown_episode_ids(index, section="train_episode_ids") == (7, 11)
    config = {
        "protocol": {
            "action_dim": 4,
            "control_mode": "pd_ee_delta_pos",
            "source_action_dim": 7,
            "action_projection_indices": [0, 1, 2, 6],
        }
    }
    assert runner._action_projection_indices(config) == (0, 1, 2, 6)


def test_action_flow_model_accepts_four_and_seven_dimensions() -> None:
    for action_dim in (4, 7):
        config = ActionFlowConfig(action_dim=action_dim)
        model = ActionOnlyFlowModel(config)
        observations = {
            BASE_IMAGE_KEY: torch.zeros((1, 2, 3, 128, 128)),
            HAND_IMAGE_KEY: torch.zeros((1, 2, 3, 128, 128)),
            STATE_KEY: torch.zeros((1, 2, 18)),
        }
        actions = torch.zeros((1, 16, action_dim))
        with torch.inference_mode():
            prediction = model(observations, actions, torch.full((1,), 0.5))
        assert tuple(prediction.shape) == (1, 16, action_dim)


def test_four_dimensional_normalization_stats_round_trip_shape() -> None:
    stats = NormalizationStats(
        image_mean=torch.zeros(3),
        image_std=torch.ones(3),
        state_mean=torch.zeros(18),
        state_std=torch.ones(18),
        action_mean=torch.zeros(4),
        action_std=torch.ones(4),
    )
    action = torch.zeros((2, 16, 4))
    normalized = stats.normalize_action(action)
    restored, counts = stats.denormalize_action(normalized)
    assert tuple(normalized.shape) == (2, 16, 4)
    assert torch.equal(restored, action)
    assert counts == {"clipped_elements": 0, "total_elements": 128}


def test_invalid_action_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="action_dim=4 or action_dim=7"):
        ActionFlowConfig(action_dim=8)


def test_output_preflight_rejects_non_empty_directory_before_ddp(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    (output_dir / "partial.log").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing existing non-empty M2 base output"):
        runner._validate_output_target(output_dir, overwrite=False)


def test_selector_ranking_prefers_closed_loop_success_before_checkpoint_age() -> None:
    lower_step = {
        "checkpoint_step": 10_000,
        "n_action_steps": 8,
        "summary": {
            "success_at_end": 0.5,
            "success_once": 0.6,
            "ever_on_cube": 0.7,
            "mean_first_success_step": 40.0,
        },
    }
    higher_step = {
        "checkpoint_step": 20_000,
        "n_action_steps": 4,
        "summary": {
            "success_at_end": 0.6,
            "success_once": 0.6,
            "ever_on_cube": 0.7,
            "mean_first_success_step": 40.0,
        },
    }
    assert selector._selection_key(higher_step) < selector._selection_key(lower_step)


def test_selector_parser_has_route_aware_development_defaults() -> None:
    parsed = selector.build_parser().parse_args(
        [
            "select-development",
            "--train-root",
            "outputs/train",
            "--output-root",
            "outputs/selection",
        ]
    )
    assert parsed.policy_device == "auto"
    assert parsed.num_gpus is None
    assert parsed.num_envs == 64
    assert parsed.num_workers == 4
    assert parsed.max_steps == 200


def test_selector_route_follows_protocol_backend() -> None:
    official = selector._resolve_rollout_route(runner.load_config(CONFIG), "auto")
    native = selector._resolve_rollout_route(runner.load_config(NATIVE_4D_100K_CONFIG), "auto")
    assert official == ("physx_cpu", "cpu")
    assert native == ("physx_cuda", "cuda")
    with pytest.raises(ValueError, match="conflicts"):
        selector._resolve_rollout_route(runner.load_config(NATIVE_4D_CONFIG), "cpu")


def test_gpu_selector_worker_isolates_one_physical_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_evaluate_task(task: object, **kwargs: object) -> dict[str, object]:
        observed["visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        observed["kwargs"] = kwargs
        return {"task": task}

    monkeypatch.setattr(selector, "_evaluate_task", fake_evaluate_task)
    monkeypatch.setattr(selector.torch.cuda, "empty_cache", lambda: None)

    result = selector._evaluate_gpu_task_batch([{"candidate": "demo"}], "7")

    assert result == [{"task": {"candidate": "demo"}}]
    assert observed["visible_devices"] == "7"
    assert observed["kwargs"] == {"gpu_index": 0, "physical_gpu_id": "7"}
