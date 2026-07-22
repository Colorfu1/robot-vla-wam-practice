from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pytest

import robot_practice.evaluation.libero_act as libero
import robot_practice.evaluation.offline_act as offline
from robot_practice.evaluation.libero_tasks import resolve_libero_task


def test_offline_parser_defaults_to_all_task_episodes() -> None:
    args = offline.build_argument_parser().parse_args(
        [
            "--policy-path",
            "/tmp/checkpoint",
            "--suite",
            "libero_object",
            "--task-id",
            "8",
        ]
    )
    assert args.suite == "libero_object"
    assert args.task_id == 8
    assert args.episodes is None
    assert args.dataset_root is None
    assert args.batch_size == 1
    assert args.num_visualized_episodes == 1
    assert args.overwrite is False
    assert args.max_frames is None
    assert args.rebuild_dataset_cache is False
    assert args.open_viewer is True
    assert args.agentview_image_convention == "auto"


def test_offline_parser_accepts_explicit_agentview_image_convention() -> None:
    args = offline.build_argument_parser().parse_args(
        [
            "--policy-path",
            "/tmp/checkpoint",
            "--suite",
            "libero_object",
            "--task-id",
            "8",
            "--agentview-image-convention",
            "lerobot_180",
        ]
    )
    assert args.agentview_image_convention == "lerobot_180"


def test_offline_config_rejects_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    config = offline.OfflineEvaluationConfig(
        policy_path=tmp_path,
        suite="libero_object",
        task_id=8,
        dataset_root=root,
        episodes=(823, 823),
    )
    with pytest.raises(ValueError, match="duplicate"):
        offline.validate_config(config)


def test_prediction_schema_has_exact_columns_and_nullable_latency() -> None:
    schema = offline._prediction_schema(20)
    assert schema.names == [
        "suite",
        "libero_task_id",
        "episode_index",
        "frame_index",
        "dataset_index",
        "timestamp_s",
        "task_index",
        "batch_id",
        "batch_row",
        "batch_size",
        "target_action",
        "predicted_action",
        "target_chunk",
        "predicted_chunk",
        "valid_horizon",
        "absolute_error",
        "model_query_latency_ms",
    ]
    assert pa.types.is_fixed_size_list(schema.field("predicted_chunk").type)
    assert schema.field("predicted_chunk").type.list_size == 140
    assert schema.field("model_query_latency_ms").nullable


def test_offline_metrics_only_does_not_create_skeleton_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = resolve_libero_task("libero_object", 8)
    monkeypatch.setattr(
        offline,
        "PandaSkeletonReconstructor",
        lambda _: pytest.fail("metrics-only evaluation created a MuJoCo environment"),
    )

    assert offline._make_skeleton_reconstructor(task, ()) is None


def test_offline_visualization_creates_skeleton_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = resolve_libero_task("libero_object", 8)
    sentinel = object()
    monkeypatch.setattr(offline, "PandaSkeletonReconstructor", lambda value: sentinel)

    assert offline._make_skeleton_reconstructor(task, (823,)) is sentinel


def test_offline_camera_metadata_requires_explicit_hwc_contract() -> None:
    metadata = SimpleNamespace(
        features={
            "observation.images.image": {
                "shape": (256, 320, 3),
                "names": ["height", "width", "channel"],
            }
        }
    )
    assert offline._camera_hwc_shape(metadata, "observation.images.image") == (
        256,
        320,
        3,
    )


@pytest.mark.parametrize(
    "feature",
    [
        {"shape": (3, 256, 320), "names": ["channel", "height", "width"]},
        {"shape": (256, 320), "names": ["height", "width"]},
        {"shape": (256, 320, 2), "names": ["height", "width", "channel"]},
    ],
)
def test_offline_camera_metadata_rejects_ambiguous_layout(feature: object) -> None:
    metadata = SimpleNamespace(features={"observation.images.image": feature})
    with pytest.raises(ValueError, match="HWC"):
        offline._camera_hwc_shape(metadata, "observation.images.image")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_episodes", 0, "positive"),
        ("num_visualized_episodes", 2, "between"),
        ("perturbation", "image_shift", "none"),
        ("max_steps", 281, "between"),
    ],
)
def test_libero_config_validation(field: str, value: object, message: str) -> None:
    values = {
        "policy_path": Path("checkpoint"),
        "suite": "libero_object",
        "task_id": 8,
        "n_episodes": 1,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        libero.validate_config(libero.LiberoEvaluationConfig(**values))


def test_libero_parser_defaults_match_approved_contract() -> None:
    args = libero.build_argument_parser().parse_args(
        [
            "--policy-path",
            "/tmp/checkpoint",
            "--suite",
            "libero_goal",
            "--task-id",
            "3",
        ]
    )
    assert args.suite == "libero_goal"
    assert args.task_id == 3
    assert args.n_episodes == 20
    assert args.num_visualized_episodes == 1
    assert args.perturbation == "none"
    assert args.seed == 1000
    assert args.max_steps is None
    assert args.overwrite is False
    assert args.open_viewer is True


def test_evaluator_parsers_accept_explicit_overwrite() -> None:
    required = [
        "--policy-path=/tmp/checkpoint",
        "--suite=libero_object",
        "--task-id=8",
        "--output-dir=/tmp/result",
        "--overwrite",
    ]
    assert offline.build_argument_parser().parse_args(required).overwrite is True
    assert libero.build_argument_parser().parse_args(required).overwrite is True


@pytest.mark.parametrize(
    "config",
    [
        offline.OfflineEvaluationConfig(
            policy_path=Path("checkpoint"),
            suite="libero_object",
            task_id=8,
            overwrite=True,
        ),
        libero.LiberoEvaluationConfig(
            policy_path=Path("checkpoint"),
            suite="libero_object",
            task_id=8,
            overwrite=True,
        ),
    ],
)
def test_overwrite_requires_explicit_output_dir(config: object) -> None:
    with pytest.raises(ValueError, match="requires an explicit --output-dir"):
        if isinstance(config, offline.OfflineEvaluationConfig):
            offline.validate_config(config)
        else:
            libero.validate_config(config)


def test_libero_environment_config_uses_resolved_task() -> None:
    task = resolve_libero_task("libero_object", 8)
    config = libero.make_libero_config(task)
    assert config.task == "libero_object"
    assert config.task_ids == [8]
    assert config.obs_type == "pixels_agent_pos"
    assert config.control_mode == "relative"
    assert config.observation_height == 256
    assert config.observation_width == 256
    assert config.episode_length == 280
    assert config.camera_name == "agentview_image,robot0_eye_in_hand_image"

    goal = libero.make_libero_config(resolve_libero_task("libero_goal", 3))
    assert goal.task == "libero_goal"
    assert goal.task_ids == [3]
    assert goal.episode_length == 300


def test_extract_terminal_success_requires_vectorized_final_info() -> None:
    info = {
        "_final_info": np.array([True]),
        "final_info": {
            "is_success": np.array([True]),
            "_is_success": np.array([True]),
        },
    }
    assert libero.extract_terminal_success(info, terminal=True) is True
    assert libero.extract_terminal_success({}, terminal=False) is False
    with pytest.raises(RuntimeError, match="_final_info"):
        libero.extract_terminal_success({}, terminal=True)


def test_set_init_state_id_controls_underlying_serial_env() -> None:
    base = SimpleNamespace(init_state_id=99)
    env = SimpleNamespace(num_envs=1, envs=[base])
    libero._set_init_state_id(env, 7)
    assert base.init_state_id == 7


def test_rollout_schema_distinguishes_instant_and_cumulative_success() -> None:
    schema = libero._rollout_schema()
    assert schema.names == [
        "suite",
        "libero_task_id",
        "episode_index",
        "seed",
        "init_state_id",
        "step_index",
        "global_step",
        "timestamp_s",
        "executed_action",
        "reward",
        "terminated",
        "truncated",
        "done_after_step",
        "success_this_step",
        "success_so_far",
        "termination_reason",
        "policy_step_latency_ms",
        "model_query_latency_ms",
        "visualized",
    ]
    assert schema.field("model_query_latency_ms").nullable
    assert schema.field("termination_reason").nullable
    assert not schema.field("policy_step_latency_ms").nullable


def test_eval_info_uses_percentage_only_for_formal_run() -> None:
    metrics = {
        "aggregate": {
            "num_episodes": 1,
            "num_successes": 1,
            "success_rate": 1.0,
            "avg_sum_reward": 2.0,
            "avg_max_reward": 1.0,
        },
        "per_episode": [
            {
                "episode_index": 0,
                "seed": 1000,
                "init_state_id": 0,
                "num_steps": 3,
                "sum_reward": 2.0,
                "max_reward": 1.0,
                "success": True,
                "termination_reason": "success",
            }
        ],
    }
    task = resolve_libero_task("libero_object", 8)
    formal = libero._make_eval_info(
        metrics,
        task=task,
        formal=True,
        elapsed_s=4.0,
    )
    metrics["aggregate"]["success_rate"] = None
    smoke = libero._make_eval_info(
        metrics,
        task=task,
        formal=False,
        elapsed_s=4.0,
    )
    assert formal["suite"] == "libero_object"
    assert formal["task_id"] == 8
    assert formal["aggregated"]["pc_success"] == 100.0
    assert formal["aggregated"]["eval_ep_s"] == 4.0
    assert smoke["aggregated"]["pc_success"] is None


def test_next_step_enum_is_the_expected_contract() -> None:
    assert gym.vector.AutoresetMode.NEXT_STEP.value == "NextStep"
