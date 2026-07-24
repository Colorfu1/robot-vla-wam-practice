from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pytest
import torch

import robot_practice.evaluation.libero_act as libero
import robot_practice.evaluation.offline_act as offline
from robot_practice.evaluation.libero_tasks import resolve_libero_task


class FakeLifecycleStream:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_flush: bool = False,
        fail_disconnect: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_flush = fail_flush
        self.fail_disconnect = fail_disconnect

    def flush(self) -> None:
        self.events.append(f"flush:{self.name}")
        if self.fail_flush:
            raise RuntimeError(f"flush failed: {self.name}")

    def disconnect(self) -> None:
        self.events.append(f"disconnect:{self.name}")
        if self.fail_disconnect:
            raise RuntimeError(f"disconnect failed: {self.name}")


def test_offline_episode_recording_state_is_independent_and_validates_timeline(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    first = offline.OfflineEpisodeRecording(
        episode_index=823,
        path=tmp_path / "823.rrd",
        stream=FakeLifecycleStream("823", events),
    )
    second = offline.OfflineEpisodeRecording(
        episode_index=826,
        path=tmp_path / "826.rrd",
        stream=FakeLifecycleStream("826", events),
    )

    assert first.trajectory_history is not second.trajectory_history
    assert first.command_history is not second.command_history
    first.accept_frame_index(0)
    first.accept_frame_index(2)
    second.accept_frame_index(0)

    with pytest.raises(ValueError, match="monotonically"):
        first.accept_frame_index(1)
    with pytest.raises(ValueError, match="start at frame zero"):
        offline.OfflineEpisodeRecording(
            episode_index=834,
            path=tmp_path / "834.rrd",
            stream=FakeLifecycleStream("834", events),
        ).accept_frame_index(3)


def test_offline_recording_cleanup_flushes_all_before_disconnect(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    recordings = [
        offline.OfflineEpisodeRecording(
            episode_index=823,
            path=tmp_path / "823.rrd",
            stream=FakeLifecycleStream("823", events, fail_flush=True),
        ),
        offline.OfflineEpisodeRecording(
            episode_index=826,
            path=tmp_path / "826.rrd",
            stream=FakeLifecycleStream("826", events),
        ),
    ]

    with pytest.raises(RuntimeError, match="flush failed: 823"):
        offline.flush_and_disconnect_recordings(recordings)

    assert events == [
        "flush:823",
        "flush:826",
        "disconnect:823",
        "disconnect:826",
    ]


def test_libero_episode_recording_state_is_independent_and_validates_timeline(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    first = libero.LiberoEpisodeRecording(
        episode_index=0,
        seed=1000,
        init_state_id=0,
        path=tmp_path / "000.rrd",
        stream=FakeLifecycleStream("000", events),
    )
    second = libero.LiberoEpisodeRecording(
        episode_index=1,
        seed=1001,
        init_state_id=1,
        path=tmp_path / "001.rrd",
        stream=FakeLifecycleStream("001", events),
    )

    assert first.trajectory_history is not second.trajectory_history
    first.accept_step_index(0)
    first.accept_step_index(2)
    second.accept_step_index(0)

    with pytest.raises(ValueError, match="monotonically"):
        first.accept_step_index(1)
    with pytest.raises(ValueError, match="start at step zero"):
        libero.LiberoEpisodeRecording(
            episode_index=2,
            seed=1002,
            init_state_id=2,
            path=tmp_path / "002.rrd",
            stream=FakeLifecycleStream("002", events),
        ).accept_step_index(3)


def test_libero_recording_cleanup_flushes_all_before_disconnect(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    recordings = [
        libero.LiberoEpisodeRecording(
            episode_index=0,
            seed=1000,
            init_state_id=0,
            path=tmp_path / "000.rrd",
            stream=FakeLifecycleStream("000", events, fail_flush=True),
        ),
        libero.LiberoEpisodeRecording(
            episode_index=1,
            seed=1001,
            init_state_id=1,
            path=tmp_path / "001.rrd",
            stream=FakeLifecycleStream("001", events, fail_disconnect=True),
        ),
    ]

    with pytest.raises(RuntimeError, match="flush failed: 000"):
        libero.flush_and_disconnect_recordings(recordings)

    assert events == [
        "flush:000",
        "flush:001",
        "disconnect:000",
        "disconnect:001",
    ]
    libero.flush_and_disconnect_recordings(recordings)
    assert len(events) == 4


@pytest.mark.parametrize(
    ("success", "termination_reason", "expected"),
    [
        (
            True,
            "success",
            "Episode 003 | seed=1003 | init=3 | success",
        ),
        (
            False,
            "time_limit",
            "Episode 003 | seed=1003 | init=3 | failure:time_limit",
        ),
        (
            False,
            "environment_done",
            "Episode 003 | seed=1003 | init=3 | failure:environment_done",
        ),
    ],
)
def test_libero_recording_name_is_exact(
    success: bool,
    termination_reason: str,
    expected: str,
) -> None:
    assert (
        libero.libero_episode_recording_name(
            episode_index=3,
            seed=1003,
            init_state_id=3,
            success=success,
            termination_reason=termination_reason,
        )
        == expected
    )


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
        ("n_action_steps", 0, "positive"),
        ("temporal_ensemble_coeff", float("nan"), "finite"),
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
    assert args.n_action_steps is None
    assert args.temporal_ensemble_coeff is None
    assert args.perturbation == "none"
    assert args.proprio_noise_std == 0.01
    assert args.seed == 1000
    assert args.max_steps is None
    assert args.overwrite is False
    assert args.open_viewer is True


def test_libero_parser_accepts_n_action_steps_override() -> None:
    args = libero.build_argument_parser().parse_args(
        [
            "--policy-path=/tmp/checkpoint",
            "--suite=libero_object",
            "--task-id=8",
            "--n-action-steps=1",
        ]
    )
    assert args.n_action_steps == 1


def test_libero_parser_accepts_temporal_ensemble_override() -> None:
    args = libero.build_argument_parser().parse_args(
        [
            "--policy-path=/tmp/checkpoint",
            "--suite=libero_object",
            "--task-id=8",
            "--n-action-steps=1",
            "--temporal-ensemble-coeff=0.01",
        ]
    )
    assert args.n_action_steps == 1
    assert args.temporal_ensemble_coeff == 0.01


@pytest.mark.parametrize("n_action_steps", [None, 2])
def test_temporal_ensemble_override_requires_explicit_single_step(
    n_action_steps: int | None,
) -> None:
    config = libero.LiberoEvaluationConfig(
        policy_path=Path("checkpoint"),
        suite="libero_object",
        task_id=8,
        n_episodes=1,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=0.01,
    )
    with pytest.raises(ValueError, match="explicit --n-action-steps=1"):
        libero.validate_config(config)


@pytest.mark.parametrize(
    ("override", "chunk_size", "temporal_ensemble_coeff", "expected"),
    [
        (None, 20, None, 5),
        (1, 20, None, 1),
        (2, 20, None, 2),
        (1, 20, 0.01, 1),
    ],
)
def test_resolve_n_action_steps(
    override: int | None,
    chunk_size: int,
    temporal_ensemble_coeff: float | None,
    expected: int,
) -> None:
    assert (
        libero.resolve_n_action_steps(
            checkpoint_n_action_steps=5,
            override=override,
            chunk_size=chunk_size,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("override", "chunk_size", "temporal_ensemble_coeff", "message"),
    [
        (21, 20, None, "cannot exceed"),
        (2, 20, 0.01, "must equal 1"),
    ],
)
def test_resolve_n_action_steps_rejects_incompatible_override(
    override: int,
    chunk_size: int,
    temporal_ensemble_coeff: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        libero.resolve_n_action_steps(
            checkpoint_n_action_steps=5,
            override=override,
            chunk_size=chunk_size,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
        )


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
        "observation_state",
        "policy_state_normalized_clean",
        "policy_state_normalized_input",
        "next_observation_state",
        "observation_joint_positions",
        "next_observation_joint_positions",
        "executed_action",
        "model_queried",
        "plan_step_index",
        "predicted_chunk",
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
    assert schema.field("plan_step_index").nullable
    assert schema.field("predicted_chunk").nullable
    assert not schema.field("observation_state").nullable
    assert not schema.field("next_observation_joint_positions").nullable
    assert not schema.field("policy_step_latency_ms").nullable
    assert not schema.field("policy_state_normalized_clean").nullable
    assert not schema.field("policy_state_normalized_input").nullable
    assert schema.metadata[b"schema_version"] == b"4"


def test_proprio_noise_is_deterministic_and_does_not_mutate_clean_input() -> None:
    state = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    image = torch.zeros(1, 3, 4, 5)
    processed = {"observation.state": state, "observation.images.image": image}

    first, clean, noisy = libero.prepare_policy_input(
        processed,
        perturbation="proprio_gaussian",
        proprio_noise_std=0.01,
        seed=1000,
        episode_index=2,
        step_index=3,
    )
    second, _, noisy_again = libero.prepare_policy_input(
        processed,
        perturbation="proprio_gaussian",
        proprio_noise_std=0.01,
        seed=1000,
        episode_index=2,
        step_index=3,
    )

    assert first is not processed
    assert first["observation.images.image"] is image
    assert first["observation.state"] is not state
    assert second["observation.state"] is not state
    np.testing.assert_array_equal(clean, np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(state.numpy()[0], clean)
    np.testing.assert_array_equal(noisy, noisy_again)
    assert not np.array_equal(clean, noisy)


def test_proprio_noise_changes_with_episode_or_step_and_none_is_identity() -> None:
    processed = {"observation.state": torch.zeros(1, 8)}

    unchanged, clean, policy_input = libero.prepare_policy_input(
        processed,
        perturbation="none",
        proprio_noise_std=0.01,
        seed=1000,
        episode_index=0,
        step_index=0,
    )
    _, _, next_step = libero.prepare_policy_input(
        processed,
        perturbation="proprio_gaussian",
        proprio_noise_std=0.01,
        seed=1000,
        episode_index=0,
        step_index=1,
    )
    _, _, next_episode = libero.prepare_policy_input(
        processed,
        perturbation="proprio_gaussian",
        proprio_noise_std=0.01,
        seed=1000,
        episode_index=1,
        step_index=0,
    )

    assert unchanged is processed
    np.testing.assert_array_equal(clean, policy_input)
    assert not np.array_equal(next_step, next_episode)


@pytest.mark.parametrize("value", [0.0, -0.01, float("nan"), float("inf")])
def test_libero_config_rejects_invalid_proprio_noise_std(value: float) -> None:
    config = libero.LiberoEvaluationConfig(
        policy_path=Path("checkpoint"),
        suite="libero_object",
        task_id=8,
        n_episodes=1,
        proprio_noise_std=value,
    )
    with pytest.raises(ValueError, match="finite and positive"):
        libero.validate_config(config)


def test_libero_extracts_policy_state_and_measured_raw_joints() -> None:
    policy_observation = {
        "observation.state": torch.arange(8, dtype=torch.float32).reshape(1, 8)
    }
    raw_observation = {
        "robot_state": {
            "joints": {
                "pos": np.arange(7, dtype=np.float64).reshape(1, 7),
            }
        }
    }

    np.testing.assert_array_equal(
        libero._extract_policy_state(policy_observation),
        np.arange(8, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        libero._extract_measured_joint_positions(raw_observation),
        np.arange(7, dtype=np.float32),
    )


def test_policy_chunk_postprocessing_matches_selected_action_processing() -> None:
    chunk = torch.arange(3 * 7, dtype=torch.float32).reshape(1, 3, 7)
    seen_shapes: list[tuple[int, ...]] = []

    def checkpoint_postprocessor(value: torch.Tensor) -> torch.Tensor:
        seen_shapes.append(tuple(value.shape))
        return value * 2 + 1

    def env_postprocessor(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"action": batch["action"] - 3}

    processed_chunk = libero._postprocess_policy_action_tensor(
        chunk,
        checkpoint_postprocessor=checkpoint_postprocessor,
        env_postprocessor=env_postprocessor,
    )
    processed_action = libero._postprocess_policy_action_tensor(
        chunk[:, 1],
        checkpoint_postprocessor=checkpoint_postprocessor,
        env_postprocessor=env_postprocessor,
    )

    assert seen_shapes == [(3, 7), (1, 7)]
    torch.testing.assert_close(processed_chunk[:, 1], processed_action)
    torch.testing.assert_close(processed_chunk, chunk * 2 - 2)


def test_controller_plan_partitions_validate_and_advance_standard_queue() -> None:
    policy_chunk = torch.arange(20 * 7, dtype=torch.float32).reshape(1, 20, 7)
    plan_step = libero.ActionPlanStep(
        model_queried=True,
        plan_step_index=0,
        queried_chunk_policy=policy_chunk,
        queued_actions_policy=policy_chunk[0, 1:5],
        discarded_actions_policy=policy_chunk[0, 5:],
        new_prediction_chunk_policy=None,
    )
    controller_chunk = policy_chunk[0].numpy()

    active, queued, discarded, temporal = libero._controller_plan_partitions(
        plan_step,
        queried_chunk_controller=controller_chunk,
        active_chunk_controller=None,
        executed_action_controller=controller_chunk[0],
        n_action_steps=5,
    )

    np.testing.assert_array_equal(active, controller_chunk)
    np.testing.assert_array_equal(queued, controller_chunk[1:5])
    np.testing.assert_array_equal(discarded, controller_chunk[5:])
    assert temporal is None


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
