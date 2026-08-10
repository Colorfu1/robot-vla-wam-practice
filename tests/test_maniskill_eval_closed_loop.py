from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import rerun.dataframe as rdf
import torch

from robot_practice.experiments.maniskill import eval_closed_loop as module
from robot_practice.experiments.maniskill.closed_loop_visualization import (
    ClosedLoopVisualizationRecorder,
)
from robot_practice.rerun_launcher import LIBERO_APPLICATION_ID, RecordingKind, detect_rrd_kind


def test_multi_gpu_worker_command_keeps_selected_policy_and_shard_arguments() -> None:
    arguments = module.build_parser().parse_args(
        [
            "--system=visual",
            "--policy-path=outputs/selected/pretrained_model",
            "--control-mode=pd_ee_delta_pose",
            "--num-envs=200",
            "--max-episodes=200",
            "--num-gpus=8",
            "--output-dir=outputs/evaluation",
        ]
    )

    command = module._multi_gpu_worker_argv(
        arguments,
        worker_index=3,
        output_dir=Path("outputs/evaluation/workers/gpu_3"),
    )

    assert "--policy-path=outputs/selected/pretrained_model" in command
    assert "--worker-index=3" in command
    assert "--worker-count=8" in command
    assert "--max-episodes=200" in command


def _rgb_observation(batch_size: int) -> dict[str, object]:
    return {
        "sensor_data": {
            "base_camera": {"rgb": np.zeros((batch_size, 128, 128, 3), dtype=np.uint8)}
        },
        "agent": {
            "qpos": np.zeros((batch_size, 9), dtype=np.float32),
            "qvel": np.zeros((batch_size, 9), dtype=np.float32),
        },
        "extra": {"goal_pos": np.zeros((batch_size, 3), dtype=np.float32)},
    }


def _signals(batch_size: int) -> dict[str, np.ndarray]:
    return {
        "success": np.zeros(batch_size, dtype=bool),
        "is_grasped": np.zeros(batch_size, dtype=bool),
        "is_obj_placed": np.zeros(batch_size, dtype=bool),
        "is_robot_static": np.ones(batch_size, dtype=bool),
        "fail": np.zeros(batch_size, dtype=bool),
    }


def test_build_policy_observation_uses_non_privileged_21d_state() -> None:
    observation = module.build_policy_observation(_rgb_observation(2))
    assert tuple(observation[module.IMAGE_KEY].shape) == (2, 3, 128, 128)
    assert tuple(observation[module.STATE_KEY].shape) == (2, 21)


def test_select_batch_preserves_nested_observation_batching() -> None:
    observation = {
        "agent": {"qpos": torch.arange(20).reshape(4, 5)},
        "extra": {"goal_pos": np.arange(12).reshape(4, 3)},
        "constant": torch.tensor(7),
    }

    selected = module._select_batch(observation, np.asarray([1, 3]), batch_size=4)

    assert selected["agent"]["qpos"].tolist() == [[5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]
    assert selected["extra"]["goal_pos"].tolist() == [[3, 4, 5], [9, 10, 11]]
    assert selected["constant"].item() == 7


def test_run_batch_policy_shard_keeps_full_reset_batch_and_returns_owned_rows() -> None:
    class FakeEnv:
        def reset(self, *, seed: list[int]):
            assert seed == [2000, 2001, 2002, 2003]
            return _rgb_observation(4), _signals(4)

        def step(self, action: np.ndarray):
            assert action.shape == (4, 4)
            assert np.all(action[[0, 2]] == 0.0)
            info = _signals(4)
            info["success"] = np.asarray([False, True, False, True])
            info["is_grasped"] = np.asarray([False, True, False, True])
            info["is_obj_placed"] = np.asarray([False, True, False, True])
            return (
                _rgb_observation(4),
                np.zeros(4),
                np.ones(4, dtype=bool),
                np.zeros(4, dtype=bool),
                info,
            )

    class FakeDriver:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def reset(self) -> None:
            pass

        def prepare_observation(self, observation: object) -> object:
            return observation

        def select_action(self, observation: dict[str, object]) -> np.ndarray:
            batch_size = observation["agent"]["qpos"].shape[0]
            self.batch_sizes.append(batch_size)
            return np.zeros((batch_size, 4), dtype=np.float32)

    driver = FakeDriver()
    rows = module.run_batch(
        env=FakeEnv(),
        seeds=[2000, 2001, 2002, 2003],
        policy_driver=driver,
        max_steps=1,
        action_dim=4,
        episode_index_offset=0,
        owned_indices=[1, 3],
    )

    assert driver.batch_sizes == [2]
    assert [row["episode_index"] for row in rows] == [1, 3]
    assert [row["success_at_end"] for row in rows] == [True, True]


def test_diffusion_driver_stacks_camera_features_for_policy_queue() -> None:
    class FakeConfig:
        device = "cpu"
        image_features = (module.IMAGE_KEY,)

    class FakePolicy:
        def __init__(self) -> None:
            self._queues: dict[str, deque] = {}
            self.seen_batch: dict[str, torch.Tensor] | None = None

        def reset(self) -> None:
            self._queues = {
                module.STATE_KEY: deque(maxlen=2),
                "action": deque(maxlen=1),
                "observation.images": deque(maxlen=2),
            }

        def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            self.seen_batch = {
                key: torch.stack(list(self._queues[key]), dim=1)
                for key in batch
                if key in self._queues
            }
            batch_size = self.seen_batch[module.STATE_KEY].shape[0]
            return torch.zeros((batch_size, 8, 7), dtype=torch.float32)

    policy = FakePolicy()
    driver = module.DiffusionPolicyDriver(
        {
            "policy": policy,
            "config": FakeConfig(),
            "preprocessor": lambda observation: observation,
            "postprocessor": lambda action: action,
        }
    )
    image = torch.zeros((2, 3, 128, 128), dtype=torch.float32)
    state = torch.zeros((2, 21), dtype=torch.float32)

    action = driver.select_action({module.IMAGE_KEY: image, module.STATE_KEY: state})

    assert action.shape == (2, 7)
    assert policy.seen_batch is not None
    assert tuple(policy.seen_batch["observation.images"].shape) == (2, 2, 1, 3, 128, 128)


def test_build_privileged_observation_matches_maniskill_state_flatten_order() -> None:
    assert module.PRIVILEGED_STATE_FIELDS == (
        "agent.qpos[0:9]",
        "agent.qvel[0:9]",
        "extra.is_grasped[0:1]",
        "extra.tcp_pose[0:7]",
        "extra.goal_pos[0:3]",
        "extra.obj_pose[0:7]",
        "extra.tcp_to_obj_pos[0:3]",
        "extra.obj_to_goal_pos[0:3]",
    )
    observation = {
        "agent": {
            "qpos": np.full((2, 9), 1, dtype=np.float32),
            "qvel": np.full((2, 9), 2, dtype=np.float32),
        },
        "extra": {
            "is_grasped": np.full((2, 1), 3, dtype=np.float32),
            "tcp_pose": np.full((2, 7), 4, dtype=np.float32),
            "goal_pos": np.full((2, 3), 5, dtype=np.float32),
            "obj_pose": np.full((2, 7), 6, dtype=np.float32),
            "tcp_to_obj_pos": np.full((2, 3), 7, dtype=np.float32),
            "obj_to_goal_pos": np.full((2, 3), 8, dtype=np.float32),
        },
    }

    state = module.build_privileged_observation(observation)

    assert tuple(state.shape) == (2, 42)
    expected = torch.cat(
        (
            torch.full((2, 9), 1, dtype=torch.float32),
            torch.full((2, 9), 2, dtype=torch.float32),
            torch.full((2, 1), 3, dtype=torch.float32),
            torch.full((2, 7), 4, dtype=torch.float32),
            torch.full((2, 3), 5, dtype=torch.float32),
            torch.full((2, 7), 6, dtype=torch.float32),
            torch.full((2, 3), 7, dtype=torch.float32),
            torch.full((2, 3), 8, dtype=torch.float32),
        ),
        dim=-1,
    )
    torch.testing.assert_close(state, expected)


def test_build_privileged_observation_handles_batched_vector_and_singleton_fields() -> None:
    observation = {
        "agent": {
            "qpos": np.zeros((2, 9), dtype=np.float32),
            "qvel": np.ones((2, 9), dtype=np.float32),
        },
        "extra": {
            "is_grasped": np.asarray([False, True]),
            "tcp_pose": np.zeros((2, 7), dtype=np.float32),
            "goal_pos": np.zeros((2, 3), dtype=np.float32),
            "obj_pose": np.zeros((2, 7), dtype=np.float32),
            "tcp_to_obj_pos": np.zeros((2, 3), dtype=np.float32),
            "obj_to_goal_pos": np.zeros((2, 3), dtype=np.float32),
        },
    }

    state = module.build_privileged_observation(observation)

    assert tuple(state.shape) == (2, 42)
    assert state[:, 18].tolist() == [0.0, 1.0]


def test_native_signals_reads_success_and_stage_signals_without_custom_predicate() -> None:
    info = {
        "success": np.asarray([True, False]),
        "is_grasped": np.asarray([True, True]),
        "is_obj_placed": np.asarray([True, False]),
        "is_robot_static": np.asarray([True, True]),
    }
    signals = module.native_signals(info, batch_size=2)
    assert signals["success"].tolist() == [True, False]
    assert signals["is_obj_placed"].tolist() == [True, False]


def test_summarize_rows_keeps_success_once_and_success_at_end_separate() -> None:
    rows = [
        {
            "success_once": True,
            "success_at_end": False,
            "first_success_step": 3,
            "steps": 5,
            "termination_reason": "truncated",
            "is_grasped": True,
            "is_obj_placed": True,
            "is_robot_static": True,
            "translation_norm_mean": 0.2,
            "rotation_norm_mean": 0.1,
            "gripper_abs_mean": 0.3,
            "action_jitter_mean": 0.01,
            "gripper_flip_count": 1,
        },
        {
            "success_once": False,
            "success_at_end": False,
            "first_success_step": None,
            "steps": 5,
            "termination_reason": "episode_limit",
            "is_grasped": False,
            "is_obj_placed": False,
            "is_robot_static": True,
            "translation_norm_mean": 0.4,
            "rotation_norm_mean": 0.2,
            "gripper_abs_mean": 0.5,
            "action_jitter_mean": 0.02,
            "gripper_flip_count": 2,
        },
    ]
    metrics = module.summarize_rows(rows)
    assert metrics["success_once"]["rate"] == 0.5
    assert metrics["success_at_end"]["rate"] == 0.0
    assert metrics["first_success_step"]["median"] == 3.0


def test_compare_metrics_files_is_offline_and_preserves_role_note(tmp_path: Path) -> None:
    metrics_paths = []
    for system, once, end in (("random", 0.0, 0.0), ("visual", 0.5, 0.4)):
        path = tmp_path / system / "metrics.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "system": system,
                    "evaluation": {
                        "success_once": {"rate": once},
                        "success_at_end": {"rate": end},
                        "first_success_step": {"median": None},
                        "stage_signal_rates": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        metrics_paths.append(path)
    output = tmp_path / "comparison.json"
    result = module.compare_metrics_files(metrics_paths, output)
    assert result["systems"]["visual"]["success_once_rate"] == 0.5
    assert "privileged expert" in result["role_note"]
    assert output.is_file()


def test_closed_loop_visualization_recorder_writes_local_viewer_artifacts(
    tmp_path: Path,
) -> None:
    observation = _rgb_observation(1)
    observation["extra"]["obj_pose"] = np.zeros((1, 7), dtype=np.float32)
    observation["extra"]["tcp_pose"] = np.zeros((1, 7), dtype=np.float32)
    signals = {
        "success": np.asarray([True]),
        "is_grasped": np.asarray([True]),
        "is_obj_placed": np.asarray([True]),
        "is_robot_static": np.asarray([True]),
    }
    recorder = ClosedLoopVisualizationRecorder(max_episodes=1)
    recorder.start_batch(
        observation,
        seeds=[2021],
        episode_index_offset=0,
        signals=signals,
    )
    recorder.record_step(
        observation,
        action=np.zeros((1, 7), dtype=np.float32),
        signals=signals,
        terminated=np.asarray([True]),
        truncated=np.asarray([False]),
    )
    recorder.end_batch()

    manifest = recorder.finalize(tmp_path / "visualizations", control_mode="pd_ee_delta_pose")
    recording = tmp_path / "visualizations" / "episode_0000_success.rrd"
    video = tmp_path / "visualizations" / "episode_0000_success.mp4"
    assert recording.is_file()
    assert video.is_file()
    assert manifest["episodes"][0]["result"] == "success"
    assert manifest["viewer"]["application_id"] == LIBERO_APPLICATION_ID
    assert detect_rrd_kind(recording) is RecordingKind.LIBERO
    assert {
        str(item.application_id()) for item in rdf.load_archive(recording).all_recordings()
    } == {LIBERO_APPLICATION_ID}
