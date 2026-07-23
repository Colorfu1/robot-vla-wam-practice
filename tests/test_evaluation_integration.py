from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.constants import ACTION

from robot_practice.evaluation.common import validate_complete_checkpoint
from robot_practice.evaluation.libero_act import (
    LiberoEvaluationConfig,
    _set_init_state_id,
    evaluate_libero,
    extract_terminal_success,
    make_libero_config,
)
from robot_practice.evaluation.libero_dataset_cache import COMBINED_DATASET_REPO_ID
from robot_practice.evaluation.libero_tasks import resolve_libero_task
from robot_practice.evaluation.offline_act import (
    OfflineEvaluationConfig,
    evaluate_offline,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ACT_EVAL_INTEGRATION") != "1",
    reason="set RUN_ACT_EVAL_INTEGRATION=1 for the bounded ACT/LIBERO smoke tests",
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PRACTICE_ROOT / "datasets" / "lerobot" / "lerobot" / "libero"


@pytest.fixture(scope="module")
def random_act_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    metadata = LeRobotDatasetMetadata(COMBINED_DATASET_REPO_ID, root=DATASET_ROOT)
    features = dataset_to_policy_features(metadata.features)
    config = ACTConfig(
        input_features={key: value for key, value in features.items() if key != ACTION},
        output_features={ACTION: features[ACTION]},
        device="cuda" if torch.cuda.is_available() else "cpu",
        chunk_size=20,
        n_action_steps=5,
        use_vae=False,
        pretrained_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    checkpoint = tmp_path_factory.mktemp("random-act") / "pretrained_model"
    policy = ACTPolicy(config)
    policy.save_pretrained(checkpoint)
    preprocessor, postprocessor = make_act_pre_post_processors(
        config,
        dataset_stats=metadata.stats,
    )
    preprocessor.save_pretrained(
        checkpoint,
        config_filename="policy_preprocessor.json",
    )
    postprocessor.save_pretrained(
        checkpoint,
        config_filename="policy_postprocessor.json",
    )
    validate_complete_checkpoint(checkpoint)
    return checkpoint


def test_one_frame_offline_prediction_writes_all_schemas(
    tmp_path: Path,
    random_act_checkpoint: Path,
) -> None:
    output = tmp_path / "offline"
    result = evaluate_offline(
        OfflineEvaluationConfig(
            policy_path=random_act_checkpoint,
            suite="libero_object",
            task_id=8,
            episodes=(823,),
            dataset_root=DATASET_ROOT,
            output_dir=output,
            max_frames=1,
            open_viewer=False,
            agentview_image_convention="lerobot_180",
        )
    )
    assert result.metrics["status"] == "smoke"
    assert result.metrics["aggregate"]["num_frames"] == 1
    assert (output / "prediction.rrd").stat().st_size > 0
    assert (output / "metrics.json").is_file()
    assert not (output / ".incomplete").exists()
    table = pq.read_table(output / "predictions.parquet")
    assert table.num_rows == 1
    assert table.schema.metadata[b"chunk_shape"] == b"[20,7]"


def test_offline_visualization_does_not_change_policy_predictions(
    tmp_path: Path,
    random_act_checkpoint: Path,
) -> None:
    outputs: list[tuple[dict, object]] = []
    for name, visualized in (("metrics_only", 0), ("visualized", 1)):
        output = tmp_path / name
        result = evaluate_offline(
            OfflineEvaluationConfig(
                policy_path=random_act_checkpoint,
                suite="libero_object",
                task_id=8,
                episodes=(823,),
                dataset_root=DATASET_ROOT,
                num_visualized_episodes=visualized,
                output_dir=output,
                max_frames=1,
                open_viewer=False,
                agentview_image_convention="lerobot_180",
            )
        )
        outputs.append((result.metrics, pq.read_table(output / "predictions.parquet")))

    metrics_only, visualized = outputs
    for column in ("target_action", "predicted_action", "target_chunk", "predicted_chunk"):
        assert metrics_only[1][column].to_pylist() == visualized[1][column].to_pylist()
    ignored = {"mean_model_query_latency_ms", "p95_model_query_latency_ms"}
    assert {
        key: value
        for key, value in metrics_only[0]["aggregate"].items()
        if key not in ignored
    } == {
        key: value
        for key, value in visualized[0]["aggregate"].items()
        if key not in ignored
    }


def test_one_step_real_libero_rollout_writes_video_and_null_formal_success(
    tmp_path: Path,
    random_act_checkpoint: Path,
) -> None:
    output = tmp_path / "libero"
    result = evaluate_libero(
        LiberoEvaluationConfig(
            policy_path=random_act_checkpoint,
            suite="libero_object",
            task_id=8,
            n_episodes=2,
            num_visualized_episodes=1,
            output_dir=output,
            max_steps=1,
            open_viewer=False,
        )
    )
    assert result.metrics["status"] == "smoke"
    assert result.metrics["aggregate"]["success_rate"] is None
    assert result.metrics["aggregate"]["num_policy_steps"] == 2
    assert result.metrics["aggregate"]["num_model_queries"] == 2
    assert (output / "rollout.rrd").stat().st_size > 0
    assert (output / "videos" / "episode_000.mp4").stat().st_size > 0
    assert not (output / ".incomplete").exists()
    table = pq.read_table(output / "rollouts.parquet")
    assert table.num_rows == 2
    assert table.column("init_state_id").to_pylist() == [0, 1]
    assert table.column("truncated").to_pylist() == [True, True]
    assert table.column("termination_reason").to_pylist() == ["time_limit", "time_limit"]
    assert table.column("model_queried").to_pylist() == [True, True]
    assert table.column("plan_step_index").to_pylist() == [0, 0]
    assert all(len(value) == 8 for value in table["observation_state"].to_pylist())
    assert all(
        len(value) == 7
        for value in table["next_observation_joint_positions"].to_pylist()
    )
    assert all(len(value) == 20 for value in table["predicted_chunk"].to_pylist())


def test_real_vector_final_info_and_controlled_init_state_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_config = make_libero_config(resolve_libero_task("libero_object", 8))
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs["libero_object"][8]
    try:
        base_env = env.envs[0]
        _set_init_state_id(env, 0)
        env.reset(seed=[1000])
        assert base_env.init_state_id == 1
        monkeypatch.setattr(base_env._env, "check_success", lambda: True)
        real_reset = base_env.reset

        def sentinel_reset(*args: object, **kwargs: object) -> tuple[dict, dict]:
            reset_observation, reset_info = real_reset(*args, **kwargs)
            reset_observation["robot_state"]["joints"]["pos"] = np.full(7, 123.0)
            return reset_observation, reset_info

        monkeypatch.setattr(base_env, "reset", sentinel_reset)
        terminal_observation, _, terminated, _, info = env.step(
            np.zeros((1, 7), dtype=np.float32)
        )
        assert bool(terminated[0])
        assert extract_terminal_success(info, terminal=True)
        assert base_env.init_state_id == 2
        assert not np.all(
            terminal_observation["robot_state"]["joints"]["pos"] == 123.0
        )

        _set_init_state_id(env, 1)
        env.reset(seed=[1001])
        assert base_env.init_state_id == 2
    finally:
        close_envs(envs)
