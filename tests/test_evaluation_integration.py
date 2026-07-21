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
from robot_practice.evaluation.offline_act import (
    DATASET_REPO_ID,
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
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID, root=DATASET_ROOT)
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
            episodes=(823,),
            dataset_root=DATASET_ROOT,
            output_dir=output,
            max_frames=1,
            open_viewer=False,
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


def test_one_step_real_libero_rollout_writes_video_and_null_formal_success(
    tmp_path: Path,
    random_act_checkpoint: Path,
) -> None:
    output = tmp_path / "libero"
    result = evaluate_libero(
        LiberoEvaluationConfig(
            policy_path=random_act_checkpoint,
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


def test_real_vector_final_info_and_controlled_init_state_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_config = make_libero_config()
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    env = envs["libero_object"][8]
    try:
        base_env = env.envs[0]
        _set_init_state_id(env, 0)
        env.reset(seed=[1000])
        assert base_env.init_state_id == 1
        monkeypatch.setattr(base_env._env, "check_success", lambda: True)
        _, _, terminated, _, info = env.step(np.zeros((1, 7), dtype=np.float32))
        assert bool(terminated[0])
        assert extract_terminal_success(info, terminal=True)
        assert base_env.init_state_id == 2

        _set_init_state_id(env, 1)
        env.reset(seed=[1001])
        assert base_env.init_state_id == 2
    finally:
        close_envs(envs)
