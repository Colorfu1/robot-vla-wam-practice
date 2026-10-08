import numpy as np
import pytest

from robot_practice.experiments.rl.fastwam_dual_official_eval import (
    pack_dual_observation,
    summarize_shard,
    validate_official_shard,
)


def test_pack_channel_stacked_official_cameras_into_training_width_layout():
    rgb = np.zeros((2, 1, 6, 224, 224), dtype=np.float32)
    rgb[:, :, :3] = 11
    rgb[:, :, 3:] = 29
    state = np.zeros((2, 1, 9), dtype=np.float32)
    result = pack_dual_observation({"rgb": rgb, "state": state})
    assert result["rgb"].shape == (2, 1, 3, 224, 448)
    assert np.all(result["rgb"][..., :224] == 11)
    assert np.all(result["rgb"][..., 224:] == 29)
    assert result["state"] is state


@pytest.mark.parametrize("shape", [(1, 1, 3, 224, 448), (1, 6, 224, 224)])
def test_pack_rejects_wrong_layout(shape):
    with pytest.raises(ValueError):
        pack_dual_observation({"rgb": np.zeros(shape), "state": np.zeros((1, 1, 9))})


def test_official_shard_requires_seed_42_and_complete_environment_range():
    assert validate_official_shard(42, 18, 2, 400)
    assert not validate_official_shard(42, 0, 1, 100)
    for args in ((43, 0, 1, 400), (42, 19, 2, 400), (42, 0, 1, 200)):
        with pytest.raises(ValueError):
            validate_official_shard(*args)


def test_official_reward_split_requires_four_continuous_resets_per_env():
    rewards = np.zeros((400, 2))
    firsts = np.zeros((401, 2), dtype=bool)
    firsts[[0, 100, 200, 300, 400]] = True
    rewards[50, 0] = 4
    rewards[350, 1] = 4
    rows = summarize_shard(rewards, firsts, env_start=5, seed=42, policy_steps=400)
    assert len(rows) == 8
    assert [row["environment_seed"] for row in rows] == [47] * 4 + [48] * 4
    assert [row["episode_sequence"] for row in rows] == [0, 1, 2, 3] * 2
    assert [row["success"] for row in rows] == [True, False, False, False,
                                                   False, False, False, True]
    firsts[200, 1] = False
    with pytest.raises(ValueError, match="reset boundaries"):
        summarize_shard(rewards, firsts, env_start=5, seed=42, policy_steps=400)
