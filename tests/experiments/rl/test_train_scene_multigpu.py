import copy

import pytest

from robot_practice.experiments.rl.train_scene_multigpu import merge_shards, split_scenes


def test_exact_once_partition():
    ids = [i for i in range(100) if i not in {0, 10, 20, 30, 40, 50, 60, 70}][:24]
    shards = split_scenes(ids, list(map(str, range(8))))
    assert all(len(part) == 3 for _, part in shards)
    assert sorted(i for _, part in shards for i in part) == ids
    with pytest.raises(ValueError):
        split_scenes([1, 1], ["0"])
    with pytest.raises(ValueError):
        split_scenes([1], ["0", "0"])


def shard(tid, success=True):
    return {"status": "complete", "replay": {str(tid): {"eligible": True}},
            "capture_verification": {"passed": 2, "total": 2},
            "models": {"step": {"checkpoint": "weights.pt", "weight_variant": "model",
                                "episodes": {str(tid): {"task_success_any_step": success,
                                                       "official_chunk_success": success,
                                                       "prefix_replay_verification": {
                                                           "passed": True}}}}}}


def test_merge_preserves_artifact_locations_and_failures():
    result = merge_shards([("shard_00", shard(1)), ("shard_01", shard(2, False))], [1, 2])
    assert result["models"]["step"]["task_success_rate"] == 0.5
    assert result["capture_verification"]["passed"] == 4
    assert result["failure_candidates"][0]["artifact_directory"] == "shard_01/step/trajectory_2"
    assert result["failure_candidates"][0]["failure_type"] == "unreviewed"


def test_merge_rejects_missing_duplicate_and_incomplete():
    with pytest.raises(ValueError):
        merge_shards([("a", shard(1))], [1, 2])
    with pytest.raises(ValueError):
        merge_shards([("a", shard(1)), ("b", shard(1))], [1])
    bad = copy.deepcopy(shard(1))
    bad["status"] = "running"
    with pytest.raises(ValueError):
        merge_shards([("a", bad)], [1])


def test_replay_mismatch_is_not_hidden():
    bad = shard(1)
    bad["status"] = "complete_with_prefix_replay_mismatch"
    bad["capture_verification"]["passed"] = 1
    result = merge_shards([("a", bad)], [1])
    assert result["status"] == "complete_with_prefix_replay_mismatch"
