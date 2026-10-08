import json

import numpy as np
import pytest

from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL
from robot_practice.experiments.rl.fastwam_dual_train_fit_multigpu import merge


def test_merge_preserves_exclusions_and_weights_open_loop_by_windows(tmp_path):
    plan = [("0", [0]), ("1", [10])]
    for gpu, ids in plan:
        directory = tmp_path / f"gpu_{gpu}"
        directory.mkdir()
        scene = {"eligible": gpu == "0"}
        if gpu == "0":
            scene["policy"] = {"task_success_any_step": True,
                               "within_400_steps": {"task_success_any_step": False}}
        data = {"status": "complete" if gpu == "0" else "partial_excluded_scenes",
                "trajectory_ids": ids, "checkpoint_sha256": "hash", "max_steps": 600,
                "protocol": PROTOCOL, "scenes": {str(ids[0]): scene}}
        (directory / "metrics.json").write_text(json.dumps(data))
        count = 1 if gpu == "0" else 3
        np.savez(directory / "open_loop.npz", predicted=np.ones((count, 32, 7)) * int(gpu),
                 target=np.zeros((count, 32, 7)))
    result = merge(tmp_path, plan, "hash", 600)
    assert result["evaluated"] == 1
    assert result["requested"] == 2
    assert result["status"] == "partial_excluded_scenes"
    assert result["task_success_rate"] == 1
    assert result["success_rate_400"] == 0
    assert result["open_loop"]["executed_first4"]["mse_clipped"] == .75
    with pytest.raises(ValueError, match="mismatched"):
        merge(tmp_path, plan, "wrong-hash", 600)


def test_pair_merge_does_not_report_rollout_success(tmp_path):
    directory = tmp_path / "gpu_0"
    directory.mkdir()
    data = {"status": "complete", "trajectory_ids": [0], "checkpoint_sha256": "hash",
            "max_steps": 600, "protocol": PROTOCOL, "image_pair_only": True,
            "pairs": [{"trajectory": 0, "frame": 0}]}
    (directory / "metrics.json").write_text(json.dumps(data))
    arrays = np.zeros((1, 32, 7))
    np.savez(directory / "image_pairs.npz", saved=arrays, live=arrays + .1, target=arrays)
    result = merge(tmp_path, [("0", [0])], "hash", 600, image_pair_only=True)
    assert result["evaluated"] == 1
    assert "task_success_rate" not in result
    assert result["paired_summary"]["saved_rgb"]["full32"]["mse_raw"] == 0


def test_parity_merge_does_not_report_rollout_success(tmp_path):
    directory = tmp_path / "gpu_0"
    directory.mkdir()
    data = {"status": "complete", "trajectory_ids": [0], "checkpoint_sha256": "hash",
            "max_steps": 600, "protocol": PROTOCOL, "protocol_parity_only": True,
            "parity_cases": [{"trajectory": 0, "frame": 0}]}
    (directory / "metrics.json").write_text(json.dumps(data))
    result = merge(tmp_path, [("0", [0])], "hash", 600, protocol_parity_only=True)
    assert result["evaluated"] == 1
    assert "task_success_rate" not in result
    assert result["parity_cases"] == data["parity_cases"]


def test_autocast_merge_uses_arrays_not_average_of_shard_averages(tmp_path):
    plan = [("0", [0]), ("1", [10])]
    for gpu, ids in plan:
        directory = tmp_path / f"gpu_{gpu}"
        directory.mkdir()
        count = 1 if gpu == "0" else 3
        data = {"status": "complete", "trajectory_ids": ids, "checkpoint_sha256": "hash",
                "max_steps": 600, "protocol": PROTOCOL, "autocast_action_only": True,
                "inference_steps": 20,
                "autocast_cases": [{"trajectory": ids[0], "frame": i} for i in range(count)]}
        (directory / "metrics.json").write_text(json.dumps(data))
        arrays = np.zeros((count, 32, 7))
        np.savez(directory / "autocast_actions.npz", normal=arrays,
                 autocast=arrays + int(gpu), target=arrays)
    result = merge(tmp_path, plan, "hash", 600, autocast_action_only=True)
    assert result["evaluated"] == 2
    assert result["autocast_summary"]["windows"] == 4
    assert result["autocast_summary"]["bf16_autocast"]["executed_first4"]["mse_clipped"] == .75
    assert "task_success_rate" not in result
