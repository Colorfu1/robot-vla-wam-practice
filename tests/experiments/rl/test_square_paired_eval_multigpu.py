import json
from types import SimpleNamespace

import pytest

from robot_practice.experiments.rl.square_paired_eval_multigpu import assignments, collect


def test_eight_gpu_plan_preserves_batches():
    plan = assignments([str(i) for i in range(8)], 80, 8)
    assert plan[0] == ("0", [0, 64])
    assert plan[1] == ("1", [8, 72])
    assert sorted(s for _, starts in plan for s in starts) == list(range(0, 80, 8))


@pytest.mark.parametrize("gpus", [[], ["0", "0"], ["-1"], ["foo"]])
def test_invalid_gpu_plan(gpus):
    with pytest.raises(ValueError):
        assignments(gpus, 80, 8)


def fixture_shards(tmp_path):
    args = SimpleNamespace(policy="dp", seed=10001, episodes=10, batch_size=8)
    plan = assignments(["2", "3"], args.episodes, args.batch_size)
    for gpu, starts in plan:
        directory = tmp_path / f"gpu_{gpu}"
        directory.mkdir()
        rows = [
            {"environment_seed": args.seed + i, "success_400": False, "success_600": True}
            for start in starts
            for i in range(start, min(start + 8, 10))
        ]
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "arguments": vars(args),
                    "checkpoint_sha256": "weights",
                    "config_sha256": "config",
                    "weight_variant": "model",
                    "status": "complete",
                    "batch_starts": starts,
                    "per_episode": rows,
                }
            )
        )
    return args, plan


def test_merge_exact_coverage_and_partial_batch(tmp_path):
    args, plan = fixture_shards(tmp_path)
    result = collect(tmp_path, plan, args, "weights", "config", final=True)
    assert result["evaluated"] == result["requested"] == 10
    assert result["success_rate_400"] == 0
    assert result["success_rate_600"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "checkpoint", "status", "seeds"])
def test_reject_bad_final_merge(tmp_path, mutation):
    args, plan = fixture_shards(tmp_path)
    path = tmp_path / "gpu_2/metrics.json"
    data = json.loads(path.read_text())
    if mutation == "duplicate":
        data["per_episode"].append(data["per_episode"][0])
    elif mutation == "missing":
        data["per_episode"].pop()
    elif mutation == "checkpoint":
        data["checkpoint_sha256"] = "wrong"
    elif mutation == "status":
        data["status"] = "failed"
    else:
        data["per_episode"][0]["environment_seed"] = 99999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        collect(tmp_path, plan, args, "weights", "config", final=True)


def test_partial_read_does_not_claim_completion(tmp_path):
    args, plan = fixture_shards(tmp_path)
    (tmp_path / "gpu_3/metrics.json").write_text('{"status":')
    result = collect(tmp_path, plan, args, "weights", "config")
    assert result["evaluated"] == 8
    with pytest.raises(json.JSONDecodeError):
        collect(tmp_path, plan, args, "weights", "config", final=True)
