import json
from types import SimpleNamespace

import pytest

from robot_practice.experiments.rl.dppo_square_official import REVISION
from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL
from robot_practice.experiments.rl.fastwam_dual_official_multigpu import (
    collect,
    official_assignments,
    shard_command,
)


def test_official_plan_uses_20_contiguous_environment_indices():
    plan = official_assignments([str(index) for index in range(8)])
    assert plan == [("0", 0, 3), ("1", 3, 3), ("2", 6, 3), ("3", 9, 3),
                    ("4", 12, 2), ("5", 14, 2), ("6", 16, 2), ("7", 18, 2)]


def test_official_smoke_only_uses_one_gpu():
    assert official_assignments(["0", "1", "2"], 1) == [("0", 0, 1)]


def test_official_command_has_seed_42_and_400_policy_calls(tmp_path):
    args = SimpleNamespace(checkpoint="state.pt", dppo_root="third_party/dppo_official",
                           hdf5="demo.hdf5",
                           normalization="norm.npz")
    command = shard_command(args, tmp_path, "1", 3, 3, 7, 400)
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--policy-steps") + 1] == "400"
    assert command[command.index("--env-start") + 1] == "3"
    assert command[command.index("--env-count") + 1] == "3"
    assert command[command.index("--dppo-root") + 1] == "third_party/dppo_official"


def test_collect_requires_80_distinct_official_episodes(tmp_path):
    plan = official_assignments(["0", "1"])
    mapping = {"0": {"egl_id": 6}, "1": {"egl_id": 7}}
    for gpu, start, count in plan:
        directory = tmp_path / f"gpu_{gpu}"
        directory.mkdir()
        rows = [{"env_index": env, "environment_seed": 42 + env,
                 "episode_sequence": sequence, "start_policy_step": sequence * 100,
                 "policy_steps": 100, "success": env == 0}
                for env in range(start, start + count) for sequence in range(4)]
        (directory / "metrics.json").write_text(json.dumps({
            "status": "complete", "checkpoint_sha256": "weights",
            "dppo_revision": REVISION, "protocol": PROTOCOL,
            "evaluation_mode": "official_dppo", "policy_steps_completed": 400,
            "arguments": {"env_start": start, "env_count": count, "seed": 42,
                          "policy_steps": 400, "egl_device_id": mapping[gpu]["egl_id"]},
            "requested": len(rows), "evaluated": len(rows), "episodes": rows,
        }))
    result = collect(tmp_path, plan, mapping, "weights", policy_steps=400, final=True)
    assert result["requested"] == result["evaluated"] == 80
    assert result["success_rate"] == 4 / 80
    path = tmp_path / "gpu_0/metrics.json"
    data = json.loads(path.read_text())
    data["episodes"][0]["environment_seed"] = 10001
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        collect(tmp_path, plan, mapping, "weights", policy_steps=400, final=True)
