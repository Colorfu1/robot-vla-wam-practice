from __future__ import annotations

import json
from pathlib import Path

from robot_practice.experiments.maniskill.eval_cpu_holdout import (
    load_cpu_holdout_split,
)


def test_cpu_holdout_expands_to_400_independent_seed_labels() -> None:
    split = load_cpu_holdout_split(Path("configs/experiments/maniskill/split_cpu.json"))
    assert split["seeds"][:3] == (3000, 3001, 3002)
    assert split["seeds"][-1] == 3399
    assert len(split["seeds"]) == 400
    assert split["tolerance"] == 1e-4


def test_cpu_holdout_manifest_has_the_five_required_protocol_fields(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    payload = {
        "task": "PickCube-v1",
        "sim_backend": "physx_cpu",
        "num_envs": 1,
        "max_episode_steps": 100,
        "ignore_terminations": True,
        "state_match_tolerance": 1e-4,
        "sets": {"holdout": {"start": 3000, "stop": 3400, "step": 1, "count": 400}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    split = load_cpu_holdout_split(path)
    assert split["payload"]["sim_backend"] == "physx_cpu"
    assert split["payload"]["num_envs"] == 1
    assert split["payload"]["max_episode_steps"] == 100
    assert split["payload"]["ignore_terminations"] is True
    assert split["payload"]["sets"]["holdout"]["count"] == 400
