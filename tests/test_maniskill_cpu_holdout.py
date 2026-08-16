from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robot_practice.experiments.maniskill.eval_cpu_holdout import (
    _select_holdout_shard,
    load_cpu_holdout_split,
    merge_holdout_shards,
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


def test_cpu_holdout_shards_cover_the_fixed_seed_set_in_order() -> None:
    seeds = tuple(range(3000, 3400))
    shards = [
        _select_holdout_shard(seeds, shard_index=index, shard_count=8)[0] for index in range(8)
    ]

    assert [len(shard) for shard in shards] == [50] * 8
    assert tuple(seed for shard in shards for seed in shard) == seeds


def test_cpu_holdout_shard_merge_rebuilds_canonical_outputs(tmp_path: Path) -> None:
    shards_root = tmp_path / "shards"
    split_file = Path("configs/experiments/maniskill/split_cpu.json")
    for shard_index in range(8):
        shard_dir = shards_root / f"worker_{shard_index:02d}"
        shard_dir.mkdir(parents=True)
        start = shard_index * 50
        rows = [
            {
                "task": "PickCube-v1",
                "episode_index": start + episode,
                "seed": 3000 + start + episode,
                "success_once": episode % 2 == 0,
                "success_at_end": episode % 2 == 0,
                "success": episode % 2 == 0,
                "fail_once": False,
                "fail_at_end": False,
                "first_success_step": 10 if episode % 2 == 0 else None,
                "steps": 100,
                "termination_reason": "truncated",
                "is_grasped": episode % 2 == 0,
                "is_obj_placed": episode % 2 == 0,
                "is_robot_static": True,
                "translation_norm_mean": 0.1,
                "rotation_norm_mean": None,
                "gripper_abs_mean": 0.2,
                "action_jitter_mean": 0.3,
                "gripper_flip_count": 1,
            }
            for episode in range(50)
        ]
        pq.write_table(pa.Table.from_pylist(rows), shard_dir / "episodes.parquet")
        metrics = {
            "schema_version": 1,
            "label": "CPU fixed holdout",
            "system": "random",
            "task": "PickCube-v1",
            "control_mode": "pd_ee_delta_pos",
            "sim_backend": "physx_cpu",
            "num_episodes": 50,
            "latency": {
                "policy_forward": {"count": 0, "mean_ms": None},
                "environment_step": {"count": 5000, "mean_ms": 2.0},
            },
        }
        manifest = {"system": "random", "sim_backend": "physx_cpu"}
        (shard_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (shard_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    output_dir = tmp_path / "merged"
    metrics = merge_holdout_shards(
        shards_root=shards_root,
        split_file=split_file,
        output_dir=output_dir,
    )

    assert metrics["num_episodes"] == 400
    assert metrics["evaluation"]["success_once"]["num_success"] == 200
    assert metrics["seed_shard"] == {"index": None, "count": 8, "num_episodes": 400}
    assert metrics["latency"]["environment_step"]["mean_ms"] == 2.0
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "episodes.parquet").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    merged_rows = pq.read_table(output_dir / "episodes.parquet").to_pylist()
    assert [row["episode_index"] for row in merged_rows] == list(range(400))
