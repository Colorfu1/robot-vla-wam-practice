from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robot_practice.experiments.maniskill.merge_closed_loop_workers import merge_workers


def _episode(index: int, success: bool) -> dict[str, object]:
    return {
        "task": "PickCube-v1",
        "episode_index": index,
        "seed": 2000 + index,
        "success_once": success,
        "success_at_end": success,
        "success": success,
        "first_success_step": 1 if success else None,
        "steps": 1,
        "termination_reason": "success" if success else "truncated",
        "is_grasped": success,
        "is_obj_placed": success,
        "is_robot_static": True,
        "translation_norm_mean": 0.1,
        "rotation_norm_mean": 0.1,
        "gripper_abs_mean": 0.2,
        "action_jitter_mean": 0.0,
        "gripper_flip_count": 0,
    }


def test_merge_workers_reconstructs_complete_episode_set(tmp_path: Path) -> None:
    workers_root = tmp_path / "workers"
    policy_path = "outputs/maniskill/m0_probe/train_main/checkpoints/010000/pretrained_model"
    for worker_index, rows in enumerate(
        ([_episode(0, True), _episode(2, False)], [_episode(1, False), _episode(3, True)])
    ):
        worker_dir = workers_root / f"gpu_{worker_index}"
        worker_dir.mkdir(parents=True)
        metrics = {
            "system": "visual",
            "task": "PickCube-v1",
            "control_mode": "pd_ee_delta_pose",
            "seed_set": "development",
            "reset_batches": [[2000, 2001, 2002, 2003]],
            "num_episodes": len(rows),
            "evaluation": {"num_episodes": len(rows)},
            "worker": {"index": worker_index, "count": 2},
            "latency": {
                "policy_forward": {"count": 2, "mean_ms": 10.0},
                "environment_step": {"count": 2, "mean_ms": 1.0},
            },
        }
        (worker_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (worker_dir / "run_manifest.json").write_text(
            json.dumps({"policy_path": policy_path, "sim_backend": "physx_cuda"}),
            encoding="utf-8",
        )
        pq.write_table(pa.Table.from_pylist(rows), worker_dir / "episodes.parquet")

    output_dir = tmp_path / "merged"
    result = merge_workers(
        workers_root=workers_root,
        output_dir=output_dir,
        expected_episodes=4,
        expected_workers=2,
    )

    assert result["num_episodes"] == 4
    assert result["evaluation"]["success_at_end"]["rate"] == 0.5
    assert result["execution"]["num_gpus"] == 2
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "episodes.parquet").is_file()
