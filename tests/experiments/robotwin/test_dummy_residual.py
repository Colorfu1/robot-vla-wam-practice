from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from robot_practice.experiments.rl.robotwin_residual import (
    run_dummy_rollout,
    run_dummy_training,
)


def _arguments(output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=str(output),
        episodes=2,
        replan_steps=4,
        seed=1,
        eval_seed=42,
        task="move_stapler_pad",
        task_config="demo_randomized",
        zero_residual=True,
        res_scale=0.1,
        base_checkpoint=None,
        base_run=None,
        max_steps=8,
        total_low_level_actions=32,
        learning_starts=4,
        num_envs=2,
        batch_size=8,
        actor_lr=1e-4,
        critic_lr=1e-4,
        prog_explore=30000,
    )


def test_dummy_rollout_writes_zero_residual_contract(tmp_path: Path) -> None:
    output = tmp_path / "rollout"
    metrics = run_dummy_rollout(_arguments(output))

    assert metrics["max_zero_residual_action_deviation"] == 0.0
    assert metrics["max_gripper_residual"] == 0.0
    assert metrics["base_hash_before"] == metrics["base_hash_after"]
    assert (output / "episode_manifest.jsonl").is_file()
    assert (output / "episodes.jsonl").is_file()


def test_dummy_training_counts_aggregate_actions_and_updates(tmp_path: Path) -> None:
    output = tmp_path / "train"
    metrics = run_dummy_training(_arguments(output))

    assert metrics["executed_low_level_actions"] == 32
    assert metrics["num_envs"] == 2
    assert metrics["sac_updates"] > 0
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["weights_written"] is False

