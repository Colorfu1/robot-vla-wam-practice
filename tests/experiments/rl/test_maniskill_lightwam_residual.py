from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_practice.experiments.rl.maniskill_lightwam_residual import (
    ChunkReplayBuffer,
    ResidualContract,
    build_parser,
    run_evaluation,
    run_training,
    _residual_enable_probability,
)


def test_residual_action_dim_matches_replan_window() -> None:
    contract = ResidualContract()
    assert contract.action_horizon == 16
    assert contract.replan_steps == 4
    assert contract.residual_dim == 4 * 7


def test_chunk_replay_buffer_overwrites_in_constant_time() -> None:
    replay = ChunkReplayBuffer(2)
    for value in range(3):
        vector = np.full(1, value, dtype=np.float32)
        replay.add((vector, vector, float(value), vector, False, 0.97))

    assert len(replay) == 2
    assert {int(item[0][0]) for item in replay.sample(2)} == {1, 2}


def test_progressive_residual_enable_probability_matches_official_schedule() -> None:
    assert _residual_enable_probability(0, 30_000) == 0.0
    assert _residual_enable_probability(15_000, 30_000) == 0.5
    assert _residual_enable_probability(30_000, 30_000) == 1.0
    assert _residual_enable_probability(1, 0) == 1.0


def test_dummy_peg_training_writes_real_sac_checkpoint_and_progress(tmp_path: Path) -> None:
    output = tmp_path / "train"
    arguments = build_parser().parse_args(
        [
            "--phase=train",
            "--dummy",
            "--device=cpu",
            "--total-low-level-actions=8",
            "--learning-starts=1",
            "--batch-size=2",
            "--utd=1",
            "--hidden-dim=32",
            "--log-every=4",
            "--save-every=8",
            f"--output-dir={output}",
        ]
    )
    metrics = run_training(arguments)

    assert metrics["executed_low_level_actions"] == 8
    assert metrics["sac_updates"] > 0
    assert metrics["base_parameters_frozen"] is True
    assert metrics["base_parameter_hash_before"] == metrics["base_parameter_hash_after"]
    assert (output / "run.log").is_file()
    assert (output / "progress.jsonl").is_file()
    checkpoint = output / "checkpoints" / "latest.pt"
    assert checkpoint.is_file()


def test_dummy_peg_base_evaluation_writes_native_success_rows(tmp_path: Path) -> None:
    output = tmp_path / "eval"
    arguments = build_parser().parse_args(
        [
            "--phase=evaluate",
            "--dummy",
            "--arm=base",
            "--device=cpu",
            "--eval-seeds=1,2",
            "--episodes-per-seed=2",
            "--max-steps=4",
            f"--output-dir={output}",
        ]
    )
    metrics = run_evaluation(arguments)

    assert metrics["episodes"] == 4
    assert metrics["result_label"] == "Smoke"
    assert metrics["formal_result"] is False
    assert metrics["success_rate"] == 1.0
    assert metrics["base_parameters_frozen"] is True
    rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(row["success"] for row in rows)
    assert all(row["requested_env_id"] == "PegInsertionSide-v2" for row in rows)
    assert len({row["reset_seed"] for row in rows}) == 4


def test_dummy_residual_evaluation_uses_checkpoint_hidden_dim(tmp_path: Path) -> None:
    train_output = tmp_path / "train"
    train_arguments = build_parser().parse_args(
        [
            "--phase=train",
            "--dummy",
            "--device=cpu",
            "--total-low-level-actions=8",
            "--learning-starts=1",
            "--batch-size=2",
            "--utd=1",
            "--hidden-dim=32",
            f"--output-dir={train_output}",
        ]
    )
    run_training(train_arguments)

    eval_output = tmp_path / "residual_eval"
    eval_arguments = build_parser().parse_args(
        [
            "--phase=evaluate",
            "--dummy",
            "--arm=residual",
            "--device=cpu",
            "--eval-seeds=1",
            "--episodes-per-seed=1",
            "--max-steps=4",
            f"--residual-checkpoint={train_output / 'checkpoints' / 'latest.pt'}",
            f"--output-dir={eval_output}",
        ]
    )
    metrics = run_evaluation(eval_arguments)

    assert metrics["episodes"] == 1
    assert (eval_output / "episodes.jsonl").is_file()
