from __future__ import annotations

import json
import subprocess
from pathlib import Path

from robot_practice.experiments.maniskill import select_m2_diffusion_checkpoint as selection


def _write_candidate(
    tmp_path: Path,
    *,
    step: int,
    success_at_end: float,
    success_once: float,
    on_cube: float = 0.5,
    static: float = 0.2,
    mean_latency_ms: float = 10.0,
    episodes: int = 200,
    policy_seed: int = 1000,
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    policy_path = run_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    policy_path.mkdir(parents=True, exist_ok=True)
    (policy_path / "config.json").write_text("{}", encoding="utf-8")
    (policy_path / "model.safetensors").write_bytes(b"weights")

    selection_dir = tmp_path / "selection"
    metrics_dir = selection.evaluation_dir(selection_dir, step)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": 1,
        "system": "visual",
        "task": "StackCube-v1",
        "control_mode": "pd_ee_delta_pose",
        "seed_set": "development",
        "policy_seed": policy_seed,
        "num_episodes": episodes,
        "max_episode_steps": 500,
        "evaluation": {
            "num_episodes": episodes,
            "success_at_end": {"rate": success_at_end},
            "success_once": {"rate": success_once},
            "stage_signal_rates": {
                "is_cubeA_grasped": 0.8,
                "is_cubeA_on_cubeB": on_cube,
                "is_cubeA_static": static,
            },
            "first_success_step": {"count": 1, "mean": 20.0},
            "termination_reasons": {"success": int(success_at_end * episodes)},
        },
        "latency": {"policy_forward": {"count": 10, "mean_ms": mean_latency_ms}},
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    manifest = {
        "task": "StackCube-v1",
        "control_mode": "pd_ee_delta_pose",
        "sim_backend": "physx_cuda",
        "policy_path": str(policy_path.resolve()),
    }
    (metrics_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, selection_dir


def test_discover_checkpoint_steps_and_build_command_are_m2_specific(tmp_path: Path) -> None:
    run_dir, selection_dir = _write_candidate(
        tmp_path,
        step=20000,
        success_at_end=0.4,
        success_once=0.5,
    )
    _write_candidate(tmp_path, step=10000, success_at_end=0.3, success_once=0.4)

    assert selection.discover_checkpoint_steps(run_dir) == (10000, 20000)
    command = selection.build_evaluation_command(
        policy_path=selection.checkpoint_policy_path(run_dir, 10000),
        dataset_repo_id="local/test_video",
        dataset_root=tmp_path / "dataset",
        state_bank_root=tmp_path / "states",
        output_dir=selection.evaluation_dir(selection_dir, 10000),
        expected_episodes=4,
        max_steps=50,
        n_action_steps=8,
        num_envs=4,
        num_gpus=1,
        policy_seed=1000,
    )
    assert "--seed-set" in command
    assert command[command.index("--seed-set") + 1] == "development"
    assert "test" not in command
    assert command[command.index("--policy-seed") + 1] == "1000"


def test_rank_candidates_uses_closed_loop_success_then_stage_and_latency() -> None:
    base = {
        "success_at_end_rate": 0.5,
        "success_once_rate": 0.6,
        "is_cubeA_on_cubeB_rate": 0.7,
        "is_cubeA_static_rate": 0.4,
        "policy_forward_mean_ms": 10.0,
    }
    candidates = [
        dict(base, checkpoint_step=20000, is_cubeA_on_cubeB_rate=0.8),
        dict(base, checkpoint_step=10000, is_cubeA_on_cubeB_rate=0.8),
        dict(base, checkpoint_step=5000, is_cubeA_on_cubeB_rate=0.7),
    ]

    ranked = selection.rank_candidates(candidates)

    assert [row["checkpoint_step"] for row in ranked] == [10000, 20000, 5000]
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_select_completed_publishes_only_inference_policy_and_provenance(tmp_path: Path) -> None:
    run_dir, selection_dir = _write_candidate(
        tmp_path,
        step=10000,
        success_at_end=0.4,
        success_once=0.5,
    )
    _write_candidate(
        tmp_path,
        step=20000,
        success_at_end=0.6,
        success_once=0.7,
    )

    result = selection.select_completed_checkpoints(
        run_dir=run_dir,
        selection_dir=selection_dir,
        checkpoint_steps=(10000, 20000),
        dataset_repo_id="local/test_video",
        dataset_root=tmp_path / "dataset",
        state_bank_root=tmp_path / "states",
        expected_episodes=200,
        policy_seed=1000,
    )

    assert result["selected_checkpoint_step"] == 20000
    assert result["dataset"]["repo_id"] == "local/test_video"
    selected = selection_dir / "selected"
    assert (selected / "pretrained_model" / "model.safetensors").read_bytes() == b"weights"
    assert not (selected / "optimizer.pt").exists()
    saved = json.loads((selection_dir / "selected_checkpoint.json").read_text(encoding="utf-8"))
    assert saved["selected_checkpoint_step"] == 20000
    assert len(saved["ranking"]) == 2


def test_candidate_runner_leaves_output_creation_to_gpu_evaluator(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "evaluation"
    log_path = tmp_path / "logs" / "evaluation.log"
    output_dir.mkdir()

    def fake_run(command, *, cwd, env, stdout, stderr, text, check):
        assert not output_dir.exists()
        output_dir.mkdir(parents=True)
        (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(selection.subprocess, "run", fake_run)

    assert selection._run_candidate_evaluation(
        command=["fake-evaluator"],
        output_dir=output_dir,
        log_path=log_path,
        reuse_completed=False,
    ) is True
