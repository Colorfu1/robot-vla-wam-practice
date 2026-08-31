from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import robot_practice.experiments.rl.validation_runner as validation_runner
from robot_practice.experiments.rl.validation_runner import _parser
from robot_practice.experiments.rl.validation_runner import _s1_training_command
from robot_practice.experiments.rl.validation_runner import _s1_training_contract
from robot_practice.experiments.rl.validation_runner import _load_config
from robot_practice.experiments.rl.validation_runner import _summarize


def test_s1_train_accepts_guide_base_policy_checkpoint_alias() -> None:
    arguments = _parser().parse_args(
        [
            "--phase",
            "s1-train",
            "--output-root",
            "outputs/test-s1-train",
            "--base-policy-ckpt",
            "outputs/base/best.pt",
        ]
    )

    assert arguments.checkpoint == "outputs/base/best.pt"


def test_s1_training_command_forwards_frozen_recipe(tmp_path: Path) -> None:
    contract = _s1_training_contract(
        _load_config(Path("configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml"))
    )
    command = _s1_training_command(
        Namespace(seed=3),
        contract,
        checkpoint=tmp_path / "best.pt",
        output=tmp_path / "seed_3",
    )
    rendered = " ".join(command)

    assert "online/pi_dec_diffusion_maniskill2.py" in rendered
    assert "--total-timesteps 4000000" in rendered
    assert "--num-envs 16" in rendered
    assert "--training-freq 64" in rendered
    assert "--utd 0.25" in rendered
    assert "--prog-explore 30000" in rendered
    assert "--eval-freq 50000" in rendered
    assert "--save-freq 1000000" in rendered
    assert "--policy-frequency 1" in rendered
    assert "--target-network-frequency 1" in rendered
    assert "--bootstrap-at-done truncated" in rendered
    assert "--actor-input obs" in rendered
    assert "--critic-input sum" in rendered


def test_s1_formal_runner_records_final_checkpoint_without_launching_training(
    monkeypatch, tmp_path: Path
) -> None:
    reference = tmp_path / "reference_repo"
    (reference / "envs").mkdir(parents=True)
    (reference / "online").mkdir()
    (reference / "online/pi_dec_diffusion_maniskill2.py").write_text("# fake\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"base-checkpoint")
    output = tmp_path / "seed_1"
    formal_config = tmp_path / "formal-config.json"
    formal_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stages": {
                    "s1": {
                        "execution_mode": "formal",
                        "formal_residual_training": True,
                        "formal_env_id": "PegInsertionSide-v2",
                        "train": {
                            "formal_training": True,
                            "total_timesteps": 4_000_000,
                            "num_envs": 16,
                            "seeds": [1, 2, 3],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            output_dir = Path(command[command.index("--output-dir") + 1])
            run_dir = output_dir / "PegInsertionSide" / "PolicyDecorator-DiffusionPolicy" / "run"
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "checkpoints/4000000.pt").write_bytes(b"residual")
            (run_dir / "args.json").write_text("{}\n", encoding="utf-8")
            read_fd, write_fd = os.pipe()
            os.write(write_fd, b"global_step=4000000, fake child\n")
            os.close(write_fd)
            self.stdout = os.fdopen(read_fd, "r", encoding="utf-8")

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            return None

    monkeypatch.setattr(
        validation_runner.subprocess,
        "Popen",
        lambda command, **_kwargs: FakeProcess(command),
    )
    arguments = Namespace(
        # Formal S1 is disabled in the project config; exercise this optional
        # compatibility path only with an explicit opt-in fixture.
        config=str(formal_config),
        checkpoint=str(checkpoint),
        reference_root=str(reference),
        output_root=str(output),
        seed=1,
    )

    assert validation_runner._run_s1_formal_training(arguments) == 0
    artifact = json.loads(
        (output / "training_artifact.json").read_text(encoding="utf-8")
    )
    assert artifact["final_step"] == 4_000_000
    assert artifact["base_parameters_frozen"] is True
    assert (output / "run.log").read_text(encoding="utf-8").endswith(
        "DONE completed=4000000 total=4000000\n"
    )


def test_formal_paired_summary_is_marked_formal(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    row = {
        "episode_key": "a",
        "reset_state_id": "state-a",
        "reset_seed": 1001,
        "policy_seed": 1,
        "requested_env_id": "PegInsertionSide-v2",
        "runtime_env_id": "PegInsertionSide-v2",
        "action_dim": 7,
        "n_obs_steps": 2,
        "action_horizon": 4,
        "execution_action_steps": 4,
        "success": True,
    }
    base.write_text(json.dumps({**row, "success": False}) + "\n", encoding="utf-8")
    treatment.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "summary"
    arguments = Namespace(
        output_root=str(output),
        base=str(base),
        treatment=str(treatment),
        formal=True,
        dummy=False,
    )

    assert _summarize(arguments, phase="s1-summarize") == 0
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert payload["result_label"] == "Closed-loop"
    assert payload["formal_result"] is True
