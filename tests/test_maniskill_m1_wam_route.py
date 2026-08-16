from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_practice.experiments.maniskill import m1_protocol_audit as audit

WAM_CONFIG = Path("configs/experiments/maniskill/m1_wam_protocol_audit.json")


def test_wam_config_is_stack_only_and_keeps_controller_contract() -> None:
    config = audit.load_config(WAM_CONFIG)

    assert audit.validate_config(config)["tasks"] == ["StackCube-v1"]
    assert audit.configured_tasks(config) == ("StackCube-v1",)
    assert "PegInsertionSide-v1" not in json.dumps(config, ensure_ascii=False)
    assert "PlugCharger-v1" not in json.dumps(config, ensure_ascii=False)
    assert config["tasks"]["StackCube-v1"]["formal_horizon"] == 500
    assert config["tasks"]["StackCube-v1"]["native_horizon"] == 50
    assert config["tasks"]["StackCube-v1"]["target_control_mode"] == "pd_ee_delta_pose"
    assert config["tasks"]["StackCube-v1"]["action_dim"] == 7
    assert config["protocol"]["replay_survival_threshold"] == 1.0
    assert config["protocol"]["backend"] == "physx_cuda"
    assert config["protocol"]["replay_backend"] == "physx_cuda"
    assert config["protocol"]["data_backend"] == "physx_cuda"
    assert config["protocol"]["rollout_backend"] == "physx_cuda"
    assert config["protocol"]["data_route"] == "frozen_gpu_carrier_live_rgb_replay_7d"
    assert config["protocol"]["native_source_backend"] == "physx_cuda"
    assert config["protocol"]["native_source_control_mode"] == "pd_ee_delta_pose"
    assert config["protocol"]["baseline_replay_h5"].endswith(
        "03_gpu_replay_stable_none/replay.h5"
    )
    assert config["protocol"]["repair_replay_h5"].endswith(
        "04_gpu_failed_ee_repair/repair.h5"
    )
    assert "native_source_input_h5" not in config["protocol"]
    assert config["tasks"]["StackCube-v1"]["native_source_dir"] == "gpu_native_source"


def test_wam_task_selection_rejects_peg() -> None:
    config = audit.load_config(WAM_CONFIG)

    assert audit.selected_tasks(config, "all") == ("StackCube-v1",)
    with pytest.raises(ValueError, match="not enabled"):
        audit.selected_tasks(config, "PegInsertionSide-v1")
    with pytest.raises(ValueError, match="not enabled"):
        audit.selected_tasks(config, "PlugCharger-v1")


def test_wam_finalize_and_frozen_validation_are_stackcube_only(tmp_path: Path) -> None:
    config = audit.load_config(WAM_CONFIG)
    for task in audit.configured_tasks(config):
        task_root = tmp_path / task
        (task_root / "state_banks").mkdir(parents=True)
        (task_root / "source_audit.json").write_text(
            json.dumps({"gate_pass": True, "summary": {"num_episodes": 1000}}),
            encoding="utf-8",
        )
        (task_root / "environment_contract.json").write_text(
            json.dumps({"gate_pass": True}), encoding="utf-8"
        )
        (task_root / "trajectory_audit.json").write_text(
            json.dumps({"gate_pass": True, "gates": {"replay_survival": {"passed": True}}}),
            encoding="utf-8",
        )
        (task_root / "d_core_manifest.json").write_text(
            json.dumps({"gate_pass": True, "episode_count": 900}), encoding="utf-8"
        )
        (task_root / "state_banks" / "state_bank_manifest.json").write_text(
            json.dumps({"gate_pass": True}), encoding="utf-8"
        )
        (task_root / "final_live_rgb_full").mkdir(parents=True)
        (task_root / "final_live_rgb_full" / "replay_manifest.json").write_text(
            json.dumps(
                {
                    "gate_pass": True,
                    "episode_count": 1000,
                    "route": "gpu_live_rgb_replay",
                }
            ),
            encoding="utf-8",
        )

    result = audit.finalize(config, override_output=tmp_path)
    frozen = result["frozen_protocol"]
    assert list(frozen["frozen_protocol"]["tasks"]) == ["StackCube-v1"]
    assert "PegInsertionSide-v1" not in json.dumps(frozen, ensure_ascii=False)
    assert "PlugCharger-v1" not in json.dumps(frozen, ensure_ascii=False)

    validation = audit.validate_frozen(tmp_path / "frozen_protocol.json", config)
    assert validation["tasks"] == ["StackCube-v1"]


def test_wam_parser_accepts_config_driven_task_selector() -> None:
    parsed = audit.build_parser().parse_args(
        [
            "source-audit",
            "--config",
            str(WAM_CONFIG),
            "--task",
            "StackCube-v1",
        ]
    )
    assert parsed.task == "StackCube-v1"


def test_wam_parser_accepts_gpu_data_phases() -> None:
    for command in (
        "gpu-source-smoke",
        "gpu-source-full",
        "gpu-baseline-smoke",
        "gpu-baseline-full",
        "gpu-repair-smoke",
        "gpu-repair-full",
        "gpu-merge-smoke",
        "gpu-merge-full",
        "gpu-final-smoke",
        "gpu-final-full",
    ):
        parsed = audit.build_parser().parse_args(
            [command, "--config", str(WAM_CONFIG), "--task", "StackCube-v1"]
        )
        assert parsed.task == "StackCube-v1"
