import json
from pathlib import Path

import h5py
import numpy as np

from robot_practice.experiments.maniskill.native_4d_data_pipeline import (
    ACTION_DIM,
    CONTROL_MODE,
    _paths,
    load_config,
)
from robot_practice.experiments.maniskill.state_identity import start_state_fingerprint
from robot_practice.experiments.maniskill.validate_gpu_ee_candidates import (
    build_carrier_batches,
)


def test_native_4d_config_is_not_a_projection_route() -> None:
    config = load_config("configs/experiments/maniskill/m1_wam_native_4d.json")

    assert config["task"] == "StackCube-v1"
    assert config["control_mode"] == CONTROL_MODE == "pd_ee_delta_pos"
    assert config["action_dim"] == ACTION_DIM == 4
    assert config["protocol"]["source_control_mode"] == "pd_joint_pos"
    assert "m1_wam_protocol_audit" not in config["output_root"]


def test_native_4d_phase_outputs_are_separate() -> None:
    config = {
        "output_root": "outputs/maniskill/example_native_4d/StackCube-v1",
    }
    paths = _paths(config)

    assert paths["convert"] != paths["baseline"]
    assert paths["baseline"] != paths["repair"]
    assert paths["capture"] != paths["merge"]
    assert paths["layout"].name == "replay_layout.json"
    assert all(isinstance(value, Path) for value in paths.values())


def test_carrier_layout_preserves_baseline_replay_order(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.h5"
    repair = tmp_path / "repair.h5"
    source_ids = [30, 10, 20]
    with h5py.File(baseline, "w") as handle:
        for group_id, source_id in enumerate(source_ids):
            group = handle.create_group(f"traj_{group_id}")
            group.create_dataset("actions", data=np.zeros((1, 4), dtype=np.float32))
            group.create_dataset("success", data=np.asarray([True], dtype=bool))
            states = group.create_group("env_states")
            states.create_group("actors").create_dataset(
                "cube", data=np.asarray([[float(source_id)]], dtype=np.float32)
            )
    with h5py.File(baseline, "r") as handle:
        records = []
        for group_id, source_id in enumerate(source_ids):
            records.append(
                {
                    "episode_id": group_id,
                    "source_episode_id": source_id,
                    "source_start_state_fingerprint": start_state_fingerprint(
                        handle[f"traj_{group_id}"]
                    ),
                }
            )
    baseline.with_suffix(".json").write_text(
        json.dumps({"episodes": records}), encoding="utf-8"
    )
    with h5py.File(repair, "w"):
        pass
    repair.with_suffix(".json").write_text(json.dumps({"episodes": []}), encoding="utf-8")

    batches, _ = build_carrier_batches(
        baseline,
        repair,
        num_envs=2,
        expected_baseline_accepted=None,
        expected_repair_accepted=None,
        expected_accepted=None,
    )

    assert [[item.source_episode_id for item in batch] for batch in batches] == [
        [30, 10],
        [20],
    ]
