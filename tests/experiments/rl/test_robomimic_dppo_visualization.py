from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_practice.experiments.rl.robomimic_dppo_visualization import (
    DPPOEpisode,
    _checkpoint_for_seed,
    _infer_checkpoint_step,
    audit,
    load_episode,
    save_episode,
    visualize,
)


def _episode(*, episode_id: int = 0, seed: int = 1) -> DPPOEpisode:
    steps = 3
    return DPPOEpisode(
        episode_id=episode_id,
        policy_seed=seed,
        env_index=episode_id,
        environment_seed=seed + episode_id,
        frames=np.zeros((steps, 8, 10, 3), dtype=np.uint8),
        state_normalized=np.zeros((steps, 9), dtype=np.float32),
        state_raw=np.zeros((steps, 9), dtype=np.float32),
        action_chunks=np.zeros((steps, 4, 7), dtype=np.float32),
        rewards=np.asarray([0.0, 4.0, 0.0], dtype=np.float32),
        terminated=np.asarray([False, False, False]),
        truncated=np.asarray([False, False, True]),
        act_steps=4,
        success_threshold=1.0,
        return_total=4.0,
        success=True,
        first_success_step=2,
    )


def test_dppo_episode_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "episode_0000.npz"
    save_episode(source, _episode())
    restored = load_episode(source)

    assert restored.frames.shape == (3, 8, 10, 3)
    assert restored.state_normalized.shape == (3, 9)
    assert restored.action_chunks.shape == (3, 4, 7)
    assert restored.success is True
    assert restored.first_success_step == 2


def test_audit_reports_three_seed_contract_and_missing_results(tmp_path: Path, capsys) -> None:
    eval_root = tmp_path / "eval"
    for seed, success_rate in ((1, 0.48), (2, 0.34)):
        seed_root = eval_root / f"seed_{seed}"
        seed_root.mkdir(parents=True)
        np.savez(
            seed_root / "result.npz",
            num_episode=50,
            eval_success_rate=success_rate,
        )

    result = audit(eval_root=eval_root, seeds=(1, 2, 3), episodes_per_seed=50)
    assert result["expected"]["total_episodes"] == 150
    assert result["materialized_episode_total"] == 100
    assert result["missing_result_seeds"] == [3]
    assert result["contract_ok"] is False
    assert "expected_total" not in capsys.readouterr().out


def test_checkpoint_resolution_supports_shared_and_per_seed_layouts(tmp_path: Path) -> None:
    shared = tmp_path / "state_4000.pt"
    shared.write_bytes(b"shared")
    assert _infer_checkpoint_step(shared) == 4000
    assert _checkpoint_for_seed(None, shared, 2, None) == shared

    per_seed_root = tmp_path / "finetune"
    per_seed = per_seed_root / "seed_2" / "checkpoint" / "state_300.pt"
    per_seed.parent.mkdir(parents=True)
    per_seed.write_bytes(b"per-seed")
    assert _checkpoint_for_seed(per_seed_root, None, 2, 300) == per_seed


def test_visualize_writes_rerun_archive(tmp_path: Path) -> None:
    pytest.importorskip("rerun")
    raw_root = tmp_path / "raw"
    seed_root = raw_root / "seed_1"
    episode = _episode()
    save_episode(seed_root / "episode_0000.npz", episode)
    (seed_root / "manifest.json").write_text(
        json.dumps({"episodes": [{"file": "episode_0000.npz"}]})
    )
    (raw_root / "manifest.json").write_text(
        json.dumps({"episodes": [{"file": "seed_1/episode_0000.npz"}]})
    )

    manifest = visualize(raw_root=raw_root, output_dir=tmp_path / "rerun")

    assert manifest["selected_episode_count"] == 1
    assert manifest["fps"] == 5.0
    assert (tmp_path / "rerun" / "robomimic_dppo_visualization.rrd").is_file()
