from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robot_practice.experiments.pusht_diffusion.fixed_observation_action_modes import (
    analyze_action_samples,
    bc_averaging_signature,
    reconstruct_action_chunks,
)


def _chunk_rows(*, num_samples: int = 20, num_steps: int = 8) -> pd.DataFrame:
    rows = []
    for sample in range(num_samples):
        for step in range(num_steps):
            rows.append(
                {
                    "sample_ordinal": sample,
                    "step": step,
                    "normalized_action_x": sample / 100.0 + step / 1000.0,
                    "normalized_action_y": -sample / 100.0 - step / 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _separated_actions() -> np.ndarray:
    rng = np.random.default_rng(7)
    left = rng.normal(-0.4, 0.005, size=(10, 8, 2))
    right = rng.normal(0.4, 0.005, size=(10, 8, 2))
    return np.concatenate([left, right], axis=0)


def test_reconstruct_action_chunks_is_order_invariant() -> None:
    rows = _chunk_rows()
    shuffled = rows.sample(frac=1.0, random_state=9)

    chunks = reconstruct_action_chunks(shuffled, expected_samples=20, expected_steps=8)

    assert chunks.shape == (20, 8, 2)
    assert chunks[3, 4].tolist() == pytest.approx([0.034, -0.034])


def test_reconstruct_action_chunks_rejects_duplicate_steps() -> None:
    rows = _chunk_rows()
    duplicated = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="exactly once"):
        reconstruct_action_chunks(duplicated, expected_samples=20, expected_steps=8)


def test_degenerate_actions_are_deterministically_unimodal() -> None:
    actions = np.zeros((20, 8, 2), dtype=np.float64)

    result = analyze_action_samples(
        actions,
        candidate_id="degenerate",
        protocol="full_sampling_seed",
        subsample_iterations=5,
    )

    assert result.degenerate_unimodal is True
    assert result.selected_k == 1
    assert result.multimodal is False


def test_separated_actions_pass_hybrid_multimodal_gate() -> None:
    result = analyze_action_samples(
        _separated_actions(),
        candidate_id="separated",
        protocol="full_sampling_seed",
        subsample_iterations=20,
    )

    assert result.selected_k == 2
    assert result.minimum_gmm_cluster_size >= 3
    assert result.minimum_distance_cluster_size >= 3
    assert result.bic_improvement >= 10.0
    assert result.subsample_multimodal_fraction >= 0.7
    assert result.silhouette >= 0.25
    assert result.adjusted_rand_index >= 0.5
    assert result.multimodal is True


def test_bc_averaging_signature_requires_mean_like_low_density_action() -> None:
    actions = _separated_actions()
    result = analyze_action_samples(
        actions,
        candidate_id="bc-signature",
        protocol="full_sampling_seed",
        subsample_iterations=20,
    )

    between = bc_averaging_signature(actions, np.zeros((8, 2)), result)
    on_mode = bc_averaging_signature(actions, actions[0], result)

    assert between.mean_like is True
    assert between.low_density is True
    assert between.averaging_consistent_signature is True
    assert on_mode.averaging_consistent_signature is False


def test_action_mode_launcher_is_registered() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "pusht"
        / "analyze_fixed_observation_action_modes.sh"
    )

    assert launcher.is_file()
