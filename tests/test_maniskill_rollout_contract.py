from __future__ import annotations

import pytest

from robot_practice.experiments.maniskill.rollout_contract import (
    M1W_DATA_PROVENANCE,
    OFFICIAL_DATA_PROVENANCE,
    require_rollout_backend,
)


def test_official_data_requires_cpu_rollout() -> None:
    assert require_rollout_backend("official_rgb_18d_4d", "cpu") == {
        "data_provenance": OFFICIAL_DATA_PROVENANCE,
        "required_rollout_backend": "physx_cpu",
        "actual_rollout_backend": "physx_cpu",
    }

    with pytest.raises(ValueError, match="requires 'physx_cpu'"):
        require_rollout_backend("official", "physx_cuda")


def test_m1w_data_requires_gpu_rollout() -> None:
    metadata = require_rollout_backend("m1w_projected_rgb_4d", "physx_cuda:3")

    assert metadata == {
        "data_provenance": M1W_DATA_PROVENANCE,
        "required_rollout_backend": "physx_cuda",
        "actual_rollout_backend": "physx_cuda",
    }

    with pytest.raises(ValueError, match="requires 'physx_cuda'"):
        require_rollout_backend("m1w", "physx_cpu")
