from __future__ import annotations

import pytest

from robot_practice.experiments.pusht_diffusion.noise import (
    full_sampling_reverse_noise_seed,
    initial_noise_seed,
    reverse_noise_seed,
)
from robot_practice.experiments.pusht_diffusion.report import _validate_e2_schema
from robot_practice.experiments.pusht_diffusion.rollout import (
    FULL_DDPM_SAMPLING,
    INITIAL_NOISE_SENSITIVITY,
)
from robot_practice.experiments.pusht_diffusion.rollout_eval import (
    _validate_num_gpus,
    partition_environment_seeds,
    resolve_protocol_plan,
    rollout_recording_identity,
    success_variance,
)


def test_formal_e2_protocol_counts_are_exact() -> None:
    e2a = resolve_protocol_plan(
        protocol=INITIAL_NOISE_SENSITIVITY,
        label="Closed-loop",
    )
    e2b = resolve_protocol_plan(
        protocol=FULL_DDPM_SAMPLING,
        label="Closed-loop",
    )

    assert e2a.environment_seeds == (2000, 2001)
    assert e2a.policy_seeds == tuple(range(5))
    assert e2a.num_diffusion_rollouts == 10
    assert e2a.num_bc_rollouts == 0
    assert e2b.environment_seeds == tuple(range(2000, 2020))
    assert e2b.policy_seeds == tuple(range(10))
    assert e2b.num_diffusion_rollouts == 200
    assert e2b.num_bc_rollouts == 20
    assert e2b.num_rollouts == 220


def test_reduced_counts_are_smoke_only() -> None:
    with pytest.raises(ValueError, match="requires exactly"):
        resolve_protocol_plan(
            protocol=FULL_DDPM_SAMPLING,
            label="Closed-loop",
            num_environment_seeds=2,
            num_policy_seeds=2,
        )

    smoke = resolve_protocol_plan(
        protocol=FULL_DDPM_SAMPLING,
        label="Smoke",
        num_environment_seeds=2,
        num_policy_seeds=2,
    )
    assert smoke.num_diffusion_rollouts == 4
    assert smoke.num_bc_rollouts == 2


def test_environment_seed_shards_are_round_robin_disjoint_and_complete() -> None:
    environment_seeds = tuple(range(2000, 2020))

    shards = partition_environment_seeds(environment_seeds, num_gpus=4)

    assert shards == (
        (2000, 2004, 2008, 2012, 2016),
        (2001, 2005, 2009, 2013, 2017),
        (2002, 2006, 2010, 2014, 2018),
        (2003, 2007, 2011, 2015, 2019),
    )
    flattened = [seed for shard in shards for seed in shard]
    assert sorted(flattened) == list(environment_seeds)
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize("num_gpus", [0, -1, True])
def test_environment_seed_shards_reject_invalid_gpu_counts(num_gpus: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        partition_environment_seeds((2000, 2001), num_gpus=num_gpus)


def test_environment_seed_shards_reject_more_gpus_than_environments() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        partition_environment_seeds((2000, 2001), num_gpus=3)


@pytest.mark.parametrize(
    ("protocol", "model", "environment_seed", "policy_seed", "success", "expected"),
    [
        (
            INITIAL_NOISE_SENSITIVITY,
            "diffusion",
            2000,
            0,
            True,
            (
                "e2a_dp_env_2000_seed_00",
                "E2a | DP | env 2000 | seed 00 | success",
            ),
        ),
        (
            FULL_DDPM_SAMPLING,
            "diffusion",
            2019,
            9,
            False,
            (
                "e2b_dp_env_2019_seed_09",
                "E2b | DP | env 2019 | seed 09 | failure",
            ),
        ),
        (
            FULL_DDPM_SAMPLING,
            "bc_chunk",
            2007,
            0,
            True,
            (
                "e2b_bc_env_2007",
                "E2b | BC | env 2007 | success",
            ),
        ),
    ],
)
def test_e2_recording_identity_distinguishes_protocol_model_and_seed(
    protocol: str,
    model: str,
    environment_seed: int,
    policy_seed: int,
    success: bool,
    expected: tuple[str, str],
) -> None:
    assert (
        rollout_recording_identity(
            protocol=protocol,
            model=model,
            environment_seed=environment_seed,
            policy_seed=policy_seed,
            success=success,
        )
        == expected
    )


def test_multi_gpu_validation_checks_cuda_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "robot_practice.experiments.pusht_diffusion.rollout_eval.torch.cuda.is_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="requires CUDA"):
        _validate_num_gpus(num_gpus=2, num_environment_seeds=20)


def test_multi_gpu_validation_checks_visible_device_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "robot_practice.experiments.pusht_diffusion.rollout_eval.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "robot_practice.experiments.pusht_diffusion.rollout_eval.torch.cuda.device_count",
        lambda: 2,
    )

    with pytest.raises(RuntimeError, match="only 2 CUDA device"):
        _validate_num_gpus(num_gpus=3, num_environment_seeds=20)


def test_e2_rng_protocols_pair_initial_noise_and_split_reverse_noise() -> None:
    initial = {
        initial_noise_seed(
            environment_seed=2000,
            policy_seed=policy_seed,
            replan_index=3,
        )
        for policy_seed in range(10)
    }
    e2a_reverse = {reverse_noise_seed(environment_seed=2000, replan_index=3) for _ in range(10)}
    e2b_reverse = {
        full_sampling_reverse_noise_seed(
            environment_seed=2000,
            policy_seed=policy_seed,
            replan_index=3,
        )
        for policy_seed in range(10)
    }

    assert len(initial) == 10
    assert len(e2a_reverse) == 1
    assert len(e2b_reverse) == 10
    assert e2a_reverse.isdisjoint(e2b_reverse)


def test_success_variance_uses_environment_rates() -> None:
    within, between = success_variance(
        {
            2000: [0, 0, 0, 0],
            2001: [1, 1, 1, 1],
            2002: [0, 1, 0, 1],
        }
    )

    assert within == pytest.approx((0.0 + 0.0 + 0.25) / 3.0)
    assert between == pytest.approx(1.0 / 6.0)


def test_report_accepts_only_exact_closed_loop_e2b(tmp_path) -> None:
    manifest = {
        "schema_version": 2,
        "experiment_schema": "pusht-e2-closed-loop-v2",
        "sampling_protocol": FULL_DDPM_SAMPLING,
        "all_samples_closed_loop": True,
        "environment_seeds": list(range(2000, 2020)),
        "policy_seeds": list(range(10)),
        "num_diffusion_rollouts": 200,
        "num_bc_rollouts": 20,
        "num_rollouts": 220,
    }
    _validate_e2_schema(tmp_path, manifest)

    manifest["sampling_protocol"] = INITIAL_NOISE_SENSITIVITY
    with pytest.raises(ValueError, match="not the exact"):
        _validate_e2_schema(tmp_path, manifest)
