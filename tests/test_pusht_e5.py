from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from robot_practice.experiments.pusht_diffusion import training_common
from robot_practice.experiments.pusht_diffusion.dataset_split import load_split
from robot_practice.experiments.pusht_diffusion.e5_dataset import (
    PrototypeConditionedDataset,
    load_prototype_sidecar,
)
from robot_practice.experiments.pusht_diffusion.e5_fixed_observation import (
    evaluate_e5_fixed_observations,
)
from robot_practice.experiments.pusht_diffusion.e5_prototypes import (
    build_episode_sketch,
    component_distances,
    fit_component_scales,
    fit_kmedoids,
    trajectory_distance,
)
from robot_practice.experiments.pusht_diffusion.e5_rollout_eval import (
    _aggregate,
    partition_environment_seeds,
)
from robot_practice.experiments.pusht_diffusion.e5_runtime import (
    FixedPrototypePreprocessor,
    add_prototype_id,
    move_processed_batch,
)
from robot_practice.experiments.pusht_diffusion.model_configs.mode_diffusion import (
    make_mode_diffusion_config,
)
from robot_practice.experiments.pusht_diffusion.noise import make_initial_noise
from robot_practice.experiments.pusht_diffusion.policy_loading import controlled_sample
from robot_practice.experiments.pusht_diffusion.provenance import (
    sha256_file,
    state_dict_hashes,
)
from robot_practice.experiments.pusht_diffusion.rollout import ExplicitChunkExecutor
from robot_practice.experiments.pusht_diffusion.training_common import (
    TrainArguments,
    build_train_pipeline_config,
    validate_training_inputs,
)
from robot_practice.policies.mode_bc_chunk import ModeBCChunkConfig, ModeBCChunkPolicy
from robot_practice.policies.mode_conditioning import PROTOTYPE_ID
from robot_practice.policies.mode_diffusion import ModeDiffusionConfig, ModeDiffusionPolicy


def _features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (2,)),
        },
        {ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
    )


def _mode_diffusion_config() -> ModeDiffusionConfig:
    inputs, outputs = _features()
    return ModeDiffusionConfig(
        input_features=inputs,
        output_features=outputs,
        device="cpu",
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=(16, 32),
        n_groups=8,
        num_train_timesteps=100,
        num_inference_steps=1,
        pretrained_backbone_weights=None,
        num_prototypes=3,
        mode_embedding_dim=32,
        condition_dropout_prob=0.2,
        push_to_hub=False,
    )


def _mode_bc_config() -> ModeBCChunkConfig:
    inputs, outputs = _features()
    return ModeBCChunkConfig(
        input_features=inputs,
        output_features=outputs,
        device="cpu",
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=(16, 32),
        n_groups=8,
        pretrained_backbone_weights=None,
        num_prototypes=3,
        mode_embedding_dim=32,
        condition_dropout_prob=0.2,
        fixed_timestep=0,
        push_to_hub=False,
    )


def _trajectory_records(*, episode: int = 0, direction: float = 1.0) -> list[dict]:
    records = []
    for frame in range(24):
        motion = max(0, frame - 6) * 3.0 * direction
        block = [200.0 + motion, 220.0]
        if frame < 5:
            agent = [140.0, 220.0]
            distance = 40.0
        else:
            agent = [block[0] - 10.0, block[1]]
            distance = 5.0
        records.append(
            {
                "episode_index": episode,
                "frame_index": frame,
                "split": "train",
                "agent_position": agent,
                "block_pose_valid": True,
                "block_position": block,
                "block_angle_rad": 0.0,
                "agent_block_surface_distance": distance,
            }
        )
    return records


def _write_sidecar(root: Path) -> Path:
    assignments = root / "episode_assignments.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": 0,
                    "prototype_id": 1,
                    "training_condition_id": 1,
                    "eligible": True,
                    "num_frames": 4,
                },
                {
                    "episode_index": 1,
                    "prototype_id": None,
                    "training_condition_id": 3,
                    "eligible": False,
                    "num_frames": 3,
                },
            ]
        ),
        assignments,
    )
    manifest = root / "prototype_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "selected_k": 3,
                "null_prototype_id": 3,
                "artifact_hashes": {
                    "episode_assignments.parquet": sha256_file(assignments),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_e5_sketch_distance_and_kmedoids_are_deterministic() -> None:
    right = build_episode_sketch(_trajectory_records(episode=10, direction=1.0))
    left = build_episode_sketch(_trajectory_records(episode=20, direction=-1.0))
    invalid_records = _trajectory_records(episode=30)
    for record in invalid_records:
        record["agent_block_surface_distance"] = 100.0
    invalid = build_episode_sketch(invalid_records)

    assert right.eligible
    assert left.eligible
    assert not invalid.eligible
    assert invalid.exclusion_reason == "no_stable_contact"
    assert all(value == pytest.approx(0.0) for value in component_distances(right, right).values())
    scales = fit_component_scales([right, left])
    assert trajectory_distance(right, right, scales=scales) == pytest.approx(0.0)
    assert trajectory_distance(right, left, scales=scales) > 0.0

    matrix = np.asarray(
        [
            [0.0, 0.1, 4.0, 4.1],
            [0.1, 0.0, 4.2, 4.0],
            [4.0, 4.2, 0.0, 0.1],
            [4.1, 4.0, 0.1, 0.0],
        ]
    )
    fitted = fit_kmedoids(matrix, k=2, seed=7, episode_ids=[40, 10, 30, 20])
    assert tuple([40, 10, 30, 20][index] for index in fitted.medoid_indices) == tuple(
        sorted([40, 10, 30, 20][index] for index in fitted.medoid_indices)
    )
    assert (
        fitted.labels.tolist()
        == fit_kmedoids(
            matrix,
            k=2,
            seed=7,
            episode_ids=[40, 10, 30, 20],
        ).labels.tolist()
    )


def test_e5_sidecar_dataset_keeps_original_data_read_only_and_maps_ineligible_to_null(
    tmp_path: Path,
) -> None:
    manifest = _write_sidecar(tmp_path)
    sidecar = load_prototype_sidecar(manifest)

    class Dataset:
        def __init__(self) -> None:
            self.episodes = [0, 1]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict:
            return {"episode_index": torch.tensor(index), "value": torch.tensor(index)}

    base = Dataset()
    wrapped = PrototypeConditionedDataset(base, sidecar)

    assert wrapped[0][PROTOTYPE_ID].item() == 1
    assert wrapped[1][PROTOTYPE_ID].item() == 3
    assert PROTOTYPE_ID not in base[0]
    assert wrapped.accounting == {
        "eligible_conditional_frames": 4,
        "ineligible_always_null_frames": 3,
        "total_frames": 7,
    }


def test_mode_policies_share_initialization_and_use_the_embedding() -> None:
    torch.manual_seed(3)
    diffusion = ModeDiffusionPolicy(_mode_diffusion_config()).eval()
    torch.manual_seed(3)
    bc = ModeBCChunkPolicy(_mode_bc_config()).eval()

    missing, unexpected = bc.load_state_dict(diffusion.state_dict(), strict=True)
    assert not missing
    assert not unexpected
    assert state_dict_hashes(bc.state_dict()) == state_dict_hashes(diffusion.state_dict())

    base = {
        OBS_STATE: torch.randn(1, 2, 2),
        OBS_ENV_STATE: torch.randn(1, 2, 2),
    }
    first = diffusion.diffusion._prepare_global_conditioning(
        add_prototype_id(base, prototype_id=0, num_prototypes=3)
    )
    second = diffusion.diffusion._prepare_global_conditioning(
        add_prototype_id(base, prototype_id=1, num_prototypes=3)
    )
    assert torch.equal(first[:, :-32], second[:, :-32])
    assert not torch.equal(first[:, -32:], second[:, -32:])


def test_e5_training_uses_seeded_random_initialization_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experiments"
        / "pusht_diffusion"
        / "split.json"
    )
    split = load_split(split_path)
    sidecar_root = tmp_path / "prototypes"
    sidecar_root.mkdir()
    prototype_manifest = _write_sidecar(sidecar_root)
    payload = json.loads(prototype_manifest.read_text(encoding="utf-8"))
    payload["split_manifest_sha256"] = split.manifest_sha256
    prototype_manifest.write_text(json.dumps(payload), encoding="utf-8")

    output_root = tmp_path / "outputs"
    dataset_root = output_root / "dataset_views" / "smoke"
    dataset_root.mkdir(parents=True)
    (dataset_root / "view_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(training_common, "default_output_root", lambda: output_root)

    arguments = TrainArguments(
        mode="smoke",
        steps=1,
        train_seed=1000,
        split_file=split_path,
        output_dir=tmp_path / "run",
        batch_size=1,
        num_workers=0,
        requested_device="cpu",
        print_command=False,
    )
    policy = make_mode_diffusion_config(device="cpu", num_prototypes=3)
    resolved = build_train_pipeline_config(
        arguments=arguments,
        policy=policy,
        policy_type="mode_diffusion",
        entry_module="test.mode_diffusion",
        model_config_module="test.mode_diffusion_config",
        prototype_manifest=prototype_manifest,
        random_initialization=True,
    )

    assert resolved.initialization_root is None
    assert resolved.config.policy.pretrained_path is None
    assert resolved.summary()["training"]["initialization"] == "random_from_train_seed"
    assert resolved.summary()["training"]["initialization_seed"] == 1000
    validate_training_inputs(resolved)


def test_mode_diffusion_controlled_sampling_and_executor_keep_the_categorical_id() -> None:
    policy = ModeDiffusionPolicy(_mode_diffusion_config()).eval()
    history = {
        OBS_STATE: torch.randn(1, 2, 2),
        OBS_ENV_STATE: torch.randn(1, 2, 2),
        PROTOTYPE_ID: torch.tensor([2]),
    }
    noise = make_initial_noise(
        seed=4,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )
    first = controlled_sample(policy, history, initial_noise=noise.tensor, reverse_seed=5)
    repeated = controlled_sample(policy, history, initial_noise=noise.tensor, reverse_seed=5)
    assert torch.equal(first.actions, repeated.actions)

    executor = ExplicitChunkExecutor(policy, environment_seed=2000, policy_seed=0)
    current = {
        OBS_STATE: torch.randn(1, 2),
        OBS_ENV_STATE: torch.randn(1, 2),
        PROTOTYPE_ID: torch.tensor([2]),
    }
    executor.select_action(current)
    assert executor.replans[0].initial_noise_seed is not None
    assert executor.replans[0].sampling_protocol is not None
    changed = {**current, PROTOTYPE_ID: torch.tensor([1])}
    with pytest.raises(ValueError, match="changed during an active rollout"):
        executor.select_action(changed)


def test_mode_bc_executor_stays_deterministic_despite_diffusion_policy_inheritance() -> None:
    policy = ModeBCChunkPolicy(_mode_bc_config()).eval()
    executor = ExplicitChunkExecutor(policy, environment_seed=2000, policy_seed=0)
    current = {
        OBS_STATE: torch.randn(1, 2),
        OBS_ENV_STATE: torch.randn(1, 2),
        PROTOTYPE_ID: torch.tensor([2]),
    }

    executor.select_action(current)

    assert executor.replans[0].initial_noise_seed is None
    assert executor.replans[0].reverse_noise_seed is None
    assert executor.replans[0].sampling_protocol is None


def test_fixed_prototype_preprocessor_validates_and_injects_after_processing() -> None:
    wrapper = FixedPrototypePreprocessor(
        lambda batch: {"x": batch["x"] + 1},
        prototype_id=2,
        num_prototypes=3,
    )

    result = wrapper({"x": torch.tensor([1.0])})

    assert result["x"].item() == 2.0
    assert result[PROTOTYPE_ID].dtype == torch.long
    assert result[PROTOTYPE_ID].tolist() == [2]
    with pytest.raises(ValueError, match="prototype_id"):
        add_prototype_id({}, prototype_id=4, num_prototypes=3)
    moved = move_processed_batch(
        {"tensor": torch.ones(1), "metadata": "kept"},
        device="meta",
    )
    assert moved["tensor"].device.type == "meta"
    assert moved["metadata"] == "kept"


def test_e5_condition_dropout_records_effective_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = ModeDiffusionPolicy(_mode_diffusion_config()).train()
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, device=None: torch.zeros(shape, device=device),
    )
    batch = {
        OBS_STATE: torch.randn(2, 2, 2),
        OBS_ENV_STATE: torch.randn(2, 2, 2),
        ACTION: torch.randn(2, 16, 2),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
        PROTOTYPE_ID: torch.tensor([0, 3]),
    }

    policy(batch)

    assert policy.diffusion.last_condition_stats == {
        "input_null_count": 1,
        "dropout_to_null_count": 1,
        "effective_null_count": 2,
        "conditional_count": 0,
    }


def test_e5_policy_checkpoint_round_trip(tmp_path: Path) -> None:
    policy = ModeDiffusionPolicy(_mode_diffusion_config()).eval()
    policy.save_pretrained(tmp_path)

    restored = ModeDiffusionPolicy.from_pretrained(tmp_path, local_files_only=True)

    assert state_dict_hashes(restored.state_dict()) == state_dict_hashes(policy.state_dict())
    assert restored.config.num_prototypes == 3


def test_e5_formal_contracts_reject_reduced_runs_before_loading_artifacts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires 20"):
        evaluate_e5_fixed_observations(
            bc_policy_path=tmp_path / "bc",
            diffusion_policy_path=tmp_path / "dp",
            prototype_manifest=tmp_path / "manifest",
            candidates_path=tmp_path / "candidates",
            dataset_root=tmp_path / "dataset",
            output_dir=tmp_path / "output",
            num_policy_seeds=2,
            num_inference_steps=100,
            label="Offline",
            overwrite=False,
        )
    assert partition_environment_seeds([2000, 2001, 2002], num_gpus=2) == (
        (2000, 2002),
        (2001,),
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        partition_environment_seeds([2000], num_gpus=2)


def test_e5_aggregate_requires_two_successful_null_seeds_per_mode() -> None:
    rows = []
    for condition_id in (0, 1, 2, 3):
        rows.append(
            {
                "model": "mode_bc_chunk",
                "condition_id": condition_id,
                "environment_seed": 2000,
                "success": True,
                "classifiable": True,
                "prototype_adherent": condition_id < 3,
                "controlled_success": condition_id < 3,
                "mean_replan_latency_ms": 1.0,
                "nearest_prototype_id": 0,
            }
        )
        for policy_seed in range(4):
            nearest = 0 if policy_seed < 2 else 1
            rows.append(
                {
                    "model": "mode_diffusion",
                    "condition_id": condition_id,
                    "environment_seed": 2000,
                    "policy_seed": policy_seed,
                    "success": True,
                    "classifiable": True,
                    "prototype_adherent": condition_id < 3,
                    "controlled_success": condition_id < 3,
                    "mean_replan_latency_ms": 2.0,
                    "nearest_prototype_id": nearest,
                }
            )

    summary, _, null_rows = _aggregate(
        rows,
        selected_k=3,
        environment_seeds=[2000],
        policy_seeds=range(4),
    )

    assert null_rows[0]["supported_prototype_ids"] == [0, 1]
    assert null_rows[0]["multi_mode_success"] is True
    assert summary["null_environments_with_multi_mode_success"] == 1
