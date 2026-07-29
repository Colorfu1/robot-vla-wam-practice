from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from robot_practice.experiments.pusht_diffusion.dataset_split import load_split
from robot_practice.experiments.pusht_diffusion.diversity_metrics import (
    complete_linkage_mode_labels,
    resample_agent_block_trajectory,
    summarize_action_diversity,
    trajectory_distance_matrix,
    trajectory_mode_distance,
)
from robot_practice.experiments.pusht_diffusion.fixed_observation_eval import (
    _aggregate_e1_metrics,
    _observation_only,
    _order_and_validate_candidates,
    _validate_e1_rows,
    evaluate_fixed_observations,
)
from robot_practice.experiments.pusht_diffusion.model_configs import (
    model_config_fingerprint,
)
from robot_practice.experiments.pusht_diffusion.model_configs.bc_chunk import (
    make_bc_chunk_config,
)
from robot_practice.experiments.pusht_diffusion.model_configs.diffusion import (
    make_diffusion_config,
)
from robot_practice.experiments.pusht_diffusion.noise import (
    fixed_observation_e1a_reverse_seed,
    fixed_observation_e1b_reverse_seed,
    fixed_observation_initial_noise_seed,
    initial_noise_seed,
    make_initial_noise,
    reverse_noise_seed,
)
from robot_practice.experiments.pusht_diffusion.output import (
    AtomicOutputDirectory,
    write_metrics,
)
from robot_practice.experiments.pusht_diffusion.padding_stats import (
    padding_counts_for_episode,
)
from robot_practice.experiments.pusht_diffusion.policy_loading import (
    controlled_sample,
    load_legacy_normalization_stats,
    replace_scheduler,
)
from robot_practice.experiments.pusht_diffusion.provenance import state_dict_hashes
from robot_practice.experiments.pusht_diffusion.report import (
    _validate_e1_schema,
    _validate_e3_schema,
    make_report,
)
from robot_practice.experiments.pusht_diffusion.rollout import (
    ExplicitChunkExecutor,
    check_environment_replay,
    run_rollout,
)
from robot_practice.experiments.pusht_diffusion.statistics import (
    environment_cluster_bootstrap,
    wilson_interval,
)
from robot_practice.experiments.pusht_diffusion.training_runtime import (
    SubsetEpisodeAwareSampler,
    persistent_training_log,
)
from robot_practice.policies.bc_chunk.configuration_bc_chunk import BCChunkConfig
from robot_practice.policies.bc_chunk.modeling_bc_chunk import BCChunkPolicy


def _features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (2,)),
        },
        {ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
    )


def _diffusion_config() -> DiffusionConfig:
    inputs, outputs = _features()
    return DiffusionConfig(
        input_features=inputs,
        output_features=outputs,
        device="cpu",
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=(16, 32),
        n_groups=8,
        num_train_timesteps=100,
        num_inference_steps=3,
        pretrained_backbone_weights=None,
        push_to_hub=False,
    )


def _bc_config() -> BCChunkConfig:
    inputs, outputs = _features()
    return BCChunkConfig(
        input_features=inputs,
        output_features=outputs,
        device="cpu",
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=(16, 32),
        n_groups=8,
        pretrained_backbone_weights=None,
        fixed_timestep=0,
        push_to_hub=False,
    )


def test_committed_split_is_deterministic_disjoint_and_complete() -> None:
    split = load_split()

    assert len(split.train) == 165
    assert len(split.development) == 20
    assert len(split.audit) == 21
    assert split.smoke_train == split.train[:8]
    assert set(split.train).isdisjoint(split.development)
    assert set(split.train).isdisjoint(split.audit)
    assert set(split.development).isdisjoint(split.audit)
    assert set(split.train + split.development + split.audit) == set(range(206))
    assert split.e2_environment_seeds == tuple(range(2000, 2020))


def test_subset_episode_sampler_emits_relative_indices_for_filtered_dataset() -> None:
    sampler = SubsetEpisodeAwareSampler(
        dataset_from_indices=[0, 10, 20, 30],
        dataset_to_indices=[10, 20, 30, 40],
        episode_indices_to_use=[3, 1],
        drop_n_last_frames=2,
        shuffle=False,
    )

    assert list(sampler) == [*range(8), *range(10, 18)]
    assert max(sampler.indices) < 20


def test_explicit_model_configs_are_complete_and_capacity_matched() -> None:
    bc = make_bc_chunk_config(device="cpu")
    diffusion = make_diffusion_config(device="cpu")

    assert isinstance(bc, BCChunkConfig)
    assert isinstance(diffusion, DiffusionConfig)
    for field in (
        "n_obs_steps",
        "horizon",
        "n_action_steps",
        "drop_n_last_frames",
        "vision_backbone",
        "crop_shape",
        "down_dims",
        "kernel_size",
        "n_groups",
        "diffusion_step_embed_dim",
        "optimizer_lr",
        "optimizer_betas",
        "optimizer_weight_decay",
        "scheduler_name",
        "scheduler_warmup_steps",
    ):
        assert getattr(bc, field) == getattr(diffusion, field)
    assert bc.fixed_timestep == 0
    assert diffusion.noise_scheduler_type == "DDPM"
    assert diffusion.num_train_timesteps == 100
    assert diffusion.num_inference_steps == 100


def test_legacy_policy_normalization_stats_are_recovered_without_loading_model(
    tmp_path: Path,
) -> None:
    from safetensors.torch import save_file

    legacy = tmp_path / "legacy_policy"
    legacy.mkdir()
    save_file(
        {
            "diffusion.fake_weight": torch.ones(2, 2),
            "normalize_inputs.buffer_observation_image.mean": torch.tensor(
                [[[0.485]], [[0.456]], [[0.406]]]
            ),
            "normalize_inputs.buffer_observation_image.std": torch.tensor(
                [[[0.229]], [[0.224]], [[0.225]]]
            ),
            "normalize_inputs.buffer_observation_state.min": torch.tensor([13.0, 32.0]),
            "normalize_inputs.buffer_observation_state.max": torch.tensor([496.0, 511.0]),
            "normalize_targets.buffer_action.min": torch.tensor([12.0, 25.0]),
            "normalize_targets.buffer_action.max": torch.tensor([511.0, 511.0]),
            "unnormalize_outputs.buffer_action.min": torch.tensor([12.0, 25.0]),
            "unnormalize_outputs.buffer_action.max": torch.tensor([511.0, 511.0]),
        },
        legacy / "model.safetensors",
    )

    stats = load_legacy_normalization_stats(
        legacy,
        revision=None,
        local_files_only=True,
    )

    assert set(stats) == {"observation.image", "observation.state", "action"}
    assert stats["observation.image"]["mean"].shape == (3, 1, 1)
    assert stats["observation.state"]["min"].tolist() == [13.0, 32.0]
    assert stats["action"]["max"].tolist() == [511.0, 511.0]


def test_model_config_fingerprint_ignores_runtime_device() -> None:
    cpu = make_diffusion_config(device="cpu")
    cuda = make_diffusion_config(device="cuda")

    assert model_config_fingerprint(cpu) == model_config_fingerprint(cuda)
    cuda.horizon = 20
    assert model_config_fingerprint(cpu) != model_config_fingerprint(cuda)


def test_training_log_is_persisted_inside_completed_lerobot_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "run"

    with persistent_training_log(output):
        print("visible and persistent")
        output.mkdir()

    assert "visible and persistent" in capsys.readouterr().out
    assert (output / "train.log").read_text(encoding="utf-8") == "visible and persistent\n"
    assert not (tmp_path / "run.train.log").exists()


def test_rng_derivation_changes_only_the_intended_stream() -> None:
    seed = initial_noise_seed(environment_seed=2000, policy_seed=3, replan_index=4)

    assert seed == initial_noise_seed(environment_seed=2000, policy_seed=3, replan_index=4)
    assert seed != initial_noise_seed(environment_seed=2000, policy_seed=4, replan_index=4)
    assert seed != initial_noise_seed(environment_seed=2000, policy_seed=3, replan_index=5)
    assert reverse_noise_seed(
        environment_seed=2000,
        replan_index=4,
    ) == reverse_noise_seed(environment_seed=2000, replan_index=4)


def test_e1_seed_derivation_is_stable_and_uses_candidate_frame_key() -> None:
    initial_zero = fixed_observation_initial_noise_seed(
        episode_index=205,
        frame_index=74,
        sample_ordinal=0,
    )
    initial_nineteen = fixed_observation_initial_noise_seed(
        episode_index=205,
        frame_index=74,
        sample_ordinal=19,
    )
    e1a_reverse = fixed_observation_e1a_reverse_seed(
        episode_index=205,
        frame_index=74,
    )
    e1b_reverse = [
        fixed_observation_e1b_reverse_seed(
            episode_index=205,
            frame_index=74,
            sample_ordinal=sample_ordinal,
        )
        for sample_ordinal in range(20)
    ]

    assert initial_zero == 2434392847
    assert initial_nineteen == 410941779
    assert e1a_reverse == 3761541897
    assert e1b_reverse[0] == 435344550
    assert e1b_reverse[-1] == 2816169923
    assert len(set(e1b_reverse)) == 20
    assert initial_zero != fixed_observation_initial_noise_seed(
        episode_index=205,
        frame_index=75,
        sample_ordinal=0,
    )


def test_e1_aggregate_metrics_use_ratio_of_protocol_means_and_null_zero_ratio() -> None:
    def row(protocol: str, trajectory: float, endpoint: float) -> dict[str, object]:
        return {
            "sampling_protocol": protocol,
            "normalized_trajectory_pairwise_rms": trajectory,
            "normalized_endpoint_pairwise_distance": endpoint,
            "denormalized_trajectory_pairwise_rms": trajectory * 10,
            "denormalized_endpoint_pairwise_distance": endpoint * 10,
            "repeat_max_error": 0.0,
        }

    metrics = _aggregate_e1_metrics(
        [
            row("initial_noise_only", 0.0, 2.0),
            row("initial_noise_only", 0.0, 4.0),
            row("full_sampling_seed", 4.0, 6.0),
            row("full_sampling_seed", 8.0, 12.0),
        ],
        num_candidates=2,
        num_action_samples=20,
        bc_repeat_max_error=0.0,
    )

    assert metrics["initial_noise_only_mean_normalized_endpoint_pairwise_distance"] == 3.0
    assert metrics["full_sampling_seed_mean_normalized_endpoint_pairwise_distance"] == 9.0
    assert (
        metrics[
            "full_sampling_seed_to_initial_noise_only_normalized_endpoint_pairwise_distance_ratio"
        ]
        == 3.0
    )
    assert (
        metrics[
            "full_sampling_seed_to_initial_noise_only_normalized_trajectory_pairwise_rms_ratio"
        ]
        is None
    )


def test_metric_markdown_renders_null_ratio_as_na(tmp_path: Path) -> None:
    write_metrics(
        tmp_path,
        {"diversity_ratio": None},
        comments={"diversity_ratio": "E1b divided by E1a"},
    )

    assert json.loads((tmp_path / "metrics.json").read_text())["diversity_ratio"] is None
    assert "| `diversity_ratio` | N/A |" in (tmp_path / "metrics.md").read_text()


def test_report_rejects_legacy_e1_schema_with_rerun_instruction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--overwrite"):
        _validate_e1_schema(
            tmp_path,
            {"schema_version": 1, "experiment_schema": "legacy"},
        )


def test_report_rejects_legacy_e4_sampling_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="produce E3 evidence"):
        _validate_e3_schema(
            tmp_path,
            {"schema_version": 1, "experiment": "E4"},
        )


def test_e1_schema_two_rows_enforce_paired_noise_and_reverse_seed_contract() -> None:
    candidate = {
        "candidate_id": "case",
        "candidate_ordinal": 0,
        "episode_index": 205,
        "frame_index": 74,
    }
    action_rows: list[dict[str, object]] = []
    for repeat in range(3):
        for step in range(8):
            action_rows.append(
                {
                    "candidate_id": "case",
                    "sampling_protocol": "bc_deterministic",
                    "sample_ordinal": repeat,
                    "step": step,
                }
            )
    e1a_reverse = fixed_observation_e1a_reverse_seed(
        episode_index=205,
        frame_index=74,
    )
    for sample in range(20):
        initial_seed = fixed_observation_initial_noise_seed(
            episode_index=205,
            frame_index=74,
            sample_ordinal=sample,
        )
        for protocol, reverse_seed in (
            ("initial_noise_only", e1a_reverse),
            (
                "full_sampling_seed",
                fixed_observation_e1b_reverse_seed(
                    episode_index=205,
                    frame_index=74,
                    sample_ordinal=sample,
                ),
            ),
        ):
            for step in range(8):
                action_rows.append(
                    {
                        "candidate_id": "case",
                        "sampling_protocol": protocol,
                        "sample_ordinal": sample,
                        "step": step,
                        "initial_noise_seed": initial_seed,
                        "initial_noise_sha256": f"tensor-{sample}",
                        "initial_noise_file_sha256": f"file-{sample}",
                        "reverse_noise_seed": reverse_seed,
                        "reverse_generator_state_sha256": f"generator-{reverse_seed}",
                    }
                )

    _validate_e1_rows(
        action_rows,
        [
            {"candidate_id": "case", "sampling_protocol": "initial_noise_only"},
            {"candidate_id": "case", "sampling_protocol": "full_sampling_seed"},
        ],
        candidates=[candidate],
        num_action_samples=20,
        n_action_steps=8,
        label="Offline",
    )
    assert len(action_rows) == 344

    broken = [dict(row) for row in action_rows]
    target = next(
        row
        for row in broken
        if row["sampling_protocol"] == "full_sampling_seed"
        and row["sample_ordinal"] == 1
    )
    target["initial_noise_sha256"] = "not-paired"
    with pytest.raises(RuntimeError, match="not paired|inconsistent"):
        _validate_e1_rows(
            broken,
            [
                {"candidate_id": "case", "sampling_protocol": "initial_noise_only"},
                {"candidate_id": "case", "sampling_protocol": "full_sampling_seed"},
            ],
            candidates=[candidate],
            num_action_samples=20,
            n_action_steps=8,
            label="Offline",
        )


def test_explicit_noise_is_bit_reproducible() -> None:
    first = make_initial_noise(
        seed=123,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )
    second = make_initial_noise(
        seed=123,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )
    different = make_initial_noise(
        seed=124,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )

    assert torch.equal(first.tensor, second.tensor)
    assert first.tensor_sha256 == second.tensor_sha256
    assert not torch.equal(first.tensor, different.tensor)


def test_fixed_observation_dataset_item_gets_an_explicit_batch_dimension() -> None:
    policy = SimpleNamespace(config=SimpleNamespace(input_features={OBS_STATE: object()}))
    result = _observation_only({OBS_STATE: torch.zeros(2, 2)}, policy)

    assert result[OBS_STATE].shape == (1, 2, 2)


def test_fixed_observation_candidates_are_ordered_by_validated_ordinal() -> None:
    candidates = [
        {"candidate_id": "second", "candidate_ordinal": 1},
        {"candidate_id": "first", "candidate_ordinal": 0},
    ]

    ordered = _order_and_validate_candidates(candidates)

    assert [candidate["candidate_id"] for candidate in ordered] == ["first", "second"]


@pytest.mark.parametrize(
    "ordinals",
    [
        [0, 0],
        [0, 2],
    ],
)
def test_fixed_observation_candidate_ordinals_reject_ambiguous_timelines(
    ordinals: list[int],
) -> None:
    candidates = [
        {"candidate_id": f"candidate_{index}", "candidate_ordinal": ordinal}
        for index, ordinal in enumerate(ordinals)
    ]

    with pytest.raises(ValueError, match="candidate_ordinal"):
        _order_and_validate_candidates(candidates)


def test_formal_fixed_observation_eval_requires_twenty_diffusion_samples(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="exactly 20"):
        evaluate_fixed_observations(
            bc_policy_path=tmp_path / "bc",
            diffusion_policy_path=tmp_path / "diffusion",
            candidates_path=tmp_path / "candidates.json",
            dataset_root=tmp_path / "dataset",
            output_dir=tmp_path / "output",
            num_action_samples=3,
            train_seed=1000,
            label="Offline",
            overwrite=False,
        )


def test_controlled_ddpm_sampling_and_scheduler_replacement() -> None:
    torch.manual_seed(7)
    policy = DiffusionPolicy(_diffusion_config()).eval()
    batch = {
        OBS_STATE: torch.randn(1, 2, 2),
        OBS_ENV_STATE: torch.randn(1, 2, 2),
    }
    noise = make_initial_noise(
        seed=9,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )
    canonical_noise = noise.tensor.clone()
    before = state_dict_hashes(policy.state_dict())
    first = controlled_sample(policy, batch, initial_noise=noise.tensor, reverse_seed=10)
    repeated = controlled_sample(policy, batch, initial_noise=noise.tensor, reverse_seed=10)

    assert first.full_horizon.shape == (1, 16, 2)
    assert first.actions.shape == (1, 8, 2)
    assert torch.equal(first.actions, repeated.actions)
    assert torch.equal(noise.tensor, canonical_noise)
    assert torch.equal(first.actions, first.full_horizon[:, 1:9])

    replace_scheduler(policy, scheduler_name="DDIM", num_inference_steps=2)
    assert before == state_dict_hashes(policy.state_dict())
    ddim = controlled_sample(policy, batch, initial_noise=noise.tensor, reverse_seed=10)
    assert ddim.scheduler == "DDIM"
    assert ddim.num_inference_steps == 2


def test_bc_chunk_uses_clean_horizon_mse_and_exact_execution_slice() -> None:
    torch.manual_seed(11)
    policy = BCChunkPolicy(_bc_config()).eval()
    batch = {
        OBS_STATE: torch.randn(2, 2, 2),
        OBS_ENV_STATE: torch.randn(2, 2, 2),
        ACTION: torch.randn(2, 16, 2),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }

    prediction = policy.predict_full_action_horizon(batch)
    loss, metrics = policy(batch)

    assert prediction.shape == (2, 16, 2)
    assert torch.allclose(loss, torch.mean((prediction - batch[ACTION]) ** 2))
    assert metrics["clean_action_mse"] == pytest.approx(float(loss.detach()))
    assert policy.bc_config.fixed_timestep == 0


def test_bc_and_diffusion_have_strictly_compatible_state_dicts() -> None:
    torch.manual_seed(13)
    diffusion = DiffusionPolicy(_diffusion_config())
    bc = BCChunkPolicy(_bc_config())

    missing, unexpected = bc.load_state_dict(diffusion.state_dict(), strict=True)

    assert not missing
    assert not unexpected
    assert state_dict_hashes(bc.state_dict()) == state_dict_hashes(diffusion.state_dict())


def test_replan_index_advances_only_after_eight_actions_are_consumed() -> None:
    policy = BCChunkPolicy(_bc_config()).eval()
    executor = ExplicitChunkExecutor(policy, environment_seed=2000, policy_seed=0)
    current = {
        OBS_STATE: torch.randn(1, 2),
        OBS_ENV_STATE: torch.randn(1, 2),
    }

    for _ in range(8):
        executor.select_action(current)
    assert executor.replan_index == 1
    assert len(executor.replans) == 1

    executor.select_action(current)
    assert executor.replan_index == 2
    assert len(executor.replans) == 2


def test_action_and_rollout_diversity_metrics() -> None:
    actions = np.asarray(
        [
            [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0]],
            [[10.0, 10.0], [10.0, 20.0], [10.0, 30.0]],
        ]
    )
    summary = summarize_action_diversity(actions, agent_position=np.asarray([5.0, 5.0]))
    first = resample_agent_block_trajectory(
        np.asarray([[0.0, 0.0], [100.0, 0.0]]),
        np.asarray([[10.0, 10.0, 3.1], [20.0, 10.0, -3.1]]),
        num_points=100,
    )
    second = first.copy()
    second[:, 0] += 100.0
    matrix = trajectory_distance_matrix([first, second])

    assert summary.trajectory_pairwise_rms > 0
    assert summary.endpoint_pairwise_distance > 0
    assert trajectory_mode_distance(first, first) == 0
    assert matrix[0, 1] > 0
    labels = complete_linkage_mode_labels(matrix, threshold=0.05)
    assert set(labels.tolist()) == {0, 1}


def test_statistical_intervals_and_atomic_output(tmp_path: Path) -> None:
    interval = wilson_interval(8, 10)
    clustered = environment_cluster_bootstrap(
        {1: [1.0, 0.0], 2: [1.0, 1.0]},
        iterations=100,
        seed=5,
    )
    output = tmp_path / "result"

    with AtomicOutputDirectory(output) as temporary:
        (temporary / "value.txt").write_text("complete\n", encoding="utf-8")

    assert interval.estimate == pytest.approx(0.8)
    assert interval.lower < interval.estimate < interval.upper
    assert clustered.estimate == pytest.approx(0.75)
    assert (output / "value.txt").is_file()
    assert not (output / ".incomplete").exists()


def test_drop_seven_tail_semantics_keep_execution_slice_valid() -> None:
    counts = padding_counts_for_episode(100)

    assert len(counts) == 93
    assert counts[0] == 1
    assert counts[-1] == 7
    assert int(np.max(counts)) == 7


def test_pusht_replay_and_pymunk_compatibility_gate() -> None:
    pymunk_major = int(metadata.version("pymunk").split(".", maxsplit=1)[0])

    result = check_environment_replay(
        environment_seed=2000,
        actions=np.asarray([[256.0, 256.0], [260.0, 260.0]], dtype=np.float32),
    )

    assert pymunk_major < 7
    assert result["reproduced"] is True
    assert result["pixels_exact"] is True


def test_two_step_controlled_diffusion_rollout() -> None:
    inputs = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
    }
    outputs = {ACTION: PolicyFeature(FeatureType.ACTION, (2,))}
    policy = DiffusionPolicy(
        DiffusionConfig(
            input_features=inputs,
            output_features=outputs,
            device="cpu",
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            crop_shape=(84, 84),
            pretrained_backbone_weights=None,
            spatial_softmax_num_keypoints=8,
            down_dims=(16, 32),
            n_groups=8,
            num_train_timesteps=100,
            num_inference_steps=1,
            push_to_hub=False,
        )
    ).eval()

    result = run_rollout(
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda action: action,
        environment_seed=2000,
        policy_seed=0,
        record_frames=False,
        episode_length=2,
    )

    assert result.episode_length == 2
    assert len(result.replans) == 1
    assert result.agent_path.shape == (3, 2)
    assert result.block_path.shape == (3, 3)


def test_report_rejects_smoke_evidence(tmp_path: Path) -> None:
    roots = {name: tmp_path / name for name in ("e0", "e1", "e2", "e3")}
    for root in roots.values():
        root.mkdir()
    labels = {"e0": "Offline", "e1": "Smoke", "e2": "Closed-loop", "e3": "Closed-loop"}
    for name, root in roots.items():
        (root / "run_manifest.json").write_text(
            json.dumps({"label": labels[name]}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Smoke outputs"):
        make_report(
            e0_root=roots["e0"],
            e1_root=roots["e1"],
            e2_root=roots["e2"],
            e3_root=roots["e3"],
            output_dir=tmp_path / "report",
            overwrite=False,
        )
