from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from robot_practice.experiments.pusht_diffusion import development_comparison
from robot_practice.experiments.pusht_diffusion.development_recordings import (
    TaskRecordingMetadata,
)
from robot_practice.experiments.pusht_diffusion.rollout import (
    ReplanRecord,
    RolloutResult,
)


def _result(model: str, seed: int, success: bool) -> RolloutResult:
    agent_start = np.asarray([float(seed), float(seed + 1)])
    block_start = np.asarray([float(seed + 2), float(seed + 3), 0.25])
    return RolloutResult(
        model=model,
        environment_seed=seed,
        policy_seed=0,
        success=success,
        sum_reward=2.0 if success else 0.5,
        max_reward=1.0 if success else 0.4,
        final_reward=1.0 if success else 0.1,
        episode_length=2,
        agent_path=np.stack([agent_start, agent_start + 1, agent_start + 2]),
        block_path=np.stack([block_start, block_start + 1, block_start + 2]),
        action_targets=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        rewards=np.asarray([0.0, 1.0 if success else 0.1]),
        frames=(np.zeros((8, 8, 3), dtype=np.uint8),) * 3,
        replans=(
            ReplanRecord(
                replan_index=0,
                environment_seed=seed,
                policy_seed=0,
                initial_noise_seed=seed if model == "diffusion" else None,
                initial_noise_sha256="noise" if model == "diffusion" else None,
                reverse_noise_seed=seed + 1 if model == "diffusion" else None,
                reverse_generator_state_sha256=("generator" if model == "diffusion" else None),
                scheduler="DDPM" if model == "diffusion" else None,
                num_inference_steps=100 if model == "diffusion" else None,
                latency_ms=1.5,
                normalized_chunk=np.zeros((8, 2), dtype=np.float32),
            ),
        ),
        condition_id=development_comparison.EXPERIMENT_ID,
        scheduler="DDPM" if model == "diffusion" else None,
        num_inference_steps=100 if model == "diffusion" else None,
    )


def test_classify_results_covers_all_four_categories() -> None:
    seeds = (1000, 1001, 1002, 1003)
    bc_results = [
        _result("bc_chunk", 1000, True),
        _result("bc_chunk", 1001, False),
        _result("bc_chunk", 1002, False),
        _result("bc_chunk", 1003, True),
    ]
    diffusion_results = [
        _result("diffusion", 1000, False),
        _result("diffusion", 1001, True),
        _result("diffusion", 1002, False),
        _result("diffusion", 1003, True),
    ]

    categories, cases = development_comparison.classify_results(
        bc_results,
        diffusion_results,
        environment_seeds=seeds,
    )

    assert categories == {
        "bc_pass_diffusion_fail": [1000],
        "diffusion_pass_bc_fail": [1001],
        "both_fail": [1002],
        "both_pass": [1003],
    }
    assert [case["category"] for case in cases] == [
        "bc_pass_diffusion_fail",
        "diffusion_pass_bc_fail",
        "both_fail",
        "both_pass",
    ]
    assert cases[0]["initial_state"]["agent_position"] == [1000.0, 1001.0]


def test_development_comparison_writes_portable_result_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selected_checkpoints.json"
    checkpoint_paths = {
        "bc_chunk": tmp_path / "bc_chunk",
        "diffusion": tmp_path / "diffusion",
    }
    checkpoint_ids = {}
    for model, checkpoint_path in checkpoint_paths.items():
        checkpoint_path.mkdir()
        model_bytes = f"{model}-weights".encode()
        (checkpoint_path / "model.safetensors").write_bytes(model_bytes)
        checkpoint_ids[model] = hashlib.sha256(model_bytes).hexdigest()
    selection_path.write_text(
        json.dumps(
            {
                "bc_chunk": {
                    "model": "bc_chunk",
                    "step": 150000,
                    "checkpoint_path": str(checkpoint_paths["bc_chunk"]),
                    "checkpoint_id": checkpoint_ids["bc_chunk"],
                    "train_seed": 1000,
                    "dataset_revision": development_comparison.DATASET_REVISION,
                    "development_environment_seed_count": 20,
                    "num_rollouts": 20,
                    "scheduler": None,
                    "num_inference_steps": None,
                    "closed_loop_success_rate": 0.55,
                    "mean_max_reward": 0.86,
                },
                "diffusion": {
                    "model": "diffusion",
                    "step": 175000,
                    "checkpoint_path": str(checkpoint_paths["diffusion"]),
                    "checkpoint_id": checkpoint_ids["diffusion"],
                    "train_seed": 1000,
                    "dataset_revision": development_comparison.DATASET_REVISION,
                    "development_environment_seed_count": 20,
                    "num_rollouts": 20,
                    "scheduler": "DDPM",
                    "num_inference_steps": 100,
                    "closed_loop_success_rate": 0.8,
                    "mean_max_reward": 0.88,
                },
            }
        ),
        encoding="utf-8",
    )
    seeds = tuple(range(1000, 1020))
    monkeypatch.setattr(
        development_comparison,
        "load_split",
        lambda: SimpleNamespace(development_environment_seeds=seeds),
    )

    class FakeBCPolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(type="bc_chunk")

    class FakeDiffusionPolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(type="diffusion")

    monkeypatch.setattr(development_comparison, "BCChunkPolicy", FakeBCPolicy)
    monkeypatch.setattr(
        development_comparison,
        "DiffusionPolicy",
        FakeDiffusionPolicy,
    )

    load_order: list[str] = []

    def fake_load(path: Path, **_kwargs: object) -> SimpleNamespace:
        assert _kwargs["device"] == "cpu"
        model = "bc_chunk" if Path(path) == checkpoint_paths["bc_chunk"] else "diffusion"
        load_order.append(model)
        policy = FakeBCPolicy() if model == "bc_chunk" else FakeDiffusionPolicy()
        return SimpleNamespace(
            policy=policy,
            config=SimpleNamespace(
                n_obs_steps=2,
                horizon=16,
                n_action_steps=8,
            ),
            preprocessor=object(),
            postprocessor=object(),
        )

    monkeypatch.setattr(
        development_comparison,
        "load_policy_with_processors",
        fake_load,
    )
    monkeypatch.setattr(
        development_comparison,
        "replace_scheduler",
        lambda *_args, **_kwargs: None,
    )

    def fake_rollout(**kwargs: object) -> RolloutResult:
        policy = kwargs["policy"]
        seed = kwargs["environment_seed"]
        assert isinstance(seed, int)
        model = policy.config.type
        remainder = seed % 4
        success = remainder in (0, 1) if model == "bc_chunk" else remainder in (0, 2)
        return _result(model, seed, success)

    monkeypatch.setattr(development_comparison, "run_rollout", fake_rollout)

    class FakeRecorder:
        def __init__(self, recordings_directory: Path) -> None:
            self.recordings_directory = Path(recordings_directory)
            self.recordings_directory.mkdir()
            for model in ("bc_chunk", "diffusion"):
                (self.recordings_directory / model).mkdir()
            self.results: list[RolloutResult] = []

        def add(
            self,
            result: RolloutResult,
            metadata: TaskRecordingMetadata,
        ) -> None:
            assert metadata.model == result.model
            assert metadata.environment_seed == result.environment_seed
            self.results.append(result)
            path = self.recordings_directory / metadata.model / metadata.recording_filename
            path.write_bytes(f"{metadata.model}-{metadata.environment_seed}".encode())

        def finalize(
            self,
            *,
            comparison_path: Path,
        ) -> dict[str, tuple[Path, ...]]:
            assert comparison_path.is_file()
            return {
                model: tuple(sorted((self.recordings_directory / model).glob("*.rrd")))
                for model in ("bc_chunk", "diffusion")
            }

    monkeypatch.setattr(development_comparison, "PairedTaskRecordings", FakeRecorder)

    def fake_save_video(_result: RolloutResult, destination: Path) -> Path:
        Path(destination).write_bytes(b"video")
        return Path(destination)

    monkeypatch.setattr(development_comparison, "save_rollout_video", fake_save_video)
    monkeypatch.setattr(
        development_comparison,
        "collect_runtime_provenance",
        lambda _root: {
            "schema_version": 1,
            "repository": {"commit": "commit", "dirty": False},
            "runtime": {"gpu": None},
        },
    )
    output = tmp_path / "comparison"

    result = development_comparison.compare_development_checkpoints(
        selection_path=selection_path,
        output_dir=output,
        device="cpu",
        overwrite=False,
    )

    assert result == output.resolve()
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["category_counts"] == {
        "bc_pass_diffusion_fail": 5,
        "diffusion_pass_bc_fail": 5,
        "both_fail": 5,
        "both_pass": 5,
    }
    assert comparison["summaries"]["bc_chunk"]["success_count"] == 10
    assert comparison["summaries"]["diffusion"]["success_count"] == 10
    assert comparison["num_rollouts_per_policy"] == 20
    assert len(comparison["cases"]) == 20
    assert comparison["cases"][0]["bc_chunk"]["video_path"] == "videos/bc_chunk_env_1000.mp4"
    assert comparison["cases"][0]["diffusion"]["rerun_entity_path"] == "/rollout"
    assert comparison["cases"][0]["diffusion"]["rerun_recording_path"] == (
        "recordings/diffusion/env_1000.rrd"
    )
    rollouts = pq.read_table(output / "rollouts.parquet")
    assert rollouts.num_rows == 40
    assert set(rollouts.column("num_rollouts").to_pylist()) == {20}
    assert pq.read_table(output / "replans.parquet").num_rows == 40
    assert len(list((output / "videos").glob("*.mp4"))) == 40
    assert not (output / "recording.rrd").exists()
    assert not (output / ".rerun_recordings").exists()
    assert len(list((output / "recordings" / "bc_chunk").glob("*.rrd"))) == 20
    assert len(list((output / "recordings" / "diffusion").glob("*.rrd"))) == 20
    assert (output / "recordings" / "bc_chunk" / "env_1000.rrd").read_bytes() == b"bc_chunk-1000"
    assert (output / "recordings" / "diffusion" / "env_1019.rrd").read_bytes() == b"diffusion-1019"
    assert (output / "metrics.json").is_file()
    assert (output / "metrics.md").is_file()
    assert (output / "run_manifest.json").is_file()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rerun_recording_directories"] == {
        "bc_chunk": "recordings/bc_chunk",
        "diffusion": "recordings/diffusion",
    }
    assert manifest["rerun_recordings_per_model"] == 20
    assert not (output / ".incomplete").exists()
    assert load_order == ["bc_chunk", "diffusion"]


def test_selection_must_come_from_full_development_protocol(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bc_chunk"
    checkpoint.mkdir()
    weights = b"weights"
    (checkpoint / "model.safetensors").write_bytes(weights)
    entry = {
        "model": "bc_chunk",
        "step": 150000,
        "checkpoint_path": str(checkpoint),
        "checkpoint_id": hashlib.sha256(weights).hexdigest(),
        "train_seed": 1000,
        "dataset_revision": development_comparison.DATASET_REVISION,
        "development_environment_seed_count": 5,
        "num_rollouts": 5,
        "scheduler": None,
        "num_inference_steps": None,
        "closed_loop_success_rate": 0.6,
        "mean_max_reward": 0.8,
    }

    with pytest.raises(
        ValueError,
        match=r"development_environment_seed_count must be 20",
    ):
        development_comparison._selection_entry(
            {"bc_chunk": entry},
            model_name="bc_chunk",
        )
