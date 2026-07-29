from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_practice.experiments.pusht_diffusion import rerun_visualization
from robot_practice.experiments.pusht_diffusion.report import _validate_e3_schema
from robot_practice.experiments.pusht_diffusion.sampling_steps_eval import (
    _aggregate_parallel_latency_rows,
    _render_progress_bar,
    _validate_num_gpus,
    partition_environment_seeds,
    partition_latency_iterations,
)


def test_e3_environment_seed_shards_are_disjoint_and_complete() -> None:
    seeds = tuple(range(2000, 2010))

    shards = partition_environment_seeds(seeds, num_gpus=3)

    assert shards == (
        (2000, 2003, 2006, 2009),
        (2001, 2004, 2007),
        (2002, 2005, 2008),
    )
    flattened = [seed for shard in shards for seed in shard]
    assert sorted(flattened) == list(seeds)
    assert len(flattened) == len(set(flattened))


def test_e3_latency_iterations_are_distributed_as_one_aggregate_budget() -> None:
    assert partition_latency_iterations(200, num_gpus=8) == (25,) * 8
    assert partition_latency_iterations(10, num_gpus=3) == (4, 3, 3)
    assert partition_latency_iterations(2, num_gpus=8) == (1, 1)


@pytest.mark.parametrize(
    ("measurement_iterations", "num_gpus"),
    [(0, 1), (-1, 1), (10, 0), (10, -1), (True, 1), (10, True)],
)
def test_e3_latency_iteration_partition_rejects_invalid_counts(
    measurement_iterations: int,
    num_gpus: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        partition_latency_iterations(
            measurement_iterations,
            num_gpus=num_gpus,
        )


def test_e3_parallel_latency_rows_are_reindexed_and_summarized() -> None:
    rows = []
    for gpu_index in range(2):
        for condition_id in ("ddim_005", "ddpm_100"):
            for latency_kind in ("model_only", "end_to_end"):
                for local_iteration in range(2):
                    rows.append(
                        {
                            "condition_id": condition_id,
                            "latency_kind": latency_kind,
                            "latency_gpu_index": gpu_index,
                            "iteration": local_iteration,
                            "latency_ms": float(
                                10 * gpu_index
                                + local_iteration
                                + (5 if latency_kind == "end_to_end" else 0)
                            ),
                            "peak_memory_bytes": 100 + gpu_index,
                        }
                    )

    aggregated, summaries = _aggregate_parallel_latency_rows(
        rows,
        conditions=(("DDIM", 5), ("DDPM", 100)),
        measurement_iterations=4,
        n_action_steps=8,
    )

    assert len(aggregated) == 16
    assert [
        row["iteration"]
        for row in aggregated
        if row["condition_id"] == "ddim_005" and row["latency_kind"] == "model_only"
    ] == [0, 1, 2, 3]
    assert summaries["ddim_005"]["model_only_iterations"] == 4
    assert summaries["ddim_005"]["model_only_peak_memory_bytes"] == 101
    assert summaries["ddim_005"]["amortized_end_to_end_p95_ms"] == pytest.approx(
        summaries["ddim_005"]["end_to_end_p95_ms"] / 8
    )


def test_e3_latency_progress_bar_reaches_total(capsys: pytest.CaptureFixture[str]) -> None:
    _render_progress_bar(completed=0, total=4, width=4)
    _render_progress_bar(completed=4, total=4, width=4)

    assert "E3 latency [----] 0/4\r" in capsys.readouterr().out


def test_e3_condition_titles_remove_numeric_zero_padding() -> None:
    assert rerun_visualization._e3_condition_title("ddim_005") == "DDIM-5"
    assert rerun_visualization._e3_condition_title("ddpm_100") == "DDPM-100"


@pytest.mark.parametrize("num_gpus", [0, -1, True])
def test_e3_environment_seed_shards_reject_invalid_gpu_counts(num_gpus: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        partition_environment_seeds((2000, 2001), num_gpus=num_gpus)


def test_e3_environment_seed_shards_reject_more_gpus_than_environments() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        partition_environment_seeds((2000, 2001), num_gpus=3)


def test_e3_multi_gpu_validation_checks_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "robot_practice.experiments.pusht_diffusion.sampling_steps_eval.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "robot_practice.experiments.pusht_diffusion.sampling_steps_eval.torch.cuda.device_count",
        lambda: 2,
    )

    with pytest.raises(RuntimeError, match="only 2 CUDA device"):
        _validate_num_gpus(num_gpus=3, num_environment_seeds=50)


def test_e3_scene_recording_uses_one_synchronized_timeline_and_active_blueprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "env_2000.rrd"
    calls: list[dict[str, object]] = []
    blueprints: list[dict[str, object]] = []

    class Recording:
        def save(self, path: Path) -> None:
            path.write_bytes(b"rrd")

        def send_recording_name(self, name: str) -> None:
            calls.append({"name": name})

        def log(self, path: str, _value: object, **kwargs: object) -> None:
            calls.append({"path": path, **kwargs})

        def flush(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

    recording = Recording()
    blueprint = object()
    timeline_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        rerun_visualization.rr,
        "RecordingStream",
        lambda *_args, **_kwargs: recording,
    )
    monkeypatch.setattr(
        rerun_visualization,
        "build_e3_sampling_steps_blueprint",
        lambda _condition_ids: blueprint,
    )
    monkeypatch.setattr(
        rerun_visualization.rr,
        "send_blueprint",
        lambda value, **kwargs: blueprints.append({"blueprint": value, **kwargs}),
    )

    def log_rollout(
        _recording: object,
        rollout: object,
        **kwargs: object,
    ) -> int:
        timeline_calls.append(dict(kwargs))
        return int(rollout.agent_path.shape[0])

    monkeypatch.setattr(rerun_visualization, "log_rollout_to_stream", log_rollout)
    writer = rerun_visualization.E3SceneRecording(
        destination,
        environment_seed=2000,
        condition_ids=("ddim_005", "ddpm_100"),
    )
    for condition_id in ("ddim_005", "ddpm_100"):
        writer.add(
            SimpleNamespace(
                environment_seed=2000,
                condition_id=condition_id,
                agent_path=np.zeros((3, 2)),
            )
        )
    writer.close()

    assert destination.read_bytes() == b"rrd"
    assert blueprints == [
        {
            "blueprint": blueprint,
            "make_active": True,
            "make_default": True,
            "recording": recording,
        }
    ]
    assert [call["global_frame_start"] for call in timeline_calls] == [0, 0]
    assert [call["entity_root"] for call in timeline_calls] == [
        "/conditions/ddim_005",
        "/conditions/ddpm_100",
    ]
    assert calls[0] == {"name": "E3 | env 2000 | 2 sampling conditions"}


def test_e3_scene_recording_rejects_incomplete_condition_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Recording:
        def save(self, path: Path) -> None:
            path.write_bytes(b"rrd")

        def send_recording_name(self, _name: str) -> None:
            pass

        def log(self, _path: str, _value: object, **_kwargs: object) -> None:
            pass

        def flush(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

    monkeypatch.setattr(
        rerun_visualization.rr,
        "RecordingStream",
        lambda *_args, **_kwargs: Recording(),
    )
    monkeypatch.setattr(
        rerun_visualization.rr,
        "send_blueprint",
        lambda *_args, **_kwargs: None,
    )
    writer = rerun_visualization.E3SceneRecording(
        tmp_path / "env_2000.rrd",
        environment_seed=2000,
        condition_ids=("ddim_005", "ddpm_100"),
    )

    with pytest.raises(RuntimeError, match="missing conditions"):
        writer.close()


def test_report_accepts_exact_e3_scene_recording_schema(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    for environment_seed in range(2000, 2050):
        (recordings / f"env_{environment_seed:04d}.rrd").write_bytes(b"rrd")
    manifest = {
        "schema_version": 2,
        "experiment_schema": "pusht-e3-sampling-steps-v2",
        "experiment": "E3",
        "conditions": [
            *[
                {"scheduler": "DDIM", "num_inference_steps": steps}
                for steps in (5, 10, 20, 50, 100)
            ],
            {"scheduler": "DDPM", "num_inference_steps": 100},
        ],
        "environment_seeds": list(range(2000, 2050)),
        "num_rollouts": 300,
        "recordings": {
            "count": 50,
            "conditions_per_recording": 6,
            "all_rollouts_include_simulator_frames": True,
        },
    }

    _validate_e3_schema(tmp_path, manifest)


def test_report_rejects_legacy_single_recording_e3_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="legacy or incompatible"):
        _validate_e3_schema(
            tmp_path,
            {
                "schema_version": 1,
                "experiment": "E3",
            },
        )
