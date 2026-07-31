from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_practice.experiments.pusht_diffusion import (
    checkpoint_selection,
    e5_oracle_audit,
    e5_oracle_mode,
)
from robot_practice.experiments.pusht_diffusion.e5_checkpoint_selection import (
    _separate_rollout_rows,
)
from robot_practice.experiments.pusht_diffusion.e5_oracle_audit import (
    MODE_BC,
    MODE_DIFFUSION,
    _selected_rows,
    wilson_interval,
)
from robot_practice.experiments.pusht_diffusion.e5_oracle_mode import (
    PUSHT_T_GEOMETRIC_CENTROID_LOCAL,
    body_pose_from_geometric_pose,
    wrapped_angle_difference,
)
from robot_practice.experiments.pusht_diffusion.provenance import sha256_file


def _write_completed_selection(
    selection_dir: Path,
    selected: dict[str, object],
) -> None:
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "experiment": "e5_oracle_mode_development_checkpoint_selection",
            }
        ),
        encoding="utf-8",
    )
    (selection_dir / "selected_checkpoints.json").write_text(
        json.dumps(selected),
        encoding="utf-8",
    )


def _repository_checkpoint_path(root: Path, model: str, step: int = 25_000) -> Path:
    return (
        root
        / "outputs"
        / "pusht_diffusion"
        / "pilot"
        / "e5_mode_conditioned"
        / "training"
        / model
        / "checkpoints"
        / f"{step:06d}"
        / "pretrained_model"
    )


def test_body_pose_conversion_preserves_geometric_center() -> None:
    center = np.asarray([170.25, 412.5])
    body_origin, body_angle = body_pose_from_geometric_pose(center, -2.3)
    rotation = np.asarray(
        [
            [math.cos(body_angle), -math.sin(body_angle)],
            [math.sin(body_angle), math.cos(body_angle)],
        ]
    )
    rebuilt_center = np.asarray(body_origin) + rotation @ PUSHT_T_GEOMETRIC_CENTROID_LOCAL

    assert rebuilt_center == pytest.approx(center)
    assert wrapped_angle_difference(body_angle, -2.3 + math.pi / 2.0) == pytest.approx(0.0)


def test_held_out_rollout_uses_explicit_commanded_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakePreprocessor:
        def __init__(
            self,
            _base: object,
            *,
            prototype_id: int,
            num_prototypes: int,
            device: torch.device,
        ) -> None:
            captured["prototype_id"] = prototype_id
            captured["num_prototypes"] = num_prototypes
            captured["device"] = device

    monkeypatch.setattr(e5_oracle_mode, "FixedPrototypePreprocessor", FakePreprocessor)
    monkeypatch.setattr(
        e5_oracle_mode,
        "get_device_from_parameters",
        lambda _policy: torch.device("cpu"),
    )
    monkeypatch.setattr(
        e5_oracle_mode,
        "run_rollout",
        lambda **kwargs: kwargs,
    )
    loaded = SimpleNamespace(
        policy=object(),
        preprocessor=object(),
        postprocessor=object(),
    )
    episode = SimpleNamespace(
        split="audit",
        episode_index=17,
        prototype_id=1,
    )

    result = e5_oracle_mode.run_oracle_mode_rollout(
        loaded,
        episode=episode,
        commanded_condition_id=3,
        selected_k=3,
        null_prototype_id=3,
        record_frames=False,
    )

    assert captured["prototype_id"] == 3
    assert captured["num_prototypes"] == 3
    assert result["condition_id"] == "audit_episode_000017_command_NULL_gt_P1"
    with pytest.raises(ValueError, match="Commanded condition"):
        e5_oracle_mode.run_oracle_mode_rollout(
            loaded,
            episode=episode,
            commanded_condition_id=4,
            selected_k=3,
            null_prototype_id=3,
            record_frames=False,
        )


def test_e5_checkpoint_tasks_keep_diffusion_first_and_use_mode_names(
    tmp_path: Path,
) -> None:
    tasks = checkpoint_selection._checkpoint_tasks(
        ((25_000, tmp_path / "bc"),),
        ((25_000, tmp_path / "diffusion"),),
        bc_model_name=MODE_BC,
        diffusion_model_name=MODE_DIFFUSION,
    )

    assert [(task.model_name, task.step) for task in tasks] == [
        (MODE_DIFFUSION, 25_000),
        (MODE_BC, 25_000),
    ]


def test_e5_selection_separates_checkpoint_and_episode_rows() -> None:
    checkpoints, rollouts = _separate_rollout_rows(
        [
            {
                "model": MODE_BC,
                "step": 25_000,
                "closed_loop_success_rate": 0.5,
                "_rollouts": [
                    {"model": MODE_BC, "step": 25_000, "episode_index": 1},
                    {"model": MODE_BC, "step": 25_000, "episode_index": 2},
                ],
            }
        ]
    )

    assert "_rollouts" not in checkpoints[0]
    assert [row["episode_index"] for row in rollouts] == [1, 2]


def test_selection_order_is_success_then_reward_then_earliest_step() -> None:
    rows = [
        {
            "model": MODE_BC,
            "step": 75_000,
            "closed_loop_success_rate": 0.8,
            "mean_max_reward": 0.6,
        },
        {
            "model": MODE_BC,
            "step": 50_000,
            "closed_loop_success_rate": 0.8,
            "mean_max_reward": 0.7,
        },
        {
            "model": MODE_BC,
            "step": 25_000,
            "closed_loop_success_rate": 0.8,
            "mean_max_reward": 0.7,
        },
    ]

    assert checkpoint_selection._select(rows, MODE_BC)["step"] == 25_000


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(11, 21)

    assert 0.0 <= lower <= 11 / 21 <= upper <= 1.0


def test_audit_rejects_incomplete_selection(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "running",
                "experiment": "e5_oracle_mode_development_checkpoint_selection",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not complete"):
        _selected_rows(tmp_path)


def test_audit_validates_both_selected_checkpoint_hashes(tmp_path: Path) -> None:
    selected = {}
    for model in (MODE_BC, MODE_DIFFUSION):
        policy = tmp_path / model
        policy.mkdir()
        weights = policy / "model.safetensors"
        weights.write_bytes(model.encode())
        selected[model] = {
            "model": model,
            "step": 25_000,
            "checkpoint_path": str(policy),
            "checkpoint_id": sha256_file(weights),
        }
    _write_completed_selection(tmp_path, selected)

    loaded = _selected_rows(tmp_path)

    assert set(loaded) == {MODE_BC, MODE_DIFFUSION}
    for model in (MODE_BC, MODE_DIFFUSION):
        assert loaded[model]["checkpoint_path_resolution"] == "recorded"
        assert loaded[model]["recorded_checkpoint_path"] == selected[model]["checkpoint_path"]


def test_audit_relocates_missing_recorded_checkpoints_to_repository(
    tmp_path: Path,
) -> None:
    selection_dir = tmp_path / "selection"
    repository_root = tmp_path / "repository"
    selection_dir.mkdir()
    selected = {}
    expected_paths = {}
    for model in (MODE_BC, MODE_DIFFUSION):
        policy = _repository_checkpoint_path(repository_root, model)
        policy.mkdir(parents=True)
        weights = policy / "model.safetensors"
        weights.write_bytes(model.encode())
        recorded = tmp_path / "missing-remote-repository" / model
        selected[model] = {
            "model": model,
            "step": 25_000,
            "checkpoint_path": str(recorded),
            "checkpoint_id": sha256_file(weights),
        }
        expected_paths[model] = policy.resolve()
    _write_completed_selection(selection_dir, selected)

    loaded = _selected_rows(selection_dir, repository_root=repository_root)
    resolution_summary = e5_oracle_audit._checkpoint_path_resolution_summary(loaded)

    for model in (MODE_BC, MODE_DIFFUSION):
        assert Path(loaded[model]["checkpoint_path"]) == expected_paths[model]
        assert loaded[model]["recorded_checkpoint_path"] == selected[model]["checkpoint_path"]
        assert loaded[model]["checkpoint_path_resolution"] == "repository_relative"
        assert resolution_summary[model] == {
            "recorded_checkpoint_path": selected[model]["checkpoint_path"],
            "checkpoint_path": str(expected_paths[model]),
            "checkpoint_path_resolution": "repository_relative",
            "checkpoint_id": selected[model]["checkpoint_id"],
        }


def test_audit_uses_matching_repository_checkpoint_after_recorded_hash_mismatch(
    tmp_path: Path,
) -> None:
    selection_dir = tmp_path / "selection"
    repository_root = tmp_path / "repository"
    selected = {}
    for model in (MODE_BC, MODE_DIFFUSION):
        recorded = tmp_path / "recorded" / model
        recorded.mkdir(parents=True)
        (recorded / "model.safetensors").write_bytes(b"wrong")
        repository = _repository_checkpoint_path(repository_root, model)
        repository.mkdir(parents=True)
        weights = repository / "model.safetensors"
        weights.write_bytes(model.encode())
        selected[model] = {
            "model": model,
            "step": 25_000,
            "checkpoint_path": str(recorded),
            "checkpoint_id": sha256_file(weights),
        }
    _write_completed_selection(selection_dir, selected)

    loaded = _selected_rows(selection_dir, repository_root=repository_root)

    for model in (MODE_BC, MODE_DIFFUSION):
        assert loaded[model]["checkpoint_path_resolution"] == "repository_relative"
        assert (
            Path(loaded[model]["checkpoint_path"])
            == _repository_checkpoint_path(
                repository_root,
                model,
            ).resolve()
        )


def test_audit_rejects_repository_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    selection_dir = tmp_path / "selection"
    repository_root = tmp_path / "repository"
    selected = {}
    for model in (MODE_BC, MODE_DIFFUSION):
        repository = _repository_checkpoint_path(repository_root, model)
        repository.mkdir(parents=True)
        weights = repository / "model.safetensors"
        weights.write_bytes(b"wrong")
        selected[model] = {
            "model": model,
            "step": 25_000,
            "checkpoint_path": str(tmp_path / "missing" / model),
            "checkpoint_id": sha256_file(Path(__file__)),
        }
    _write_completed_selection(selection_dir, selected)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _selected_rows(selection_dir, repository_root=repository_root)


def test_audit_rejects_missing_recorded_and_repository_checkpoints(tmp_path: Path) -> None:
    selection_dir = tmp_path / "selection"
    selected = {
        model: {
            "model": model,
            "step": 25_000,
            "checkpoint_path": str(tmp_path / "missing-recorded" / model),
            "checkpoint_id": "0" * 64,
        }
        for model in (MODE_BC, MODE_DIFFUSION)
    }
    _write_completed_selection(selection_dir, selected)

    with pytest.raises(FileNotFoundError, match="missing model.safetensors"):
        _selected_rows(selection_dir, repository_root=tmp_path / "missing-repository")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "other_model", "must declare model"),
        ("step", True, "positive integer"),
        ("step", 25_000.0, "positive integer"),
        ("step", 0, "positive integer"),
        ("checkpoint_id", "A" * 64, "lowercase SHA256"),
        ("checkpoint_id", "0" * 63, "lowercase SHA256"),
    ],
)
def test_audit_rejects_invalid_selected_checkpoint_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    selection_dir = tmp_path / "selection"
    selected = {
        model: {
            "model": model,
            "step": 25_000,
            "checkpoint_path": str(tmp_path / "missing" / model),
            "checkpoint_id": "0" * 64,
        }
        for model in (MODE_BC, MODE_DIFFUSION)
    }
    selected[MODE_BC][field] = value
    _write_completed_selection(selection_dir, selected)

    with pytest.raises(ValueError, match=message):
        _selected_rows(selection_dir, repository_root=tmp_path / "repository")


def test_audit_rejects_invalid_top_level_model_name(tmp_path: Path) -> None:
    selected = {
        MODE_BC: {},
        "other_model": {},
    }
    _write_completed_selection(tmp_path, selected)

    with pytest.raises(ValueError, match="must contain exactly"):
        _selected_rows(tmp_path, repository_root=tmp_path / "repository")


def test_audit_repository_resolution_is_independent_of_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection_dir = tmp_path / "selection"
    repository_root = tmp_path / "repository"
    outside = tmp_path / "outside"
    outside.mkdir()
    selected = {}
    for model in (MODE_BC, MODE_DIFFUSION):
        policy = _repository_checkpoint_path(repository_root, model)
        policy.mkdir(parents=True)
        weights = policy / "model.safetensors"
        weights.write_bytes(model.encode())
        selected[model] = {
            "model": model,
            "step": 25_000,
            "checkpoint_path": str(tmp_path / "missing" / model),
            "checkpoint_id": sha256_file(weights),
        }
    _write_completed_selection(selection_dir, selected)
    monkeypatch.setattr(e5_oracle_audit, "practice_root", lambda: repository_root)
    monkeypatch.chdir(outside)

    loaded = _selected_rows(selection_dir)

    assert {row["checkpoint_path_resolution"] for row in loaded.values()} == {"repository_relative"}


def test_audit_valid_mask_follows_squared_error_device() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        valid_steps = torch.ones((2, 3), dtype=torch.bool, device="cpu")
        squared_error = torch.ones((2, 3, 2), device="cuda:0")

        mask = e5_oracle_audit._expanded_valid_mask(valid_steps, squared_error)

    assert mask.device == squared_error.device
    assert mask.dtype == torch.bool
    assert mask.shape == squared_error.shape


def test_eight_gpu_audit_plan_covers_every_model_episode_once() -> None:
    episode_indices = tuple(range(100, 121))

    plans = e5_oracle_audit._build_worker_plans(
        episode_indices=episode_indices,
        num_gpus=8,
    )

    assert len(plans) == 8
    assert [plan.gpu_index for plan in plans] == list(range(8))
    assert sum(plan.model == MODE_BC for plan in plans) == 4
    assert sum(plan.model == MODE_DIFFUSION for plan in plans) == 4
    assert sum(plan.open_loop_leader for plan in plans if plan.model == MODE_BC) == 1
    assert sum(plan.open_loop_leader for plan in plans if plan.model == MODE_DIFFUSION) == 1
    assignments = [
        (plan.model, episode_index) for plan in plans for episode_index in plan.episode_indices
    ]
    assert len(assignments) == 42
    assert len(set(assignments)) == 42
    assert set(assignments) == {
        (model, episode_index)
        for model in (MODE_BC, MODE_DIFFUSION)
        for episode_index in episode_indices
    }


def test_odd_gpu_audit_plan_gives_extra_worker_to_diffusion() -> None:
    plans = e5_oracle_audit._build_worker_plans(
        episode_indices=tuple(range(21)),
        num_gpus=3,
    )

    assert sum(plan.model == MODE_BC for plan in plans) == 1
    assert sum(plan.model == MODE_DIFFUSION for plan in plans) == 2
    with pytest.raises(ValueError, match="exceeds 42"):
        e5_oracle_audit._build_worker_plans(
            episode_indices=tuple(range(21)),
            num_gpus=43,
        )


def test_audit_execution_request_validates_multi_gpu_constraints() -> None:
    assert (
        e5_oracle_audit._validate_execution_request(
            num_gpus=1,
            device="cpu",
        )
        == "cpu"
    )
    assert (
        e5_oracle_audit._validate_execution_request(
            num_gpus=8,
            device=None,
            cuda_available=True,
            visible_device_count=8,
        )
        is None
    )
    with pytest.raises(ValueError, match="device cannot be set"):
        e5_oracle_audit._validate_execution_request(
            num_gpus=2,
            device="cuda:3",
            cuda_available=True,
            visible_device_count=8,
        )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        e5_oracle_audit._validate_execution_request(
            num_gpus=2,
            device=None,
            cuda_available=False,
            visible_device_count=0,
        )
    with pytest.raises(RuntimeError, match="only 4 CUDA"):
        e5_oracle_audit._validate_execution_request(
            num_gpus=8,
            device=None,
            cuda_available=True,
            visible_device_count=4,
        )


def test_audit_result_validation_requires_exact_coverage_and_recordings(
    tmp_path: Path,
) -> None:
    episode_indices = (10, 11)
    selected_k = 3
    null_prototype_id = 3
    closed_loop_rows = []
    open_loop_episode_rows = []
    for model in (MODE_BC, MODE_DIFFUSION):
        for episode_index in episode_indices:
            open_loop_episode_rows.append(
                {
                    "model": model,
                    "episode_index": episode_index,
                }
            )
            gt_prototype_id = episode_index % selected_k
            gt_mode_name = f"P{gt_prototype_id}"
            for condition_id in range(selected_k + 1):
                condition_name = e5_oracle_audit._condition_name(
                    condition_id,
                    null_prototype_id=null_prototype_id,
                )
                success = condition_id == gt_prototype_id
                metadata = e5_oracle_audit._recording_metadata(
                    model=model,
                    condition_name=condition_name,
                    episode_index=episode_index,
                    gt_mode_name=gt_mode_name,
                    success=success,
                )
                closed_loop_rows.append(
                    {
                        "model": model,
                        "episode_index": episode_index,
                        "oracle_prototype_id": gt_prototype_id,
                        "commanded_condition_id": condition_id,
                        "commanded_condition_name": condition_name,
                        "success": success,
                        "recording_name": e5_oracle_audit._recording_name(
                            model=model,
                            condition_name=condition_name,
                            episode_index=episode_index,
                            gt_mode_name=gt_mode_name,
                            success=success,
                        ),
                        "recording_metadata": "\n".join(
                            f"{key}: {value}" for key, value in metadata.items()
                        ),
                        "recording_has_identity_panel": True,
                    }
                )
                recording = e5_oracle_audit._recording_path(
                    tmp_path,
                    model=model,
                    condition_name=condition_name,
                    episode_index=episode_index,
                )
                recording.parent.mkdir(parents=True, exist_ok=True)
                recording.write_bytes(b"rrd")
    open_loop_rows = [
        {
            "model": model,
            "episode_count": len(episode_indices),
        }
        for model in (MODE_BC, MODE_DIFFUSION)
    ]

    validated = e5_oracle_audit._validate_audit_results(
        list(reversed(closed_loop_rows)),
        list(reversed(open_loop_rows)),
        list(reversed(open_loop_episode_rows)),
        episode_indices=episode_indices,
        selected_k=selected_k,
        null_prototype_id=null_prototype_id,
        recordings_root=tmp_path,
    )

    assert [
        (row["model"], row["commanded_condition_id"], row["episode_index"]) for row in validated[0]
    ] == [
        (model, condition_id, episode_index)
        for model in (MODE_BC, MODE_DIFFUSION)
        for condition_id in range(selected_k + 1)
        for episode_index in episode_indices
    ]
    with pytest.raises(ValueError, match="duplicate"):
        e5_oracle_audit._validate_audit_results(
            [*closed_loop_rows, closed_loop_rows[0]],
            open_loop_rows,
            open_loop_episode_rows,
            episode_indices=episode_indices,
            selected_k=selected_k,
            null_prototype_id=null_prototype_id,
            recordings_root=tmp_path,
        )
    with pytest.raises(ValueError, match="coverage mismatch"):
        e5_oracle_audit._validate_audit_results(
            closed_loop_rows[:-1],
            open_loop_rows,
            open_loop_episode_rows,
            episode_indices=episode_indices,
            selected_k=selected_k,
            null_prototype_id=null_prototype_id,
            recordings_root=tmp_path,
        )


def test_audit_condition_metrics_cover_every_model_condition_and_null_is_nullable() -> None:
    rows = []
    for model in (MODE_BC, MODE_DIFFUSION):
        for episode_index, gt_prototype_id in enumerate((0, 1, 2)):
            for condition_id in range(4):
                is_null = condition_id == 3
                success = condition_id == gt_prototype_id if model == MODE_BC else not is_null
                rows.append(
                    {
                        "model": model,
                        "episode_index": episode_index,
                        "oracle_prototype_id": gt_prototype_id,
                        "commanded_condition_id": condition_id,
                        "success": success,
                        "max_reward": float(success),
                        "classifiable": True,
                        "prototype_adherent": (
                            None if is_null else condition_id == gt_prototype_id
                        ),
                        "controlled_success": (
                            None if is_null else success and condition_id == gt_prototype_id
                        ),
                        "replan_latencies_ms": [1.0, 2.0],
                    }
                )

    condition_rows = e5_oracle_audit._condition_metric_rows(
        rows,
        selected_k=3,
        null_prototype_id=3,
    )
    matrix_rows = e5_oracle_audit._gt_commanded_metric_rows(
        rows,
        selected_k=3,
        null_prototype_id=3,
    )
    flattened = e5_oracle_audit._flatten_condition_metrics(condition_rows)

    assert len(condition_rows) == 8
    assert len(matrix_rows) == 24
    bc_p0 = next(
        row
        for row in condition_rows
        if row["model"] == MODE_BC and row["commanded_condition_name"] == "P0"
    )
    dp_null = next(
        row
        for row in condition_rows
        if row["model"] == MODE_DIFFUSION and row["commanded_condition_name"] == "NULL"
    )
    assert bc_p0["success_rate"] == pytest.approx(1 / 3)
    assert dp_null["success_rate"] == 0.0
    assert dp_null["prototype_adherence_rate"] is None
    assert dp_null["controlled_success_rate"] is None
    assert flattened[f"{MODE_BC}_P0_all_tasks_success_rate"] == pytest.approx(1 / 3)
    assert flattened[f"{MODE_DIFFUSION}_NULL_all_tasks_controlled_success_rate"] is None


def test_audit_worker_waits_at_ready_barrier_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeQueue:
        def __init__(self) -> None:
            self.events = []

        def put(self, event: object) -> None:
            self.events.append(event)

    event_queue = FakeQueue()

    class FakeStartEvent:
        waited = False

        def wait(self) -> None:
            assert event_queue.events[0].kind == "ready"
            self.waited = True

    start_event = FakeStartEvent()
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "set_num_threads", lambda _threads: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _threads: None)
    monkeypatch.setattr(
        e5_oracle_audit,
        "load_policy_with_processors",
        lambda *_args, **_kwargs: SimpleNamespace(policy=object()),
    )
    monkeypatch.setattr(e5_oracle_audit, "_configure_loaded_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(e5_oracle_audit, "load_prototype_sidecar", lambda _path: object())
    monkeypatch.setattr(e5_oracle_audit, "_load_classifier", lambda _sidecar: object())

    def fake_open_loop(*_args: object, **_kwargs: object) -> tuple[dict, list[dict]]:
        assert start_event.waited
        return (
            {"model": MODE_BC, "episode_count": 1},
            [{"model": MODE_BC, "episode_index": 10}],
        )

    def fake_closed_loop(*_args: object, **kwargs: object) -> dict:
        assert start_event.waited
        return {
            "model": MODE_BC,
            "episode_index": 10,
            "commanded_condition_id": kwargs["commanded_condition_id"],
        }

    monkeypatch.setattr(e5_oracle_audit, "evaluate_full_open_loop", fake_open_loop)
    monkeypatch.setattr(e5_oracle_audit, "_evaluate_closed_loop_episode", fake_closed_loop)
    plan = e5_oracle_audit.AuditWorkerPlan(
        gpu_index=0,
        model=MODE_BC,
        model_worker_index=0,
        model_worker_count=1,
        episode_indices=(10,),
        open_loop_leader=True,
    )
    config = e5_oracle_audit.AuditWorkerConfig(
        selected={
            MODE_BC: {
                "checkpoint_path": str(tmp_path),
                "step": 25_000,
                "checkpoint_id": "0" * 64,
            }
        },
        prototype_manifest=tmp_path / "prototype.json",
        dataset_root=tmp_path,
        temporary=tmp_path,
        episodes=(SimpleNamespace(episode_index=10),),
        selected_k=3,
        null_prototype_id=3,
        open_loop_batch_size=2,
    )

    e5_oracle_audit._audit_worker(plan, config, start_event, event_queue)

    assert start_event.waited
    assert [event.kind for event in event_queue.events] == [
        "ready",
        "progress",
        "progress",
        "progress",
        "progress",
        "progress",
        "progress",
        "progress",
        "progress",
        "progress",
        "result",
    ]
    result_event = event_queue.events[-1]
    assert [row["commanded_condition_id"] for row in result_event.closed_loop_rows] == [0, 1, 2, 3]


def test_full_open_loop_uses_every_valid_execution_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeDataset:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            actions = torch.arange(8, dtype=torch.float32).reshape(4, 2) + index
            return {
                "observation.state": torch.zeros(2, 2),
                "action": actions,
                "action_is_pad": torch.tensor([False, False, False, True]),
                "episode_index": torch.tensor(10 + index),
            }

    class FakePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                type=MODE_BC,
                input_features={"observation.state": object()},
                image_features={},
                n_obs_steps=2,
                n_action_steps=2,
                horizon=4,
            )

        def predict_full_action_horizon(
            self,
            batch: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            return batch["action"]

    monkeypatch.setattr(e5_oracle_audit, "ModeBCChunkPolicy", FakePolicy)
    monkeypatch.setattr(e5_oracle_audit, "LeRobotDataset", lambda *_args, **_kwargs: FakeDataset())
    monkeypatch.setattr(
        e5_oracle_audit,
        "LeRobotDatasetMetadata",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(e5_oracle_audit, "resolve_delta_timestamps", lambda *_args: {})
    loaded = SimpleNamespace(
        policy=FakePolicy(),
        config=SimpleNamespace(n_obs_steps=2, n_action_steps=2),
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
    )
    episodes = [
        SimpleNamespace(episode_index=10, prototype_id=0, classifiable=True),
        SimpleNamespace(episode_index=11, prototype_id=1, classifiable=True),
    ]

    summary, rows = e5_oracle_audit.evaluate_full_open_loop(
        loaded,
        dataset_root=tmp_path,
        episodes=episodes,
        selected_k=3,
        batch_size=2,
    )

    assert summary["frame_count"] == 2
    assert summary["valid_action_elements"] == 8
    assert summary["denormalized_action_mse"] == pytest.approx(0.0)
    assert [row["episode_index"] for row in rows] == [10, 11]
