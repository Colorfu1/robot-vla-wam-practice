from __future__ import annotations

import json
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.utils.constants import ACTION

from robot_practice.experiments.pusht_diffusion import route_field_eval
from robot_practice.experiments.pusht_diffusion.config import PUSHT_GOAL_BLOCK_ANGLE_RAD
from robot_practice.experiments.pusht_diffusion.route_field_data import (
    ROUTE_FIELD_GRID_SIZE,
    cell_center,
    permute_macro_route_middle,
    reverse_macro_route,
    route_segment_direction,
    route_to_field,
)
from robot_practice.experiments.pusht_diffusion.route_field_episodes import (
    CONDITION_ORACLE,
    CONDITION_PERMUTED,
    CONDITION_REVERSED,
    OBEDIENCE_PROBE_ROLE,
    TASK_REFERENCE_ROLE,
    achieved_route_metrics,
    episode_condition_labels,
    episode_route_conditions,
    route_field_tensor,
)
from robot_practice.experiments.pusht_diffusion.route_field_runtime import (
    FixedRouteFieldPreprocessor,
    add_route_field,
)


def _route_track(cells: tuple[int, ...], *, steps: int = 80) -> np.ndarray:
    def center(cell: int) -> tuple[float, float]:
        return cell_center(cell, grid_size=7)

    points = []
    for first, second in pairwise(cells):
        first_center = center(first)
        second_center = center(second)
        points.append(
            np.column_stack(
                [
                    np.linspace(first_center[0], second_center[0], steps // len(cells)),
                    np.linspace(first_center[1], second_center[1], steps // len(cells)),
                ]
            )
        )
    positions = np.vstack(points)
    block_path = np.zeros((len(positions), 3), dtype=np.float64)
    block_path[:, :2] = positions
    block_path[:, 2] = 0.02
    return block_path


def _episode(route_cells: tuple[int, ...], *, episode_index: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        episode_index=episode_index,
        split="audit",
        route_cells=route_cells,
        field=route_field_tensor(route_cells),
        block_center_workspace=cell_center(route_cells[0], grid_size=7),
    )


# --- route transforms -------------------------------------------------------


def test_reverse_macro_route_preserves_occupancy_and_flips_order() -> None:
    cells = (16, 17, 24)
    reversed_cells = reverse_macro_route(cells)
    assert reversed_cells == (24, 17, 16)
    forward = route_to_field(cells, grid_size=7)
    backward = route_to_field(reversed_cells, grid_size=7)
    # Occupancy support is identical, so the probe isolates order/direction.
    assert np.array_equal(forward[0], backward[0])
    # Progress is mirrored on the occupied cells.
    occupied = forward[0] > 0
    assert np.allclose(backward[1][occupied], 1.0 - forward[1][occupied])
    # Direction differs and is not a plain negation: the zero-direction terminal
    # cell moves from the end of the route to its start.
    # Cell 16 is (row 2, col 2), 17 is (row 2, col 3), 24 is (row 3, col 3).
    assert (forward[2][2, 2], forward[3][2, 2]) == (1.0, 0.0)
    assert (forward[2][2, 3], forward[3][2, 3]) == (0.0, 1.0)
    assert (forward[2][3, 3], forward[3][3, 3]) == (0.0, 0.0)
    assert (backward[2][3, 3], backward[3][3, 3]) == (0.0, -1.0)
    assert (backward[2][2, 3], backward[3][2, 3]) == (-1.0, 0.0)
    assert (backward[2][2, 2], backward[3][2, 2]) == (0.0, 0.0)


def test_reverse_macro_route_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="at least 2 cells"):
        reverse_macro_route((16,))
    with pytest.raises(ValueError, match="without repeated cells"):
        reverse_macro_route((16, 17, 16))


def test_permute_macro_route_middle_keeps_endpoints() -> None:
    assert permute_macro_route_middle((16, 17, 24, 31)) == (16, 24, 17, 31)
    assert permute_macro_route_middle((16, 17, 24, 31, 38)) == (16, 31, 24, 17, 38)
    permuted = permute_macro_route_middle((16, 17, 24, 31))
    original = route_to_field((16, 17, 24, 31), grid_size=7)
    probe = route_to_field(permuted, grid_size=7)
    assert np.array_equal(original[0], probe[0])
    assert not np.allclose(original[1], probe[1])


def test_permute_macro_route_middle_requires_four_cells() -> None:
    with pytest.raises(ValueError, match="at least 4 cells"):
        permute_macro_route_middle((16, 17, 24))


def test_route_segment_direction_matches_cell_geometry() -> None:
    assert route_segment_direction((16, 17), grid_size=7) == pytest.approx((1.0, 0.0))
    assert route_segment_direction((16, 23), grid_size=7) == pytest.approx((0.0, 1.0))
    assert route_segment_direction((16,), grid_size=7) is None


# --- per-episode condition sets --------------------------------------------


def test_episode_route_conditions_scale_with_route_length() -> None:
    assert episode_condition_labels(_episode((16,))) == (CONDITION_ORACLE,)
    assert episode_condition_labels(_episode((16, 17))) == (
        CONDITION_ORACLE,
        CONDITION_REVERSED,
    )
    assert episode_condition_labels(_episode((16, 17, 24))) == (
        CONDITION_ORACLE,
        CONDITION_REVERSED,
    )
    assert episode_condition_labels(_episode((16, 17, 24, 31))) == (
        CONDITION_ORACLE,
        CONDITION_REVERSED,
        CONDITION_PERMUTED,
    )


def test_episode_route_conditions_carry_roles_and_distinct_fields() -> None:
    conditions = episode_route_conditions(_episode((16, 17, 24, 31)))
    roles = {condition.label: condition.role for condition in conditions}
    assert roles[CONDITION_ORACLE] == TASK_REFERENCE_ROLE
    assert roles[CONDITION_REVERSED] == OBEDIENCE_PROBE_ROLE
    assert roles[CONDITION_PERMUTED] == OBEDIENCE_PROBE_ROLE
    fields = [condition.field for condition in conditions]
    for index in range(len(fields)):
        for other in range(index + 1, len(fields)):
            assert not torch.equal(fields[index], fields[other])
        assert tuple(fields[index].shape) == (4, ROUTE_FIELD_GRID_SIZE, ROUTE_FIELD_GRID_SIZE)


# --- obedience metrics ------------------------------------------------------


def test_achieved_route_metrics_dual_adherence_and_signed_progress() -> None:
    oracle_cells = (16, 17, 24)
    track = _route_track(oracle_cells)
    forward = achieved_route_metrics(
        track,
        requested_cells=oracle_cells,
        oracle_cells=oracle_cells,
    )
    assert forward["ordered_waypoint_adherence"] == pytest.approx(1.0)
    assert forward["commanded_minus_oracle_adherence"] == pytest.approx(0.0)
    assert forward["commanded_first_segment_progress_cells"] > 0.0

    # The same executed path scored against the reversed command: adherence must
    # not beat the ground truth, and the commanded first segment points backwards.
    reversed_cells = reverse_macro_route(oracle_cells)
    probe = achieved_route_metrics(
        track,
        requested_cells=reversed_cells,
        oracle_cells=oracle_cells,
    )
    assert probe["commanded_minus_oracle_adherence"] < 0.0
    assert probe["commanded_first_segment_progress_cells"] < 0.0
    assert probe["oracle_ordered_waypoint_adherence"] == pytest.approx(1.0)
    assert probe["oracle_first_segment_progress_cells"] == pytest.approx(
        forward["commanded_first_segment_progress_cells"]
    )

    # Rotation error is measured against the gym_pusht goal angle pi/4, not 0.
    goal_track = np.asarray(track, dtype=np.float64)
    goal_track[:, 2] = PUSHT_GOAL_BLOCK_ANGLE_RAD
    goal_metrics = achieved_route_metrics(
        goal_track,
        requested_cells=oracle_cells,
        oracle_cells=oracle_cells,
    )
    assert goal_metrics["final_rotation_error_deg"] == pytest.approx(0.0, abs=1e-6)
    assert goal_metrics["goal_block_angle_rad"] == pytest.approx(PUSHT_GOAL_BLOCK_ANGLE_RAD)


def test_achieved_route_metrics_rotation_error_reports_45_at_zero_angle() -> None:
    oracle_cells = (16, 17, 24)
    track = np.asarray(_route_track(oracle_cells), dtype=np.float64)
    track[:, 2] = 0.0
    metrics = achieved_route_metrics(
        track,
        requested_cells=oracle_cells,
        oracle_cells=oracle_cells,
    )
    assert metrics["final_rotation_error_deg"] == pytest.approx(45.0, abs=1e-6)


def test_achieved_route_metrics_reports_empty_route_instead_of_raising() -> None:
    # Two frames cannot confirm a cell under the three-frame debounce; a short
    # rollout must not abort the whole evaluation.
    block_path = np.tile(np.asarray([[120.0, 120.0, 0.0]]), (2, 1))
    metrics = achieved_route_metrics(
        block_path,
        requested_cells=(16, 17, 24),
        oracle_cells=(16, 17, 24),
    )
    assert metrics["achieved_route_is_empty"] is True
    assert metrics["achieved_macro_route_cells"] == ()
    assert metrics["ordered_waypoint_adherence"] == pytest.approx(0.0)
    assert metrics["commanded_minus_oracle_adherence"] == pytest.approx(0.0)


def test_achieved_route_metrics_still_rejects_unusable_paths() -> None:
    with pytest.raises(ValueError, match="no finite"):
        achieved_route_metrics(
            np.full((4, 3), np.nan),
            requested_cells=(16, 17),
        )


# --- runtime conditioning ---------------------------------------------------


def test_add_route_field_copies_batch_and_validates_shape() -> None:
    base = {"observation.state": torch.zeros((1, 2, 2))}
    field = torch.zeros((4, 7, 7))
    processed = add_route_field(base, field=field)
    assert "route_field" in processed
    assert tuple(processed["route_field"].shape) == (1, 4, 7, 7)
    assert base.get("route_field") is None
    with pytest.raises(ValueError, match="already contains"):
        add_route_field(processed, field=field)
    with pytest.raises(ValueError, match="shape"):
        add_route_field(base, field=torch.zeros((4, 7)))


def test_fixed_route_field_preprocessor_appends_field_after_base() -> None:
    def base_preprocessor(batch):
        return {"observation.state": batch["observation.state"] * 2}

    preprocessor = FixedRouteFieldPreprocessor(
        base_preprocessor,
        field=torch.ones((4, 7, 7)),
        device="cpu",
    )
    batch = {"observation.state": torch.ones((1, 2, 2))}
    processed = preprocessor(batch)
    assert tuple(processed["observation.state"].shape) == (1, 2, 2)
    assert torch.equal(processed["route_field"], torch.ones((1, 4, 7, 7)))


# --- fixed-observation gate plumbing ---------------------------------------


class _FakeDataset:
    """Minimal dataset stub returning unbatched items keyed by row index."""

    def __init__(self, rows: dict[int, dict[str, torch.Tensor]]) -> None:
        self.rows = rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def _fake_loaded() -> SimpleNamespace:
    config = SimpleNamespace(
        input_features={"observation.state": None, "observation.image": None},
    )
    return SimpleNamespace(policy=SimpleNamespace(config=config))


def test_fixed_observation_batch_selects_middle_frame_and_adds_batch_dim() -> None:
    rows = {
        index: {
            "observation.state": torch.full((2, 2), float(index)),
            "observation.image": torch.zeros((2, 3, 96, 96)),
            ACTION: torch.full((16, 2), float(index)),
        }
        for index in (10, 11, 12, 13, 14)
    }
    batch = route_field_eval._fixed_observation_batch(
        _fake_loaded(),
        dataset=_FakeDataset(rows),
        lookup={7: [10, 11, 12, 13, 14]},
        episode_index=7,
    )
    # A leading batch dimension is mandatory; without it the observation-history
    # axis would silently be interpreted as the batch axis.
    assert tuple(batch["observation.state"].shape) == (1, 2, 2)
    assert tuple(batch["observation.image"].shape) == (1, 2, 3, 96, 96)
    assert tuple(batch[ACTION].shape) == (1, 16, 2)
    # Index 12 is the middle of five frames.
    assert batch["observation.state"][0, 0, 0].item() == pytest.approx(12.0)


def test_fixed_observation_batch_rejects_empty_frame_list() -> None:
    with pytest.raises(ValueError, match="no dataset frames"):
        route_field_eval._fixed_observation_batch(
            _fake_loaded(),
            dataset=_FakeDataset({}),
            lookup={7: []},
            episode_index=7,
        )


# --- aggregation and coverage ----------------------------------------------


def _closed_loop_stub(
    *,
    model: str,
    episode_index: int,
    condition: str,
    success: bool = True,
    advantage: float | None = 0.0,
    progress: float | None = 1.0,
    requested_route_length: int = 1,
) -> dict[str, object]:
    return {
        "model": model,
        "episode_index": episode_index,
        "commanded_condition": condition,
        "condition_role": (
            TASK_REFERENCE_ROLE if condition == CONDITION_ORACLE else OBEDIENCE_PROBE_ROLE
        ),
        "success": success,
        "max_reward": 0.9,
        "ordered_waypoint_adherence": 0.5,
        "normalized_edit_distance": 0.5,
        "oracle_ordered_waypoint_adherence": 0.5,
        "commanded_minus_oracle_adherence": advantage,
        "commanded_first_segment_progress_cells": progress,
        "oracle_first_segment_progress_cells": 1.0,
        "requested_route_length": requested_route_length,
        "achieved_route_is_empty": False,
        "final_rotation_error_deg": 3.0,
        "replan_latencies_ms": [10.0, 20.0],
    }


def test_condition_metric_rows_skip_absent_conditions_and_keep_roles() -> None:
    rows = [
        _closed_loop_stub(
            model="route_field_diffusion", episode_index=1, condition=CONDITION_ORACLE
        ),
        _closed_loop_stub(
            model="route_field_diffusion",
            episode_index=1,
            condition=CONDITION_REVERSED,
            success=False,
        ),
    ]
    metrics = route_field_eval._condition_metric_rows(
        rows,
        models=("route_field_diffusion",),
    )
    conditions = [row["commanded_condition"] for row in metrics]
    # No episode supported the permuted probe, which is a split property.
    assert conditions == [CONDITION_ORACLE, CONDITION_REVERSED]
    assert metrics[0]["condition_role"] == TASK_REFERENCE_ROLE
    assert metrics[1]["condition_role"] == OBEDIENCE_PROBE_ROLE
    assert metrics[1]["success_rate"] == pytest.approx(0.0)


def test_paired_condition_rows_count_obedience_only_when_command_wins() -> None:
    rows = []
    for episode_index, advantage, progress in ((1, 0.5, 2.0), (2, -0.5, -2.0)):
        rows.append(
            _closed_loop_stub(
                model="route_field_diffusion",
                episode_index=episode_index,
                condition=CONDITION_ORACLE,
            )
        )
        rows.append(
            _closed_loop_stub(
                model="route_field_diffusion",
                episode_index=episode_index,
                condition=CONDITION_REVERSED,
                success=False,
                advantage=advantage,
                progress=progress,
            )
        )
    paired = route_field_eval._paired_condition_rows(
        rows,
        models=("route_field_diffusion",),
    )
    assert len(paired) == 1
    row = paired[0]
    assert row["probe_condition"] == CONDITION_REVERSED
    assert row["pair_count"] == 2
    assert row["obeyed_count"] == 1
    assert row["obeyed_rate"] == pytest.approx(0.5)
    assert row["commanded_first_segment_positive_count"] == 1
    assert row["mean_commanded_minus_oracle_adherence"] == pytest.approx(0.0)
    assert row["reference_success_count"] == 2
    assert row["probe_success_count"] == 0


def test_paired_condition_rows_separate_undecidable_two_cell_ties() -> None:
    rows = [
        _closed_loop_stub(
            model="route_field_diffusion",
            episode_index=1,
            condition=CONDITION_ORACLE,
        ),
        _closed_loop_stub(
            model="route_field_diffusion",
            episode_index=1,
            condition=CONDITION_REVERSED,
            success=False,
            advantage=0.0,
            progress=0.0,
            requested_route_length=2,
        ),
        _closed_loop_stub(
            model="route_field_diffusion",
            episode_index=2,
            condition=CONDITION_ORACLE,
        ),
        _closed_loop_stub(
            model="route_field_diffusion",
            episode_index=2,
            condition=CONDITION_REVERSED,
            success=False,
            advantage=0.5,
            progress=1.0,
            requested_route_length=3,
        ),
    ]
    paired = route_field_eval._paired_condition_rows(
        rows,
        models=("route_field_diffusion",),
    )
    assert len(paired) == 1
    row = paired[0]
    assert row["pair_count"] == 2
    assert row["obeyed_count"] == 1
    assert row["obeyed_rate"] == pytest.approx(0.5)
    assert row["undecidable_pair_count"] == 1
    assert row["decidable_pair_count"] == 1
    assert row["obeyed_count_decidable"] == 1
    assert row["obeyed_rate_decidable"] == pytest.approx(1.0)
    assert row["obeyed_wilson95_low_decidable"] is not None
    assert row["obeyed_wilson95_high_decidable"] is not None
    assert row["two_cell_probe_count"] == 1


def test_rederive_route_field_metrics_corrects_rotation_read_only(tmp_path) -> None:
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    from robot_practice.experiments.pusht_diffusion.provenance import sha256_file
    from robot_practice.experiments.pusht_diffusion.route_field_metrics_rederive import (
        rederive_route_field_metrics,
    )

    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for condition, count in (
        (CONDITION_ORACLE, 21),
        (CONDITION_REVERSED, 19),
        (CONDITION_PERMUTED, 5),
    ):
        for ordinal in range(count):
            rows.append(
                {
                    "model": "route_field_diffusion",
                    "episode_index": 1000 + ordinal,
                    "commanded_condition": condition,
                    "success": ordinal == 0 and condition == CONDITION_ORACLE,
                    "max_reward": 0.5,
                    "ordered_waypoint_adherence": 0.4,
                    "normalized_edit_distance": 0.6,
                    "oracle_ordered_waypoint_adherence": 0.45,
                    "commanded_minus_oracle_adherence": (0.0 if ordinal < 13 else -0.2),
                    "commanded_first_segment_progress_cells": (0.0 if ordinal < 13 else -1.0),
                    "oracle_first_segment_progress_cells": 1.0,
                    "requested_route_length": (2 if ordinal < 9 else 3),
                    "achieved_route_is_empty": False,
                    "final_block_angle_rad": PUSHT_GOAL_BLOCK_ANGLE_RAD,
                    "final_rotation_error_deg": 45.0,
                    "replan_latencies_ms": [100.0],
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), source / "rollouts.parquet")
    source_manifest = {
        "schema_version": 1,
        "status": "complete",
        "label": "Closed-loop",
        "experiment": "route_field_held_out_validation",
        "dataset_revision": "revision-test",
        "episode_indices": list(range(45)),
        "policy_seed": 0,
        "route_manifest": "outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json",
        "selection_dir": "outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection",
        "runtime": {"repository": {"commit": None}},
    }
    (source / "run_manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    (source / "metrics.json").write_text(
        json.dumps(
            {
                "validation_split": "audit",
                "episode_count": 21,
                "rollout_count": 45,
                "condition_response_episode_cap": 5,
                "condition_response_row_count": 5,
                "route_field_diffusion_condition_response_chunk_l2_mean": 100.0,
                "route_field_diffusion_condition_response_endpoint_mean": 50.0,
            }
        ),
        encoding="utf-8",
    )
    before = sha256_file(source / "rollouts.parquet")

    output = rederive_route_field_metrics(
        source_dir=source,
        output_dir=tmp_path / "corrected",
        overwrite=False,
    )

    assert sha256_file(source / "rollouts.parquet") == before
    corrected = pq.read_table(output / "rollouts.parquet").to_pylist()
    assert len(corrected) == 45
    assert all(row["final_rotation_error_deg"] == pytest.approx(0.0, abs=1e-6) for row in corrected)
    assert all(
        row["goal_block_angle_rad"] == pytest.approx(PUSHT_GOAL_BLOCK_ANGLE_RAD)
        for row in corrected
    )
    paired = pq.read_table(output / "paired_condition_metrics.parquet").to_pylist()
    reversed_row = next(row for row in paired if row["probe_condition"] == CONDITION_REVERSED)
    assert reversed_row["undecidable_pair_count"] == 13
    assert reversed_row["decidable_pair_count"] == 6
    assert reversed_row["two_cell_probe_count"] == 9
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["label"] == "Re-derived"
    assert manifest["derived_from_rollouts_sha256"] == before
    assert manifest["source_repository_commit"] is None
    assert manifest["route_manifest"].startswith("outputs/")
    assert manifest["selection_dir"].startswith("outputs/")
    assert manifest["dataset_revision"] == "revision-test"
    assert hashlib.sha256((source / "rollouts.parquet").read_bytes()).hexdigest() == before


def test_expected_rollout_count_uses_per_episode_conditions() -> None:
    episodes = [
        _episode((16,), episode_index=1),
        _episode((16, 17), episode_index=2),
        _episode((16, 17, 24, 31), episode_index=3),
    ]
    assert route_field_eval._expected_rollout_count(episodes=episodes) == 1 + 2 + 3
    plans = (
        route_field_eval.RouteFieldEvalWorkerPlan(
            gpu_index=0,
            model="route_field_diffusion",
            episode_indices=(1, 3),
            leader=True,
        ),
    )
    assert route_field_eval._expected_rollout_count(episodes=episodes, plans=plans) == 4


def test_validate_coverage_requires_exactly_the_supported_conditions(tmp_path) -> None:
    episodes = [_episode((16, 17), episode_index=5)]
    model = "route_field_diffusion"
    rows = []
    for condition in (CONDITION_ORACLE, CONDITION_REVERSED):
        row = _closed_loop_stub(model=model, episode_index=5, condition=condition)
        row["recording_name"] = route_field_eval._recording_name(
            model=model,
            condition=condition,
            episode_index=5,
            success=True,
        )
        row["recording_metadata"] = "\n".join(
            f"{key}: {value}"
            for key, value in route_field_eval._recording_metadata(
                model=model,
                condition=condition,
                episode_index=5,
                success=True,
            ).items()
        )
        row["recording_has_identity_panel"] = True
        recording = route_field_eval._recording_path(
            tmp_path,
            model=model,
            condition=condition,
            episode_index=5,
        )
        recording.parent.mkdir(parents=True, exist_ok=True)
        recording.write_bytes(b"rrd")
        rows.append(row)

    ordered = route_field_eval._validate_coverage(
        rows,
        episodes=episodes,
        models=(model,),
        recordings_root=tmp_path,
        figures_root=None,
        figures_enabled=False,
    )
    assert [row["commanded_condition"] for row in ordered] == [
        CONDITION_ORACLE,
        CONDITION_REVERSED,
    ]

    # A permuted row is unexpected because this episode's route has two cells.
    extra = dict(rows[0])
    extra["commanded_condition"] = CONDITION_PERMUTED
    with pytest.raises(ValueError, match="coverage mismatch"):
        route_field_eval._validate_coverage(
            [*rows, extra],
            episodes=episodes,
            models=(model,),
            recordings_root=tmp_path,
            figures_root=None,
            figures_enabled=False,
        )


def test_build_worker_plans_distributes_models_and_marks_leaders() -> None:
    episodes = [SimpleNamespace(episode_index=index) for index in range(10)]
    plans = route_field_eval._build_worker_plans(
        episodes=episodes,
        models=("route_field_bc_chunk", "route_field_diffusion"),
        num_gpus=4,
    )
    assert len(plans) == 4
    bc_plans = [plan for plan in plans if plan.model == "route_field_bc_chunk"]
    dp_plans = [plan for plan in plans if plan.model == "route_field_diffusion"]
    assert len(bc_plans) == 2 and len(dp_plans) == 2
    assert [plan.leader for plan in bc_plans] == [True, False]
    assigned = [index for plan in bc_plans for index in plan.episode_indices]
    assert sorted(assigned) == list(range(10))


# --- gate-only path --------------------------------------------------------


def test_gate_only_requires_at_least_one_fixed_observation_episode() -> None:
    with pytest.raises(ValueError, match="fixed_observation_episodes >= 1"):
        route_field_eval._evaluate_gate_only(
            selected={},
            dataset_root="unused",
            episodes=(),
            num_inference_steps=100,
            fixed_observation_episodes=0,
            device="cpu",
        )


def test_gate_only_outputs_are_labelled_offline(tmp_path) -> None:
    rows = [
        {
            "model": "route_field_diffusion",
            "episode_index": 826,
            "split": "audit",
            "comparison": "reversed_minus_oracle",
            "chunk_l2_distance": 0.4,
            "endpoint_distance": 0.1,
            "max_chunk_distance": 0.2,
        },
        {
            "model": "route_field_diffusion",
            "episode_index": 830,
            "split": "audit",
            "comparison": "reversed_minus_oracle",
            "chunk_l2_distance": 0.2,
            "endpoint_distance": 0.05,
            "max_chunk_distance": 0.1,
        },
    ]
    route_field_eval._write_gate_only_outputs(
        tmp_path,
        rows=rows,
        models=("route_field_diffusion",),
        split_name="audit",
        episodes=(_episode((16, 17, 24), episode_index=826),),
        route_manifest=tmp_path / "route_manifest.json",
        route_manifest_sha256="0" * 64,
        selection_dir=tmp_path / "selection",
        selection_manifest={"experiment": "route_field_oracle_development_checkpoint_selection"},
        num_inference_steps=100,
        fixed_observation_episodes=5,
        device="cuda:0",
        runtime={"repository": {"commit": "abc"}},
    )
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    # The gate executes nothing in the simulator, so it must never be labelled
    # Closed-loop and must not be able to report success.
    assert manifest["label"] == "Offline"
    assert manifest["experiment"] == "route_field_condition_gate"
    assert manifest["comparison"] == "reversed_minus_oracle"
    assert "gate_criterion" in manifest

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["condition_response_row_count"] == 2
    assert metrics["route_field_diffusion_condition_response_chunk_l2_mean"] == pytest.approx(0.3)
    assert metrics["route_field_diffusion_condition_response_chunk_l2_min"] == pytest.approx(0.2)
    assert not any(key.endswith("success_rate") for key in metrics)
    assert (tmp_path / "condition_response.parquet").is_file()
    assert not (tmp_path / "rollouts.parquet").exists()
