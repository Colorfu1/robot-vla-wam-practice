"""Dataset-first-frame reconstruction and RF7 route-conditioned rollout helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.utils import get_device_from_parameters

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    PUSHT_GOAL_BLOCK_ANGLE_RAD,
    PUSHT_WORKSPACE_SIZE,
)
from .e0_geometry import extract_moving_block_pose
from .e5_oracle_mode import (
    body_pose_from_geometric_pose,
    make_dataset_first_frame_env,
    refine_reconstruction_pose,
    wrapped_angle_difference,
)
from .rollout import FULL_DDPM_SAMPLING, RolloutResult, run_rollout
from .route_field_data import (
    ROUTE_FIELD_GRID_SIZE,
    ROUTE_FIELD_STABLE_FRAMES,
    permute_macro_route_middle,
    position_to_cell,
    reverse_macro_route,
    route_segment_direction,
    route_to_field,
    simplify_macro_route,
)
from .route_field_dataset import RouteFieldSidecar, load_route_field_sidecar
from .route_field_runtime import FixedRouteFieldPreprocessor

CONDITION_ORACLE = "oracle"
CONDITION_REVERSED = "reversed"
CONDITION_PERMUTED = "permuted"
CONDITION_ORDER = (CONDITION_ORACLE, CONDITION_REVERSED, CONDITION_PERMUTED)
TASK_REFERENCE_ROLE = "task_reference"
OBEDIENCE_PROBE_ROLE = "obedience_probe"
CONDITION_ROLES = {
    CONDITION_ORACLE: TASK_REFERENCE_ROLE,
    CONDITION_REVERSED: OBEDIENCE_PROBE_ROLE,
    CONDITION_PERMUTED: OBEDIENCE_PROBE_ROLE,
}
# A reversed route needs two cells to define an order; a middle permutation
# needs two interior cells, so four in total.
MINIMUM_REVERSED_ROUTE_CELLS = 2
MINIMUM_PERMUTED_ROUTE_CELLS = 4


@dataclass(frozen=True)
class RouteFieldEpisode:
    """One held-out episode with its exact route as the oracle condition."""

    episode_index: int
    split: str
    route_cells: tuple[int, ...]
    raw_route_cells: tuple[int, ...]
    field: torch.Tensor
    # Fields shared with e5_oracle_mode reconstruction helpers.
    prototype_id: int | None = None
    classifiable: bool = True
    distance_to_medoid: float | None = None
    agent_position: tuple[float, float] = (0.0, 0.0)
    block_center_workspace: tuple[float, float] = (0.0, 0.0)
    extractor_angle_rad: float = 0.0
    block_body_origin: tuple[float, float] = (0.0, 0.0)
    block_body_angle_rad: float = 0.0
    reconstruction_angle_adjustment_deg: float = 0.0
    reconstruction_center_adjustment_px: tuple[float, float] = (0.0, 0.0)
    first_frame: np.ndarray | None = None


@dataclass(frozen=True)
class RouteCondition:
    """One commanded route for one episode, plus its reporting role.

    ``role`` separates the task reference (the episode's own ground-truth
    route, which the policy can satisfy while completing the task) from
    obedience probes (routes the policy was never trained to receive from this
    initial state). Probe task success is an obedience readout with inverted
    sign, not a performance number.
    """

    label: str
    role: str
    route_cells: tuple[int, ...]
    field: torch.Tensor


def route_field_tensor(route_cells: Sequence[int]) -> torch.Tensor:
    """Encode one macro route as the `[4,7,7]` tensor the policies consume."""

    return torch.as_tensor(
        route_to_field(tuple(route_cells), grid_size=ROUTE_FIELD_GRID_SIZE),
        dtype=torch.float32,
    )


def final_rotation_error_degrees(final_angle: float) -> float:
    """Return the angular distance of one final block pose from the PushT goal.

    gym_pusht's goal pose is ``[256, 256, pi/4]``, so the reference angle is
    ``PUSHT_GOAL_BLOCK_ANGLE_RAD`` rather than zero. The signed wrapped
    difference is reported as an absolute angle in degrees.
    """

    return float(
        abs(wrapped_angle_difference(float(final_angle), PUSHT_GOAL_BLOCK_ANGLE_RAD))
        * 180.0
        / np.pi
    )


def episode_route_conditions(episode: RouteFieldEpisode) -> tuple[RouteCondition, ...]:
    """Build the commanded routes for one episode.

    Every episode receives its ground-truth macro route. Routes with at least
    two cells additionally receive the reversed route, and routes with at least
    four cells receive the middle-permuted route. Single-cell routes reverse to
    themselves, so they would contribute a guaranteed-zero difference and are
    deliberately excluded from the probes.
    """

    cells = tuple(int(cell) for cell in episode.route_cells)
    if not cells:
        raise ValueError(f"Episode {episode.episode_index} has an empty macro route")
    conditions = [
        RouteCondition(
            label=CONDITION_ORACLE,
            role=CONDITION_ROLES[CONDITION_ORACLE],
            route_cells=cells,
            field=episode.field,
        )
    ]
    if len(cells) >= MINIMUM_REVERSED_ROUTE_CELLS:
        reversed_cells = reverse_macro_route(cells)
        conditions.append(
            RouteCondition(
                label=CONDITION_REVERSED,
                role=CONDITION_ROLES[CONDITION_REVERSED],
                route_cells=reversed_cells,
                field=route_field_tensor(reversed_cells),
            )
        )
    if len(cells) >= MINIMUM_PERMUTED_ROUTE_CELLS:
        permuted_cells = permute_macro_route_middle(cells)
        conditions.append(
            RouteCondition(
                label=CONDITION_PERMUTED,
                role=CONDITION_ROLES[CONDITION_PERMUTED],
                route_cells=permuted_cells,
                field=route_field_tensor(permuted_cells),
            )
        )
    return tuple(conditions)


def episode_condition_labels(episode: RouteFieldEpisode) -> tuple[str, ...]:
    """Return only the condition labels one episode supports."""

    return tuple(condition.label for condition in episode_route_conditions(episode))


def _frame_lookup(dataset: LeRobotDataset) -> dict[int, int]:
    dataset._ensure_hf_dataset_loaded()
    result: dict[int, int] = {}
    episode_frames: dict[int, list[tuple[int, int]]] = defaultdict(list)
    episodes = dataset.hf_dataset["episode_index"]
    frames = dataset.hf_dataset["frame_index"]
    for index, (raw_episode, raw_frame) in enumerate(zip(episodes, frames, strict=True)):
        episode = int(raw_episode.item() if hasattr(raw_episode, "item") else raw_episode)
        frame = int(raw_frame.item() if hasattr(raw_frame, "item") else raw_frame)
        episode_frames[episode].append((frame, index))
    for episode, candidates in episode_frames.items():
        result[episode] = min(candidates)[1]
    return result


def load_route_field_episodes(
    *,
    dataset_root: str | Path,
    route_manifest: str | Path,
    episode_indices: Sequence[int],
    split_name: str,
) -> tuple[RouteFieldEpisode, ...]:
    """Load first frames and exact per-episode route conditions for one split."""

    requested = tuple(int(value) for value in episode_indices)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("episode_indices must be a non-empty unique sequence")
    sidecar: RouteFieldSidecar = load_route_field_sidecar(route_manifest)
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=Path(dataset_root).expanduser().resolve(),
        episodes=list(requested),
        revision=DATASET_REVISION,
    )
    lookup = _frame_lookup(dataset)
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"Dataset is missing requested first frames: {missing}")

    episodes: list[RouteFieldEpisode] = []
    for episode_index in requested:
        try:
            assignment = sidecar.assignments[episode_index]
        except KeyError as exc:
            raise KeyError(f"Route-field sidecar has no episode {episode_index}") from exc
        if assignment.split != split_name:
            raise ValueError(
                f"Episode {episode_index} sidecar split is {assignment.split!r}, "
                f"expected {split_name!r}"
            )
        if not assignment.route_cells:
            raise ValueError(f"Episode {episode_index} has an empty route field")
        field = route_field_tensor(assignment.route_cells)

        item = dataset[lookup[episode_index]]
        from .rerun_visualization import image_to_hwc_uint8

        image = image_to_hwc_uint8(item["observation.image"])
        raw_agent = item["observation.state"]
        agent = np.asarray(
            raw_agent.detach().cpu().numpy() if hasattr(raw_agent, "detach") else raw_agent,
            dtype=np.float64,
        ).reshape(-1)
        if agent.shape != (2,) or not np.isfinite(agent).all():
            raise ValueError(f"Episode {episode_index} has invalid first-frame agent state")
        pose = extract_moving_block_pose(image, agent_position=agent)
        if not pose.valid:
            raise ValueError(
                f"Episode {episode_index} first-frame moving-block pose is not reconstructable"
            )
        body_origin, body_angle = body_pose_from_geometric_pose(
            pose.position_workspace,
            pose.angle_rad,
        )
        episode = RouteFieldEpisode(
            episode_index=episode_index,
            split=split_name,
            route_cells=assignment.route_cells,
            raw_route_cells=assignment.raw_route_cells,
            field=field,
            agent_position=(float(agent[0]), float(agent[1])),
            block_center_workspace=pose.position_workspace,
            extractor_angle_rad=pose.angle_rad,
            block_body_origin=body_origin,
            block_body_angle_rad=body_angle,
            first_frame=image,
        )
        episodes.append(refine_reconstruction_pose(episode))
    return tuple(episodes)


def pilot_route_fields(
    route_manifest: str | Path,
) -> tuple[tuple[str, tuple[int, ...], torch.Tensor], ...]:
    """Return the committed Route A/B labels, cells, and exact canonical tensors."""

    sidecar = load_route_field_sidecar(route_manifest)
    pilot_pair = sidecar.manifest.get("pilot_pair")
    if not isinstance(pilot_pair, list) or len(pilot_pair) != 2:
        raise ValueError("RF7 validation requires exactly two pilot routes")
    result = []
    for record in pilot_pair:
        label = str(record["label"])
        family_id = int(record["family_id"])
        cells = tuple(int(cell) for cell in record["route_cells"])
        if not 0 <= family_id < sidecar.num_families:
            raise ValueError(f"Pilot route {label} family {family_id} is out of range")
        result.append((label, cells, sidecar.fields[family_id].clone()))
    return tuple(result)


def make_episode_env(
    episode: RouteFieldEpisode,
    *,
    episode_length: int = 300,
    render: bool = True,
) -> gym.Env:
    return make_dataset_first_frame_env(
        episode,
        episode_length=episode_length,
        render=render,
    )


def run_route_field_rollout(
    loaded: Any,
    *,
    episode: RouteFieldEpisode,
    field: torch.Tensor,
    condition_label: str,
    record_frames: bool,
    episode_length: int = 300,
    policy_seed: int = 0,
) -> RolloutResult:
    """Roll out once from a held-out episode state under one explicit route."""

    device = get_device_from_parameters(loaded.policy)
    preprocessor = FixedRouteFieldPreprocessor(
        loaded.preprocessor,
        field=field,
        device=device,
    )

    def environment_factory(*, episode_length: int, render: bool) -> gym.Env:
        return make_episode_env(
            episode,
            episode_length=episode_length,
            render=render,
        )

    return run_rollout(
        policy=loaded.policy,
        preprocessor=preprocessor,
        postprocessor=loaded.postprocessor,
        environment_seed=episode.episode_index,
        policy_seed=policy_seed,
        record_frames=record_frames,
        episode_length=episode_length,
        condition_id=(
            f"{episode.split}_episode_{episode.episode_index:06d}_command_{condition_label}"
        ),
        sampling_protocol=FULL_DDPM_SAMPLING,
        environment_factory=environment_factory,
    )


def _ordered_subsequence_length(
    requested: Sequence[int],
    achieved: Sequence[int],
) -> int:
    """Longest common subsequence length over short route cell sequences."""

    first = tuple(requested)
    second = tuple(achieved)
    table = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for first_index in range(1, len(first) + 1):
        for second_index in range(1, len(second) + 1):
            if first[first_index - 1] == second[second_index - 1]:
                table[first_index][second_index] = table[first_index - 1][second_index - 1] + 1
            else:
                table[first_index][second_index] = max(
                    table[first_index - 1][second_index],
                    table[first_index][second_index - 1],
                )
    return table[len(first)][len(second)]


def _levenshtein(first: Sequence[int], second: Sequence[int]) -> int:
    left = tuple(first)
    right = tuple(second)
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[len(right)]


def _stable_cells(
    cells: Sequence[int],
    *,
    stable_frames: int,
) -> tuple[int, ...]:
    """Debounce an already-grid-mapped cell sequence (mirror of extract_stable_route)."""

    if stable_frames <= 0:
        raise ValueError("stable_frames must be positive")
    route: list[int] = []
    confirmed: int | None = None
    candidate: int | None = None
    candidate_frames = 0
    for cell in cells:
        if cell == confirmed:
            candidate = None
            candidate_frames = 0
            continue
        if cell == candidate:
            candidate_frames += 1
        else:
            candidate = cell
            candidate_frames = 1
        if candidate_frames >= stable_frames:
            confirmed = cell
            route.append(cell)
            candidate = None
            candidate_frames = 0
    return tuple(route)


def _route_comparison(
    requested: Sequence[int],
    achieved: Sequence[int],
) -> tuple[int, float | None, int, float | None]:
    """Return matches, adherence, edit distance, and normalized edit distance."""

    requested_cells = tuple(int(cell) for cell in requested)
    achieved_cells = tuple(int(cell) for cell in achieved)
    matched = _ordered_subsequence_length(requested_cells, achieved_cells)
    edit_distance = _levenshtein(requested_cells, achieved_cells)
    if not requested_cells:
        return matched, None, edit_distance, None
    denominator = max(len(requested_cells), len(achieved_cells))
    return (
        matched,
        matched / len(requested_cells),
        edit_distance,
        edit_distance / denominator if denominator else None,
    )


def _signed_first_segment_progress(
    positions: np.ndarray,
    route_cells: Sequence[int],
    *,
    grid_size: int,
) -> float | None:
    """Project net T-center displacement onto the commanded first segment.

    The value is expressed in grid cells and keeps its sign, so a positive
    number means the block moved along the direction that was commanded first.
    This stays informative when the block never completes the route, which the
    cell-sequence metrics cannot distinguish from not moving at all.
    """

    direction = route_segment_direction(route_cells, grid_size=grid_size)
    if direction is None or positions.shape[0] < 2:
        return None
    delta = positions[-1, :2] - positions[0, :2]
    cell_size = PUSHT_WORKSPACE_SIZE / grid_size
    return float((delta[0] * direction[0] + delta[1] * direction[1]) / cell_size)


def achieved_route_metrics(
    block_path: np.ndarray,
    *,
    requested_cells: Sequence[int],
    oracle_cells: Sequence[int] | None = None,
    stable_frames: int = ROUTE_FIELD_STABLE_FRAMES,
) -> dict[str, Any]:
    """Debounce the executed T-center path and compare it with commanded routes.

    When ``oracle_cells`` is given, the same executed path is additionally
    scored against the episode's ground-truth route. The difference between the
    two adherence values is the decisive readout for an obedience probe:
    positive means the executed path followed the commanded route rather than
    the route the policy would traverse anyway.
    """

    positions = np.asarray(block_path, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2 or positions.shape[0] < 1:
        raise ValueError(f"block_path must have shape [steps,>=2], got {tuple(positions.shape)}")
    valid = np.isfinite(positions[:, :2]).all(axis=1)
    if not bool(valid.any()):
        raise ValueError("block_path contains no finite T-center positions")
    finite = positions[valid]
    cells = [
        position_to_cell(tuple(position), grid_size=ROUTE_FIELD_GRID_SIZE)
        for position in finite[:, :2]
    ]
    raw_route = _stable_cells(cells, stable_frames=stable_frames)
    if raw_route:
        macro_route, backtrack_removed, loop_removed = simplify_macro_route(raw_route)
    else:
        # An episode can terminate before any cell is confirmed for
        # ``stable_frames``. Report an empty achieved route instead of failing
        # the whole evaluation on one short rollout.
        macro_route, backtrack_removed, loop_removed = (), 0, 0
    requested = tuple(int(cell) for cell in requested_cells)
    matched, adherence, edit_distance, normalized_edit_distance = _route_comparison(
        requested,
        macro_route,
    )
    final_angle = float(finite[-1, 2]) if finite.shape[1] >= 3 else float("nan")
    commanded_progress = _signed_first_segment_progress(
        finite,
        requested,
        grid_size=ROUTE_FIELD_GRID_SIZE,
    )
    result: dict[str, Any] = {
        "achieved_raw_route_cells": raw_route,
        "achieved_macro_route_cells": macro_route,
        "achieved_macro_route_length": len(macro_route),
        "achieved_route_is_empty": not raw_route,
        "backtrack_removed_cells": backtrack_removed,
        "loop_removed_cells": loop_removed,
        "requested_route_cells": requested,
        "requested_route_length": len(requested),
        "ordered_waypoint_matches": matched,
        "ordered_waypoint_adherence": adherence,
        "edit_distance": edit_distance,
        "normalized_edit_distance": normalized_edit_distance,
        "commanded_first_segment_cells": (requested[:2] if len(requested) >= 2 else ()),
        "commanded_first_segment_progress_cells": commanded_progress,
        "final_block_angle_rad": final_angle,
        "final_rotation_error_deg": (
            final_rotation_error_degrees(final_angle) if np.isfinite(final_angle) else None
        ),
        "goal_block_angle_rad": PUSHT_GOAL_BLOCK_ANGLE_RAD,
    }
    if oracle_cells is None:
        return result
    oracle = tuple(int(cell) for cell in oracle_cells)
    (
        oracle_matched,
        oracle_adherence,
        oracle_edit_distance,
        oracle_normalized_edit_distance,
    ) = _route_comparison(oracle, macro_route)
    result.update(
        {
            "oracle_route_cells": oracle,
            "oracle_route_length": len(oracle),
            "oracle_ordered_waypoint_matches": oracle_matched,
            "oracle_ordered_waypoint_adherence": oracle_adherence,
            "oracle_edit_distance": oracle_edit_distance,
            "oracle_normalized_edit_distance": oracle_normalized_edit_distance,
            "oracle_first_segment_progress_cells": _signed_first_segment_progress(
                finite,
                oracle,
                grid_size=ROUTE_FIELD_GRID_SIZE,
            ),
            "commanded_minus_oracle_adherence": (
                adherence - oracle_adherence
                if adherence is not None and oracle_adherence is not None
                else None
            ),
        }
    )
    return result


__all__ = [
    "CONDITION_ORACLE",
    "CONDITION_ORDER",
    "CONDITION_PERMUTED",
    "CONDITION_REVERSED",
    "CONDITION_ROLES",
    "MINIMUM_PERMUTED_ROUTE_CELLS",
    "MINIMUM_REVERSED_ROUTE_CELLS",
    "OBEDIENCE_PROBE_ROLE",
    "TASK_REFERENCE_ROLE",
    "RouteCondition",
    "RouteFieldEpisode",
    "achieved_route_metrics",
    "episode_condition_labels",
    "episode_route_conditions",
    "load_route_field_episodes",
    "make_episode_env",
    "pilot_route_fields",
    "route_field_tensor",
    "run_route_field_rollout",
]
