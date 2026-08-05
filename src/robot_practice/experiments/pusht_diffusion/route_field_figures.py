"""PNG trajectory figures for RF7 validation rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import PUSHT_WORKSPACE_SIZE
from .route_field_visualization import _plot_route

REQUESTED_COLOR = "#2563eb"
ACHIEVED_COLOR = "#ea580c"
TRAJECTORY_COLOR = "#111111"


def _grid_position(block_path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(block_path, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"block_path must have shape [steps,>=2], got {tuple(positions.shape)}")
    scale = PUSHT_WORKSPACE_SIZE / 7.0
    return positions[:, 0] / scale, positions[:, 1] / scale


def _draw_panel(
    axis: Any,
    *,
    route: Sequence[int],
    color: str,
    title: str,
    block_path: np.ndarray | None = None,
) -> None:
    _plot_route(axis, route=route, grid_size=7, color=color)
    if block_path is not None and len(block_path):
        x, y = _grid_position(block_path)
        axis.plot(x, y, color=TRAJECTORY_COLOR, linewidth=1.0, alpha=0.75)
    axis.set_title(title, fontsize=12)


def _metric_text(row: Mapping[str, Any]) -> str:
    adherence = row.get("ordered_waypoint_adherence")
    oracle_adherence = row.get("oracle_ordered_waypoint_adherence")
    advantage = row.get("commanded_minus_oracle_adherence")
    progress = row.get("commanded_first_segment_progress_cells")
    rotation = row.get("final_rotation_error_deg")
    p95 = row.get("p95_replan_latency_ms")

    def rendered(value: Any, spec: str = ".2f") -> str:
        return "-" if value is None else format(float(value), spec)

    parts = [
        f"success={bool(row.get('success'))}",
        f"adherence={rendered(adherence)}",
        f"gt_adherence={rendered(oracle_adherence)}",
        f"cmd-gt={rendered(advantage, '+.2f')}",
        f"seg1={rendered(progress, '+.2f')}cell",
        f"rot_err={rendered(rotation, '.1f')}deg",
        f"p95={rendered(p95, '.0f')}ms",
        f"steps={int(row.get('episode_length', 0))}",
    ]
    return " | ".join(parts)


def write_rollout_figure(
    output_path: str | Path,
    *,
    model_display: str,
    episode_index: int,
    condition: str,
    row: Mapping[str, Any],
    block_path: np.ndarray,
) -> Path:
    """Render requested vs achieved route for one closed-loop rollout."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.2), constrained_layout=True)
    requested = tuple(int(cell) for cell in row.get("requested_route_cells", ()))
    achieved = tuple(int(cell) for cell in row.get("achieved_macro_route_cells", ()))
    _draw_panel(
        axes[0],
        route=requested,
        color=REQUESTED_COLOR,
        title="requested",
    )
    _draw_panel(
        axes[1],
        route=achieved,
        color=ACHIEVED_COLOR,
        title="achieved (macro)",
        block_path=block_path,
    )
    figure.suptitle(
        f"RF7 | {model_display} | command {condition} | episode {episode_index}\n"
        f"{_metric_text(row)}",
        fontsize=13,
        fontweight="bold",
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.incomplete.png")
    figure.savefig(temporary, dpi=160, facecolor="white")
    plt.close(figure)
    temporary.replace(destination)
    return destination


def write_condition_pair_figure(
    output_path: str | Path,
    *,
    model_display: str,
    episode_index: int,
    reference_label: str,
    probe_label: str,
    reference_row: Mapping[str, Any],
    probe_row: Mapping[str, Any],
    reference_block_path: np.ndarray,
    probe_block_path: np.ndarray,
) -> Path:
    """Render a task-reference route against one obedience probe, same initial state."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    _draw_panel(
        axes[0, 0],
        route=tuple(int(cell) for cell in reference_row.get("requested_route_cells", ())),
        color=REQUESTED_COLOR,
        title=f"{reference_label} requested",
    )
    _draw_panel(
        axes[0, 1],
        route=tuple(int(cell) for cell in reference_row.get("achieved_macro_route_cells", ())),
        color=ACHIEVED_COLOR,
        title=f"{reference_label} achieved",
        block_path=reference_block_path,
    )
    _draw_panel(
        axes[1, 0],
        route=tuple(int(cell) for cell in probe_row.get("requested_route_cells", ())),
        color=REQUESTED_COLOR,
        title=f"{probe_label} requested",
    )
    _draw_panel(
        axes[1, 1],
        route=tuple(int(cell) for cell in probe_row.get("achieved_macro_route_cells", ())),
        color=ACHIEVED_COLOR,
        title=f"{probe_label} achieved",
        block_path=probe_block_path,
    )
    axes[0, 1].set_title(
        f"{reference_label} achieved\n{_metric_text(reference_row)}",
        fontsize=11,
    )
    axes[1, 1].set_title(
        f"{probe_label} achieved\n{_metric_text(probe_row)}",
        fontsize=11,
    )
    figure.suptitle(
        f"RF7 {reference_label} vs {probe_label} | {model_display} | "
        f"episode {episode_index} (same initial state, same noise)",
        fontsize=14,
        fontweight="bold",
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.incomplete.png")
    figure.savefig(temporary, dpi=150, facecolor="white")
    plt.close(figure)
    temporary.replace(destination)
    return destination


def write_contact_sheet(
    output_path: str | Path,
    *,
    model_display: str,
    condition: str,
    rows: Sequence[Mapping[str, Any]],
    max_columns: int = 5,
) -> Path:
    """Overlay requested and achieved routes for every episode in one image."""

    ordered = sorted(rows, key=lambda row: int(row["episode_index"]))
    if not ordered:
        raise ValueError("Cannot build a contact sheet from empty rows")
    columns = min(max_columns, len(ordered))
    rows_count = (len(ordered) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(4.2 * columns, 4.2 * rows_count),
        constrained_layout=True,
        squeeze=False,
    )
    for ordinal, row in enumerate(ordered):
        axis = axes[ordinal // columns][ordinal % columns]
        requested = tuple(int(cell) for cell in row.get("requested_route_cells", ()))
        achieved = tuple(int(cell) for cell in row.get("achieved_macro_route_cells", ()))
        _plot_route(axis, route=requested, grid_size=7, color=REQUESTED_COLOR)
        _plot_route(axis, route=achieved, grid_size=7, color=ACHIEVED_COLOR)
        axis.set_title(
            f"ep {int(row['episode_index'])} | {_metric_text(row)}",
            fontsize=9,
        )
    for flat_index in range(len(ordered), rows_count * columns):
        axis = axes[flat_index // columns][flat_index % columns]
        axis.set_axis_off()
    figure.suptitle(
        f"RF7 | {model_display} | command {condition} | requested (blue) vs achieved (orange)",
        fontsize=15,
        fontweight="bold",
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.incomplete.png")
    figure.savefig(temporary, dpi=140, facecolor="white")
    plt.close(figure)
    temporary.replace(destination)
    return destination


__all__ = [
    "write_condition_pair_figure",
    "write_contact_sheet",
    "write_rollout_figure",
]
