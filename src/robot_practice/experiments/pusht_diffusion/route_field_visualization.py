"""Visualize the canonical RF7 pilot routes and their four input channels."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .config import default_output_root
from .route_field_data import ROUTE_FIELD_CHANNELS, ROUTE_FIELD_NAME
from .route_field_dataset import RouteFieldSidecar, load_route_field_sidecar


def _style_grid(axis: Axes, *, grid_size: int) -> None:
    axis.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xticks(range(grid_size))
    axis.set_yticks(range(grid_size))
    axis.set_xlabel("grid x")
    axis.set_ylabel("grid y")


def _plot_route(
    axis: Axes,
    *,
    route: Sequence[int],
    grid_size: int,
    color: str,
) -> None:
    axis.set_xlim(-0.5, grid_size - 0.5)
    axis.set_ylim(grid_size - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_facecolor("#f2f2f2")
    for cell in range(grid_size * grid_size):
        row, column = divmod(cell, grid_size)
        axis.text(column, row, str(cell), ha="center", va="center", color="#777777")
    for order, cell in enumerate(route):
        row, column = divmod(int(cell), grid_size)
        axis.add_patch(
            Rectangle(
                (column - 0.46, row - 0.46),
                0.92,
                0.92,
                facecolor=color,
                edgecolor="black",
                # Cap the alpha so the last cell of a 6-cell route (order=5)
                # stays inside matplotlib's [0, 1] range.
                alpha=min(0.35 + 0.15 * order, 0.95),
                linewidth=1.5,
            )
        )
        axis.text(
            column,
            row - 0.27,
            f"t{order}",
            ha="center",
            va="center",
            color="black",
            fontsize=9,
            fontweight="bold",
        )
    for current, following in pairwise(route):
        row, column = divmod(int(current), grid_size)
        next_row, next_column = divmod(int(following), grid_size)
        axis.annotate(
            "",
            xy=(next_column, next_row),
            xytext=(column, row),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 3,
                "shrinkA": 13,
                "shrinkB": 13,
            },
        )
    _style_grid(axis, grid_size=grid_size)


def _plot_channel(
    axis: Axes,
    values: np.ndarray,
    *,
    title: str,
    grid_size: int,
    cmap: str,
    vmin: float,
    vmax: float,
) -> object:
    image = axis.imshow(values, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    for row in range(grid_size):
        for column in range(grid_size):
            value = float(values[row, column])
            label = f"{value:.2g}"
            normalized = (value - vmin) / (vmax - vmin)
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if normalized > 0.55 or normalized < 0.12 else "black",
                fontsize=8,
            )
    axis.set_title(title)
    _style_grid(axis, grid_size=grid_size)
    return image


def visualize_pilot_routes(
    *,
    route_manifest: str | Path,
    output_path: str | Path,
) -> Path:
    """Render Route A/B and the exact canonical tensors read by the policies."""

    sidecar: RouteFieldSidecar = load_route_field_sidecar(route_manifest)
    pilot_pair = sidecar.manifest.get("pilot_pair")
    if not isinstance(pilot_pair, list) or len(pilot_pair) != 2:
        raise ValueError("RF7 visualization requires exactly two pilot routes")
    grid_size = int(sidecar.manifest["grid"]["size"])
    display_name = str(sidecar.manifest["field"].get("name", ROUTE_FIELD_NAME))

    figure, axes = plt.subplots(
        2,
        5,
        figsize=(20, 8),
        constrained_layout=True,
        squeeze=False,
    )
    route_colors = ("#2563eb", "#ea580c")
    channel_specs = (
        ("occupancy", "Blues", 0.0, 1.0),
        ("progress", "viridis", 0.0, 1.0),
        ("direction x", "coolwarm", -1.0, 1.0),
        ("direction y", "coolwarm", -1.0, 1.0),
    )
    images: list[object] = []
    for row_index, route_record in enumerate(pilot_pair):
        label = str(route_record["label"])
        route = tuple(int(cell) for cell in route_record["route_cells"])
        family_id = int(route_record["family_id"])
        field = sidecar.fields[family_id].numpy()
        axes[row_index, 0].set_title(
            f"Route {label}: {' -> '.join(str(cell) for cell in route)}\n"
            f"family={family_id}, train support={route_record['train_episode_support']}"
        )
        _plot_route(
            axes[row_index, 0],
            route=route,
            grid_size=grid_size,
            color=route_colors[row_index],
        )
        for channel_index, (title, cmap, vmin, vmax) in enumerate(channel_specs):
            images.append(
                _plot_channel(
                    axes[row_index, channel_index + 1],
                    field[channel_index],
                    title=f"{ROUTE_FIELD_CHANNELS[channel_index]} ({title})",
                    grid_size=grid_size,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            )
    figure.suptitle(
        f"{display_name} canonical route conditions: "
        f"exact {list(sidecar.fields.shape[1:])} policy inputs",
        fontsize=16,
        fontweight="bold",
    )
    for column_index in range(1, 5):
        figure.colorbar(
            images[column_index - 1],
            ax=axes[:, column_index],
            shrink=0.72,
            pad=0.02,
        )

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.incomplete.png")
    figure.savefig(temporary, dpi=180, facecolor="white")
    plt.close(figure)
    temporary.replace(destination)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route-manifest",
        type=Path,
        default=default_output_root()
        / "pilot"
        / "route_field_7x7"
        / "sidecar"
        / "route_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_root()
        / "pilot"
        / "route_field_7x7"
        / "visualizations"
        / "rf7_route_a_b.png",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    output = visualize_pilot_routes(
        route_manifest=args.route_manifest,
        output_path=args.output,
    )
    print(f"RF7 visualization: {output}")


if __name__ == "__main__":
    main()


__all__ = ["visualize_pilot_routes"]
