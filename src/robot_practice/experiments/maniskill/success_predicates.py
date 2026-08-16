"""Pure success predicates shared by the M1 data and environment audits.

The functions in this module intentionally do not import ManiSkill.  That keeps
the geometric contract testable without constructing a simulator and lets the
repo-local corrected environment use exactly the same predicate at runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PEG_INSERTION_LOWER_X = -0.015


def _array(value: Any, *, name: str, last_dim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if last_dim is not None and (result.ndim == 0 or result.shape[-1] != last_dim):
        raise ValueError(f"{name} must have final dimension {last_dim}, got {result.shape}")
    return result


def peg_insertion_strict(
    peg_head_pos_at_hole: Any,
    box_hole_radius: Any,
    peg_half_length: Any,
    *,
    lower_x: float = PEG_INSERTION_LOWER_X,
    boundary_tolerance: float = 0.0,
) -> np.ndarray:
    """Return the corrected PegInsertionSide success label.

    The native ManiSkill predicate has no insertion-axis upper bound.  M1 uses
    the per-instance peg half length as the upper bound, so a peg passing
    through the far side of the hole cannot be labelled successful.
    """

    relative = _array(peg_head_pos_at_hole, name="peg_head_pos_at_hole", last_dim=3)
    radius = _array(box_hole_radius, name="box_hole_radius")
    half_length = _array(peg_half_length, name="peg_half_length")
    lower = float(lower_x) - float(boundary_tolerance)
    upper = half_length + float(boundary_tolerance)
    x_ok = (relative[..., 0] >= lower) & (relative[..., 0] <= upper)
    y_ok = np.abs(relative[..., 1]) <= radius + float(boundary_tolerance)
    z_ok = np.abs(relative[..., 2]) <= radius + float(boundary_tolerance)
    return np.asarray(x_ok & y_ok & z_ok, dtype=bool)


def peg_insertion_native(
    peg_head_pos_at_hole: Any,
    box_hole_radius: Any,
    *,
    lower_x: float = PEG_INSERTION_LOWER_X,
    boundary_tolerance: float = 0.0,
) -> np.ndarray:
    """Return the ManiSkill 3.0.1 native PegInsertionSide label."""

    relative = _array(peg_head_pos_at_hole, name="peg_head_pos_at_hole", last_dim=3)
    radius = _array(box_hole_radius, name="box_hole_radius")
    x_ok = relative[..., 0] >= float(lower_x) - float(boundary_tolerance)
    y_ok = np.abs(relative[..., 1]) <= radius + float(boundary_tolerance)
    z_ok = np.abs(relative[..., 2]) <= radius + float(boundary_tolerance)
    return np.asarray(x_ok & y_ok & z_ok, dtype=bool)


def quaternion_canonicalize(value: Any) -> np.ndarray:
    """Canonicalize a quaternion so ``q`` and ``-q`` have identical values."""

    quaternion = _array(value, name="quaternion", last_dim=4).copy()
    flat = quaternion.reshape(-1, 4)
    for row in flat:
        nonzero = np.flatnonzero(np.abs(row) > 0.0)
        if nonzero.size and row[nonzero[0]] < 0:
            row *= -1.0
    return quaternion


def peg_boundary_probe_cases(*, half_length: float, radius: float) -> list[dict[str, Any]]:
    """Build the frozen ±1e-4 boundary probes for one geometry instance."""

    delta = 1e-4
    cases: list[dict[str, Any]] = []
    for axis, values in {
        "x_lower": (
            PEG_INSERTION_LOWER_X - delta,
            PEG_INSERTION_LOWER_X,
            PEG_INSERTION_LOWER_X + delta,
        ),
        "x_upper": (half_length - delta, half_length, half_length + delta),
        "y_radius": (radius - delta, radius, radius + delta),
        "z_radius": (radius - delta, radius, radius + delta),
    }.items():
        for value in values:
            relative = np.zeros(3, dtype=np.float64)
            if axis.startswith("x"):
                relative[0] = value
            elif axis == "y_radius":
                relative[1] = value
            else:
                relative[2] = value
            expected = bool(peg_insertion_strict(relative, radius, half_length))
            cases.append(
                {
                    "axis": axis,
                    "value": float(value),
                    "relative": relative.tolist(),
                    "expected_strict": expected,
                }
            )
    return cases
