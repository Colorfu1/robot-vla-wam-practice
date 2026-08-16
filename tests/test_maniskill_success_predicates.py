from __future__ import annotations

import numpy as np

from robot_practice.experiments.maniskill.m1_peg_predicate_audit import _geometry_probes
from robot_practice.experiments.maniskill.success_predicates import (
    PEG_INSERTION_LOWER_X,
    peg_insertion_native,
    peg_insertion_strict,
    quaternion_canonicalize,
)


def test_strict_predicate_has_per_instance_upper_bound() -> None:
    relative = np.asarray(
        [
            [PEG_INSERTION_LOWER_X, 0.0, 0.0],
            [0.085, 0.0, 0.0],
            [0.085 + 1e-4, 0.0, 0.0],
        ]
    )
    assert peg_insertion_native(relative, 0.02).tolist() == [True, True, True]
    assert peg_insertion_strict(relative, 0.02, 0.085).tolist() == [True, True, False]


def test_strict_predicate_keeps_y_z_radius_boundaries() -> None:
    relative = np.asarray([[0.0, 0.02, 0.0], [0.0, 0.0201, 0.0]])
    assert peg_insertion_strict(relative, 0.02, 0.1).tolist() == [True, False]


def test_quaternion_sign_is_equivalent() -> None:
    positive = quaternion_canonicalize([1.0, 0.0, 0.0, 0.0])
    negative = quaternion_canonicalize([-1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(positive, negative)


def test_geometry_probe_matrix_is_fully_passed() -> None:
    result = _geometry_probes()
    assert result["num_cases"] == 48
    assert result["gate_pass"] is True
