from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from robot_practice.experiments.pusht_diffusion.dataset_split import load_split
from robot_practice.experiments.pusht_diffusion.provenance import sha256_file
from robot_practice.experiments.pusht_diffusion.route_field_data import (
    PILOT_ROUTE_A,
    PILOT_ROUTE_B,
    ROUTE_FIELD_SCHEMA_VERSION,
    extract_stable_route,
    position_to_cell,
    route_to_field,
    simplify_macro_route,
)
from robot_practice.experiments.pusht_diffusion.route_field_dataset import (
    RouteFieldConditionedDataset,
    load_route_field_sidecar,
)
from robot_practice.experiments.pusht_diffusion.train_route_field_bc_chunk import (
    resolve_training as resolve_bc_training,
)
from robot_practice.experiments.pusht_diffusion.training_common import (
    validate_training_inputs,
)
from robot_practice.policies.route_field_bc_chunk import (
    RouteFieldBCChunkConfig,
    RouteFieldBCChunkPolicy,
)
from robot_practice.policies.route_field_conditioning import ROUTE_FIELD
from robot_practice.policies.route_field_diffusion import (
    RouteFieldDiffusionConfig,
    RouteFieldDiffusionPolicy,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]


def _write_sidecar(root: Path) -> Path:
    split = load_split()
    assignments_path = root / "episode_assignments.parquet"
    rows = []
    for episode in range(split.num_episodes):
        if episode in split.train:
            split_name = "train"
        elif episode in split.development:
            split_name = "development"
        else:
            split_name = "audit"
        rows.append(
            {
                "episode_index": episode,
                "split": split_name,
                "raw_route_cells": list(PILOT_ROUTE_A),
                "route_cells": list(PILOT_ROUTE_A),
                "family_id": 0,
                "training_condition_id": 0 if split_name == "train" else None,
                "eligible": True,
                "exclusion_reason": None,
                "backtrack_removed_cells": 0,
                "loop_removed_cells": 0,
                "total_removed_cells": 0,
                "num_frames": 4,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), assignments_path)
    fields_path = root / "route_fields.npz"
    np.savez_compressed(
        fields_path,
        family_ids=np.asarray([0], dtype=np.int64),
        fields=route_to_field(PILOT_ROUTE_A)[None],
    )
    manifest_path = root / "route_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": ROUTE_FIELD_SCHEMA_VERSION,
                "status": "complete",
                "split_manifest_sha256": split.manifest_sha256,
                "grid": {"size": 7},
                "field": {
                    "channels": [
                        "occupancy",
                        "progress",
                        "direction_x",
                        "direction_y",
                    ],
                    "shape": [4, 7, 7],
                },
                "canonical_families": [
                    {
                        "family_id": 0,
                        "route_cells": list(PILOT_ROUTE_A),
                        "train_episode_support": len(split.train),
                    }
                ],
                "training": {"episode_ids": sorted(split.train)},
                "artifact_hashes": {
                    assignments_path.name: sha256_file(assignments_path),
                    fields_path.name: sha256_file(fields_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _features() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (2,)),
        },
        {ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
    )


def _policy_config(kind: str):
    inputs, outputs = _features()
    shared = {
        "input_features": inputs,
        "output_features": outputs,
        "device": "cpu",
        "n_obs_steps": 2,
        "horizon": 16,
        "n_action_steps": 8,
        "down_dims": (16, 32),
        "n_groups": 8,
        "num_train_timesteps": 10,
        "pretrained_backbone_weights": None,
        "route_grid_size": 7,
        "route_field_channels": 4,
        "route_embedding_dim": 8,
        "push_to_hub": False,
    }
    if kind == "bc":
        return RouteFieldBCChunkConfig(**shared, fixed_timestep=0)
    return RouteFieldDiffusionConfig(**shared, num_inference_steps=1)


def test_route_extraction_debounces_boundaries_and_uses_row_major_cells() -> None:
    assert position_to_cell((0.0, 0.0)) == 0
    assert position_to_cell((511.9, 511.9)) == 48

    def cell_center(cell: int) -> tuple[float, float]:
        row, column = divmod(cell, 7)
        return ((column + 0.5) * 512 / 7, (row + 0.5) * 512 / 7)

    cell_16 = cell_center(16)
    cell_17 = cell_center(17)
    cell_24 = cell_center(24)
    positions = [
        *([cell_16] * 3),
        cell_17,
        cell_16,
        *([cell_17] * 3),
        *([cell_24] * 3),
    ]

    assert extract_stable_route(positions) == PILOT_ROUTE_A


def test_route_field_encodes_occupancy_progress_and_direction() -> None:
    field_a = route_to_field(PILOT_ROUTE_A)
    field_b = route_to_field(PILOT_ROUTE_B)

    assert field_a.shape == (4, 7, 7)
    assert field_a.dtype == np.float32
    assert field_a[0].sum() == pytest.approx(3.0)
    assert field_a[1, 2, 2] == pytest.approx(0.0)
    assert field_a[1, 2, 3] == pytest.approx(0.5)
    assert field_a[1, 3, 3] == pytest.approx(1.0)
    assert field_a[2, 2, 2] == pytest.approx(1.0)
    assert field_a[3, 2, 3] == pytest.approx(1.0)
    assert field_b[3, 2, 2] == pytest.approx(1.0)
    assert field_b[2, 3, 2] == pytest.approx(1.0)
    assert not np.array_equal(field_a, field_b)
    with pytest.raises(ValueError, match="repeated"):
        route_to_field((16, 17, 16))


def test_route_field_supports_single_cell_and_audits_route_simplification() -> None:
    single = route_to_field((16,))
    assert single[0, 2, 2] == pytest.approx(1.0)
    assert single[1, 2, 2] == pytest.approx(1.0)
    assert single[2:].sum() == pytest.approx(0.0)

    assert simplify_macro_route((1, 2, 1, 3)) == ((1, 3), 2, 0)
    assert simplify_macro_route((1, 2, 3, 1, 4)) == ((1, 4), 0, 3)


def test_route_sidecar_and_dataset_attach_canonical_field_read_only(tmp_path: Path) -> None:
    manifest = _write_sidecar(tmp_path)
    sidecar = load_route_field_sidecar(manifest)

    class Dataset:
        def __init__(self) -> None:
            self.episodes = [205, 74]

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict:
            return {
                "episode_index": torch.tensor(self.episodes[index]),
                "value": torch.tensor(index),
            }

    dataset = Dataset()
    wrapped = RouteFieldConditionedDataset(dataset, sidecar)

    assert torch.equal(wrapped[0][ROUTE_FIELD], wrapped[1][ROUTE_FIELD])
    assert ROUTE_FIELD not in dataset[0]
    assert sidecar.training_episodes(load_split().smoke_train) == load_split().smoke_train


@pytest.mark.parametrize(
    ("policy_class", "kind"),
    [
        (RouteFieldBCChunkPolicy, "bc"),
        (RouteFieldDiffusionPolicy, "diffusion"),
    ],
)
def test_route_field_policies_use_the_spatial_condition(policy_class, kind: str) -> None:
    policy = policy_class(_policy_config(kind)).eval()
    base = {
        OBS_STATE: torch.randn(1, 2, 2),
        OBS_ENV_STATE: torch.randn(1, 2, 2),
    }
    first = policy.diffusion._prepare_global_conditioning(
        {**base, ROUTE_FIELD: torch.from_numpy(route_to_field(PILOT_ROUTE_A))[None]}
    )
    second = policy.diffusion._prepare_global_conditioning(
        {**base, ROUTE_FIELD: torch.from_numpy(route_to_field(PILOT_ROUTE_B))[None]}
    )

    assert torch.equal(first[:, :-8], second[:, :-8])
    assert not torch.equal(first[:, -8:], second[:, -8:])


def test_route_field_training_resolution_keeps_all_smoke_episodes(tmp_path: Path) -> None:
    manifest = _write_sidecar(tmp_path)
    arguments, resolved = resolve_bc_training(
        [
            "--mode=smoke",
            f"--route-manifest={manifest}",
            "--steps=3",
            "--batch-size=2",
            "--num-workers=0",
            "--device=cpu",
            f"--output-dir={tmp_path / 'run'}",
            "--print-command",
        ]
    )

    assert arguments.print_command
    assert resolved.episodes == load_split().smoke_train
    assert resolved.config.dataset.episodes == list(load_split().smoke_train)
    assert resolved.route_manifest == manifest.resolve()
    assert resolved.summary()["route_field_conditioning"] == {
        "manifest": str(manifest.resolve()),
        "grid_size": 7,
        "field_channels": 4,
        "embedding_dim": 64,
    }
    validate_training_inputs(resolved, check_run_destination=False)


def test_route_field_scripts_and_vscode_entries_are_direct(tmp_path: Path) -> None:
    del tmp_path
    script_root = PRACTICE_ROOT / "scripts" / "experiments" / "pusht"
    for name, module in (
        ("build_route_fields.sh", "route_field_data"),
        ("train_route_field_bc_chunk.sh", "train_route_field_bc_chunk"),
        ("train_route_field_diffusion.sh", "train_route_field_diffusion"),
    ):
        script = script_root / name
        text = script.read_text(encoding="utf-8")
        assert script.stat().st_mode & 0o111
        assert f"pusht_diffusion.{module}" in text
        assert '"$@"' in text

    launch = json.loads((PRACTICE_ROOT / ".vscode" / "launch.json").read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in launch["configurations"]}
    assert by_name["PushT E6: RouteField-BCChunk train (3-step debug)"]["module"].endswith(
        "train_route_field_bc_chunk"
    )
    assert by_name["PushT E6: RouteField-Diffusion train (3-step debug)"]["module"].endswith(
        "train_route_field_diffusion"
    )
