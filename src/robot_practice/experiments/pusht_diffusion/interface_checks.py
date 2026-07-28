"""Phase-0 PushT interface, RNG, environment replay, and provenance checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .config import (
    ACTION_DIM,
    DATASET_REPO_ID,
    DATASET_REVISION,
    EXECUTION_SLICE,
    HORIZON,
    N_ACTION_STEPS,
    N_OBS_STEPS,
    default_dataset_root,
    default_output_root,
    practice_root,
)
from .dataset_split import load_split
from .noise import make_initial_noise
from .output import AtomicOutputDirectory, write_metrics
from .policy_loading import controlled_sample
from .provenance import collect_runtime_provenance, sha256_file, write_json_atomic
from .rollout import check_environment_replay


def _feature_maps() -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature]]:
    return (
        {
            "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
            OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
        },
        {ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
    )


def _reference_policy(device: str) -> DiffusionPolicy:
    inputs, outputs = _feature_maps()
    config = DiffusionConfig(
        input_features=inputs,
        output_features=outputs,
        device=device,
        n_obs_steps=N_OBS_STEPS,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        crop_shape=(84, 84),
        crop_is_random=True,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=8,
        down_dims=(16, 32),
        n_groups=8,
        num_train_timesteps=100,
        num_inference_steps=3,
        noise_scheduler_type="DDPM",
        do_mask_loss_for_padding=False,
        drop_n_last_frames=7,
        push_to_hub=False,
    )
    policy = DiffusionPolicy(config).eval()
    return policy


def _frozen_batch(policy: DiffusionPolicy) -> dict[str, torch.Tensor]:
    parameter = next(policy.parameters())
    generator = torch.Generator(device=parameter.device).manual_seed(20260727)
    return {
        OBS_STATE: torch.rand(
            (1, N_OBS_STEPS, 2),
            generator=generator,
            device=parameter.device,
            dtype=parameter.dtype,
        ),
        OBS_IMAGES: torch.rand(
            (1, N_OBS_STEPS, 1, 3, 96, 96),
            generator=generator,
            device=parameter.device,
            dtype=parameter.dtype,
        ),
    }


def _dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return {
            "present": False,
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_revision": DATASET_REVISION,
        }
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    split = load_split()
    info_hash = sha256_file(info_path)
    if info_hash != split.metadata_info_sha256:
        raise ValueError("Local PushT metadata does not match the committed revision")
    total_episodes = info.get("total_episodes")
    if total_episodes is not None and int(total_episodes) != 206:
        raise ValueError(f"Expected 206 PushT episodes, got {total_episodes}")
    return {
        "present": True,
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "metadata_info_sha256": info_hash,
        "total_episodes": total_episodes,
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "features": info.get("features"),
    }


def run_interface_checks(
    *,
    output_dir: Path,
    dataset_root: Path,
    device: str,
    overwrite: bool,
) -> Path:
    resolved_device = device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy = _reference_policy(resolved_device)
    frozen = _frozen_batch(policy)
    parameter = next(policy.parameters())
    noise_a = make_initial_noise(
        seed=11,
        batch_size=1,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    noise_b = make_initial_noise(
        seed=12,
        batch_size=1,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    with torch.inference_mode():
        first = controlled_sample(
            policy,
            frozen,
            initial_noise=noise_a.tensor,
            reverse_seed=31,
        )
        repeated = controlled_sample(
            policy,
            frozen,
            initial_noise=noise_a.tensor,
            reverse_seed=31,
        )
        different = controlled_sample(
            policy,
            frozen,
            initial_noise=noise_b.tensor,
            reverse_seed=31,
        )
    same_noise_error = float(torch.max(torch.abs(first.actions - repeated.actions)).cpu())
    different_noise_error = float(torch.max(torch.abs(first.actions - different.actions)).cpu())
    if tuple(first.full_horizon.shape) != (1, HORIZON, ACTION_DIM):
        raise RuntimeError(f"Unexpected full horizon shape: {tuple(first.full_horizon.shape)}")
    if tuple(first.actions.shape) != (1, N_ACTION_STEPS, ACTION_DIM):
        raise RuntimeError(f"Unexpected public action chunk shape: {tuple(first.actions.shape)}")
    if same_noise_error > 1e-6:
        raise RuntimeError(
            f"Same-noise reproducibility error is {same_noise_error}, expected <=1e-6"
        )
    if different_noise_error <= 1e-6:
        raise RuntimeError(
            "Different explicit initial noise did not change the sampled action chunk"
        )

    replay = check_environment_replay(
        environment_seed=2000,
        actions=np.asarray([[256.0, 256.0], [260.0, 260.0]], dtype=np.float32),
    )
    runtime = collect_runtime_provenance(practice_root())
    dataset_info = _dataset_info(dataset_root)
    diffusion_config = {
        "n_obs_steps": N_OBS_STEPS,
        "horizon": HORIZON,
        "n_action_steps": N_ACTION_STEPS,
        "action_dim": ACTION_DIM,
        "execution_slice": [EXECUTION_SLICE.start, EXECUTION_SLICE.stop],
        "noise_shape": [1, HORIZON, ACTION_DIM],
        "reference_scheduler": "DDPM",
        "reference_num_inference_steps": 100,
    }
    bc_config = {
        **diffusion_config,
        "policy_type": "bc_chunk",
        "fixed_timestep": 0,
        "training_target": "clean_action",
    }
    checks = {
        "same_noise_max_error": same_noise_error,
        "different_noise_max_error": different_noise_error,
        "full_horizon_shape": list(first.full_horizon.shape),
        "action_chunk_shape": list(first.actions.shape),
        "execution_slice": [EXECUTION_SLICE.start, EXECUTION_SLICE.stop],
        "environment_replay": replay,
        "passed": True,
    }

    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        write_json_atomic(temporary / "dataset_info.json", dataset_info)
        write_json_atomic(temporary / "diffusion_config.json", diffusion_config)
        write_json_atomic(temporary / "bc_config.json", bc_config)
        write_json_atomic(temporary / "interface_checks.json", checks)
        write_json_atomic(temporary / "run_manifest.json", runtime)
        commit = runtime["repository"]["commit"] or "unknown"
        (temporary / "git_commit.txt").write_text(f"{commit}\n", encoding="utf-8")
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        (temporary / "environment.txt").write_text(freeze, encoding="utf-8")
        write_metrics(
            temporary,
            {
                "same_noise_max_error": same_noise_error,
                "different_noise_max_error": different_noise_error,
                "environment_replay": replay["reproduced"],
                "dataset_present": dataset_info["present"],
            },
            comments={
                "same_noise_max_error": "Maximum normalized chunk error for identical RNG inputs",
                "different_noise_max_error": "Maximum normalized chunk change from initial noise",
                "environment_replay": "Same reset seed and actions reproduced the trajectory",
                "dataset_present": "Local public dataset metadata was available for inspection",
            },
        )
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "metadata",
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run_interface_checks(
        output_dir=args.output_dir.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"Phase-0 checks passed: {result}")


if __name__ == "__main__":
    main()
