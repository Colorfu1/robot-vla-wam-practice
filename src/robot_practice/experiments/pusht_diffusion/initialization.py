"""Create paired BCChunk and Diffusion step-zero checkpoints."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed

from robot_practice.policies.bc_chunk.configuration_bc_chunk import BCChunkConfig
from robot_practice.policies.bc_chunk.modeling_bc_chunk import BCChunkPolicy

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    default_output_root,
)
from .dataset_split import EpisodeSplit, load_split
from .model_configs import model_config_fingerprint
from .model_configs.bc_chunk import make_bc_chunk_config
from .model_configs.diffusion import make_diffusion_config
from .provenance import (
    collect_runtime_provenance,
    sha256_file,
    state_dict_hashes,
    write_json_atomic,
)
from .training_common import resolve_device


@dataclass(frozen=True)
class PairedInitialization:
    root: Path
    diffusion_path: Path
    bc_chunk_path: Path
    shared_tensor_count: int


def _train_config(
    *,
    dataset_root: Path,
    episodes: Sequence[int],
    policy: DiffusionConfig | BCChunkConfig,
    output_dir: Path,
    train_seed: int,
) -> TrainPipelineConfig:
    return TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=DATASET_REPO_ID,
            root=str(dataset_root),
            episodes=list(episodes),
            revision=DATASET_REVISION,
            use_imagenet_stats=False,
        ),
        policy=policy,
        output_dir=output_dir,
        seed=train_seed,
        batch_size=64,
        steps=0,
        eval_freq=0,
        log_freq=100,
        save_checkpoint=True,
        save_freq=25_000,
        wandb=WandBConfig(enable=False),
    )


def _save_checkpoint(
    *,
    destination: Path,
    policy: DiffusionPolicy | BCChunkPolicy,
    train_config: TrainPipelineConfig,
    dataset_stats: dict[str, Any],
) -> None:
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset_stats,
    )
    policy.save_pretrained(destination)
    train_config.save_pretrained(destination)
    preprocessor.save_pretrained(
        destination,
        config_filename="policy_preprocessor.json",
    )
    postprocessor.save_pretrained(
        destination,
        config_filename="policy_postprocessor.json",
    )
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    }
    missing = sorted(name for name in required if not (destination / name).is_file())
    if missing:
        raise RuntimeError(f"Incomplete initialization checkpoint: {missing}")


def create_paired_initialization(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    split: EpisodeSplit,
    mode: str,
    train_seed: int = DEFAULT_TRAIN_SEED,
    device: str | None = None,
    overwrite: bool = False,
) -> PairedInitialization:
    episodes = split.episodes_for_mode(mode)
    if mode not in {"smoke", "pilot"}:
        raise ValueError("Paired initialization mode must be smoke or pilot")
    dataset_path = Path(dataset_root).expanduser().resolve()
    final_root = Path(output_dir).expanduser().resolve()
    if final_root.exists():
        if not overwrite:
            raise FileExistsError(f"Initialization output already exists: {final_root}")
        shutil.rmtree(final_root)
    view_manifest = dataset_path / "view_manifest.json"
    if not view_manifest.is_file():
        raise FileNotFoundError(
            f"Split-specific dataset view is missing at {dataset_path}; run prepare_dataset first"
        )

    resolved_device = resolve_device(device or "auto")
    diffusion_config = make_diffusion_config(device=resolved_device)
    bc_config = make_bc_chunk_config(device=resolved_device)
    config_fingerprints = {
        "diffusion": model_config_fingerprint(diffusion_config),
        "bc_chunk": model_config_fingerprint(bc_config),
    }
    diffusion_train_config = _train_config(
        dataset_root=dataset_path,
        episodes=episodes,
        policy=diffusion_config,
        output_dir=final_root / "unused_diffusion_train_root",
        train_seed=train_seed,
    )

    set_seed(train_seed)
    dataset = make_dataset(diffusion_train_config)
    if tuple(int(value) for value in dataset.episodes) != tuple(episodes):
        raise ValueError("Loaded dataset episodes do not match the committed split")

    diffusion_policy = make_policy(cfg=diffusion_config, ds_meta=dataset.meta)
    if not isinstance(diffusion_policy, DiffusionPolicy):
        raise TypeError(f"Expected DiffusionPolicy, got {type(diffusion_policy).__name__}")

    bc_train_config = _train_config(
        dataset_root=dataset_path,
        episodes=episodes,
        policy=bc_config,
        output_dir=final_root / "unused_bc_train_root",
        train_seed=train_seed,
    )
    bc_policy = make_policy(cfg=bc_config, ds_meta=dataset.meta)
    if not isinstance(bc_policy, BCChunkPolicy):
        raise TypeError(f"Expected BCChunkPolicy, got {type(bc_policy).__name__}")
    missing, unexpected = bc_policy.load_state_dict(diffusion_policy.state_dict(), strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Shared initialization failed: missing={missing}, unexpected={unexpected}"
        )

    diffusion_hashes = state_dict_hashes(diffusion_policy.state_dict())
    bc_hashes = state_dict_hashes(bc_policy.state_dict())
    if diffusion_hashes != bc_hashes:
        raise RuntimeError("BCChunk and Diffusion common initialization hashes differ")

    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_root.with_name(f".{final_root.name}.incomplete.{os.getpid()}")
    temporary.mkdir()
    try:
        diffusion_path = temporary / "diffusion"
        bc_path = temporary / "bc_chunk"
        _save_checkpoint(
            destination=diffusion_path,
            policy=diffusion_policy,
            train_config=diffusion_train_config,
            dataset_stats=dataset.meta.stats,
        )
        _save_checkpoint(
            destination=bc_path,
            policy=bc_policy,
            train_config=bc_train_config,
            dataset_stats=dataset.meta.stats,
        )
        manifest = {
            "schema_version": 2,
            "status": "complete",
            "mode": mode,
            "train_seed": train_seed,
            "device": resolved_device,
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_revision": DATASET_REVISION,
            "episodes": list(episodes),
            "split_manifest_sha256": split.manifest_sha256,
            "dataset_view_manifest_sha256": sha256_file(view_manifest),
            "model_config_modules": {
                "bc_chunk": ("robot_practice.experiments.pusht_diffusion.model_configs.bc_chunk"),
                "diffusion": ("robot_practice.experiments.pusht_diffusion.model_configs.diffusion"),
            },
            "model_config_fingerprints": config_fingerprints,
            "shared_tensor_count": len(diffusion_hashes),
            "shared_tensor_hashes": diffusion_hashes,
            "model_files": {
                "diffusion": sha256_file(diffusion_path / "model.safetensors"),
                "bc_chunk": sha256_file(bc_path / "model.safetensors"),
            },
            "runtime": collect_runtime_provenance(Path(__file__).resolve().parents[4]),
        }
        write_json_atomic(temporary / "initialization_manifest.json", manifest)
        temporary.rename(final_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return PairedInitialization(
        root=final_root,
        diffusion_path=final_root / "diffusion",
        bc_chunk_path=final_root / "bc_chunk",
        shared_tensor_count=len(diffusion_hashes),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    split = load_split(args.split_file)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            default_output_root() / "initialization" / args.mode / f"seed_{args.train_seed}"
        )
    result = create_paired_initialization(
        dataset_root=args.dataset_root,
        output_dir=output_dir,
        split=split,
        mode=args.mode,
        train_seed=args.train_seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"Diffusion step-zero checkpoint: {result.diffusion_path}")
    print(f"BCChunk step-zero checkpoint: {result.bc_chunk_path}")
    print(f"Shared tensor hashes verified: {result.shared_tensor_count}")


if __name__ == "__main__":
    main()
