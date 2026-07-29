"""Create paired Mode-BCChunk and Mode-Diffusion step-zero checkpoints."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.random_utils import set_seed

from robot_practice.policies.mode_bc_chunk import ModeBCChunkPolicy
from robot_practice.policies.mode_diffusion import ModeDiffusionPolicy

from .config import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_TRAIN_SEED,
    default_output_root,
)
from .dataset_split import EpisodeSplit, load_split
from .e5_dataset import load_prototype_sidecar
from .initialization import _save_checkpoint, _train_config
from .model_configs import model_config_fingerprint
from .model_configs.mode_bc_chunk import make_mode_bc_chunk_config
from .model_configs.mode_diffusion import make_mode_diffusion_config
from .provenance import (
    collect_runtime_provenance,
    sha256_file,
    state_dict_hashes,
    write_json_atomic,
)
from .training_common import resolve_device


@dataclass(frozen=True)
class E5PairedInitialization:
    root: Path
    mode_diffusion_path: Path
    mode_bc_chunk_path: Path
    shared_tensor_count: int


def create_e5_paired_initialization(
    *,
    dataset_root: str | Path,
    prototype_manifest: str | Path,
    output_dir: str | Path,
    split: EpisodeSplit,
    mode: str,
    train_seed: int = DEFAULT_TRAIN_SEED,
    device: str | None = None,
    overwrite: bool = False,
) -> E5PairedInitialization:
    if mode not in {"smoke", "pilot"}:
        raise ValueError("E5 initialization mode must be smoke or pilot")
    episodes = split.episodes_for_mode(mode)
    dataset_path = Path(dataset_root).expanduser().resolve()
    final_root = Path(output_dir).expanduser().resolve()
    sidecar = load_prototype_sidecar(prototype_manifest)
    if sidecar.manifest.get("split_manifest_sha256") != split.manifest_sha256:
        raise ValueError("E5 prototype sidecar does not match the selected split")
    for episode in episodes:
        sidecar.condition_for_episode(episode)

    view_manifest = dataset_path / "view_manifest.json"
    if not view_manifest.is_file():
        raise FileNotFoundError(
            f"Split-specific dataset view is missing at {dataset_path}; run prepare_dataset first"
        )
    if final_root.exists():
        if not overwrite:
            raise FileExistsError(f"E5 initialization output already exists: {final_root}")
        shutil.rmtree(final_root)

    resolved_device = resolve_device(device or "auto")
    diffusion_config = make_mode_diffusion_config(
        device=resolved_device,
        num_prototypes=sidecar.selected_k,
    )
    bc_config = make_mode_bc_chunk_config(
        device=resolved_device,
        num_prototypes=sidecar.selected_k,
    )
    config_fingerprints = {
        "mode_diffusion": model_config_fingerprint(diffusion_config),
        "mode_bc_chunk": model_config_fingerprint(bc_config),
    }
    diffusion_train_config = _train_config(
        dataset_root=dataset_path,
        episodes=episodes,
        policy=diffusion_config,
        output_dir=final_root / "unused_mode_diffusion_train_root",
        train_seed=train_seed,
    )

    set_seed(train_seed)
    dataset = make_dataset(diffusion_train_config)
    if tuple(int(value) for value in dataset.episodes) != tuple(episodes):
        raise ValueError("Loaded dataset episodes do not match the committed split")
    diffusion_policy = make_policy(cfg=diffusion_config, ds_meta=dataset.meta)
    if not isinstance(diffusion_policy, ModeDiffusionPolicy):
        raise TypeError(f"Expected ModeDiffusionPolicy, got {type(diffusion_policy).__name__}")

    bc_train_config = _train_config(
        dataset_root=dataset_path,
        episodes=episodes,
        policy=bc_config,
        output_dir=final_root / "unused_mode_bc_train_root",
        train_seed=train_seed,
    )
    bc_policy = make_policy(cfg=bc_config, ds_meta=dataset.meta)
    if not isinstance(bc_policy, ModeBCChunkPolicy):
        raise TypeError(f"Expected ModeBCChunkPolicy, got {type(bc_policy).__name__}")
    missing, unexpected = bc_policy.load_state_dict(
        diffusion_policy.state_dict(),
        strict=True,
    )
    if missing or unexpected:
        raise RuntimeError(
            f"E5 shared initialization failed: missing={missing}, unexpected={unexpected}"
        )
    diffusion_hashes = state_dict_hashes(diffusion_policy.state_dict())
    bc_hashes = state_dict_hashes(bc_policy.state_dict())
    if diffusion_hashes != bc_hashes:
        raise RuntimeError("Mode-BCChunk and Mode-Diffusion initialization hashes differ")

    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_root.with_name(f".{final_root.name}.incomplete.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        diffusion_path = temporary / "mode_diffusion"
        bc_path = temporary / "mode_bc_chunk"
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
            "experiment": "E5 mode-aware paired initialization",
            "mode": mode,
            "train_seed": train_seed,
            "device": resolved_device,
            "dataset_repo_id": DATASET_REPO_ID,
            "dataset_revision": DATASET_REVISION,
            "episodes": list(episodes),
            "split_manifest_sha256": split.manifest_sha256,
            "dataset_view_manifest_sha256": sha256_file(view_manifest),
            "prototype_manifest_sha256": sha256_file(sidecar.manifest_path),
            "num_prototypes": sidecar.selected_k,
            "null_prototype_id": sidecar.null_prototype_id,
            "model_config_modules": {
                "mode_bc_chunk": (
                    "robot_practice.experiments.pusht_diffusion.model_configs.mode_bc_chunk"
                ),
                "mode_diffusion": (
                    "robot_practice.experiments.pusht_diffusion.model_configs.mode_diffusion"
                ),
            },
            "model_config_fingerprints": config_fingerprints,
            "shared_tensor_count": len(diffusion_hashes),
            "shared_tensor_hashes": diffusion_hashes,
            "model_files": {
                "mode_diffusion": sha256_file(diffusion_path / "model.safetensors"),
                "mode_bc_chunk": sha256_file(bc_path / "model.safetensors"),
            },
            "runtime": collect_runtime_provenance(Path(__file__).resolve().parents[4]),
        }
        write_json_atomic(temporary / "initialization_manifest.json", manifest)
        temporary.rename(final_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return E5PairedInitialization(
        root=final_root,
        mode_diffusion_path=final_root / "mode_diffusion",
        mode_bc_chunk_path=final_root / "mode_bc_chunk",
        shared_tensor_count=len(diffusion_hashes),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prototype-manifest", type=Path, required=True)
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
            default_output_root()
            / "initialization"
            / args.mode
            / f"seed_{args.train_seed}"
            / "e5_mode_conditioned"
        )
    result = create_e5_paired_initialization(
        dataset_root=args.dataset_root,
        prototype_manifest=args.prototype_manifest,
        output_dir=output_dir,
        split=split,
        mode=args.mode,
        train_seed=args.train_seed,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"Mode-Diffusion step-zero checkpoint: {result.mode_diffusion_path}")
    print(f"Mode-BCChunk step-zero checkpoint: {result.mode_bc_chunk_path}")
    print(f"Shared tensor hashes verified: {result.shared_tensor_count}")


if __name__ == "__main__":
    main()


__all__ = ["E5PairedInitialization", "create_e5_paired_initialization"]
