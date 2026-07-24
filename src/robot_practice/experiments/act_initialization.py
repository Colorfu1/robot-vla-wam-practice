"""Create and validate reproducible step-0 ACT comparison checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.random_utils import set_seed

from robot_practice.evaluation.common import (
    iso_timestamp,
    package_version,
    practice_root,
    validate_complete_checkpoint,
    write_json_atomic,
)
from robot_practice.experiments.act_a_sanity import (
    BACKBONE_WEIGHTS,
    CAMERA_KEYS,
    IMAGE_SHAPE,
    verify_pretrained_backbone,
    verify_saved_policy,
)

DATASET_REPO_ID = "lerobot/libero"
TASK_INDEX = 29
TASK_SUITE = "libero_object"
SIMULATOR_TASK_ID = 8
EPISODES = (
    823,
    826,
    834,
    840,
    841,
    845,
    847,
    869,
    875,
    878,
    910,
    918,
    920,
    929,
    944,
    948,
    949,
    953,
    966,
    972,
)
EXPECTED_FRAME_COUNT = 3233
STATE_SHAPE = (8,)
ACTION_SHAPE = (7,)
FPS = 10.0
SEED = 1000
BATCH_SIZE = 8
PRETRAINED_MODEL_RELATIVE_PATH = Path("checkpoints/000000/pretrained_model")


@dataclass(frozen=True)
class ACTVariant:
    name: str
    chunk_size: int
    n_action_steps: int
    output_subdirectory: str


VARIANTS = {
    "a0": ACTVariant(
        name="a0",
        chunk_size=1,
        n_action_steps=1,
        output_subdirectory="a0_single_step",
    ),
    "a1": ACTVariant(
        name="a1",
        chunk_size=20,
        n_action_steps=5,
        output_subdirectory="a1_chunking",
    ),
}


@dataclass(frozen=True)
class InitializationResult:
    run_root: Path
    policy_path: Path
    model_sha256: str
    total_parameters: int
    learnable_parameters: int


def default_dataset_root() -> Path:
    return practice_root() / "datasets" / "lerobot" / "lerobot" / "libero"


def resolve_variant(name: str) -> ACTVariant:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ACT comparison variant {name!r}; expected one of {sorted(VARIANTS)}"
        ) from exc


def build_step_zero_train_config(
    *,
    variant: str,
    output_dir: Path,
    dataset_root: Path,
    device: str | None = None,
) -> TrainPipelineConfig:
    definition = resolve_variant(variant)
    policy = ACTConfig(
        device=device,
        chunk_size=definition.chunk_size,
        n_action_steps=definition.n_action_steps,
        use_vae=False,
        pretrained_backbone_weights=BACKBONE_WEIGHTS,
        push_to_hub=False,
    )
    dataset = DatasetConfig(
        repo_id=DATASET_REPO_ID,
        root=str(dataset_root.expanduser().resolve()),
        episodes=list(EPISODES),
        use_imagenet_stats=True,
    )
    return TrainPipelineConfig(
        dataset=dataset,
        policy=policy,
        output_dir=output_dir.expanduser().resolve(),
        seed=SEED,
        batch_size=BATCH_SIZE,
        steps=0,
        eval_freq=0,
        log_freq=100,
        save_checkpoint=True,
        save_freq=20_000,
        wandb=WandBConfig(enable=False),
    )


def validate_policy_contract(config: ACTConfig, *, variant: str) -> None:
    definition = resolve_variant(variant)
    mismatches: list[str] = []
    expected_scalars = {
        "chunk_size": definition.chunk_size,
        "n_action_steps": definition.n_action_steps,
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": BACKBONE_WEIGHTS,
    }
    for name, expected in expected_scalars.items():
        actual = getattr(config, name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r}, expected {expected!r}")
    if config.use_vae:
        mismatches.append("use_vae must be false")
    if config.temporal_ensemble_coeff is not None:
        mismatches.append("temporal_ensemble_coeff must be null")

    state = config.robot_state_feature
    if state is None or tuple(state.shape) != STATE_SHAPE:
        actual = None if state is None else tuple(state.shape)
        mismatches.append(f"observation.state shape={actual}, expected {STATE_SHAPE}")
    action = config.action_feature
    if action is None or tuple(action.shape) != ACTION_SHAPE:
        actual = None if action is None else tuple(action.shape)
        mismatches.append(f"action shape={actual}, expected {ACTION_SHAPE}")

    images = config.image_features
    if set(images) != set(CAMERA_KEYS):
        mismatches.append(
            f"camera keys={sorted(images)}, expected {sorted(CAMERA_KEYS)}"
        )
    else:
        for key in CAMERA_KEYS:
            if tuple(images[key].shape) != IMAGE_SHAPE:
                mismatches.append(
                    f"{key} shape={tuple(images[key].shape)}, expected {IMAGE_SHAPE}"
                )

    expected_inputs = {OBS_STATE, *CAMERA_KEYS}
    if config.input_features is None or set(config.input_features) != expected_inputs:
        actual = [] if config.input_features is None else sorted(config.input_features)
        mismatches.append(f"input feature keys={actual}, expected {sorted(expected_inputs)}")
    if config.output_features is None or set(config.output_features) != {ACTION}:
        actual = [] if config.output_features is None else sorted(config.output_features)
        mismatches.append(f"output feature keys={actual}, expected {[ACTION]}")

    if mismatches:
        raise ValueError(
            f"Incompatible ACT comparison checkpoint for {variant}: "
            + "; ".join(mismatches)
        )


def validate_policy_path(policy_path: Path, *, variant: str) -> Path:
    manifest = validate_complete_checkpoint(policy_path)
    config = PreTrainedConfig.from_pretrained(manifest.path, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise TypeError(f"Expected ACTConfig, got {type(config).__name__}")
    validate_policy_contract(config, variant=variant)
    return manifest.path


def _validate_dataset_contract(dataset: Any) -> None:
    if tuple(int(index) for index in dataset.episodes) != EPISODES:
        raise ValueError(f"Expected episodes {list(EPISODES)}, got {list(dataset.episodes)}")
    if len(dataset) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FRAME_COUNT} selected frames, got {len(dataset)}"
        )
    if float(dataset.fps) != FPS:
        raise ValueError(f"Expected dataset fps {FPS}, got {dataset.fps}")
    if tuple(dataset.meta.camera_keys) != CAMERA_KEYS:
        raise ValueError(
            f"Expected camera keys {list(CAMERA_KEYS)}, got {list(dataset.meta.camera_keys)}"
        )
    task_indices = {int(value) for value in dataset.hf_dataset.unique("task_index")}
    if task_indices != {TASK_INDEX}:
        raise ValueError(
            f"All comparison episodes must use task index {TASK_INDEX}, got "
            f"{sorted(task_indices)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_to_workspace(path: Path) -> str:
    return os.path.relpath(path.resolve(), start=practice_root().resolve())


def _camera_stats(dataset: Any) -> dict[str, dict[str, list[Any]]]:
    return {
        key: {
            "mean": torch.as_tensor(dataset.meta.stats[key]["mean"]).tolist(),
            "std": torch.as_tensor(dataset.meta.stats[key]["std"]).tolist(),
        }
        for key in CAMERA_KEYS
    }


def _initialization_manifest(
    *,
    run_root: Path,
    policy_path: Path,
    config: ACTConfig,
    dataset: Any,
    variant: ACTVariant,
    model_sha256: str,
    backbone_tensor_count: int,
    serialized_tensor_count: int,
    total_parameters: int,
    learnable_parameters: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at": iso_timestamp(),
        "experiment": "act_action_chunking_comparison",
        "variant": variant.name,
        "initialization_step": 0,
        "optimizer_updates": 0,
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "root": _relative_to_workspace(Path(dataset.root)),
            "task_suite": TASK_SUITE,
            "simulator_task_id": SIMULATOR_TASK_ID,
            "task_index": TASK_INDEX,
            "episodes": list(EPISODES),
            "num_frames": len(dataset),
            "fps": float(dataset.fps),
            "use_imagenet_stats": True,
            "camera_stats": _camera_stats(dataset),
        },
        "observations": {
            "camera_keys": list(CAMERA_KEYS),
            "camera_shape": list(IMAGE_SHAPE),
            "state_key": OBS_STATE,
            "state_shape": list(STATE_SHAPE),
            "state_description": "eef_position_3 + axis_angle_3 + gripper_qpos_2",
        },
        "action": {
            "key": ACTION,
            "shape": list(ACTION_SHAPE),
            "space": "relative_eef_translation_3 + relative_rotation_3 + gripper_1",
            "chunk_size": config.chunk_size,
            "n_action_steps": config.n_action_steps,
        },
        "policy": {
            "type": "act",
            "seed": SEED,
            "use_vae": config.use_vae,
            "vision_backbone": config.vision_backbone,
            "pretrained_backbone_weights": config.pretrained_backbone_weights,
            "pretrained_backbone_verified": True,
            "backbone_tensor_count": backbone_tensor_count,
            "total_parameters": total_parameters,
            "learnable_parameters": learnable_parameters,
            "push_to_hub": config.push_to_hub,
        },
        "runtime": {
            "python": platform.python_version(),
            "lerobot": package_version("lerobot"),
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "device": str(config.device),
        },
        "checkpoint": {
            "path": str(policy_path.relative_to(run_root)),
            "model_sha256": model_sha256,
            "serialized_tensor_count": serialized_tensor_count,
            "resumable_training_state": False,
        },
    }


def create_initialization_checkpoint(
    *,
    variant: str,
    output_dir: Path,
    dataset_root: Path,
    device: str | None = None,
) -> InitializationResult:
    definition = resolve_variant(variant)
    final_root = output_dir.expanduser().resolve()
    resolved_dataset_root = dataset_root.expanduser().resolve()
    if final_root.exists():
        raise FileExistsError(f"Output directory already exists: {final_root}")
    if not (resolved_dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LIBERO dataset is missing at {resolved_dataset_root}")

    cfg = build_step_zero_train_config(
        variant=definition.name,
        output_dir=final_root,
        dataset_root=resolved_dataset_root,
        device=device,
    )
    cfg.validate()
    if cfg.policy is None or not isinstance(cfg.policy, ACTConfig):
        raise TypeError("ACT comparison initialization requires an ACTConfig")

    set_seed(SEED)
    dataset = make_dataset(cfg)
    _validate_dataset_contract(dataset)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    if not isinstance(policy, ACTPolicy):
        raise TypeError(f"Expected ACTPolicy, got {type(policy).__name__}")
    validate_policy_contract(policy.config, variant=definition.name)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset.meta.stats,
    )
    backbone_tensor_count = verify_pretrained_backbone(policy)
    total_parameters = sum(parameter.numel() for parameter in policy.parameters())
    learnable_parameters = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )

    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = final_root.with_name(f".{final_root.name}.incomplete.{os.getpid()}")
    temporary_root.mkdir(exist_ok=False)
    (temporary_root / ".incomplete").write_text(
        "Step-0 ACT comparison checkpoint creation did not complete atomically.\n",
        encoding="utf-8",
    )

    try:
        temporary_policy_path = temporary_root / PRETRAINED_MODEL_RELATIVE_PATH
        policy.save_pretrained(temporary_policy_path)
        cfg.save_pretrained(temporary_policy_path)
        preprocessor.save_pretrained(
            temporary_policy_path,
            config_filename="policy_preprocessor.json",
        )
        postprocessor.save_pretrained(
            temporary_policy_path,
            config_filename="policy_postprocessor.json",
        )
        validate_complete_checkpoint(temporary_policy_path)
        serialized_tensor_count = verify_saved_policy(
            policy,
            temporary_policy_path / "model.safetensors",
        )
        model_sha256 = _sha256(temporary_policy_path / "model.safetensors")
        final_policy_path = final_root / PRETRAINED_MODEL_RELATIVE_PATH
        manifest = _initialization_manifest(
            run_root=final_root,
            policy_path=final_policy_path,
            config=policy.config,
            dataset=dataset,
            variant=definition,
            model_sha256=model_sha256,
            backbone_tensor_count=backbone_tensor_count,
            serialized_tensor_count=serialized_tensor_count,
            total_parameters=total_parameters,
            learnable_parameters=learnable_parameters,
        )
        write_json_atomic(temporary_root / "initialization_manifest.json", manifest)
        temporary_root.rename(final_root)
        (final_root / ".incomplete").unlink()
    except BaseException:
        if temporary_root.exists():
            import shutil

            shutil.rmtree(temporary_root)
        raise

    return InitializationResult(
        run_root=final_root,
        policy_path=final_root / PRETRAINED_MODEL_RELATIVE_PATH,
        model_sha256=model_sha256,
        total_parameters=total_parameters,
        learnable_parameters=learnable_parameters,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create one fixed step-0 checkpoint")
    create.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    create.add_argument("--output-dir", required=True, type=Path)
    create.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    create.add_argument("--device", choices=("cpu", "cuda", "mps"))

    validate = subparsers.add_parser(
        "validate-policy",
        help="Validate a checkpoint before positive-step comparison training",
    )
    validate.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    validate.add_argument("--policy-path", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    if args.command == "validate-policy":
        validate_policy_path(args.policy_path, variant=args.variant)
        return
    if args.command == "create":
        result = create_initialization_checkpoint(
            variant=args.variant,
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
            device=args.device,
        )
        print(f"ACT {args.variant} step-0 checkpoint: {result.policy_path}")
        print(f"model.safetensors sha256: {result.model_sha256}")
        print(f"total parameters: {result.total_parameters}")
        print(f"learnable parameters: {result.learnable_parameters}")
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
