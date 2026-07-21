"""Create and validate the fixed ACT A-Sanity initialization checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file
from torchvision.models import ResNet18_Weights

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

DATASET_REPO_ID = "lerobot/libero"
TASK_INDEX = 29
EPISODES = (823, 826, 834, 840, 841)
EXPECTED_FRAME_COUNT = 859
CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)
STATE_SHAPE = (8,)
ACTION_SHAPE = (7,)
IMAGE_SHAPE = (3, 256, 256)
FPS = 10.0
CHUNK_SIZE = 20
N_ACTION_STEPS = 5
SEED = 1000
BACKBONE_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"
PRETRAINED_MODEL_RELATIVE_PATH = Path("checkpoints/000000/pretrained_model")


@dataclass(frozen=True)
class InitializationResult:
    run_root: Path
    policy_path: Path
    model_sha256: str


def default_dataset_root() -> Path:
    return practice_root() / "datasets" / "lerobot" / "lerobot" / "libero"


def build_step_zero_train_config(
    *,
    output_dir: Path,
    dataset_root: Path,
    device: str | None = None,
) -> TrainPipelineConfig:
    """Build the exact fixed A-Sanity config needed before ``make_dataset``."""

    policy = ACTConfig(
        device=device,
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
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
        batch_size=8,
        steps=0,
        eval_freq=0,
        log_freq=100,
        save_checkpoint=True,
        save_freq=20_000,
        wandb=WandBConfig(enable=False),
    )


def validate_policy_contract(config: ACTConfig) -> None:
    """Require the policy interface and architecture fixed by A-Sanity."""

    mismatches: list[str] = []
    if config.chunk_size != CHUNK_SIZE:
        mismatches.append(f"chunk_size={config.chunk_size}, expected {CHUNK_SIZE}")
    if config.n_action_steps != N_ACTION_STEPS:
        mismatches.append(
            f"n_action_steps={config.n_action_steps}, expected {N_ACTION_STEPS}"
        )
    if config.use_vae:
        mismatches.append("use_vae must be false")
    if config.temporal_ensemble_coeff is not None:
        mismatches.append("temporal_ensemble_coeff must be null")
    if config.vision_backbone != "resnet18":
        mismatches.append(f"vision_backbone={config.vision_backbone!r}, expected 'resnet18'")
    if config.pretrained_backbone_weights != BACKBONE_WEIGHTS:
        mismatches.append(
            "pretrained_backbone_weights="
            f"{config.pretrained_backbone_weights!r}, expected {BACKBONE_WEIGHTS!r}"
        )

    state = config.robot_state_feature
    if state is None or tuple(state.shape) != STATE_SHAPE:
        shape = None if state is None else tuple(state.shape)
        mismatches.append(f"observation.state shape={shape}, expected {STATE_SHAPE}")

    action = config.action_feature
    if action is None or tuple(action.shape) != ACTION_SHAPE:
        shape = None if action is None else tuple(action.shape)
        mismatches.append(f"action shape={shape}, expected {ACTION_SHAPE}")

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
        raise ValueError("Incompatible A-Sanity ACT checkpoint: " + "; ".join(mismatches))


def validate_a_sanity_policy_path(policy_path: Path) -> Path:
    """Validate a complete checkpoint before passing it to ``lerobot-train``."""

    manifest = validate_complete_checkpoint(policy_path)
    config = PreTrainedConfig.from_pretrained(manifest.path, local_files_only=True)
    if not isinstance(config, ACTConfig):
        raise TypeError(f"Expected ACTConfig, got {type(config).__name__}")
    validate_policy_contract(config)
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
            f"All A-Sanity episodes must use task index {TASK_INDEX}, got "
            f"{sorted(task_indices)}"
        )


def _assert_tensor_mappings_equal(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> int:
    missing = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f"{label} reference is missing tensors: {missing}")

    mismatched: list[str] = []
    for name, tensor in actual.items():
        reference = expected[name]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            mismatched.append(name)
            continue
        if not torch.equal(tensor.detach().cpu(), reference.detach().cpu()):
            mismatched.append(name)
    if mismatched:
        preview = ", ".join(mismatched[:10])
        raise ValueError(f"{label} tensor mismatch: {preview}")
    return len(actual)


def verify_pretrained_backbone(
    policy: ACTPolicy,
    *,
    reference_state: Mapping[str, torch.Tensor] | None = None,
) -> int:
    """Prove that the saved ACT backbone still equals torchvision ImageNet weights."""

    if reference_state is None:
        reference_state = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(
            progress=False,
            check_hash=True,
        )
    return _assert_tensor_mappings_equal(
        policy.model.backbone.state_dict(),
        reference_state,
        label="ResNet18 ImageNet backbone",
    )


def verify_saved_policy(policy: ACTPolicy, model_path: Path) -> int:
    """Compare every serialized policy tensor with the in-memory initialization."""

    saved = load_file(str(model_path), device="cpu")
    current = policy.state_dict()
    if set(saved) != set(current):
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        raise ValueError(
            f"Saved policy tensor keys differ: missing={missing}, unexpected={unexpected}"
        )
    return _assert_tensor_mappings_equal(
        current,
        saved,
        label="Saved ACT policy",
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
    result: dict[str, dict[str, list[Any]]] = {}
    for key in CAMERA_KEYS:
        result[key] = {
            "mean": torch.as_tensor(dataset.meta.stats[key]["mean"]).tolist(),
            "std": torch.as_tensor(dataset.meta.stats[key]["std"]).tolist(),
        }
    return result


def _manifest(
    *,
    run_root: Path,
    policy_path: Path,
    config: ACTConfig,
    dataset: Any,
    model_sha256: str,
    backbone_tensor_count: int,
    serialized_tensor_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at": iso_timestamp(),
        "initialization_step": 0,
        "optimizer_updates": 0,
        "dataset": {
            "repo_id": DATASET_REPO_ID,
            "root": _relative_to_workspace(Path(dataset.root)),
            "task_suite": "libero_object",
            "simulator_task_id": 8,
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
    output_dir: Path,
    dataset_root: Path,
    device: str | None = None,
) -> InitializationResult:
    """Create an atomic, complete, zero-update A-Sanity policy checkpoint."""

    final_root = output_dir.expanduser().resolve()
    if final_root.exists():
        raise FileExistsError(f"Output directory already exists: {final_root}")
    if not (dataset_root.expanduser().resolve() / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LIBERO dataset is missing at {dataset_root}")

    cfg = build_step_zero_train_config(
        output_dir=final_root,
        dataset_root=dataset_root,
        device=device,
    )
    cfg.validate()
    if cfg.policy is None or not isinstance(cfg.policy, ACTConfig):
        raise TypeError("A-Sanity requires an ACTConfig")

    set_seed(SEED)
    dataset = make_dataset(cfg)
    _validate_dataset_contract(dataset)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    if not isinstance(policy, ACTPolicy):
        raise TypeError(f"Expected ACTPolicy, got {type(policy).__name__}")
    validate_policy_contract(policy.config)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        dataset_stats=dataset.meta.stats,
    )
    backbone_tensor_count = verify_pretrained_backbone(policy)

    final_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = final_root.with_name(f".{final_root.name}.incomplete.{os.getpid()}")
    temporary_root.mkdir(exist_ok=False)
    incomplete_marker = temporary_root / ".incomplete"
    incomplete_marker.write_text(
        "Step-0 checkpoint creation did not complete atomically.\n",
        encoding="utf-8",
    )

    temporary_policy_path = temporary_root / PRETRAINED_MODEL_RELATIVE_PATH
    policy.save_pretrained(temporary_policy_path)
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
    manifest = _manifest(
        run_root=final_root,
        policy_path=final_policy_path,
        config=policy.config,
        dataset=dataset,
        model_sha256=model_sha256,
        backbone_tensor_count=backbone_tensor_count,
        serialized_tensor_count=serialized_tensor_count,
    )
    write_json_atomic(temporary_root / "initialization_manifest.json", manifest)
    temporary_root.rename(final_root)
    (final_root / ".incomplete").unlink()

    return InitializationResult(
        run_root=final_root,
        policy_path=final_root / PRETRAINED_MODEL_RELATIVE_PATH,
        model_sha256=model_sha256,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create the fixed step-0 checkpoint")
    create.add_argument("--output-dir", required=True, type=Path)
    create.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    create.add_argument("--device", choices=("cpu", "cuda", "mps"))

    validate = subparsers.add_parser(
        "validate-policy",
        help="Validate a checkpoint for positive-step A-Sanity initialization",
    )
    validate.add_argument("--policy-path", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    if args.command == "validate-policy":
        validate_a_sanity_policy_path(args.policy_path)
        return
    if args.command == "create":
        result = create_initialization_checkpoint(
            output_dir=args.output_dir,
            dataset_root=args.dataset_root,
            device=args.device,
        )
        print(f"A-Sanity step-0 checkpoint: {result.policy_path}")
        print(f"model.safetensors sha256: {result.model_sha256}")
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
