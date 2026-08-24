"""Deterministic one-step loss comparison for the two M2 base 4D pipelines.

This probe is intentionally separate from the formal training entry point. It
reuses the formal HDF5 window dataset and normalization path, but replaces the
DataLoader/random training loop with one explicitly selected episode/window.
The native 4D and 7D->4D pipelines therefore receive the same source episode
ID and action index while retaining their own source H5, train-only statistics,
and projection configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from robot_practice.policies.wam_tiny.action_flow import (
    ACTION_KEY,
    BASE_IMAGE_KEY,
    HAND_IMAGE_KEY,
    STATE_KEY,
    ActionOnlyFlowModel,
    NormalizationStats,
    load_action_flow_checkpoint,
)

from . import m2_base_action_flow as runner

DEFAULT_NATIVE_CONFIG = Path(
    "configs/experiments/maniskill/m2_base_m1w_native_rgb_4d_100k.json"
)
DEFAULT_PROJECTED_CONFIG = Path(
    "configs/experiments/maniskill/m2_base_m1w_rgb_4d_success_at_end.json"
)
DEFAULT_CHECKPOINT = Path(
    "outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/"
    "formal_seed1000_100k/checkpoint_step_00100000.pt"
)
DEFAULT_NATIVE_STATS = Path(
    "outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/"
    "formal_seed1000_100k/normalization_stats.json"
)
DEFAULT_PROJECTED_STATS = Path(
    "outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/"
    "formal_seed1000/normalization_stats.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/maniskill/m2_base_action_flow/"
    "loss_probe_native4d_vs_projected4d"
)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_stats(path: str | Path, *, device: torch.device) -> NormalizationStats:
    target = runner._rooted(path)
    if not target.is_file():
        raise FileNotFoundError(f"Normalization stats do not exist: {runner._portable(target)}")
    payload = runner._read_json(target)
    stats = payload.get("stats")
    if not isinstance(stats, Mapping):
        raise TypeError(f"Normalization stats have no stats object: {runner._portable(target)}")
    return NormalizationStats.from_mapping(stats).to(device)


def _sample_from_dataset(
    config: Mapping[str, Any],
    stats: NormalizationStats,
    *,
    episode_id: int,
    action_index: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    dataset = runner.RGBH5WindowDataset(config, split="train", stats=stats)
    entry = (int(episode_id), int(action_index))
    try:
        dataset_index = dataset.entries.index(entry)
    except ValueError as exc:
        raise ValueError(
            f"Fixed sample {entry} is not available in the configured train split "
            f"for {runner._portable(dataset.path)}"
        ) from exc
    sample = dataset[dataset_index]
    batch = {
        key: sample[key].unsqueeze(0)
        for key in (BASE_IMAGE_KEY, HAND_IMAGE_KEY, STATE_KEY, ACTION_KEY, "action_is_pad")
    }
    metadata = {
        "dataset_index": int(dataset_index),
        "episode_id": int(sample["episode_id"].item()),
        "action_index": int(action_index),
        "sample_index": int(sample["sample_index"].item()),
        "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
        "action_padding_count": int(batch["action_is_pad"].sum().item()),
        "read_audit": dict(dataset.read_audit),
        "batch_hashes": {key: _tensor_sha256(value) for key, value in batch.items()},
    }
    return batch, metadata


def _diff_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_left": list(left.shape),
            "shape_right": list(right.shape),
            "same_shape": False,
        }
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "shape": list(left.shape),
        "same_shape": True,
        "exact": bool(np.array_equal(left, right)),
        "mean_abs": float(np.abs(difference).mean()),
        "max_abs": float(np.abs(difference).max()),
    }


def _raw_alignment(
    native_config: Mapping[str, Any],
    projected_config: Mapping[str, Any],
    *,
    episode_id: int,
    action_index: int,
) -> dict[str, Any]:
    native_path = runner._rooted(native_config["inputs"]["source_h5"])
    projected_path = runner._rooted(projected_config["inputs"]["source_h5"])
    projection = runner._action_projection_indices(projected_config)
    context_start = max(0, int(action_index) - 1)
    context_stop = int(action_index) + 1
    action_stop = int(action_index) + runner.EXPECTED_HORIZON
    with h5py.File(native_path, "r") as native, h5py.File(projected_path, "r") as projected:
        native_group = native[f"traj_{int(episode_id)}"]
        projected_group = projected[f"traj_{int(episode_id)}"]
        native_action_count = int(native_group["actions"].shape[0])
        projected_action_count = int(projected_group["actions"].shape[0])
        if action_stop > min(native_action_count, projected_action_count):
            raise ValueError(
                f"Fixed action window [{action_index}, {action_stop}) exceeds one source: "
                f"native={native_action_count}, projected={projected_action_count}"
            )
        native_state = np.concatenate(
            (
                np.asarray(native_group["obs/agent/qpos"][context_start:context_stop]),
                np.asarray(native_group["obs/agent/qvel"][context_start:context_stop]),
            ),
            axis=-1,
        )
        projected_state = np.concatenate(
            (
                np.asarray(projected_group["obs/agent/qpos"][context_start:context_stop]),
                np.asarray(projected_group["obs/agent/qvel"][context_start:context_stop]),
            ),
            axis=-1,
        )
        native_actions = np.asarray(native_group["actions"][action_index:action_stop])
        projected_actions = np.asarray(projected_group["actions"][action_index:action_stop])[:, projection]
        return {
            "episode_id": int(episode_id),
            "action_index": int(action_index),
            "context_range": [context_start, context_stop],
            "action_range": [int(action_index), action_stop],
            "projection_indices": list(projection),
            "native_action_count": native_action_count,
            "projected_source_action_count": projected_action_count,
            "base_rgb": _diff_summary(
                np.asarray(native_group["obs/sensor_data/base_camera/rgb"][context_start:context_stop]),
                np.asarray(projected_group["obs/sensor_data/base_camera/rgb"][context_start:context_stop]),
            ),
            "hand_rgb": _diff_summary(
                np.asarray(native_group["obs/sensor_data/hand_camera/rgb"][context_start:context_stop]),
                np.asarray(projected_group["obs/sensor_data/hand_camera/rgb"][context_start:context_stop]),
            ),
            "state": _diff_summary(native_state, projected_state),
            "raw_action_after_projection": _diff_summary(native_actions, projected_actions),
        }


def _stats_summary(
    native_stats: NormalizationStats,
    projected_stats: NormalizationStats,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("image_mean", "image_std", "state_mean", "state_std", "action_mean", "action_std"):
        left = getattr(native_stats, name).detach().cpu().numpy()
        right = getattr(projected_stats, name).detach().cpu().numpy()
        result[name] = _diff_summary(left, right)
    return result


def _loss_once(
    model: ActionOnlyFlowModel,
    batch: Mapping[str, torch.Tensor],
    *,
    source_noise: torch.Tensor,
    flow_time: torch.Tensor,
    learning_rate: float,
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        betas=tuple(float(value) for value in config["training"]["betas"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: runner._cosine_warmup_lambda(
            step,
            warmup=int(config["training"]["warmup_steps"]),
            total=int(config["training"]["optimizer_steps"]),
        ),
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    observations = {
        key: batch[key].to(next(model.parameters()).device, non_blocking=True)
        for key in (BASE_IMAGE_KEY, HAND_IMAGE_KEY, STATE_KEY)
    }
    clean_actions = batch[ACTION_KEY].to(next(model.parameters()).device, non_blocking=True)
    padding = batch["action_is_pad"].to(next(model.parameters()).device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    _set_seed(seed)
    device = next(model.parameters()).device
    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        noisy = (
            (1.0 - flow_time[:, None, None]) * source_noise
            + flow_time[:, None, None] * clean_actions
        )
        target_velocity = clean_actions - source_noise
        prediction = model(observations, noisy, flow_time)
        error = (prediction - target_velocity).square()
        mask = (~padding).unsqueeze(-1).to(dtype=error.dtype)
        loss = (error * mask).sum() / mask.sum().clamp_min(1.0) / clean_actions.shape[-1]
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Fixed M2 base loss is non-finite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["training"]["gradient_clip_norm"])
    )
    optimizer.step()
    scheduler.step()
    after = {name: parameter.detach() for name, parameter in model.named_parameters()}
    deltas = [torch.max(torch.abs(after[name] - before[name])).item() for name in before]
    return {
        "loss": float(loss.detach().cpu()),
        "learning_rate_requested": float(learning_rate),
        "optimizer_learning_rate_after_step": float(scheduler.get_last_lr()[0]),
        "model_mode": "train",
        "grad_norm_after_clip": float(grad_norm.detach().cpu()),
        "parameters_exactly_unchanged": all(
            torch.equal(before[name], after[name]) for name in before
        ),
        "max_parameter_abs_delta": float(max(deltas, default=0.0)),
    }


def _copy_model(model: ActionOnlyFlowModel, device: torch.device) -> ActionOnlyFlowModel:
    result = copy.deepcopy(model).to(device)
    result.eval()
    return result


def run_probe(
    *,
    native_config_path: str | Path,
    projected_config_path: str | Path,
    checkpoint_path: str | Path,
    native_stats_path: str | Path,
    projected_stats_path: str | Path,
    output_root: str | Path,
    episode_id: int,
    action_index: int,
    flow_time_value: float,
    seed: int,
    learning_rate: float,
    include_state_swap: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if learning_rate != 0.0:
        raise ValueError("This diagnostic probe requires --learning-rate=0")
    if not 0.0 <= flow_time_value <= 1.0:
        raise ValueError("--flow-time must be in [0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("The one-step loss probe requires CUDA to match the training autocast path")
    device = torch.device("cuda:0")
    native_config = runner.load_config(native_config_path)
    projected_config = runner.load_config(projected_config_path)
    native_validation = runner.validate_config(native_config, num_gpus=1, check_inputs=True)
    projected_validation = runner.validate_config(projected_config, num_gpus=1, check_inputs=True)
    if native_validation["action_dim"] != 4 or projected_validation["action_dim"] != 4:
        raise ValueError("Both probe pipelines must expose action_dim=4")
    if projected_validation["source_action_dim"] != 7:
        raise ValueError("Projected pipeline must retain source_action_dim=7")

    native_stats = _load_stats(native_stats_path, device=device)
    projected_stats = _load_stats(projected_stats_path, device=device)
    native_batch, native_batch_meta = _sample_from_dataset(
        native_config,
        native_stats,
        episode_id=episode_id,
        action_index=action_index,
    )
    projected_batch, projected_batch_meta = _sample_from_dataset(
        projected_config,
        projected_stats,
        episode_id=episode_id,
        action_index=action_index,
    )
    checkpoint_policy, checkpoint_payload = load_action_flow_checkpoint(
        runner._rooted(checkpoint_path), device=device, n_action_steps=4
    )
    base_model = checkpoint_policy.model
    if int(checkpoint_payload["model_config"]["action_dim"]) != 4:
        raise ValueError("Pretrained checkpoint must be a 4D Action-only Flow model")
    native_model = _copy_model(base_model, device)
    projected_model = _copy_model(base_model, device)
    parameter_identity = all(
        torch.equal(left, right)
        for left, right in zip(native_model.parameters(), projected_model.parameters(), strict=True)
    )
    if not parameter_identity:
        raise AssertionError("Native and projected probe models do not start from identical parameters")

    _set_seed(seed)
    source_noise = torch.randn((1, runner.EXPECTED_HORIZON, 4), device=device, dtype=torch.float32)
    flow_time = torch.full((1,), float(flow_time_value), device=device, dtype=torch.float32)
    started = time.perf_counter()
    native_result = _loss_once(
        native_model,
        native_batch,
        source_noise=source_noise,
        flow_time=flow_time,
        learning_rate=learning_rate,
        seed=seed + 1,
        config=native_config,
    )
    projected_result = _loss_once(
        projected_model,
        projected_batch,
        source_noise=source_noise,
        flow_time=flow_time,
        learning_rate=learning_rate,
        seed=seed + 1,
        config=projected_config,
    )
    loss_difference = projected_result["loss"] - native_result["loss"]
    payload = {
        "schema_version": 1,
        "experiment": "m2_base_native4d_vs_7d_projected4d_one_step_loss",
        "checkpoint": {
            "path": runner._portable(runner._rooted(checkpoint_path)),
            "step": int(checkpoint_payload["step"]),
            "model_config": checkpoint_payload["model_config"],
            "native_and_projected_parameter_identity": parameter_identity,
        },
        "fixed_batch": {
            "episode_id": int(episode_id),
            "action_index": int(action_index),
            "observation_context": [max(0, action_index - 1), action_index + 1],
            "action_horizon": [action_index, action_index + runner.EXPECTED_HORIZON],
            "flow_time": float(flow_time_value),
            "source_noise_seed": int(seed),
            "loss_forward_seed": int(seed + 1),
            "source_noise_sha256": _tensor_sha256(source_noise),
        },
        "runtime": {
            "device": str(device),
            "learning_rate": float(learning_rate),
            "optimizer_step_called": True,
            "parameters_updated": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "pipelines": {
            "native_4d": {
                "config": runner._portable(runner._rooted(native_config_path)),
                "stats": runner._portable(runner._rooted(native_stats_path)),
                "validation": native_validation,
                "batch": native_batch_meta,
                "loss": native_result,
            },
            "projected_4d_from_7d": {
                "config": runner._portable(runner._rooted(projected_config_path)),
                "stats": runner._portable(runner._rooted(projected_stats_path)),
                "validation": projected_validation,
                "batch": projected_batch_meta,
                "loss": projected_result,
            },
        },
        "loss_comparison": {
            "native_loss": native_result["loss"],
            "projected_loss": projected_result["loss"],
            "projected_minus_native": loss_difference,
            "absolute_difference": abs(loss_difference),
            "relative_to_native": abs(loss_difference) / max(abs(native_result["loss"]), 1e-12),
        },
        "raw_source_alignment": _raw_alignment(
            native_config,
            projected_config,
            episode_id=episode_id,
            action_index=action_index,
        ),
        "normalization_alignment": _stats_summary(native_stats, projected_stats),
    }
    if include_state_swap:
        state_swapped_batch = dict(projected_batch)
        state_swapped_batch[STATE_KEY] = native_batch[STATE_KEY]
        state_swapped_model = _copy_model(base_model, device)
        state_swapped_result = _loss_once(
            state_swapped_model,
            state_swapped_batch,
            source_noise=source_noise,
            flow_time=flow_time,
            learning_rate=learning_rate,
            seed=seed + 1,
            config=projected_config,
        )
        payload["state_swap"] = {
            "description": (
                "projected 7D->4D RGB/action with the normalized native 4D state tensor"
            ),
            "state_source_config": runner._portable(runner._rooted(native_config_path)),
            "state_source_stats": runner._portable(runner._rooted(native_stats_path)),
            "batch_hash": _tensor_sha256(state_swapped_batch[STATE_KEY]),
            "loss": state_swapped_result,
            "loss_minus_projected_full": state_swapped_result["loss"]
            - projected_result["loss"],
            "relative_change_from_projected_full": (
                state_swapped_result["loss"] - projected_result["loss"]
            )
            / max(abs(projected_result["loss"]), 1e-12),
        }
    target = runner._rooted(output_root)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Probe output exists: {runner._portable(target)}")
        outputs_root = (runner.practice_root() / "outputs").resolve()
        if not target.resolve().is_relative_to(outputs_root):
            raise ValueError("--overwrite is restricted to an output directory under outputs/")
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=False)
    runner._write_json(target / "metrics.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-config", type=Path, default=DEFAULT_NATIVE_CONFIG)
    parser.add_argument("--projected-config", type=Path, default=DEFAULT_PROJECTED_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--native-stats", type=Path, default=DEFAULT_NATIVE_STATS)
    parser.add_argument("--projected-stats", type=Path, default=DEFAULT_PROJECTED_STATS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode-id", type=int, default=1)
    parser.add_argument("--action-index", type=int, default=40)
    parser.add_argument("--flow-time", type=float, default=0.37)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument(
        "--include-state-swap",
        action="store_true",
        help="Also replace the projected pipeline state tensor with native 4D state.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_probe(
        native_config_path=args.native_config,
        projected_config_path=args.projected_config,
        checkpoint_path=args.checkpoint,
        native_stats_path=args.native_stats,
        projected_stats_path=args.projected_stats,
        output_root=args.output_root,
        episode_id=args.episode_id,
        action_index=args.action_index,
        flow_time_value=args.flow_time,
        seed=args.seed,
        learning_rate=args.learning_rate,
        include_state_swap=args.include_state_swap,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["loss_comparison"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
