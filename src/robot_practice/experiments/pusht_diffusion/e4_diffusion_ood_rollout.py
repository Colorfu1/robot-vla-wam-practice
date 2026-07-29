"""Run the selected PushT Diffusion policy in the E4 rotation-only scene."""

from __future__ import annotations

import argparse
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from tqdm.auto import tqdm

from .config import default_output_root, practice_root
from .e4_rotation_only_env import (
    E4_AGENT_START,
    E4_GOAL_POSE,
    E4_LOWER_LEFT_AGENT_START,
    E4_LOWER_LEFT_GOAL_CENTER,
    E4_LOWER_LEFT_GOAL_POSE,
    E4_LOWER_LEFT_RESET_STATE,
    E4_LOWER_LEFT_START_CENTER,
    E4_RESET_STATE,
    E4_ROTATION_CENTER,
    e4_lower_left_agent_start,
    e4_lower_left_reset_state,
    make_e4_lower_left_env,
    make_e4_rotation_only_env,
)
from .output import AtomicOutputDirectory
from .policy_loading import load_policy_with_processors
from .provenance import (
    collect_runtime_provenance,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from .rerun_visualization import record_single_rollout
from .rollout import (
    FULL_DDPM_SAMPLING,
    EnvironmentFactory,
    RolloutResult,
    run_rollout,
    save_rollout_video,
)
from .training_common import resolve_device

EXPERIMENT_ID = "e4_rotation_only_dp_ood"
LOWER_LEFT_EXPERIMENT_ID = "e4_lower_left_dp_ood"
RESULT_LABEL = "Closed-loop / OOD diagnostic"
DEFAULT_POLICY_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_POLICY_PATH = (
    default_output_root() / "pilot" / "diffusion" / "checkpoints" / "175000" / "pretrained_model"
)
DEFAULT_OUTPUT_DIR = default_output_root() / "pilot" / "e4_rotation_only_dp_ood"
DEFAULT_LOWER_LEFT_OUTPUT_DIR = default_output_root() / "pilot" / "e4_lower_left_dp_ood"
REQUIRED_POLICY_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


@dataclass(frozen=True)
class E4OODScene:
    key: str
    name: str
    experiment_id: str
    environment_factory: EnvironmentFactory
    reset_state: np.ndarray
    agent_start: np.ndarray
    start_center: np.ndarray
    goal_center: np.ndarray
    goal_pose: np.ndarray
    output_dir: Path


E4_OOD_SCENES = {
    "same-center": E4OODScene(
        key="same-center",
        name="E4 same-center rotation-only scene",
        experiment_id=EXPERIMENT_ID,
        environment_factory=make_e4_rotation_only_env,
        reset_state=E4_RESET_STATE,
        agent_start=E4_AGENT_START,
        start_center=E4_ROTATION_CENTER,
        goal_center=E4_ROTATION_CENTER,
        goal_pose=E4_GOAL_POSE,
        output_dir=DEFAULT_OUTPUT_DIR,
    ),
    "lower-left": E4OODScene(
        key="lower-left",
        name="E4 lower-left-to-center 180-degree scene",
        experiment_id=LOWER_LEFT_EXPERIMENT_ID,
        environment_factory=make_e4_lower_left_env,
        reset_state=E4_LOWER_LEFT_RESET_STATE,
        agent_start=E4_LOWER_LEFT_AGENT_START,
        start_center=E4_LOWER_LEFT_START_CENTER,
        goal_center=E4_LOWER_LEFT_GOAL_CENTER,
        goal_pose=E4_LOWER_LEFT_GOAL_POSE,
        output_dir=DEFAULT_LOWER_LEFT_OUTPUT_DIR,
    ),
}


def parse_policy_seeds(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "policy seeds must be comma-separated integers"
            ) from exc
    else:
        seeds = tuple(int(item) for item in value)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one policy seed is required")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("policy seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("policy seeds must be unique")
    return seeds


def _repository_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(practice_root()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _validate_policy_path(path: str | Path) -> Path:
    policy_path = Path(path).expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Diffusion checkpoint directory is missing: {policy_path}")
    missing = [name for name in REQUIRED_POLICY_FILES if not (policy_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Diffusion checkpoint is incomplete: {missing}")
    return policy_path


def _checkpoint_step(policy_path: Path) -> int:
    raw_step = policy_path.parent.name
    if not raw_step.isdigit():
        raise ValueError(
            f"Expected checkpoint path .../checkpoints/<step>/pretrained_model, got {policy_path}"
        )
    return int(raw_step)


def _rollout_row(
    result: RolloutResult,
    *,
    experiment_id: str,
    checkpoint_step: int,
    contact_offset_px: float,
    recording_path: str,
    video_path: str,
) -> dict[str, Any]:
    return {
        "label": RESULT_LABEL,
        "experiment": experiment_id,
        "model": result.model,
        "checkpoint_step": checkpoint_step,
        "contact_offset_px": contact_offset_px,
        "environment_seed": result.environment_seed,
        "policy_seed": result.policy_seed,
        "sampling_protocol": result.sampling_protocol,
        "scheduler": result.scheduler,
        "num_inference_steps": result.num_inference_steps,
        "success": result.success,
        "sum_reward": result.sum_reward,
        "max_reward": result.max_reward,
        "final_reward": result.final_reward,
        "episode_length": result.episode_length,
        "num_replans": len(result.replans),
        "termination_reason": result.termination_reason,
        "recording_path": recording_path,
        "video_path": video_path,
    }


def _aggregate_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    experiment_id: str = EXPERIMENT_ID,
    contact_offset_px: float = 0.0,
) -> dict[str, Any]:
    successes = np.asarray([row["success"] for row in rows], dtype=np.float64)
    max_rewards = np.asarray([row["max_reward"] for row in rows], dtype=np.float64)
    final_rewards = np.asarray([row["final_reward"] for row in rows], dtype=np.float64)
    return {
        "schema_version": 1,
        "experiment": experiment_id,
        "label": RESULT_LABEL,
        "contact_offset_px": contact_offset_px,
        "interpretation": (
            "The policy was trained on original PushT, so these values are diagnostic "
            "and are not an E4 performance estimate."
        ),
        "num_rollouts": len(rows),
        "policy_seeds": [int(row["policy_seed"]) for row in rows],
        "success_count": int(successes.sum()),
        "diagnostic_success_rate": float(successes.mean()),
        "mean_max_reward": float(max_rewards.mean()),
        "best_max_reward": float(max_rewards.max()),
        "mean_final_reward": float(final_rewards.mean()),
    }


def run_e4_diffusion_ood(
    *,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    policy_seeds: Sequence[int] = DEFAULT_POLICY_SEEDS,
    device: str = "auto",
    episode_length: int = 300,
    scene: str = "same-center",
    contact_offset_px: float = 0.0,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    seeds = parse_policy_seeds(policy_seeds)
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    try:
        scene_config = E4_OOD_SCENES[scene]
    except KeyError as exc:
        raise ValueError(f"scene must be one of {sorted(E4_OOD_SCENES)}, got {scene!r}") from exc
    contact_offset_px = float(contact_offset_px)
    if not np.isfinite(contact_offset_px):
        raise ValueError("contact_offset_px must be finite")
    if scene != "lower-left" and contact_offset_px != 0.0:
        raise ValueError("contact_offset_px is only valid with scene='lower-left'")
    environment_factory = scene_config.environment_factory
    reset_state = scene_config.reset_state
    agent_start = scene_config.agent_start
    if scene == "lower-left":
        agent_start = e4_lower_left_agent_start(contact_offset_px)
        reset_state = e4_lower_left_reset_state(contact_offset_px)
        environment_factory = partial(
            make_e4_lower_left_env,
            contact_offset_px=contact_offset_px,
        )
    checkpoint = _validate_policy_path(policy_path)
    checkpoint_step = _checkpoint_step(checkpoint)
    resolved_device = resolve_device(device)
    destination = Path(output_dir or scene_config.output_dir).expanduser().resolve()
    checkpoint_relative = _repository_relative(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint / "model.safetensors")

    loaded = load_policy_with_processors(
        checkpoint,
        local_files_only=True,
        device=resolved_device,
    )
    if loaded.config.type != "diffusion" or not isinstance(loaded.policy, DiffusionPolicy):
        raise TypeError(f"E4 OOD rollout requires a Diffusion policy, got {loaded.config.type!r}")

    rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(seeds),
        desc=f"E4 {scene_config.key} offset={contact_offset_px:+g} DP OOD",
        unit="rollout",
    )
    try:
        with AtomicOutputDirectory(destination, overwrite=overwrite) as temporary:
            recordings_dir = temporary / "recordings"
            videos_dir = temporary / "videos"
            recordings_dir.mkdir()
            videos_dir.mkdir()
            for rollout_index, policy_seed in enumerate(seeds):
                progress.set_postfix(policy_seed=policy_seed, phase="rollout", refresh=True)
                result = run_rollout(
                    policy=loaded.policy,
                    preprocessor=loaded.preprocessor,
                    postprocessor=loaded.postprocessor,
                    environment_seed=0,
                    policy_seed=policy_seed,
                    record_frames=True,
                    episode_length=episode_length,
                    condition_id=scene_config.experiment_id,
                    sampling_protocol=FULL_DDPM_SAMPLING,
                    environment_factory=environment_factory,
                )
                recording_name = f"policy_seed_{policy_seed:02d}.rrd"
                video_name = f"policy_seed_{policy_seed:02d}.mp4"
                recording_relative = f"recordings/{recording_name}"
                video_relative = f"videos/{video_name}"

                progress.set_postfix(policy_seed=policy_seed, phase="recording", refresh=True)
                record_single_rollout(
                    output_path=recordings_dir / recording_name,
                    rollout=result,
                    recording_id=(
                        f"e4_{scene_config.key}_dp_{checkpoint_step:06d}"
                        f"_contact_offset_{contact_offset_px:+g}"
                        f"_policy_seed_{policy_seed:02d}"
                    ),
                    recording_name=(
                        f"E4 {scene_config.key} DP-{checkpoint_step:06d} "
                        f"· contact_offset={contact_offset_px:+g} "
                        f"· policy_seed={policy_seed}"
                    ),
                )
                progress.set_postfix(policy_seed=policy_seed, phase="video", refresh=True)
                save_rollout_video(result, videos_dir / video_name)
                row = _rollout_row(
                    result,
                    experiment_id=scene_config.experiment_id,
                    checkpoint_step=checkpoint_step,
                    contact_offset_px=contact_offset_px,
                    recording_path=recording_relative,
                    video_path=video_relative,
                )
                rows.append(row)
                progress.write(
                    f"policy_seed={policy_seed}: success={result.success}, "
                    f"max_reward={result.max_reward:.3f}, "
                    f"steps={result.episode_length}"
                )
                progress.update(1)
                del result
                gc.collect()

            write_json_atomic(temporary / "rollouts.json", rows)
            write_json_atomic(
                temporary / "metrics.json",
                _aggregate_metrics(
                    rows,
                    experiment_id=scene_config.experiment_id,
                    contact_offset_px=contact_offset_px,
                ),
            )
            write_json_atomic(
                temporary / "run_manifest.json",
                {
                    "schema_version": 1,
                    "experiment": scene_config.experiment_id,
                    "label": RESULT_LABEL,
                    "policy": {
                        "checkpoint_path": checkpoint_relative,
                        "checkpoint_step": checkpoint_step,
                        "model_sha256": checkpoint_sha256,
                        "device": resolved_device,
                    },
                    "environment": {
                        "name": scene_config.name,
                        "scene_key": scene_config.key,
                        "environment_seed": 0,
                        "start_center": scene_config.start_center.tolist(),
                        "goal_center": scene_config.goal_center.tolist(),
                        "reset_state": reset_state.tolist(),
                        "goal_pose": scene_config.goal_pose.tolist(),
                        "agent_start": agent_start.tolist(),
                        "contact_offset_px": contact_offset_px,
                        "contact_offset_convention": (
                            "negative=left-up, zero=center, positive=right-down"
                        ),
                    },
                    "rollout": {
                        "policy_seeds": list(seeds),
                        "sampling_protocol": FULL_DDPM_SAMPLING,
                        "episode_length": episode_length,
                        "control_frequency_hz": 10,
                    },
                    "initial_state_sha256": sha256_bytes(reset_state.tobytes()),
                    "runtime": collect_runtime_provenance(practice_root()),
                },
            )
    finally:
        progress.close()
        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected Diffusion policy in the E4 rotation-only OOD scene."
    )
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--policy-seeds",
        type=parse_policy_seeds,
        default=DEFAULT_POLICY_SEEDS,
        help="Comma-separated policy seeds (default: 0,1,2,3,4).",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument(
        "--scene",
        choices=tuple(E4_OOD_SCENES),
        default="same-center",
        help="E4 scene variant (default: same-center).",
    )
    parser.add_argument(
        "--contact-offset-px",
        type=float,
        default=0.0,
        help=(
            "Lower-left scene crossbar offset in physical pixels: "
            "negative=left-up, positive=right-down (default: 0)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the selected scene's output directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = run_e4_diffusion_ood(
        policy_path=args.policy_path,
        policy_seeds=args.policy_seeds,
        device=args.device,
        episode_length=args.episode_length,
        scene=args.scene,
        contact_offset_px=args.contact_offset_px,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"E4 Diffusion OOD rollouts written: {output}")


if __name__ == "__main__":
    main()
