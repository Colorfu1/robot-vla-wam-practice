"""Run the official ManiSkill Diffusion Policy evaluator on one checkpoint.

The policy class, environment factory, wrappers, and metric evaluator are
loaded from the official ManiSkill checkout.  This module only supplies the
checkpoint-loading glue and persists the official evaluator's output; it does
not implement a second success predicate or a second simulator path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from .rollout_contract import OFFICIAL_DATA_PROVENANCE, require_rollout_backend

TASK_ID = "StackCube-v1"
DATA_PROVENANCE = OFFICIAL_DATA_PROVENANCE
CONTROL_MODE = "pd_ee_delta_pos"
SIM_BACKEND = "physx_cpu"
OBS_MODE = "rgb"
OBS_HORIZON = 2
ACT_HORIZON = 8
PRED_HORIZON = 16
DEFAULT_NUM_EPISODES = 100
DEFAULT_NUM_ENVS = 10
DEFAULT_MAX_EPISODE_STEPS = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision_file = repository / "SOURCE_REVISION"
        if revision_file.is_file():
            return revision_file.read_text(encoding="utf-8").strip() or None
        return None


def _load_official_train_module(official_repo: Path) -> ModuleType:
    baseline_dir = official_repo / "examples" / "baselines" / "diffusion_policy"
    train_file = baseline_dir / "train_rgbd.py"
    if not train_file.is_file():
        raise FileNotFoundError(f"Official train_rgbd.py is missing: {train_file}")
    sys.path.insert(0, str(baseline_dir))
    spec = importlib.util.spec_from_file_location(
        "official_maniskill_diffusion_policy_train_rgbd", train_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official module from {train_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_args() -> SimpleNamespace:
    return SimpleNamespace(
        obs_horizon=OBS_HORIZON,
        act_horizon=ACT_HORIZON,
        pred_horizon=PRED_HORIZON,
        diffusion_step_embed_dim=64,
        unet_dims=[64, 128, 256],
        n_groups=8,
    )


def _metric_values(value: Any) -> list[float]:
    array = np.asarray(value).reshape(-1)
    return [float(item) for item in array]


def verify_checkpoint(
    *,
    official_repo: Path,
    checkpoint: Path,
    output_dir: Path,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    num_envs: int = DEFAULT_NUM_ENVS,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    device_name: str = "cuda",
    capture_video: bool = False,
    seed: int = 1,
    env_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint through the official CPU evaluation path."""

    if num_episodes <= 0 or num_envs <= 0 or num_episodes % num_envs != 0:
        raise ValueError("num_episodes must be a positive multiple of num_envs")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if env_seed is not None and env_seed < 0:
        raise ValueError("env_seed must be non-negative")
    official_repo = official_repo.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import torch

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be cuda or cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    official = _load_official_train_module(official_repo)
    rollout_contract = require_rollout_backend(DATA_PROVENANCE, SIM_BACKEND)
    env_kwargs = {
        "control_mode": CONTROL_MODE,
        "reward_mode": "sparse",
        "obs_mode": OBS_MODE,
        "render_mode": "rgb_array",
        "human_render_camera_configs": {"shader_pack": "default"},
        "max_episode_steps": max_episode_steps,
    }
    video_dir = output_dir / "videos" if capture_video else None
    envs = official.make_eval_envs(
        TASK_ID,
        num_envs,
        SIM_BACKEND,
        env_kwargs,
        {"obs_horizon": OBS_HORIZON},
        video_dir=str(video_dir) if video_dir is not None else None,
        # The ablation checkpoint is trained on the custom 18D state
        # contract (qpos[9] + qvel[9]).  The upstream FlattenRGBD wrapper
        # also exposes extra task fields and produces a 25D state, which
        # creates a different Agent conditioning dimension and cannot load
        # the 18D checkpoint.
        wrappers=[_stackcube_18d_observation_wrapper()],
    )
    args = _official_args()
    agent = official.Agent(envs, args).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    if not isinstance(checkpoint_data, dict):
        raise TypeError(f"Official checkpoint must be a dictionary: {checkpoint}")
    state_dict = checkpoint_data.get("ema_agent") or checkpoint_data.get("agent")
    if state_dict is None:
        raise KeyError("Official checkpoint has neither 'ema_agent' nor 'agent'")
    agent.load_state_dict(state_dict)

    try:
        official_metrics = official.evaluate(
            num_episodes,
            agent,
            envs,
            device,
            SIM_BACKEND,
            progress_bar=True,
            seed=env_seed,
        )
    finally:
        envs.close()

    evaluation: dict[str, Any] = {}
    for key, value in official_metrics.items():
        values = _metric_values(value)
        evaluation[key] = {
            "values": values,
            "num_episodes": len(values),
            "mean": float(np.mean(values)) if values else None,
        }
    result = {
        "schema_version": 1,
        "label": "Official ManiSkill Diffusion Policy RGB verification",
        "task": TASK_ID,
        "control_mode": CONTROL_MODE,
        "obs_mode": OBS_MODE,
        "sim_backend": SIM_BACKEND,
        **rollout_contract,
        "num_episodes": num_episodes,
        "num_envs": num_envs,
        "max_episode_steps": max_episode_steps,
        "policy_device": str(device),
        "evaluation": evaluation,
        "primary_metric": "success_once",
        "official_protocol": {
            "partial_reset": False,
            "ignore_terminations": True,
            "reconfiguration_freq": 1,
            "reset": (
                "official evaluate reset with contiguous episode seeds"
                if env_seed is not None
                else "official make_eval_envs reset without an explicit seed"
            ),
            "environment_seed": env_seed,
            "environment_seed_mode": (
                "episode i uses env_seed + i" if env_seed is not None else "random reset"
            ),
            "success_source": "official evaluate.py metrics",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "ema_used": "ema_agent" in checkpoint_data,
        },
        "official_repo": {
            "path": str(official_repo),
            "git_revision": _git_revision(official_repo),
            "train_rgbd_sha256": _sha256(
                official_repo / "examples" / "baselines" / "diffusion_policy" / "train_rgbd.py"
            ),
            "evaluate_sha256": _sha256(
                official_repo
                / "examples"
                / "baselines"
                / "diffusion_policy"
                / "diffusion_policy"
                / "evaluate.py"
            ),
        },
        "seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _stackcube_18d_observation_wrapper():
    """Return the observation wrapper used by the 18D/4D training adapter."""

    from robot_practice.experiments.maniskill.official_m1w_adapter import (
        M1WFlattenRGBObservationWrapper,
    )

    return M1WFlattenRGBObservationWrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help=(
            "Base seed passed to the official evaluator; episode i uses "
            "env-seed+i. Omit for random environment resets."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_checkpoint(
        official_repo=args.official_repo,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        max_episode_steps=args.max_episode_steps,
        device_name=args.device,
        capture_video=args.capture_video,
        seed=args.seed,
        env_seed=args.env_seed,
    )
    summary = {
        key: value.get("mean")
        for key, value in result["evaluation"].items()
        if isinstance(value, dict)
    }
    print(json.dumps({"primary_metric": "success_once", "evaluation": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
