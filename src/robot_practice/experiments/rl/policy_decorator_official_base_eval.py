"""Run the pinned Policy Decorator evaluation path for a frozen base policy.

The upstream online script does not expose an eval-only command.  This adapter
therefore imports the pinned upstream module and reuses its ``make_env``,
``BasePolicy`` and ``evaluate`` implementation verbatim.  The only adapter
specific piece is a zero residual actor, so the official five-environment
evaluation measures the frozen base policy without changing the upstream
checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


FORMAL_ENV_ID = "PegInsertionSide-v2"
OFFICIAL_EVAL_ENVS = 5
OFFICIAL_TRAIN_ENVS = 16
OFFICIAL_EPISODES = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path is not a directory: {path}")
        entries = list(path.iterdir())
        if entries:
            raise FileExistsError(f"refusing to reuse non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_online_args(
    checkpoint_path: Path,
    *,
    env_id: str,
    seed: int,
    episodes: int,
    num_eval_envs: int,
    ddim_steps: int,
    act_horizon: int,
) -> SimpleNamespace:
    args_path = checkpoint_path.parent.parent / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"missing frozen base args.json: {args_path}")
    payload = json.loads(args_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"base args.json must contain an object: {args_path}")
    if payload.get("env_id") != env_id:
        raise ValueError(
            f"base args.json env_id={payload.get('env_id')!r} does not match {env_id!r}"
        )

    # These are the values that the upstream online parser supplies or that
    # its __main__ path expects after loading the offline checkpoint args.
    payload.update(
        {
            "env_id": env_id,
            "base_policy_ckpt": str(checkpoint_path),
            "seed": int(seed),
            "ddim_steps": int(ddim_steps),
            "act_horizon": int(act_horizon),
            "num_envs": OFFICIAL_TRAIN_ENVS,
            "num_eval_episodes": int(episodes),
            "num_eval_envs": int(num_eval_envs),
            "sync_venv": False,
            "capture_video": False,
            "obj_ids": [],
            "actor_input": "obs",
            "res_scale": 0.0,
            "cuda": True,
        }
    )
    required = (
        "obs_horizon",
        "pred_horizon",
        "diffusion_step_embed_dim",
        "unet_dims",
        "n_groups",
        "control_mode",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"base args.json is missing model fields: {missing}")
    return SimpleNamespace(**payload)


def _load_checkpoint(torch: Any, path: Path, device: Any) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older Torch versions do not accept weights_only.
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"base checkpoint must contain a mapping: {path}")
    return checkpoint


def _make_zero_residual(torch: Any, action_dim: int) -> Any:
    class ZeroResidual(torch.nn.Module):
        def get_eval_action(self, observations: Any) -> Any:
            return torch.zeros(
                (observations.shape[0], action_dim),
                dtype=observations.dtype,
                device=observations.device,
            )

    return ZeroResidual()


def _official_eval_envs(official: Any, gym: Any, args: SimpleNamespace) -> Any:
    vector_env_cls: Callable[..., Any]
    if args.sync_venv or args.num_eval_envs == 1:
        vector_env_cls = gym.vector.SyncVectorEnv
    else:
        vector_env_cls = lambda thunks: gym.vector.AsyncVectorEnv(thunks, context="forkserver")
    eval_envs = vector_env_cls(
        [
            official.make_env(
                args.env_id,
                args.seed + 1000 + index,
                args.control_mode,
                other_kwargs=args.__dict__,
            )
            for index in range(args.num_eval_envs)
        ]
    )
    eval_envs.reset(seed=args.seed + 1000)
    return eval_envs


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    if arguments.env_id != FORMAL_ENV_ID:
        raise ValueError(f"strict official base evaluator is fixed to {FORMAL_ENV_ID}")
    if arguments.episodes != OFFICIAL_EPISODES:
        raise ValueError(f"official evaluation requires {OFFICIAL_EPISODES} episodes")
    if arguments.num_eval_envs != OFFICIAL_EVAL_ENVS:
        raise ValueError(f"official evaluation requires {OFFICIAL_EVAL_ENVS} eval environments")
    if arguments.episodes % arguments.num_eval_envs != 0:
        raise ValueError("episodes must be divisible by num_eval_envs")

    reference_root = Path(arguments.reference_root).expanduser().resolve()
    checkpoint_path = Path(arguments.base_policy_ckpt).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"missing pinned Policy Decorator checkout: {reference_root}")
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing frozen base checkpoint: {checkpoint_path}")
    _prepare_output(output_dir)
    run_log = output_dir / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with run_log.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    args = _load_online_args(
        checkpoint_path,
        env_id=arguments.env_id,
        seed=arguments.seed,
        episodes=arguments.episodes,
        num_eval_envs=arguments.num_eval_envs,
        ddim_steps=arguments.ddim_steps,
        act_horizon=arguments.act_horizon,
    )
    _write_json(
        output_dir / "resolved_config.json",
        {
            "protocol_id": "policy_decorator_official_base_eval_v1",
            "official_script": str(reference_root / "online/pi_dec_diffusion_maniskill2.py"),
            "reference_revision": _git_revision(reference_root),
            "base_policy_ckpt": str(checkpoint_path),
            "base_policy_sha256": _sha256(checkpoint_path),
            "requested_env_id": arguments.env_id,
            "runtime_env_id": arguments.env_id,
            "seed": int(arguments.seed),
            "episodes": int(arguments.episodes),
            "num_train_envs_recorded": OFFICIAL_TRAIN_ENVS,
            "num_eval_envs": int(arguments.num_eval_envs),
            "eval_reset_seed": int(arguments.seed + 1000),
            "obs_mode": "state",
            "control_mode": args.control_mode,
            "ddim_steps": int(arguments.ddim_steps),
            "act_horizon": int(arguments.act_horizon),
            "residual_scale": 0.0,
        },
    )

    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    official = importlib.import_module("online.pi_dec_diffusion_maniskill2")
    import gymnasium as gym

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Match the seeding block in the upstream online script before creating
    # the evaluation environments or the Diffusion Policy module.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(getattr(args, "torch_deterministic", True))
    official.args = args
    official.device = device
    official.global_step = 0

    eval_envs = _official_eval_envs(official, gym, args)
    try:
        if len(eval_envs.single_observation_space.shape) != 2:
            raise RuntimeError(
                "official base eval requires stacked state observations, got "
                f"{eval_envs.single_observation_space.shape}"
            )
        if not isinstance(eval_envs.single_action_space, gym.spaces.Box):
            raise RuntimeError("official base eval requires a continuous Box action space")

        act_dim = int(np.prod(eval_envs.single_action_space.shape))
        total_act_dim = act_dim * int(args.act_horizon)
        official.act_dim = act_dim
        official.total_act_dim = total_act_dim

        base_policy = official.BasePolicy(eval_envs, args).to(device)
        checkpoint = _load_checkpoint(torch, checkpoint_path, device)
        for key in ("ema_agent", "agent", "actor", "q"):
            if key in checkpoint:
                base_policy.load_state_dict(checkpoint[key])
                break
        else:
            raise RuntimeError("base checkpoint must contain ema_agent or agent")
        base_policy.eval()
        base_policy.requires_grad_(False)
        zero_residual = _make_zero_residual(torch, total_act_dim)

        completed = 0
        original_collect = official.collect_episode_info

        def collect_with_status(infos: Any, result: Any = None) -> Any:
            nonlocal completed
            before = len(result.get("return", [])) if result is not None else 0
            collected = original_collect(infos, result)
            after = len(collected.get("return", []))
            while completed < after:
                completed += 1
                log(f"STATUS completed={completed} total={arguments.episodes}")
            if after < before:
                raise RuntimeError("official evaluator result count moved backwards")
            return collected

        official.collect_episode_info = collect_with_status
        log(
            f"START official_base seed={arguments.seed} "
            f"eval_envs={arguments.num_eval_envs} episodes={arguments.episodes} device={device}"
        )
        result = official.evaluate(
            arguments.episodes,
            base_policy,
            zero_residual,
            eval_envs,
            device,
        )
    except Exception as exc:
        log(f"FAILED official_base seed={arguments.seed} error={type(exc).__name__}: {exc}")
        raise
    finally:
        if "original_collect" in locals():
            official.collect_episode_info = original_collect
        eval_envs.close()

    successes = [bool(value) for value in result.get("success", [])]
    returns = [float(value) for value in result.get("return", [])]
    lengths = [int(value) for value in result.get("len", [])]
    if len(successes) != arguments.episodes:
        raise RuntimeError(
            f"official evaluator returned {len(successes)} episodes, expected {arguments.episodes}"
        )
    metrics = {
        "schema_version": 1,
        "protocol_id": "policy_decorator_official_base_eval_v1",
        "result_label": "Official five-env closed-loop base evaluation",
        "requested_env_id": arguments.env_id,
        "runtime_env_id": arguments.env_id,
        "seed": int(arguments.seed),
        "episodes": len(successes),
        "success_count": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "returns": returns,
        "episode_lengths": lengths,
        "num_eval_envs": int(arguments.num_eval_envs),
        "num_train_envs_recorded": OFFICIAL_TRAIN_ENVS,
        "eval_reset_seed": int(arguments.seed + 1000),
        "obs_mode": "state",
        "control_mode": args.control_mode,
        "ddim_steps": int(arguments.ddim_steps),
        "act_horizon": int(arguments.act_horizon),
        "residual_scale": 0.0,
        "device": str(device),
        "official_script": str(reference_root / "online/pi_dec_diffusion_maniskill2.py"),
        "reference_revision": _git_revision(reference_root),
        "base_policy_sha256": _sha256(checkpoint_path),
        "formal_result": True,
    }
    _write_json(output_dir / "metrics.json", metrics)
    log(
        f"DONE official_base seed={arguments.seed} completed={len(successes)} "
        f"total={arguments.episodes} success_rate={metrics['success_rate']:.4f}"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--env-id", default=FORMAL_ENV_ID)
    parser.add_argument("--base-policy-ckpt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=OFFICIAL_EPISODES)
    parser.add_argument("--num-eval-envs", type=int, default=OFFICIAL_EVAL_ENVS)
    parser.add_argument("--ddim-steps", type=int, default=4)
    parser.add_argument("--act-horizon", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
