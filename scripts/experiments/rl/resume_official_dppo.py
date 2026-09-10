#!/usr/bin/env python3
"""Resume an official DPPO agent from an upstream model checkpoint.

The upstream launcher initializes ``agent.itr`` to zero and does not expose a
resume flag.  This small shim composes the same Hydra config, instantiates the
same upstream agent class, restores the checkpoint model/iteration, and then
calls the upstream ``agent.run()``.  It intentionally lives in this
repository; the upstream DPPO checkout is not modified.

Official DPPO checkpoints contain the model state and iteration only.  When a
checkpoint does not contain optimizer/scheduler state, this shim restores the
model and iteration and aligns the learning-rate schedulers to the resumed
iteration; Adam moments and RNG streams cannot be reconstructed from the
official checkpoint format.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides; pass them after '--'",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path, device: str) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older torch versions
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"checkpoint is missing the official 'model' field: {path}")
    if "itr" not in checkpoint:
        raise ValueError(f"checkpoint is missing the official 'itr' field: {path}")
    return checkpoint


def _align_scheduler(scheduler: object, next_itr: int) -> None:
    """Align an upstream scheduler without changing the upstream source."""

    # The official scheduler implements step(epoch=...), and its state is not
    # serialized in state_N.pt.  This keeps the LR schedule at the resumed
    # iteration even though Adam moments are unavailable.
    step = getattr(scheduler, "step", None)
    if step is not None:
        step(epoch=next_itr)


def main() -> int:
    args = _parse_args()
    if args.overrides and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]

    dppo_root = Path.cwd().resolve()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = (dppo_root / config_dir).resolve()
    checkpoint_path = Path(args.resume_checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

    # The upstream config imports top-level packages such as agent/model.
    sys.path.insert(0, str(dppo_root))
    os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    checkpoint = _load_checkpoint(checkpoint_path, str(cfg.device))
    checkpoint_itr = int(checkpoint["itr"])
    target_itr = int(cfg.train.n_train_itr)
    next_itr = checkpoint_itr + 1
    if checkpoint_itr >= target_itr:
        raise ValueError(
            f"checkpoint itr={checkpoint_itr} is outside target n_train_itr={target_itr}"
        )

    # Match the upstream TrainAgent.load() contract: a changed model/config is
    # a hard error, not a partially restored run with silently fresh weights.
    agent.model.load_state_dict(checkpoint["model"])
    agent.itr = next_itr

    # Official state_N.pt does not contain optimizer/scheduler state.  Restore
    # optional states if a future checkpoint provides them; otherwise at least
    # put the official schedulers at the correct iteration.
    if "actor_optimizer" in checkpoint and hasattr(agent, "actor_optimizer"):
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    if "critic_optimizer" in checkpoint and hasattr(agent, "critic_optimizer"):
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    if "actor_lr_scheduler" in checkpoint and hasattr(agent, "actor_lr_scheduler"):
        agent.actor_lr_scheduler.load_state_dict(checkpoint["actor_lr_scheduler"])
    else:
        _align_scheduler(agent.actor_lr_scheduler, next_itr)
    if "critic_lr_scheduler" in checkpoint and hasattr(agent, "critic_lr_scheduler"):
        agent.critic_lr_scheduler.load_state_dict(checkpoint["critic_lr_scheduler"])
    else:
        _align_scheduler(agent.critic_lr_scheduler, next_itr)

    logging.getLogger(__name__).info(
        "RESUME checkpoint=%s checkpoint_itr=%d next_itr=%d target_n_train_itr=%d "
        "optimizer_state_restored=%s",
        checkpoint_path,
        checkpoint_itr,
        next_itr,
        target_itr,
        "actor_optimizer" in checkpoint and "critic_optimizer" in checkpoint,
    )
    agent.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
