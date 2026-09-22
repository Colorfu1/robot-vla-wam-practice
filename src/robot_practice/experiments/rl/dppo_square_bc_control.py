"""Standalone DPPO BC control on the existing NPZ; no PPO and no asset downloads."""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def evaluate(agent, cfg, args, output):
    import hydra

    from robot_practice.experiments.rl.lightwam_base import (
        _DPPOSquareEnvClient,
        _summarize_official_square_rollout,
    )

    # Evaluation must not perturb the subsequent training RNG sequence.
    py_rng, np_rng = random.getstate(), np.random.get_state()
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        random.seed(10001)
        np.random.seed(10001)
        torch.manual_seed(10001)
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.network.augment = False
        model_cfg.use_ddim = True
        model_cfg.ddim_steps = 5
        model_cfg.randn_clip_value = 3
        output.mkdir()
        ema_path = output / "ema_weights.pt"
        torch.save({"model": agent.ema_model.state_dict()}, ema_path)
        model_cfg._target_ = "model.diffusion.diffusion_eval.DiffusionEval"
        model_cfg.network_path = str(ema_path)
        model_cfg.ft_denoising_steps = 0
        model = hydra.utils.instantiate(model_cfg).eval()
        # Evaluate EMA explicitly; do not rely on an implicit checkpoint key preference.
        model.network.load_state_dict(agent.ema_model.network.state_dict(), strict=True)
        os.environ["ROBOT_PRACTICE_ROLLOUT_FORENSICS"] = str((output / "forensics").resolve())
        rewards = np.zeros((100, args.envs))
        firsts = np.zeros((101, args.envs), dtype=bool)
        firsts[0] = True
        started = time.monotonic()
        with _DPPOSquareEnvClient(
            dppo_python=Path(sys.executable),
            dppo_root=args.dppo_root,
            normalization_path=args.normalization,
            num_envs=args.envs,
            max_episode_steps=400,
            action_horizon=4,
            policy_seed=10001,
            worker_log=output / "env_worker.log",
        ) as env:
            obs = env.reset()
            for step in range(100):
                with torch.inference_mode():
                    cond = {
                        key: torch.as_tensor(obs[key], device="cuda:0").float()
                        for key in ("rgb", "state")
                    }
                    action = model(cond=cond, deterministic=True).trajectories.cpu().numpy()
                if action.shape != (args.envs, 4, 7) or not np.isfinite(action).all():
                    raise ValueError("Invalid predicted action")
                obs, reward, terminated, truncated = env.step(np.clip(action, -1, 1))
                rewards[step] = reward
                firsts[step + 1] = terminated | truncated
                if step % 5 == 0 or step == 99:
                    logger.info(
                        "STATUS phase=eval epoch=%d completed=%d total=100", agent.epoch, step + 1
                    )
        episodes = _summarize_official_square_rollout(
            rewards,
            firsts,
            act_steps=4,
            success_threshold=1.0,
        )
        if len(episodes) != args.envs:
            raise RuntimeError(f"Expected {args.envs} completed episodes, got {len(episodes)}")
        result = {
            "epoch": agent.epoch,
            "optimizer_steps": agent.updates,
            "weight_variant": "ema",
            "episodes": episodes,
            "success_rate": float(np.mean([r["success"] for r in episodes])),
            "environment_seeds": list(range(10001, 10001 + args.envs)),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output / "metrics.json", result)
        logger.info(
            "DONE phase=eval epoch=%d episodes=%d success_rate=%.4f",
            agent.epoch,
            len(episodes),
            result["success_rate"],
        )
        return result
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(gpu_rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dppo-root", type=Path, default=Path("/opt/robot-practice/dppo"))
    parser.add_argument(
        "--data", type=Path, default=Path("datasets/raw/dppo/robomimic/square-img/train.npz")
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("datasets/raw/dppo/robomimic/square-img/normalization.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.stop_after <= 4000 or args.eval_every < 1 or args.envs < 1:
        parser.error("Invalid training/evaluation budget")
    for key in ("dppo_root", "data", "normalization", "output"):
        setattr(args, key, getattr(args, key).resolve())
    with np.load(args.data, allow_pickle=False) as data:
        if len(data["traj_lengths"]) != 100 or int(data["traj_lengths"].sum()) != 27214:
            raise ValueError("Expected the existing 100-demo / 27214-frame NPZ")
    if not args.normalization.is_file():
        raise FileNotFoundError(args.normalization)
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log")],
    )
    sys.path.insert(0, str(args.dppo_root))
    from agent.pretrain.train_diffusion_agent import TrainDiffusionAgent

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    config_path = args.dppo_root / "cfg/robomimic/pretrain/square/pre_diffusion_mlp_img.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.pop("hydra", None)
    cfg.pop("defaults", None)
    cfg.logdir = str(args.output)
    cfg.wandb = None
    cfg.seed = 42
    cfg.device = "cuda:0"
    cfg.train_dataset_path = str(args.data)
    cfg.train.n_epochs = 4000
    cfg.train.lr_scheduler.first_cycle_steps = 4000
    cfg.train.lr_scheduler.warmup_steps = 100
    cfg.train.save_model_freq = args.eval_every
    cfg.train_dataset.device = "cuda:0"
    cfg.train_dataset.max_n_episodes = 100
    if args.smoke:
        cfg.train.batch_size = 2
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, args.output / "config.yaml")
    write_json(
        args.output / "manifest.json",
        {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "data_sha256": sha256(args.data),
            "normalization_sha256": sha256(args.normalization),
            "upstream_config_sha256": sha256(config_path),
            "source_sha256": sha256(__file__),
            "upstream_sources_sha256": {
                name: sha256(args.dppo_root / name)
                for name in (
                    "agent/pretrain/train_agent.py",
                    "agent/dataset/sequence.py",
                    "model/diffusion/diffusion.py",
                    "model/diffusion/diffusion_eval.py",
                    "model/diffusion/mlp_diffusion.py",
                )
            },
            "training": "BC from scratch, upstream loss/optimizer/EMA; no PPO",
            "evaluation": "OSMesa spawn serial, 400 steps, action horizon 4, seeds starting 10001",
            "weight_variant": "ema",
            "schedule_epochs": 4000,
            "formal_result": False,
        },
    )
    os.environ.update(
        {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND": "osmesa",
            "ROBOT_PRACTICE_DIAGNOSTIC_OFFSAMPLES": "4",
            "ROBOT_PRACTICE_READPIXELS_PROBE": "1",
            "ROBOT_PRACTICE_ROLLOUT_START_METHOD": "spawn",
            "ROBOT_PRACTICE_ROLLOUT_SERIAL_STEP": "1",
        }
    )
    agent = TrainDiffusionAgent(cfg)
    agent.updates = 0
    results = []
    epochs = 1 if args.smoke else args.stop_after
    total = 2 if args.smoke else epochs * len(agent.dataloader_train)
    logger.info("START epochs=%d updates=%d windows=%d", epochs, total, len(agent.dataset_train))
    for epoch in range(1, epochs + 1):
        agent.epoch = epoch
        losses = []
        for batch in agent.dataloader_train:
            agent.model.train()
            loss = agent.model.loss(*batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            agent.optimizer.step()
            agent.optimizer.zero_grad()
            if agent.updates % agent.update_ema_freq == 0:
                agent.step_ema()
            agent.updates += 1
            losses.append(loss.item())
            if agent.updates % 25 == 0 or args.smoke:
                logger.info(
                    "STATUS phase=train completed=%d total=%d epoch=%d loss=%.6f",
                    agent.updates,
                    total,
                    epoch,
                    loss.item(),
                )
            if args.smoke and agent.updates == 2:
                break
        agent.lr_scheduler.step()
        with (args.output / "train_metrics.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {"epoch": epoch, "updates": agent.updates, "train_loss": float(np.mean(losses))}
                )
                + "\n"
            )
        if epoch % args.eval_every == 0 or epoch == epochs:
            agent.save_model()
            # Keep a full restart snapshot even though this entrypoint does not resume yet.
            torch.save(
                {
                    "epoch": epoch,
                    "updates": agent.updates,
                    "model": agent.model.state_dict(),
                    "ema": agent.ema_model.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "scheduler": agent.lr_scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                },
                args.output / "training_state.pt",
            )
            results.append(evaluate(agent, cfg, args, args.output / f"eval_epoch_{epoch:04d}"))
            write_json(args.output / "metrics.json", {"smoke": args.smoke, "results": results})
    logger.info("DONE phase=all completed=%d total=%d", agent.updates, total)


if __name__ == "__main__":
    main()
