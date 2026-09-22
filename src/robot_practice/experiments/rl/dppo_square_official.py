"""Pinned upstream Square image BC training/evaluation, with explicit runtime adapters."""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

REVISION = "cc7234ad7ff39a8f32de3af903606723a16f0648"
LEGACY_REVISION = "dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc"
logger = logging.getLogger(__name__)


class Progress(logging.Filter):
    def __init__(self, total):
        super().__init__()
        self.total = total

    def filter(self, record):
        match = re.match(r"(\d+): train loss", record.getMessage())
        if match:
            record.msg = (
                f"STATUS phase=train completed={match[1]} total={self.total} " + record.getMessage()
            )
            record.args = ()
        return True


def configure(root, output, data, mode, checkpoint=None, smoke=False, eval_overrides=None):
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    kind = "pretrain" if mode == "train" else "eval"
    name = "pre_diffusion_mlp_img.yaml" if mode == "train" else "eval_diffusion_mlp_img.yaml"
    cfg = OmegaConf.load(root / "cfg/robomimic" / kind / "square" / name)
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    cfg.logdir = str(output)
    if mode == "train":
        cfg.wandb = None
        cfg.train_dataset_path = str(data / "train.npz")
        if smoke:
            # Full batch size and full dataset; only truncate run length, not LR schedule.
            cfg.train.n_epochs = 1
    else:
        cfg.base_policy_path = str(checkpoint)
        cfg.normalization_path = str(data / "normalization.npz")
        cfg.robomimic_env_cfg_path = str(root / "cfg/robomimic/env_meta/square-img.json")
        # Upstream YAML references a nonexistent module at this revision.
        cfg.model._target_ = "model.diffusion.diffusion_eval.DiffusionEval"
        if smoke:
            cfg.env.n_envs = 1
            cfg.n_steps = 100
        for key, value in (eval_overrides or {}).items():
            OmegaConf.update(cfg, key, value)
    OmegaConf.resolve(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--root", type=Path, default=Path("third_party/dppo_official"))
    parser.add_argument("--data", type=Path, default=Path("datasets/raw/dppo/robomimic/square-img"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-version", choices=("current", "v0.6"), default="current")
    parser.add_argument("--weight-variant", choices=("model", "ema"), default="model")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ft-denoising-steps", type=int)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--eval-envs", type=int)
    parser.add_argument("--eval-steps", type=int)
    args = parser.parse_args()
    root, data, output = args.root.resolve(), args.data.resolve(), args.output.resolve()
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=no"], text=True
    )
    expected_revision = LEGACY_REVISION if args.source_version == "v0.6" else REVISION
    if revision != expected_revision or dirty:
        raise RuntimeError(
            "Expected clean pinned upstream checkout; do not use the patched runtime tree"
        )
    if args.source_version == "v0.6" and args.mode == "eval":
        parser.error(
            "v0.6 has no standalone eval entry; use current eval with explicit weight variant"
        )
    required = data / ("train.npz" if args.mode == "train" else "normalization.npz")
    if not required.is_file():
        raise FileNotFoundError(required)
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if args.mode == "eval" and (checkpoint is None or not checkpoint.is_file()):
        parser.error("eval requires an existing --checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    sys.path.insert(0, str(root))
    import hydra
    from omegaconf import OmegaConf

    overrides = {
        key: value for key, value in {
            "ft_denoising_steps": args.ft_denoising_steps,
            "seed": args.eval_seed,
            "env.n_envs": args.eval_envs,
            "n_steps": args.eval_steps,
        }.items() if value is not None
    }
    if overrides and args.mode != "eval":
        parser.error("evaluation overrides require eval mode")
    cfg = configure(root, output, data, args.mode, checkpoint, args.smoke, overrides)
    if args.mode == "eval":
        if not 0 <= cfg.ft_denoising_steps <= cfg.ddim_steps:
            parser.error("ft-denoising-steps must be between 0 and ddim_steps")
        if cfg.env.n_envs < 1 or cfg.n_steps < 1:
            parser.error("eval-envs and eval-steps must be positive")
    if args.mode == "eval" and args.weight_variant == "ema":
        import torch

        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "ema" not in weights:
            raise ValueError("Requested EMA but checkpoint has no ema entry")
        selected_path = output / "selected_ema.pt"
        torch.save({"model": weights["ema"]}, selected_path)
        cfg.base_policy_path = str(selected_path)
        # configure() has already resolved interpolations.
        cfg.model.network_path = str(selected_path)
    OmegaConf.save(cfg, output / "config.yaml")
    with required.open("rb") as stream:
        digest = (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else None
        )
    if digest is None:
        hasher = hashlib.sha256()
        with required.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    manifest = {
        "revision": revision,
        "mode": args.mode,
        "source_version": args.source_version,
        "smoke": args.smoke,
        "data_sha256": digest,
        "language_input": False,
        "training_loop": "upstream TrainDiffusionAgent.run unchanged",
        "eval_weights": args.weight_variant,
        "eval_overrides": overrides,
        "eval_adapters": [
            "correct obsolete model import",
            "OSMesa",
            "spawn",
            "retain shared buffers",
        ],
        "claim": "pinned source configuration reproduction, not historical paper score reproduction",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.mode == "train":
        progress = Progress(cfg.train.n_epochs)
        for handler in logging.getLogger().handlers:
            handler.addFilter(progress)
        agent = hydra.utils.get_class(cfg._target_)(cfg)
        logger.info(
            "START phase=train epochs=%d windows=%d batches_per_epoch=%d",
            cfg.train.n_epochs,
            len(agent.dataset_train),
            len(agent.dataloader_train),
        )
        agent.run()
        logger.info(
            "DONE phase=train completed=%d total=%d", cfg.train.n_epochs, cfg.train.n_epochs
        )
    else:
        import multiprocessing

        import agent.eval.eval_agent as eval_module
        from env.gym_utils import async_vector_env

        from robot_practice.experiments.rl.readpixels_probe import probed_shared_worker
        from robot_practice.experiments.rl.rollout_forensics import retain_shared_buffers

        multiprocessing.set_start_method("spawn", force=True)
        os.environ.update(
            MUJOCO_GL="osmesa",
            PYOPENGL_PLATFORM="osmesa",
            ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND="osmesa",
            ROBOT_PRACTICE_DIAGNOSTIC_OFFSAMPLES="4",
            ROBOT_PRACTICE_ROLLOUT_FORENSICS=str(output / "forensics"),
        )
        os.environ.pop("ROBOT_PRACTICE_GRASP_PROBE", None)
        async_vector_env._probe_original_shared_worker = async_vector_env._worker_shared_memory
        async_vector_env._worker_shared_memory = probed_shared_worker
        original_make = eval_module.make_async

        def make_with_retained_buffers(env_id, **kwargs):
            return retain_shared_buffers(async_vector_env, original_make, id=env_id, **kwargs)

        eval_module.make_async = make_with_retained_buffers
        agent = hydra.utils.get_class(cfg._target_)(cfg)
        original_step = agent.venv.step
        completed = 0

        def observed_step(action):
            nonlocal completed
            result = original_step(action)
            completed += 1
            if completed % 10 == 0 or completed == 1:
                logger.info("STATUS phase=eval completed=%d total=%d", completed, cfg.n_steps)
            return result

        agent.venv.step = observed_step
        try:
            agent.run()
        finally:
            agent.venv.close()
        logger.info("DONE phase=eval completed=%d total=%d", completed, cfg.n_steps)


if __name__ == "__main__":
    main()
