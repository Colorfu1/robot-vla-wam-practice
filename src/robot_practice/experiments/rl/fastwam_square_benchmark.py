"""Offline, single-GPU FastWAM training throughput benchmark on Square NPZ.

Uses real vendor loss/backprop/AdamW updates, not inference or cached VAE latents.
No checkpoint is saved; projected training time is not time to convergence.
"""

import argparse
import hashlib
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def project_times(seconds, effective_batch, windows, target_updates, target_batch, epochs):
    if not seconds or min(seconds) <= 0 or min(effective_batch, windows, target_batch) <= 0:
        raise ValueError("Positive timings and batch/dataset sizes required")
    mean = statistics.mean(seconds)
    throughput = effective_batch / mean
    return {
        "mean_update_seconds": mean,
        "median_update_seconds": statistics.median(seconds),
        "min_update_seconds": min(seconds),
        "max_update_seconds": max(seconds),
        "windows_per_second": throughput,
        "hours_for_target_sample_volume": target_updates * target_batch / throughput / 3600,
        "target_updates": target_updates,
        "target_global_batch": target_batch,
        "target_sample_presentations": target_updates * target_batch,
        "hours_for_epochs": epochs * windows / throughput / 3600,
        "target_epochs": epochs,
        "hours_per_epoch": windows / throughput / 3600,
        "comparable_update_batch": effective_batch == target_batch,
        "scope": "single GPU steady-state; excludes loading, eval and checkpoint I/O",
        "time_to_convergence": None,
    }


def model_kwargs(vendor, action_weights, checkpointing):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(vendor) / "configs/model/fastwam.yaml")
    # Replace task-specific interpolations before resolving the official model config.
    cfg.proprio_dim = 9
    cfg.video_dit_config.action_dim = 7
    cfg.action_dit_config.action_dim = 7
    cfg.mot_checkpoint_mixed_attn = checkpointing
    cfg.video_dit_config.use_gradient_checkpointing = checkpointing
    cfg.action_dit_config.use_gradient_checkpointing = checkpointing
    cfg.action_dit_pretrained_path = str(Path(action_weights).resolve())
    cfg.redirect_common_files = False
    cfg.load_text_encoder = False
    result = OmegaConf.to_container(cfg, resolve=True)
    result.pop("_target_")
    return result


def write_result(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vendor", default="third_party/w0_fastwam/FastWAM")
    parser.add_argument("--assets", default="pretrained/w0_fastwam")
    parser.add_argument("--npz", default="datasets/raw/dppo/robomimic/square-img/train.npz")
    parser.add_argument(
        "--context", default="outputs/lightwam_base_robomimic_square/assets/square_context.pt"
    )
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=64)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--measure-updates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--target-updates", type=int, default=420000)
    parser.add_argument("--target-batch", type=int, default=256)
    parser.add_argument("--target-epochs", type=int, default=4000)
    args = parser.parse_args()
    for name in ("microbatch", "accumulation", "warmup_updates", "measure_updates",
                 "target_updates", "target_batch", "target_epochs"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        logging.StreamHandler(), logging.FileHandler(output / "run.log")
    ])
    result = {"status": "running", "arguments": vars(args), "updates": []}
    write_result(output / "metrics.json", result)
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from robot_practice.experiments.rl.lightwam_base import RobomimicLightWAMWindowDataset

        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("This benchmark is single GPU; do not launch with torchrun")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for real training timing")
        torch.manual_seed(42)
        np.random.seed(42)
        vendor = Path(args.vendor).resolve()
        assets = Path(args.assets).resolve()
        weights = assets / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
        wan = assets / "models/Wan-AI/Wan2.2-TI2V-5B"
        required = [vendor / "src/fastwam/runtime.py", weights, wan / "Wan2.2_VAE.pth",
                    Path(args.npz), Path(args.context)]
        required += [wan / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors"
                     for i in range(1, 4)]
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Required offline asset missing: {path}")
        os.environ.update(DIFFSYNTH_MODEL_BASE_PATH=str(assets / "models"),
                          DIFFSYNTH_SKIP_DOWNLOAD="true", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1")
        sys.path.insert(0, str(vendor / "src"))
        from fastwam.runtime import create_fastwam
        from fastwam.trainer import Wan22Trainer

        kwargs = model_kwargs(vendor, weights, args.checkpointing)
        result["model_config"] = kwargs
        result["source_sha256"] = {
            name: hashlib.sha256((vendor / name).read_bytes()).hexdigest()
            for name in ("src/fastwam/models/wan22/fastwam.py", "src/fastwam/trainer.py",
                         "configs/model/fastwam.yaml")
        }
        logger.info("START phase=data completed=0 total=1")
        dataset = RobomimicLightWAMWindowDataset(
            args.npz, action_horizon=4, video_frames=5, image_size=(224, 224),
            context_path=args.context, normalized=True,
        )
        if len(dataset.traj_lengths) != 100 or dataset.states.shape[1] != 9 or dataset.actions.shape[1] != 7:
            raise ValueError("Expected the 100-trajectory Square NPZ with 9D state and 7D action")
        if len(dataset) < args.microbatch:
            raise ValueError("Dataset smaller than microbatch")
        loader = DataLoader(dataset, batch_size=args.microbatch, shuffle=True, drop_last=True,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
        iterator = iter(loader)
        result["dataset"] = {"trajectories": len(dataset.traj_lengths), "windows": len(dataset),
                             "video_shape": [3, 5, 224, 224], "action_shape": [4, 7],
                             "proprio_dim": 9, "control_hz": 20}
        logger.info("DONE phase=data completed=1 total=1 windows=%d", len(dataset))
        logger.info("START phase=load completed=0 total=1")
        model = create_fastwam(**kwargs, device="cuda:0", model_dtype=torch.bfloat16)
        Wan22Trainer._apply_dit_only_train_mode(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-2)
        result["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                                "trainable": sum(p.numel() for p in params)}
        result["runtime"] = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                             "parameter_dtype": "bf16", "optimizer": "AdamW (0.9,0.95)",
                             "lr": 1e-4, "gradient_clip": 1.0, "vae_cache": False,
                             "world_size": 1, "schedule": "constant LR short benchmark",
                             "model_source": str(Path(sys.modules[model.__module__].__file__))}
        torch.cuda.synchronize()
        result["setup_seconds"] = time.perf_counter() - started
        logger.info("DONE phase=load completed=1 total=1 trainable=%d", result["parameters"]["trainable"])
        total = args.warmup_updates + args.measure_updates
        for update in range(total):
            if update == args.warmup_updates:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            waits = 0.0
            for micro in range(args.accumulation):
                wait_start = time.perf_counter()
                try:
                    sample = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    sample = next(iterator)
                waits += time.perf_counter() - wait_start
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, detail = model.training_loss(sample)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite training loss")
                (loss / args.accumulation).backward()
                losses.append(float(loss.detach()))
                if micro % 8 == 0 or micro + 1 == args.accumulation:
                    logger.info("STATUS phase=train completed=%d total=%d update=%d/%d loss=%.6f",
                                update * args.accumulation + micro + 1, total * args.accumulation,
                                update + 1, total, losses[-1])
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            result["updates"].append({"update": update + 1,
                                      "warmup": update < args.warmup_updates,
                                      "seconds": elapsed, "data_wait_seconds": waits,
                                      "mean_loss": statistics.mean(losses), "last_loss_parts": detail})
            logger.info("DONE phase=update completed=%d total=%d seconds=%.3f", update + 1, total, elapsed)
            write_result(output / "metrics.json", result)
        measured = [r["seconds"] for r in result["updates"] if not r["warmup"]]
        result["projection"] = project_times(measured, args.microbatch * args.accumulation,
                                             len(dataset), args.target_updates, args.target_batch,
                                             args.target_epochs)
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        result["status"] = "complete"
        result["limitations"] = [
            "Short throughput run, not converged training; no weights saved.",
            "No evaluation/checkpoint I/O, LR schedule or distributed communication measured.",
            "Global-batch-equivalent accumulation is not an 8-GPU speed measurement.",
            "End-to-end update timing includes VAE, backward, optimizer, data wait and logging.",
        ]
        logger.info("DONE phase=all completed=%d total=%d windows_per_second=%.3f projected_hours=%.2f",
                    total, total, result["projection"]["windows_per_second"],
                    result["projection"]["hours_for_target_sample_volume"])
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logger.exception("FAILED benchmark; preserve run.log before changing configuration")
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        write_result(output / "metrics.json", result)


if __name__ == "__main__":
    main()
