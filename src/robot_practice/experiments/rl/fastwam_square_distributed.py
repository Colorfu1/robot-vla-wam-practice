"""DDP FastWAM Square benchmark and wall-clock-limited supervised pilot."""

import argparse
import hashlib
import logging
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from robot_practice.experiments.rl.fastwam_square_benchmark import (
    model_kwargs,
    project_times,
    write_result,
)

log = logging.getLogger(__name__)


def learning_rate(elapsed, peak, minimum, budget_seconds, warmup_fraction):
    """Wall-clock warmup + cosine; independent of optimizer-update throughput."""
    elapsed = max(0.0, elapsed)
    warmup = budget_seconds * warmup_fraction
    if warmup > 0 and elapsed < warmup:
        return minimum + (peak - minimum) * elapsed / warmup
    progress = min(1.0, (elapsed - warmup) / (budget_seconds - warmup))
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def optimizer_update(model, optimizer, params, next_sample, accumulation, device):
    """DDP averages rank gradients; loss division averages accumulation microbatches."""
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for micro in range(accumulation):
        sync = model.no_sync() if micro + 1 < accumulation and hasattr(model, "no_sync") else nullcontext()
        with sync:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss, _ = model(next_sample())
            finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite loss on at least one rank")
            (loss / accumulation).backward()
            loss_sum += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss_sum / accumulation


class Batches:
    def __init__(self, loader, sampler, epoch=0, offset=0):
        self.loader, self.sampler = loader, sampler
        self.epoch, self.offset = epoch, 0
        sampler.set_epoch(epoch)
        self.iterator = iter(loader)
        for _ in range(offset):
            next(self.iterator)
            self.offset += 1

    def __call__(self):
        try:
            sample = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.offset = 0
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            sample = next(self.iterator)
        self.offset += 1
        return sample


def save_checkpoint(model, optimizer, batches, step, elapsed, output, protocol, rank):
    local = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
             "numpy": np.random.get_state(), "python": random.getstate(),
             "epoch": batches.epoch, "offset": batches.offset}
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local)
    if rank == 0:
        log.info("START phase=save completed=0 total=1 step=%d", step)
        directory = output / "checkpoints"
        directory.mkdir(exist_ok=True)
        # Write atomically; retain weight snapshots, replace only this run's latest resume state.
        weights = directory / f"step_{step:06d}.pt"
        temporary = weights.with_suffix(".tmp")
        model.save_checkpoint(str(temporary), step=step)
        temporary.replace(weights)
        payload = {"mot": model.mot.state_dict(),
                   "proprio_encoder": model.proprio_encoder.state_dict(),
                   "optimizer": optimizer.state_dict(), "step": step,
                   "elapsed_training_seconds": elapsed, "ranks": ranks, "protocol": protocol}
        temporary = directory / "latest_state.tmp"
        torch.save(payload, temporary)
        temporary.replace(directory / "latest_state.pt")
        log.info("DONE phase=save completed=1 total=1 step=%d", step)
    dist.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["benchmark", "train"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vendor", default="third_party/w0_fastwam/FastWAM")
    p.add_argument("--assets", default="pretrained/w0_fastwam")
    p.add_argument("--npz", default="datasets/raw/dppo/robomimic/square-img/train.npz")
    p.add_argument("--dual-data", help="Completed native dual-view dataset: fixed 32/4/9/4 protocol")
    p.add_argument("--checkpointing", action="store_true", help="Enable model gradient checkpointing")
    p.add_argument("--context", default="outputs/lightwam_base_robomimic_square/assets/square_context.pt")
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--warmup-updates", type=int, default=5)
    p.add_argument("--measure-updates", type=int, default=20)
    p.add_argument("--hours", type=float, default=12)
    p.add_argument("--save-hours", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-fraction", type=float, default=0.01)
    p.add_argument("--resume", help="Trusted latest_state.pt; --hours is total budget, not extra hours")
    args = p.parse_args()
    if min(args.microbatch, args.accumulation, args.warmup_updates, args.measure_updates,
           args.hours, args.save_hours, args.lr) <= 0:
        p.error("Sizes, budgets and learning-rate parameters must be positive")
    if (args.workers < 0 or not 0 <= args.min_lr <= args.lr
            or not 0 <= args.warmup_fraction < 1
            or not all(math.isfinite(v) for v in (args.hours, args.save_hours, args.lr, args.min_lr))):
        p.error("Require finite budgets, 0 <= min-lr <= lr and 0 <= warmup-fraction < 1")
    if args.resume and args.mode != "train":
        p.error("Resume is only supported for training")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    handlers = [logging.FileHandler(output / ("run.log" if rank == 0 else f"rank_{rank}.log"))]
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    result = {"status": "running", "mode": args.mode, "arguments": vars(args), "world_size": world}
    if rank == 0:
        write_result(output / "metrics.json", result)
    try:
        from robot_practice.experiments.rl.lightwam_base import RobomimicLightWAMWindowDataset

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        vendor, assets = Path(args.vendor).resolve(), Path(args.assets).resolve()
        os.environ.update(DIFFSYNTH_MODEL_BASE_PATH=str(assets / "models"),
                          DIFFSYNTH_SKIP_DOWNLOAD="true", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1")
        sys.path.insert(0, str(vendor / "src"))
        from fastwam.runtime import create_fastwam
        from fastwam.trainer import Wan22Trainer

        kwargs = model_kwargs(vendor, assets / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
                              args.checkpointing)
        if args.dual_data:
            from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL, DualViewDataset

            dataset = DualViewDataset(args.dual_data, args.context)
            kwargs.setdefault("loss", {}).update(lambda_video=1.0, lambda_action=1.0)
        else:
            dataset = RobomimicLightWAMWindowDataset(args.npz, action_horizon=4, video_frames=5,
                                                   image_size=(224, 224), context_path=args.context)
        if len(dataset.traj_lengths) != 100 or dataset.states.shape[1] != 9 or dataset.actions.shape[1] != 7:
            raise ValueError("Expected 100-trajectory Square NPZ, 9D state, 7D action")
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=42, drop_last=True)
        loader = DataLoader(dataset, sampler=sampler, batch_size=args.microbatch, drop_last=True,
                            num_workers=args.workers, persistent_workers=args.workers > 0,
                            multiprocessing_context="spawn" if args.workers else None,
                            pin_memory=True, generator=torch.Generator().manual_seed(42 + rank))
        if not len(loader):
            raise ValueError("No complete batch on this rank")
        # Start data workers before constructing CUDA model; dataset access is deterministic.
        batches = Batches(loader, sampler)
        log.info("START phase=load completed=0 total=1 rank=%d", rank)
        model = create_fastwam(**kwargs, device=str(device), model_dtype=torch.bfloat16)
        Wan22Trainer._apply_dit_only_train_mode(model)
        params = [v for v in model.parameters() if v.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-2)
        protocol = {"world_size": world, "microbatch": args.microbatch,
                    "accumulation": args.accumulation, "model": kwargs,
                    "lr": args.lr, "min_lr": args.min_lr,
                    "scheduler": "wall_clock_warmup_cosine_v1",
                    "budget_seconds": args.hours * 3600,
                    "warmup_fraction": args.warmup_fraction,
                    "windows": len(dataset), "source_hash": hashlib.sha256(
                        (vendor / "src/fastwam/models/wan22/fastwam.py").read_bytes()).hexdigest()}
        # Dataset identity check is rank-zero only, then shared across ranks.
        identity = [None]
        if rank == 0:
            identity[0] = {}
            for label, name in (("npz", args.npz), ("context", args.context)):
                digest = hashlib.sha256()
                with open(name, "rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024**2), b""):
                        digest.update(block)
                identity[0][label] = digest.hexdigest()
        dist.broadcast_object_list(identity, src=0)
        protocol["data_sha256"] = identity[0]
        if args.dual_data:
            import json

            from robot_practice.experiments.rl.lightwam_train_scene_eval import file_hash

            manifest = json.loads((Path(args.dual_data) / "manifest.json").read_text())
            if manifest["source_sha256"]["npz"] != identity[0]["npz"]:
                raise ValueError("Dual-view source NPZ differs from requested training data")
            if rank == 0:
                LOGICAL_KEYS = ("images", "states", "actions", "traj_lengths")
                for key in LOGICAL_KEYS:
                    if file_hash(Path(args.dual_data) / f"{key}.npy") != manifest["array_sha256"][key]:
                        raise ValueError(f"Dual-view dataset hash mismatch: {key}")
            dist.barrier()
            protocol.update(observation_action=PROTOCOL,
                            dual_manifest_sha256=file_hash(Path(args.dual_data) / "manifest.json"))
        step, elapsed_before = 0, 0.0
        resume_rng = None
        if args.resume:
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            if payload["protocol"] != protocol:
                raise ValueError("Resume protocol/data/source differs from saved state")
            model.mot.load_state_dict(payload["mot"], strict=True)
            model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            step, elapsed_before = payload["step"], payload["elapsed_training_seconds"]
            clock_path = Path(args.resume).with_name("latest_state_clock.json")
            if clock_path.exists():
                import json

                clock_state = json.loads(clock_path.read_text())
                if clock_state["step"] == step:
                    elapsed_before = max(elapsed_before, clock_state["elapsed_training_seconds"])
            if elapsed_before >= args.hours * 3600:
                raise ValueError("Saved run has already exhausted its total training-time budget")
            resume_rng = payload["ranks"][rank]
            batches = Batches(loader, sampler, resume_rng["epoch"], resume_rng["offset"])
            del payload
        ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True,
                  gradient_as_bucket_view=True, broadcast_buffers=False)
        if resume_rng is None:
            torch.manual_seed(42 + rank)
            np.random.seed(42 + rank)
            random.seed(42 + rank)
        else:
            torch.set_rng_state(resume_rng["torch"])
            torch.cuda.set_rng_state(resume_rng["cuda"])
            np.random.set_state(resume_rng["numpy"])
            random.setstate(resume_rng["python"])
        batch = args.microbatch * args.accumulation * world
        result.update(protocol=protocol, global_batch=batch,
                      trainable_parameters=sum(v.numel() for v in params),
                      runtime={"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                               "ddp": True, "precision": "bf16", "optimizer": "AdamW",
                               "checkpointing": args.checkpointing, "vae_cache": False})
        if args.mode == "train" and rank == 0:
            # Retained bf16 weight snapshots plus two full-state sizes during atomic replacement.
            weight_bytes = sum(v.numel() * v.element_size() for v in params)
            required = weight_bytes * (math.ceil(args.hours / args.save_hours) + 8)
            if shutil.disk_usage(output).free < required:
                raise RuntimeError(f"Insufficient checkpoint disk space; need about {required / 2**30:.1f} GiB")
        dist.barrier()
        torch.cuda.synchronize()
        log.info("DONE phase=load completed=1 total=1 rank=%d global_batch=%d", rank, batch)
        started = time.perf_counter()
        last_save = elapsed_before
        measured = []
        durations = []
        target = args.warmup_updates + args.measure_updates
        while True:
            if args.mode == "benchmark" and step == args.warmup_updates:
                torch.cuda.reset_peak_memory_stats()
            # A single clock on rank zero determines LR for all ranks, including after resume.
            clock = torch.zeros(1, device=device, dtype=torch.float64)
            if rank == 0:
                clock[0] = elapsed_before + time.perf_counter() - started
            dist.broadcast(clock, src=0)
            lr = args.lr if args.mode == "benchmark" else learning_rate(
                clock.item(), args.lr, args.min_lr, args.hours * 3600, args.warmup_fraction)
            for group in optimizer.param_groups:
                group["lr"] = lr
            torch.cuda.synchronize()
            tick = time.perf_counter()
            loss = optimizer_update(ddp, optimizer, params, batches, args.accumulation, device)
            torch.cuda.synchronize()
            seconds = torch.tensor(time.perf_counter() - tick, device=device)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            loss_tensor = torch.tensor(loss, device=device)
            dist.all_reduce(loss_tensor)
            loss = loss_tensor.item() / world
            step += 1
            duration = seconds.item()
            durations.append(duration)
            if step > args.warmup_updates:
                measured.append(duration)
            # Rank zero owns the wall-clock stop/save decisions so all ranks enter collectives together.
            control = torch.zeros(3, device=device, dtype=torch.float64)
            if rank == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                stop = step >= target if args.mode == "benchmark" else elapsed >= args.hours * 3600
                save = args.mode == "train" and (stop or elapsed - last_save >= args.save_hours * 3600)
                control[:] = torch.tensor([elapsed, stop, save], device=device)
            dist.broadcast(control, src=0)
            elapsed, stop, save = control.tolist()
            if rank == 0:
                log.info("STATUS phase=%s completed=%d total=%d unit=%s step=%d loss=%.6f lr=%.8f update_s=%.3f",
                         args.mode, step if args.mode == "benchmark" else int(elapsed),
                         target if args.mode == "benchmark" else int(args.hours * 3600),
                         "updates" if args.mode == "benchmark" else "seconds", step, loss, lr, duration)
                result.update(step=step, elapsed_training_seconds=elapsed, last_loss=loss,
                              last_update_seconds=duration, last_learning_rate=lr)
                write_result(output / "metrics.json", result)
            if save:
                save_checkpoint(model, optimizer, batches, step, elapsed, output, protocol, rank)
                if rank == 0:
                    write_result(output / "checkpoints/latest_state_clock.json", {
                        "step": step,
                        "elapsed_training_seconds": elapsed_before + time.perf_counter() - started,
                    })
                last_save = elapsed
            if stop:
                break
        memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], device=device)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        if rank == 0:
            projection = project_times(measured or durations, batch, len(dataset), 420000, 256, 4000)
            projection["scope"] = f"measured {world}-GPU DDP updates; excludes checkpoint/eval I/O"
            result.update(status="complete", projection=projection,
                          peak_allocated_gib=memory[0].item() / 2**30,
                          peak_reserved_gib=memory[1].item() / 2**30,
                          measured_update_seconds=measured,
                          wall_seconds_including_saves=time.perf_counter() - started + elapsed_before,
                          limitations=["No closed-loop evaluation in this training run.",
                                       "Short benchmark cannot predict convergence.",
                                       "Final checkpoint save may extend beyond wall-clock budget."])
            write_result(output / "metrics.json", result)
            log.info("DONE phase=all completed=%d total=%d windows_per_second=%.3f", step, step,
                     projection["windows_per_second"])
    except Exception as exc:
        log.exception("FAILED rank=%d", rank)
        write_result(output / f"failure_rank_{rank}.json", {"error": f"{type(exc).__name__}: {exc}"})
        if rank == 0:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_result(output / "metrics.json", result)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
