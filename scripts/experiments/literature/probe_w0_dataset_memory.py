#!/usr/bin/env python3
"""Isolate the W0 Fast-WAM DataLoader memory growth in a single process.

Why this exists
---------------
Four 8-GPU training runs died to a cgroup OOM at steps 265, 270, 560 and 598. The chain was
established from stack dumps: the kernel SIGKILLs a DataLoader worker, its rank blocks in
``queue.get``, never enqueues its gradient allreduce, and the other seven spin in DeepSpeed's
``allreduce_and_copy`` until the 600 s NCCL watchdog kills the job.

What was measured, and what it ruled out:

* ``memory.stat`` puts the growth in ``rss`` (1368 GiB) not ``cache`` (35 GiB), and
  ``/dev/shm`` holds 4.8 GB, so it is anonymous heap -- not page cache, not the
  shared-memory tensor transport.
* Anonymous memory grows about 5 GiB per step, roughly 30-50 MB per sample consumed.
* ``num_workers`` 8 -> 2 left the slope unchanged while the per-rank figure stayed flat, so
  it accumulates per sample processed. Lowering ``num_workers`` cannot fix it at any value.
* ``MALLOC_ARENA_MAX=2`` with a pinned low ``MALLOC_MMAP_THRESHOLD_`` and
  ``MALLOC_TRIM_THRESHOLD_`` also left the slope unchanged, so it is not glibc arena
  fragmentation.
* Code reading found no per-sample state: ``robot_video_dataset`` assigns ``self.*`` only in
  ``__init__``, ``decode_video_frames_torchvision`` keeps ``loaded_frames`` in a local and
  closes its container, the returned dict holds no view of the big decoded tensor, and
  PyTorch's worker loop already does ``del data``.

So the next step has to be measurement rather than more reading, and it does not need eight
GPUs, DeepSpeed, or NCCL. This script instantiates only the training dataset and iterates it
in one process.

The two counters together are the discriminator:

* live torch tensors grows -> something in Python holds references, and the printed
  shape histogram says what.
* live torch tensors flat while RSS climbs -> retention is below Python: an allocator, a
  C++ cache, or a leaked native handle.

Bisect with --stage to attribute the growth to a layer:

    lerobot   only ``dataset.lerobot_dataset[i]``  (parquet + video decode)
    full      the whole ``dataset[i]``             (adds resize/crop/normalize + text cache)
    text      only the cached text embedding load

Usage
-----
    scripts/experiments/literature/probe_w0_dataset_memory.py --samples 300
    scripts/experiments/literature/probe_w0_dataset_memory.py --stage lerobot --samples 300
"""

from __future__ import annotations

import argparse
import collections
import gc
import os
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _rss_kb() -> dict[str, int]:
    """Read anonymous and total resident memory for this process.

    RssAnon is the figure that matters: the cgroup attributed the training growth to
    anonymous memory rather than page cache.
    """

    wanted = {"VmRSS", "RssAnon", "RssFile", "RssShmem"}
    out: dict[str, int] = {}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in wanted:
                out[key] = int(rest.split()[0])
    return out


def _name_referrers(tensor, limit: int = 6) -> list[str]:
    """Say who is holding a retained tensor.

    A frame referrer is the interesting case: a live frame keeps every local alive, which is
    why intermediate tensors are retained alongside the final one. Frames stay alive when
    something still references their traceback, so the referrer's code name points straight
    at the function whose exception is being kept.
    """

    import gc as _gc

    names: list[str] = []
    for holder in _gc.get_referrers(tensor):
        kind = type(holder).__name__
        if kind == "frame":
            names.append(f"frame:{holder.f_code.co_name}:{holder.f_code.co_filename.split('/')[-1]}")
        elif kind == "dict":
            keys = [k for k in list(holder)[:4] if isinstance(k, str)]
            names.append(f"dict{keys}")
        elif kind == "traceback":
            names.append("traceback")
        else:
            names.append(kind)
        if len(names) >= limit:
            break
    return names


def _tensor_census() -> tuple[int, int, collections.Counter]:
    """Count live torch tensors and the bytes they hold.

    gc.get_objects() only sees Python-visible objects, which is exactly the point: if this
    grows the retention is a Python reference, and if it does not the retention is below
    Python and no amount of dataset-code reading will find it.
    """

    import torch

    count = 0
    total_bytes = 0
    shapes: collections.Counter = collections.Counter()
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj):
                continue
        except Exception:  # pragma: no cover - torch.is_tensor can touch broken proxies
            continue
        count += 1
        try:
            nbytes = obj.untyped_storage().nbytes()
        except Exception:
            nbytes = obj.numel() * obj.element_size()
        total_bytes += nbytes
        if nbytes >= 1 << 20:  # only large tensors are worth naming
            shapes[tuple(obj.shape)] += 1
    return count, total_bytes, shapes


def _largest_retained_tensor():
    """Return the biggest live tensor, which is the one worth attributing."""

    import torch

    best = None
    best_bytes = 0
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj):
                continue
        except Exception:  # pragma: no cover
            continue
        try:
            nbytes = obj.untyped_storage().nbytes()
        except Exception:
            nbytes = obj.numel() * obj.element_size()
        if nbytes > best_bytes:
            best, best_bytes = obj, nbytes
    return best


def _build_dataset(vendor_root: Path, dataset_root: Path, task: str, work_dir: Path):
    """Instantiate only ``cfg.data.train``, the same way runtime.build_datasets does.

    ``run_training`` calls ``misc.register_work_dir(cfg.output_dir)`` before building the
    datasets, and ``RobotVideoDataset.__init__`` writes ``dataset_stats.json`` under that
    directory after its normalization pass. Skipping the registration leaves the default
    ``./runs``, which does not exist, and instantiation dies with FileNotFoundError only
    after a full 1712-episode pass has already been paid for.
    """

    sys.path.insert(0, str(vendor_root / "FastWAM" / "src"))
    from fastwam.utils import misc
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    work_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(work_dir)

    dirs = ",".join(str(dataset_root / f"{suite}_no_noops_lerobot") for suite in SUITES)
    text_cache = dataset_root.parent.parent / "text_embeds_cache/libero"
    overrides = [
        f"task={task}",
        f"data.train.dataset_dirs=[{dirs}]",
        f"data.train.text_embedding_cache_dir={text_cache}",
    ]
    config_dir = str(vendor_root / "FastWAM" / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    return instantiate(cfg.data.train)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vendor-root", default=os.environ.get("W0_VENDOR_ROOT", "/opt/robot-practice/w0-vendor"))
    parser.add_argument(
        "--dataset-root",
        default="pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2",
        help="directory holding the four <suite>_no_noops_lerobot datasets",
    )
    parser.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    parser.add_argument(
        "--work-dir",
        default="outputs/literature/w0_fastwam/dataset_memory_probe",
        help="scratch directory for the dataset_stats.json the dataset writes at init",
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument(
        "--stage",
        choices=("full", "lerobot", "text"),
        default="full",
        help="which layer to exercise, to attribute the growth",
    )
    parser.add_argument(
        "--video-backend",
        default=None,
        choices=(None, "pyav", "torchcodec"),
        help=(
            "force the decode backend on every underlying LeRobotDataset. lerobot picks it "
            "with importlib.util.find_spec('torchcodec'), which succeeds here because the "
            "Python package is installed even though its shared library cannot load "
            "(it wants FFmpeg 4-7 and the image ships FFmpeg 8). Every decode therefore "
            "raises inside decode_video_frames_torchcodec, and the traceback keeps that "
            "frame -- plus the pyav objects and decoded tensors beside it -- alive in a "
            "reference cycle. Passing pyav skips the doomed attempt entirely. "
            "BaseLerobotDataset does not forward the argument, so it is set after "
            "construction."
        ),
    )
    parser.add_argument(
        "--cycles",
        action="store_true",
        help=(
            "name the reference cycle instead of measuring growth: iterate without "
            "collecting, then use gc.DEBUG_SAVEALL to enumerate the objects that were only "
            "reachable through cycles, and report who refers to the large tensors among them"
        ),
    )
    parser.add_argument(
        "--shapes",
        action="store_true",
        help="print a histogram of large live tensor shapes at each report",
    )
    args = parser.parse_args(argv)

    vendor_root = Path(args.vendor_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not (vendor_root / "FastWAM" / "configs").is_dir():
        parser.error(f"no FastWAM/configs under --vendor-root={vendor_root}")
    if not dataset_root.is_dir():
        parser.error(f"--dataset-root does not exist: {dataset_root}")

    print(f"[probe] vendor={vendor_root}", flush=True)
    print(f"[probe] dataset={dataset_root}", flush=True)
    print(f"[probe] stage={args.stage} samples={args.samples}", flush=True)
    print(f"[probe] work_dir={Path(args.work_dir).expanduser().resolve()}", flush=True)

    dataset = _build_dataset(
        vendor_root, dataset_root, args.task, Path(args.work_dir).expanduser().resolve()
    )
    inner = getattr(dataset, "lerobot_dataset", None)

    if args.video_backend is not None:
        # Walk down to the concrete LeRobotDataset objects; only they consult video_backend.
        targets = []
        multi = getattr(inner, "multi_dataset", None)
        for candidate in getattr(multi, "_datasets", []) or []:
            if hasattr(candidate, "video_backend"):
                targets.append(candidate)
        if not targets:
            parser.error(
                "--video-backend could not find any LeRobotDataset under "
                "dataset.lerobot_dataset.multi_dataset._datasets"
            )
        was = sorted({getattr(t, "video_backend", "?") for t in targets})
        for target in targets:
            target.video_backend = args.video_backend
        print(
            f"[probe] video_backend: {was} -> {args.video_backend} "
            f"on {len(targets)} datasets",
            flush=True,
        )
    if args.stage == "lerobot" and inner is None:
        parser.error("--stage=lerobot needs dataset.lerobot_dataset, which is missing")

    import numpy as np

    size = len(dataset)
    print(f"[probe] dataset length={size}", flush=True)
    baseline = _rss_kb()
    print(f"[probe] baseline RssAnon={baseline.get('RssAnon', 0) / 1048576:.2f} GiB", flush=True)

    # Count how often the dataset falls into its own retry path. The retained group contains
    # intermediate tensors, which only a live frame explains, and __getitem__ is the one place
    # that keeps going after an exception -- so the retry rate should match the leak rate.
    retries = {"n": 0}
    original_get = type(dataset)._get

    def counting_get(self, idx):  # noqa: ANN001 - mirrors the wrapped signature
        try:
            return original_get(self, idx)
        except Exception:
            retries["n"] += 1
            raise

    type(dataset)._get = counting_get

    rng = np.random.default_rng(42)

    if args.cycles:
        # The growth is uncollected cycle garbage, not a leak: an explicit gc.collect() frees
        # every retained tensor and returns the memory, while without it a single process
        # climbs about 29 MiB per sample. So the useful question is no longer "how much" but
        # "which cycle", and DEBUG_SAVEALL answers it exactly: everything it parks in
        # gc.garbage was reachable only through a cycle.
        for i in range(args.samples):
            idx = int(rng.integers(0, size))
            item = dataset[idx] if args.stage == "full" else inner[idx]
            del item
        import torch

        # No pre-collect: it would free exactly the cycles we are trying to name. Enable
        # SAVEALL only after the iteration, so the single collect below parks the cycles that
        # were still pending at that moment -- which is the backlog that grows RSS.
        gc.set_debug(gc.DEBUG_SAVEALL)
        found = gc.collect()
        garbage = list(gc.garbage)
        gc.garbage.clear()
        gc.set_debug(0)

        kinds: collections.Counter = collections.Counter()
        tensors_in_cycles = []
        for obj in garbage:
            kinds[type(obj).__name__] += 1
            try:
                if torch.is_tensor(obj) and obj.untyped_storage().nbytes() >= (1 << 20):
                    tensors_in_cycles.append(obj)
            except Exception:  # pragma: no cover
                continue
        print(
            f"[probe] cycles: gc collected {found} objects after {args.samples} samples; "
            f"{len(tensors_in_cycles)} of them are large tensors",
            flush=True,
        )
        print(f"[probe] cycle members: {kinds.most_common(12)}", flush=True)
        for tensor in tensors_in_cycles[:3]:
            print(
                f"[probe]   {tuple(tensor.shape)} {tensor.dtype} <- {_name_referrers(tensor, 8)}",
                flush=True,
            )
        # Frames are the interesting members: a frame in a cycle keeps every local alive,
        # which is what drags the intermediate tensors along with the final one.
        frames = [obj for obj in garbage if type(obj).__name__ == "frame"]
        for frame in frames[:6]:
            print(
                f"[probe]   frame {frame.f_code.co_name} "
                f"({frame.f_code.co_filename.split('/')[-1]}:{frame.f_lineno})",
                flush=True,
            )
        return 0

    first_anon: int | None = None
    for i in range(args.samples):
        idx = int(rng.integers(0, size))
        if args.stage == "full":
            item = dataset[idx]
        elif args.stage == "lerobot":
            item = inner[idx]
        else:
            item = dataset._get_cached_text_context("probe prompt")
        del item

        if i % args.report_every == 0 or i == args.samples - 1:
            mem = _rss_kb()
            anon = mem.get("RssAnon", 0)
            if first_anon is None:
                first_anon = anon
            before, _, _ = _tensor_census()
            collected = gc.collect()
            count, tensor_bytes, shapes = _tensor_census()
            per_sample = (anon - first_anon) / max(i, 1) / 1024
            print(
                f"[probe] sample={i:5d} RssAnon={anon / 1048576:7.2f} GiB "
                f"shmem={mem.get('RssShmem', 0) / 1048576:5.2f} GiB "
                f"tensors={count:7d} (gc freed {before - count:5d}) "
                f"tensor_bytes={tensor_bytes / 1073741824:7.2f} GiB "
                f"growth={per_sample:7.1f} MiB/sample",
                flush=True,
            )
            if args.shapes and shapes:
                top = ", ".join(f"{shape}x{n}" for shape, n in shapes.most_common(4))
                print(f"[probe]        large tensors: {top}", flush=True)
                biggest = _largest_retained_tensor()
                if biggest is not None:
                    print(
                        f"[probe]        held by: {tuple(biggest.shape)} <- "
                        f"{_name_referrers(biggest)}",
                        flush=True,
                    )
                del biggest

    type(dataset)._get = original_get
    mem = _rss_kb()
    anon = mem.get("RssAnon", 0)
    grew = (anon - (first_anon or anon)) / 1048576
    print(
        f"[probe] _get raised on {retries['n']} of {args.samples} samples "
        f"({100.0 * retries['n'] / max(args.samples, 1):.1f}%)",
        flush=True,
    )
    print(
        f"[probe] done: RssAnon {(first_anon or 0) / 1048576:.2f} -> {anon / 1048576:.2f} GiB "
        f"(+{grew:.2f} GiB over {args.samples} samples)",
        flush=True,
    )
    # A flat curve here means the leak is not in the dataset at all and the next suspect is
    # the DataLoader transport or the trainer, so say so explicitly rather than leaving the
    # reader to infer it from two numbers.
    if grew < 1.0:
        print("[probe] verdict: dataset iteration does NOT leak; look at the transport", flush=True)
    else:
        print(f"[probe] verdict: leaks {grew * 1024 / args.samples:.1f} MiB/sample", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
