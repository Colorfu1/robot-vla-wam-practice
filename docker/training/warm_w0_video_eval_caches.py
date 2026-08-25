"""Bake the weights the W0 video-generation scoring path downloads on first use.

The remote training host has no outbound network, so anything fetched lazily at runtime is
a hard failure there rather than a slow first run.  Three such fetches were measured on the
base image, and pinned VBench offers a local-weight path for none of them:

  torchvision AlexNet   ``lpips.LPIPS(net="alex")`` builds ``alexnet(pretrained=True)``, so
                        the 244 MB backbone is pulled even though the guide already supplies
                        the LPIPS linear weights separately.  W0-Video-40 LPIPS depends on it.
  co-tracker2           ``camera_motion`` calls ``torch.hub.load("facebookresearch/co-tracker",
                        "cotracker2")``.  ``init_submodules`` has no ``local`` branch for this
                        dimension, so both the repo zipball and the 195 MB checkpoint are
                        always fetched.
  dreamsim ensemble     ``i2v_background`` calls ``dreamsim(pretrained=True)``.  Its
                        ``init_submodules`` entry carries only ``resolution``, so there is no
                        local path either, and the ensemble pulls about 3.2 GB.

The first two land in ``$TORCH_HOME/hub``, an absolute path, so baking them needs nothing at
run time.  dreamsim is different: its signature is
``dreamsim(pretrained=True, device="cuda", cache_dir="./models", ...)`` and VBench passes no
cache_dir, so it resolves **relative to the process working directory**.  Baking it therefore
only helps if the run exposes it as ``./models``; see docker/training/README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def silence_torch_hub_progress() -> None:
    """Replace torch.hub's tqdm so a build log is not thousands of progress lines."""
    import torch.hub

    class _Quiet:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.total = kwargs.get("total")

        def __enter__(self) -> "_Quiet":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def update(self, *args: object, **kwargs: object) -> None:
            return None

        def close(self) -> None:
            return None

    torch.hub.tqdm = _Quiet  # type: ignore[attr-defined]


def inventory(root: Path, label: str) -> int:
    if not root.exists():
        print(f"{label}: MISSING ({root})", flush=True)
        return 0
    total = 0
    entries = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            size = item.stat().st_size
            total += size
            if size > 5_000_000:
                entries.append(f"    {size / 1e6:9.1f} MB  {item.relative_to(root)}")
    print(f"{label}: {total / 1e6:.1f} MB under {root}", flush=True)
    for line in entries:
        print(line, flush=True)
    return total


def warm_torch_hub() -> None:
    import torch
    import torchvision

    silence_torch_hub_progress()
    hub_root = Path(torch.hub.get_dir())

    # LPIPS backbone.  progress=False keeps the build log clean.
    torchvision.models.alexnet(weights="DEFAULT", progress=False)
    alexnet = hub_root / "checkpoints/alexnet-owt-7be5be79.pth"
    if not alexnet.is_file():
        sys.exit(f"expected AlexNet weights at {alexnet}")

    # Constructing the real evaluator proves lpips finds both the backbone and its own
    # bundled linear weights, which is what the score path actually does.
    import lpips

    lpips.LPIPS(net="alex", verbose=False)
    print("lpips.LPIPS(net='alex') constructed from cache", flush=True)

    # camera_motion's tracker.  Built on CPU here; the build host has no GPU.
    tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2", trust_repo=True)
    print(f"co-tracker2 constructed: {type(tracker).__name__}", flush=True)
    cotracker = hub_root / "checkpoints/cotracker2.pth"
    if not cotracker.is_file():
        sys.exit(f"expected co-tracker2 weights at {cotracker}")

    inventory(hub_root, "torch hub cache")


def warm_dreamsim(target: Path) -> None:
    silence_torch_hub_progress()
    from dreamsim import dreamsim

    target.mkdir(parents=True, exist_ok=True)
    model, _preprocess = dreamsim(pretrained=True, device="cpu", cache_dir=str(target))
    print(f"dreamsim ensemble constructed: {type(model).__name__}", flush=True)

    # dreamsim reads back exactly what it wrote, so assert the layout rather than a file list
    # that its next release could rename.
    if not any(target.rglob("*.pth")) and not any(target.rglob("*.safetensors")):
        sys.exit(f"dreamsim cache at {target} contains no weight files")
    inventory(target, "dreamsim cache")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("torch-hub", "dreamsim"), required=True)
    parser.add_argument(
        "--dreamsim-target",
        default=os.environ.get("W0_DREAMSIM_CACHE", "/opt/robot-practice/w0-video-eval-cache/models"),
        help="absolute directory to bake the dreamsim ensemble into",
    )
    args = parser.parse_args()

    if args.stage == "torch-hub":
        warm_torch_hub()
    else:
        warm_dreamsim(Path(args.dreamsim_target))


if __name__ == "__main__":
    main()
