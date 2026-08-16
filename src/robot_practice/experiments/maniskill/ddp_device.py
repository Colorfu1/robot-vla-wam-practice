"""Bind ManiSkill simulation and rendering backends to a DDP local rank."""

from __future__ import annotations


def rank_sim_backend(sim_backend: str, local_rank: int) -> str:
    """Return a backend string whose CUDA device matches ``local_rank``.

    ManiSkill accepts an explicit device suffix (for example,
    ``physx_cuda:2``).  Without it, a GPU environment defaults to CUDA device
    0 even when the PyTorch DDP process has already selected another device.
    CPU backends are intentionally left unchanged.
    """

    backend = str(sim_backend)
    base = backend.split(":", 1)[0]
    if base in {"physx_cuda", "cuda", "gpu"}:
        return f"physx_cuda:{int(local_rank)}"
    return backend


def rank_render_backend(sim_backend: str, local_rank: int) -> str | None:
    """Return the matching explicit RGB renderer backend, when applicable."""

    backend = str(sim_backend).split(":", 1)[0]
    if backend in {"physx_cuda", "cuda", "gpu"}:
        return f"sapien_cuda:{int(local_rank)}"
    if backend in {"physx_cpu", "cpu"}:
        return "sapien_cpu"
    return None


__all__ = ["rank_render_backend", "rank_sim_backend"]
