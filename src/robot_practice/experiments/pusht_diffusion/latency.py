"""Synchronized chunk-latency measurement utilities."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class LatencySummary:
    iterations: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    peak_memory_bytes: int | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "iterations": self.iterations,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_ms": self.std_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(
    operation: Callable[[], object],
    *,
    device: torch.device | str,
    warmup_iterations: int = 50,
    measurement_iterations: int = 1000,
) -> tuple[LatencySummary, np.ndarray]:
    resolved_device = torch.device(device)
    if warmup_iterations < 0 or measurement_iterations <= 0:
        raise ValueError("Latency iteration counts are invalid")
    for _ in range(warmup_iterations):
        operation()
    _synchronize(resolved_device)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    measurements = np.empty(measurement_iterations, dtype=np.float64)
    for index in range(measurement_iterations):
        _synchronize(resolved_device)
        started = time.perf_counter_ns()
        operation()
        _synchronize(resolved_device)
        measurements[index] = (time.perf_counter_ns() - started) / 1_000_000.0

    peak_memory = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else None
    )
    return (
        LatencySummary(
            iterations=measurement_iterations,
            mean_ms=float(np.mean(measurements)),
            p50_ms=float(np.percentile(measurements, 50)),
            p90_ms=float(np.percentile(measurements, 90)),
            p95_ms=float(np.percentile(measurements, 95)),
            p99_ms=float(np.percentile(measurements, 99)),
            std_ms=float(np.std(measurements)),
            peak_memory_bytes=peak_memory,
        ),
        measurements,
    )
