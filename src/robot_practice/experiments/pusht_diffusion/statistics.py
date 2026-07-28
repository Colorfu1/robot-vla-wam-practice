"""Confidence intervals for PushT Pilot summaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
        }


def wilson_interval(successes: int, trials: int, *, confidence: float = 0.95) -> Interval:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0,trials]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    proportion = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    radius = (
        z * np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)) / denominator
    )
    return Interval(
        estimate=float(proportion),
        lower=float(max(0.0, center - radius)),
        upper=float(min(1.0, center + radius)),
    )


def environment_cluster_bootstrap(
    groups: Mapping[int, Sequence[float]],
    *,
    statistic: Callable[[np.ndarray], float] = lambda values: float(np.mean(values)),
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260727,
) -> Interval:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    ordered_keys = sorted(groups)
    if not ordered_keys:
        raise ValueError("At least one environment group is required")
    arrays = {key: np.asarray(groups[key], dtype=np.float64).reshape(-1) for key in ordered_keys}
    if any(values.size == 0 or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Every environment group must contain finite values")
    observed = statistic(np.concatenate([arrays[key] for key in ordered_keys]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.choice(ordered_keys, size=len(ordered_keys), replace=True)
        estimates[iteration] = statistic(np.concatenate([arrays[int(key)] for key in sampled]))
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        estimate=float(observed),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
    )
