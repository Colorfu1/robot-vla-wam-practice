"""Metric definitions and aggregation helpers for robot policy experiments."""

REQUIRED_EXPERIMENT_FIELDS = [
    "task suite",
    "dataset size",
    "observation keys",
    "action space",
    "action horizon",
    "control frequency",
    "train loss",
    "open-loop action error",
    "closed-loop success rate",
    "latency",
    "failure cases",
]

