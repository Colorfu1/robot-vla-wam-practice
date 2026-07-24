"""Windowed terminal and JSONL logging for ACT-delta training losses."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

LOSS_LOG_PATH_ENV = "ROBOT_PRACTICE_A2_LOSS_LOG_PATH"
LOSS_LOG_FREQ_ENV = "ROBOT_PRACTICE_A2_LOSS_LOG_FREQ"


def _is_main_process() -> bool:
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return rank in {"", "0"}


@dataclass
class LossComponentRecorder:
    """Accumulate scalar loss components and emit one average per fixed window."""

    output_path: Path
    log_freq: int
    logger: logging.Logger
    step: int = 0
    window_count: int = 0
    sums: dict[str, float] = field(
        default_factory=lambda: {
            "l1_loss": 0.0,
            "delta_action_loss": 0.0,
            "weighted_delta_action_loss": 0.0,
            "total_loss": 0.0,
        }
    )

    @classmethod
    def from_environment(
        cls,
        logger: logging.Logger,
    ) -> LossComponentRecorder | None:
        path_value = os.environ.get(LOSS_LOG_PATH_ENV)
        frequency_value = os.environ.get(LOSS_LOG_FREQ_ENV)
        if path_value is None and frequency_value is None:
            return None
        if not path_value or not frequency_value:
            raise ValueError(
                f"{LOSS_LOG_PATH_ENV} and {LOSS_LOG_FREQ_ENV} must be set together"
            )
        try:
            log_freq = int(frequency_value)
        except ValueError as exc:
            raise ValueError(f"{LOSS_LOG_FREQ_ENV} must be a positive integer") from exc
        if log_freq <= 0 or str(log_freq) != frequency_value:
            raise ValueError(f"{LOSS_LOG_FREQ_ENV} must be a canonical positive integer")
        if not _is_main_process():
            return None
        return cls(output_path=Path(path_value), log_freq=log_freq, logger=logger)

    def update(
        self,
        *,
        l1_loss: float,
        delta_action_loss: float,
        delta_loss_weight: float,
        total_loss: float,
    ) -> None:
        self.step += 1
        self.window_count += 1
        self.sums["l1_loss"] += l1_loss
        self.sums["delta_action_loss"] += delta_action_loss
        self.sums["weighted_delta_action_loss"] += (
            delta_loss_weight * delta_action_loss
        )
        self.sums["total_loss"] += total_loss
        if self.window_count != self.log_freq:
            return

        record: dict[str, int | float] = {
            "step": self.step,
            "window_size": self.window_count,
            **{
                name: value / self.window_count
                for name, value in self.sums.items()
            },
        }
        self.logger.info(
            "[A2 loss components] step=%d window=%d "
            "l1_loss=%.6f delta_action_loss=%.6f "
            "weighted_delta_action_loss=%.6f total_loss=%.6f",
            record["step"],
            record["window_size"],
            record["l1_loss"],
            record["delta_action_loss"],
            record["weighted_delta_action_loss"],
            record["total_loss"],
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, sort_keys=True))
            stream.write("\n")
        self.window_count = 0
        for name in self.sums:
            self.sums[name] = 0.0
