#!/usr/bin/env python
"""Compare the initial states the two W0 evaluators feed to LIBERO.

The official evaluator calls ``task_suite.get_task_init_states(task_id)``.
LeRobot reimplements that loader in ``lerobot.envs.libero.get_task_init_states``
so it can pass ``weights_only=False`` for PyTorch 2.6+ pickles.  Both then index
the array by trial, so any difference in content or ordering silently changes
which scenes are evaluated -- and a per-task success gap would then say nothing
about the policy.

Run inside the LeRobot environment, where both packages are importable.  Reads
only; no rollout, no GPU, no model.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites",
        default=",".join(SUITES),
        help="comma-separated suites to compare (default: the four W0 suites)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="how many leading trials to compare per task",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from libero.libero import benchmark
    from lerobot.envs.libero import get_task_init_states as lerobot_init_states

    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    mismatches: list[str] = []
    for suite_name in suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in range(len(suite.tasks)):
            official = np.asarray(suite.get_task_init_states(task_id))
            ported = np.asarray(lerobot_init_states(suite, task_id))
            label = f"{suite_name}/task{task_id:02d}"
            if official.shape != ported.shape:
                mismatches.append(label)
                print(
                    f"[INIT] {label:28s} SHAPE MISMATCH official={official.shape} "
                    f"lerobot={ported.shape}",
                    flush=True,
                )
                continue
            limit = min(args.trials, official.shape[0])
            delta = float(np.abs(official[:limit] - ported[:limit]).max())
            status = "same" if delta == 0.0 else "DIFFERS"
            if delta != 0.0:
                mismatches.append(label)
            print(
                f"[INIT] {label:28s} {status:8s} states={official.shape[0]} "
                f"compared={limit} max_abs_delta={delta:.3e}",
                flush=True,
            )

    print(
        f"[INIT] DONE suites={suites} mismatching_tasks={len(mismatches)}"
        + (f" -> {mismatches}" if mismatches else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
