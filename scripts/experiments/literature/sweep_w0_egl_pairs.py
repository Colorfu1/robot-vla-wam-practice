#!/usr/bin/env python3
"""Measure which (CUDA device, EGL index) pairs can render, as a matrix.

Why this exists
---------------
``MUJOCO_EGL_DEVICE_ID`` is a position in ``eglQueryDevicesEXT()``, not a CUDA device id, and
picking the wrong one makes rendering abort natively inside ``mjr_readPixels`` where no Python
handler can intervene. Two attempts to infer the correct pairing have now failed:

* ``EGL_CUDA_DEVICE_NV`` would answer directly, but returns ``EGLError`` for every index on
  driver 535.261.03.
* Joining ``nvidia-smi`` PCI ids with ``/sys/class/drm/card*/device`` and the EGL device list
  produces ``{0:6, 1:7, 2:4, 3:5, 4:2, 5:3, 6:0, 7:1}`` on an 8x H20 node, and
  ``(cuda 0, egl 6)`` then aborts immediately -- so either the join is wrong or exposing two
  devices in ``CUDA_VISIBLE_DEVICES`` (needed to satisfy robosuite's assertion when the render
  index differs from the compute id) changes the outcome.

Inference has therefore been retired in favour of measurement. This sweep runs one isolated
child per cell, because a bad cell aborts natively and would otherwise take the whole sweep
down with it.

Reading the matrix
-----------------
Every cell is ``PASS`` or ``FAIL(reason)``. What the shape means:

* Only the diagonal passes -> identity is correct on this host.
* Everything passes -> the pairing does not matter for a short run, and a longer ``--steps``
  is needed: cross-GPU rendering mostly works and only aborts after a while. One evaluation
  died at about the 141st cumulative render after a 120-step probe had accepted the pair, so
  a 120-step PASS is weak evidence.
* Exactly one cell fails -> that single cell is not explained by a systematic index offset,
  and must not be treated as one.

Usage
-----
    scripts/experiments/literature/sweep_w0_egl_pairs.py --cuda 0 --steps 120
    scripts/experiments/literature/sweep_w0_egl_pairs.py --steps 400 --output outputs/.../sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config",
        default="configs/experiments/literature/w0_fastwam_reproduction.json",
        help="W0 config, used only to build the same runtime env the evaluator uses",
    )
    parser.add_argument("--vendor-root", default=os.environ.get("W0_VENDOR_ROOT", "/opt/robot-practice/w0-vendor"))
    parser.add_argument(
        "--cuda",
        default=None,
        help="comma-separated CUDA devices to sweep (default: all visible)",
    )
    parser.add_argument(
        "--egl",
        default=None,
        help="comma-separated EGL indices to try (default: 0..egl_device_count-1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "render steps per cell. A PASS only means 'did not abort within this many renders'; "
            "an evaluation once aborted at ~141 after a 120-step probe passed, so prefer 400+ "
            "when the answer matters"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help=(
            "episodes per cell. The evaluator reuses one environment across trials, so >1 "
            "exercises the inter-episode reset -- which is where the one observed delayed abort "
            "happened (episode 2, step 22, after a hard reset)"
        ),
    )
    parser.add_argument("--suite", default=None, help="LIBERO suite (default: the probe's own)")
    parser.add_argument("--task-id", type=int, default=None, help="LIBERO task id")
    parser.add_argument("--output", default=None, help="write the matrix to this JSON file")
    args = parser.parse_args(argv)

    from robot_practice.experiments.literature.w0_fastwam_audit import (  # noqa: E402
        EGL_PROBE_STEPS,
        _artifact_paths,
        _model_runtime_env,
        _probe_render_pair,
        _read_json,
        _render_fingerprint,
        _rooted,
        _visible_gpu_ids,
    )

    config = _read_json(_rooted(args.config))
    paths = _artifact_paths(config)
    runtime_env = _model_runtime_env(paths, vendor_root=_rooted(args.vendor_root))
    fingerprint = _render_fingerprint(runtime_env)
    steps = args.steps or EGL_PROBE_STEPS

    if args.cuda:
        cuda_ids = [value.strip() for value in args.cuda.split(",") if value.strip()]
    else:
        cuda_ids = _visible_gpu_ids(int(fingerprint.get("gpu_count") or 1))
    if args.egl:
        egl_ids = [value.strip() for value in args.egl.split(",") if value.strip()]
    else:
        egl_ids = [str(index) for index in range(int(fingerprint.get("egl_device_count") or len(cuda_ids)))]

    print(f"[SWEEP] fingerprint={json.dumps(fingerprint)}", flush=True)
    print(
        f"[SWEEP] cuda={cuda_ids} egl={egl_ids} steps={steps} trials={args.trials} "
        f"suite={args.suite or '<probe default>'} task_id={args.task_id}",
        flush=True,
    )
    print(
        "[SWEEP] a PASS means 'no abort within "
        f"{steps} renders', which is not proof the pair is correct",
        flush=True,
    )

    matrix: dict[str, dict[str, bool]] = {}
    for cuda in cuda_ids:
        row: dict[str, bool] = {}
        for egl in egl_ids:
            ok = _probe_render_pair(
                cuda_device=cuda,
                egl_index=egl,
                runtime_env=runtime_env,
                steps=steps,
                trials=args.trials,
                suite=args.suite,
                task_id=args.task_id,
            )
            row[egl] = ok
            print(f"[SWEEP] cuda={cuda} egl={egl} {'PASS' if ok else 'FAIL'}", flush=True)
        matrix[cuda] = row
        passing = [egl for egl, ok in row.items() if ok]
        print(f"[SWEEP] cuda={cuda} passing={passing}", flush=True)

    header = "cuda\\egl " + " ".join(f"{egl:>3}" for egl in egl_ids)
    print(f"\n[SWEEP] {header}", flush=True)
    for cuda, row in matrix.items():
        cells = " ".join(f"{'  .' if row[egl] else '  X'}" for egl in egl_ids)
        print(f"[SWEEP] {cuda:>8} {cells}", flush=True)
    print("[SWEEP] '.' = no abort, 'X' = aborted", flush=True)

    only_diagonal = all(
        row.get(cuda, False) and not any(ok for egl, ok in row.items() if egl != cuda)
        for cuda, row in matrix.items()
    )
    all_pass = all(all(row.values()) for row in matrix.values())
    failures = sum(1 for row in matrix.values() for ok in row.values() if not ok)
    if only_diagonal:
        verdict = "identity is the only working map on this host"
    elif all_pass:
        verdict = (
            f"every pair survived {steps} renders; raise --steps, because cross-GPU rendering "
            "mostly works and aborts only after a while"
        )
    elif failures == 1:
        verdict = (
            "exactly one cell failed, which no systematic index offset explains; do not "
            "generalise from it"
        )
    else:
        verdict = "mixed; read the matrix rather than assuming a formula"
    print(f"[SWEEP] verdict: {verdict}", flush=True)

    # Emit a ready-to-use map. Assignment is a bijection: two workers pointed at the same EGL
    # index would share one render device, which is a different failure to debug later.
    chosen: dict[str, str] = {}
    taken: set[str] = set()
    unassigned: list[str] = []
    for cuda, row in matrix.items():
        passing = [egl for egl in egl_ids if row.get(egl)]
        if cuda in passing and cuda not in taken:
            pick = cuda  # identity when it works, so the map stays easy to read
        else:
            pick = next((egl for egl in passing if egl not in taken), None)
        if pick is None:
            unassigned.append(cuda)
            continue
        chosen[cuda] = pick
        taken.add(pick)

    if unassigned:
        print(
            f"[SWEEP] no free passing EGL index for CUDA {unassigned}; "
            "the matrix has fewer usable devices than requested GPUs",
            flush=True,
        )
    else:
        spec = ",".join(f"{cuda}:{chosen[cuda]}" for cuda in cuda_ids if cuda in chosen)
        print(f"[SWEEP] W0_EGL_DEVICE_MAP={spec}", flush=True)
        print(
            "[SWEEP] pass it explicitly to skip inference entirely: "
            f"W0_EGL_DEVICE_MAP={spec} bash scripts/experiments/literature/run_w0_fastwam.sh ...",
            flush=True,
        )

    if args.output:
        target = _rooted(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "render_fingerprint": fingerprint,
                    "steps_per_cell": steps,
                    "matrix": matrix,
                    "verdict": verdict,
                    "device_map": chosen,
                    "unassigned": unassigned,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[SWEEP] wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
