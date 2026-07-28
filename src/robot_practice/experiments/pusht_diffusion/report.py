"""Assemble E0/E1/E2/E4 outputs into one measured PushT Pilot report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import rerun as rr

from .config import default_output_root, practice_root
from .output import AtomicOutputDirectory, write_metrics
from .provenance import collect_runtime_provenance, write_json_atomic


def _read_json(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment output is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _validate_label(root: Path, expected: str) -> dict[str, Any]:
    manifest = _read_json(root, "run_manifest.json")
    if manifest.get("label") != expected:
        raise ValueError(
            f"{root} has label={manifest.get('label')!r}, expected {expected!r}; "
            "Smoke outputs cannot be promoted to Pilot evidence"
        )
    return manifest


def _value(metrics: Mapping[str, Any], key: str) -> Any:
    if key not in metrics:
        raise KeyError(f"Required metric {key!r} is missing")
    return metrics[key]


def _write_rerun(path: Path, markdown: str) -> None:
    recording = rr.RecordingStream(
        "robot_practice_pusht_pilot_report",
        recording_id="pusht_pilot_summary",
    )
    recording.save(path)
    try:
        recording.log("report/summary", rr.TextDocument(markdown, media_type=rr.MediaType.MARKDOWN))
    finally:
        recording.disconnect()


def make_report(
    *,
    e0_root: Path,
    e1_root: Path,
    e2_root: Path,
    e4_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    _validate_label(e0_root, "Offline")
    _validate_label(e1_root, "Offline")
    _validate_label(e2_root, "Closed-loop")
    _validate_label(e4_root, "Closed-loop")
    e0 = _read_json(e0_root, "metrics.json")
    e1 = _read_json(e1_root, "metrics.json")
    e2 = _read_json(e2_root, "metrics.json")
    e4 = _read_json(e4_root, "metrics.json")
    mode_analysis = _read_json(e2_root, "mode_analysis.json")
    condition_metrics = _read_json(e4_root, "condition_metrics.json")
    primary_counts = mode_analysis.get("primary_mode_counts", {})
    multimode_seeds = sorted(int(seed) for seed, count in primary_counts.items() if int(count) >= 2)
    action_diversity = float(_value(e1, "mean_denormalized_trajectory_pairwise_rms"))
    e1_answer = (
        "Different initial noise produced measurable action-pattern differences."
        if action_diversity > 1e-6
        else "No measurable action-pattern difference was observed at the configured tolerance."
    )
    e2_answer = (
        f"Multiple successful trajectory modes were detected for seeds {multimode_seeds}."
        if multimode_seeds
        else "No initial seed produced two successful modes at the pre-registered threshold."
    )
    e3_answer = (
        f"E3 is worth a separate review; start with seed {multimode_seeds[0]}."
        if multimode_seeds
        else "E3 is not justified by the current primary mode criterion."
    )
    best_condition = e4.get("best_qualifying_condition")
    lines = [
        "# PushT Diffusion Pilot Report",
        "",
        "## Evidence labels",
        "",
        "- E0-lite: Offline qualitative inspection.",
        "- E1: Offline fixed-observation sampling; no actions were executed.",
        "- E2: Closed-loop repeated rollouts.",
        "- E4: Closed-loop paired scheduler/step benchmark.",
        "",
        "## Measured summary",
        "",
        f"- E0 demonstrations inspected: {_value(e0, 'num_demonstrations')}.",
        f"- E1 mean action-trajectory diversity: {action_diversity:.6g} pixels.",
        f"- E2 BCChunk success: {float(_value(e2, 'bc_success_rate')):.3f}.",
        (
            f"- E2 Diffusion single-sample success: "
            f"{float(_value(e2, 'single_sample_success')):.3f}."
        ),
        (
            f"- E2 success-any@5: {float(_value(e2, 'success_any_at_5')):.3f} "
            "(oracle coverage diagnostic)."
        ),
        f"- E4 DDPM-100 success: {float(_value(e4, 'reference_success_rate')):.3f}.",
        (
            f"- E4 DDPM-100 p95 end-to-end chunk latency: "
            f"{float(_value(e4, 'reference_end_to_end_p95_ms')):.3f} ms."
        ),
        (
            f"- Best condition meeting the pre-registered speed/success target: "
            f"{best_condition or 'none'}."
        ),
        "",
        "## Required questions",
        "",
        f"1. Same-observation action patterns: {e1_answer}",
        (
            "2. Structured versus invalid noise: E1 alone cannot decide this; use E2 "
            "success-conditioned trajectory evidence."
        ),
        f"3. Multiple successful rollouts from one initial state: {e2_answer}",
        (
            "4. BCChunk mode averaging: not established automatically; inspect the E1 "
            "overlay and E2 failure cases qualitatively."
        ),
        (
            "5. Whether diversity improves ordinary single-sample success: not causally "
            "identifiable from any@5; compare fixed policy_seed=0 against BCChunk."
        ),
        f"6. Best success-latency balance: {best_condition or 'no condition met the target'}.",
        f"7. Fixed-middle-state E3 decision: {e3_answer}",
        "",
        "## Limitations",
        "",
        "- E0-lite does not prove strict conditional multimodality.",
        "- success-any@5 is not deployment success.",
        "- This Pilot does not restore dataset middle frames or serialize Pymunk state.",
    ]
    markdown = "\n".join(lines) + "\n"
    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        (temporary / "report.md").write_text(markdown, encoding="utf-8")
        aggregate = {
            "e1_action_trajectory_diversity": action_diversity,
            "e2_bc_success_rate": e2["bc_success_rate"],
            "e2_diffusion_single_sample_success": e2["single_sample_success"],
            "e2_success_any_at_5": e2["success_any_at_5"],
            "e2_multimode_environment_count": len(multimode_seeds),
            "e4_reference_success_rate": e4["reference_success_rate"],
            "e4_reference_end_to_end_p95_ms": e4["reference_end_to_end_p95_ms"],
            "e4_best_qualifying_condition": best_condition,
        }
        write_metrics(
            temporary,
            aggregate,
            comments={
                "e1_action_trajectory_diversity": "Offline fixed-input action diversity",
                "e2_bc_success_rate": "Deterministic BCChunk closed-loop success",
                "e2_diffusion_single_sample_success": "Diffusion policy_seed=0 deployment metric",
                "e2_success_any_at_5": "Oracle repeated-sampling coverage",
                "e2_multimode_environment_count": "Primary threshold 0.05",
                "e4_reference_success_rate": "DDPM-100 paired final seeds",
                "e4_reference_end_to_end_p95_ms": "Raw observation to action chunk",
                "e4_best_qualifying_condition": "2x speed and <=5-point success drop",
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "label": "Formal",
                "experiment": "PushT Diffusion Pilot report",
                "sources": {
                    "e0": str(e0_root),
                    "e1": str(e1_root),
                    "e2": str(e2_root),
                    "e4": str(e4_root),
                },
                "condition_count": len(condition_metrics.get("conditions", [])),
                "runtime": collect_runtime_provenance(practice_root()),
            },
        )
        _write_rerun(temporary / "recording.rrd", markdown)
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0", type=Path, required=True)
    parser.add_argument("--e1", type=Path, required=True)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--e4", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "pilot" / "report",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = make_report(
        e0_root=args.e0.expanduser().resolve(),
        e1_root=args.e1.expanduser().resolve(),
        e2_root=args.e2.expanduser().resolve(),
        e4_root=args.e4.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(f"Pilot report: {result}")


if __name__ == "__main__":
    main()
