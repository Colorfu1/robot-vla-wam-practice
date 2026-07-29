"""Assemble E0/E1/E2/E3 outputs into one measured PushT Pilot report."""

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


def _validate_e1_schema(root: Path, manifest: Mapping[str, Any]) -> None:
    expected = "pusht_e1_paired_sampling_v2"
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_schema") != expected
    ):
        raise ValueError(
            f"{root} is a legacy or incompatible E1 output. Rerun "
            "scripts/experiments/pusht/eval_fixed_observations.sh with --overwrite "
            f"to produce {expected}."
        )
    for filename in (
        "recording_initial_noise_only.rrd",
        "recording_full_sampling_seed.rrd",
    ):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"E1 schema version 2 recording is missing: {root / filename}")


def _validate_e2_schema(root: Path, manifest: Mapping[str, Any]) -> None:
    expected = "pusht-e2-closed-loop-v2"
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_schema") != expected
        or manifest.get("sampling_protocol") != "full-ddpm-sampling"
        or manifest.get("all_samples_closed_loop") is not True
        or manifest.get("environment_seeds") != list(range(2000, 2020))
        or manifest.get("policy_seeds") != list(range(10))
        or manifest.get("num_diffusion_rollouts") != 200
        or manifest.get("num_bc_rollouts") != 20
        or manifest.get("num_rollouts") != 220
    ):
        raise ValueError(
            f"{root} is not the exact 20-environment x 10-policy-seed E2b output. "
            "Rerun full-ddpm-sampling from scratch with the registered Closed-loop plan."
        )


def _validate_e3_schema(root: Path, manifest: Mapping[str, Any]) -> None:
    expected_conditions = [
        {"scheduler": "DDIM", "num_inference_steps": steps} for steps in (5, 10, 20, 50, 100)
    ] + [{"scheduler": "DDPM", "num_inference_steps": 100}]
    recordings = manifest.get("recordings")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("experiment_schema") != "pusht-e3-sampling-steps-v2"
        or manifest.get("experiment") != "E3"
        or manifest.get("conditions") != expected_conditions
        or manifest.get("environment_seeds") != list(range(2000, 2050))
        or manifest.get("num_rollouts") != 300
        or not isinstance(recordings, Mapping)
        or recordings.get("count") != 50
        or recordings.get("conditions_per_recording") != 6
        or recordings.get("all_rollouts_include_simulator_frames") is not True
    ):
        raise ValueError(
            f"{root} is a legacy or incompatible sampling-step output. Rerun "
            "scripts/experiments/pusht/benchmark_sampling_steps.sh to produce E3 evidence."
        )
    recordings_root = root / "recordings"
    expected_recordings = {
        recordings_root / f"env_{environment_seed:04d}.rrd"
        for environment_seed in range(2000, 2050)
    }
    observed_recordings = set(recordings_root.glob("env_*.rrd"))
    if observed_recordings != expected_recordings or any(
        not path.is_file() or path.stat().st_size == 0 for path in expected_recordings
    ):
        raise ValueError(f"{root} does not contain the exact 50 non-empty E3 scene Recordings")


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
    e3_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    _validate_label(e0_root, "Offline")
    e1_manifest = _validate_label(e1_root, "Offline")
    _validate_e1_schema(e1_root, e1_manifest)
    e2_manifest = _validate_label(e2_root, "Closed-loop")
    _validate_e2_schema(e2_root, e2_manifest)
    e3_manifest = _validate_label(e3_root, "Closed-loop")
    _validate_e3_schema(e3_root, e3_manifest)
    e0 = _read_json(e0_root, "metrics.json")
    e1 = _read_json(e1_root, "metrics.json")
    e2 = _read_json(e2_root, "metrics.json")
    e3 = _read_json(e3_root, "metrics.json")
    mode_analysis = _read_json(e2_root, "mode_analysis.json")
    condition_metrics = _read_json(e3_root, "condition_metrics.json")
    primary_counts = mode_analysis.get("primary_mode_counts", {})
    multimode_seeds = sorted(int(seed) for seed, count in primary_counts.items() if int(count) >= 2)
    action_diversity = float(
        _value(
            e1,
            "full_sampling_seed_mean_denormalized_trajectory_pairwise_rms",
        )
    )
    e1_answer = (
        "Full stochastic Diffusion sampling produced measurable action-pattern differences."
        if action_diversity > 1e-6
        else "No measurable action-pattern difference was observed at the configured tolerance."
    )
    e2_answer = (
        f"Multiple successful trajectory modes were detected for seeds {multimode_seeds}."
        if multimode_seeds
        else "No initial seed produced two successful modes at the pre-registered threshold."
    )
    best_condition = e3.get("best_qualifying_condition")
    lines = [
        "# PushT Diffusion Pilot Report",
        "",
        "## Evidence labels",
        "",
        "- E0-lite: Offline qualitative inspection.",
        "- E1: Offline fixed-observation sampling; no actions were executed.",
        "- E2: Closed-loop repeated rollouts.",
        "- E3: Closed-loop paired scheduler/step benchmark.",
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
            f"- E2 success-any@10: {float(_value(e2, 'success_any_at_10')):.3f} "
            "(oracle coverage diagnostic)."
        ),
        f"- E3 DDPM-100 success: {float(_value(e3, 'reference_success_rate')):.3f}.",
        (
            f"- E3 DDPM-100 p95 end-to-end chunk latency: "
            f"{float(_value(e3, 'reference_end_to_end_p95_ms')):.3f} ms."
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
            "identifiable from any@10; compare fixed policy_seed=0 against BCChunk."
        ),
        f"6. Best success-latency balance: {best_condition or 'no condition met the target'}.",
        "",
        "## Limitations",
        "",
        "- E0-lite does not prove strict conditional multimodality.",
        "- success-any@10 is not deployment success.",
        "- This Pilot does not restore dataset middle frames or serialize Pymunk state.",
    ]
    markdown = "\n".join(lines) + "\n"
    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        (temporary / "report.md").write_text(markdown, encoding="utf-8")
        aggregate = {
            "e1_action_trajectory_diversity": action_diversity,
            "e2_bc_success_rate": e2["bc_success_rate"],
            "e2_diffusion_single_sample_success": e2["single_sample_success"],
            "e2_success_any_at_10": e2["success_any_at_10"],
            "e2_multimode_environment_count": len(multimode_seeds),
            "e3_reference_success_rate": e3["reference_success_rate"],
            "e3_reference_end_to_end_p95_ms": e3["reference_end_to_end_p95_ms"],
            "e3_best_qualifying_condition": best_condition,
        }
        write_metrics(
            temporary,
            aggregate,
            comments={
                "e1_action_trajectory_diversity": "Offline fixed-input action diversity",
                "e2_bc_success_rate": "Deterministic BCChunk closed-loop success",
                "e2_diffusion_single_sample_success": "Diffusion policy_seed=0 deployment metric",
                "e2_success_any_at_10": "Oracle repeated-sampling coverage",
                "e2_multimode_environment_count": "Primary threshold 0.05",
                "e3_reference_success_rate": "DDPM-100 paired final seeds",
                "e3_reference_end_to_end_p95_ms": "Raw observation to action chunk",
                "e3_best_qualifying_condition": "2x speed and <=5-point success drop",
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 2,
                "experiment_schema": "pusht-pilot-report-e0-e3-v2",
                "label": "Formal",
                "experiment": "PushT Diffusion Pilot report",
                "sources": {
                    "e0": str(e0_root),
                    "e1": str(e1_root),
                    "e2": str(e2_root),
                    "e3": str(e3_root),
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
    parser.add_argument("--e3", type=Path, required=True)
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
        e3_root=args.e3.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(f"Pilot report: {result}")


if __name__ == "__main__":
    main()
