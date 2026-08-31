#!/usr/bin/env python3
"""Print the paired LIBERO-Wan read-out for one operating point.

Kept as a file rather than a heredoc inside ``run_w0_libero_wan.sh``: a nested ``<<'PY'`` inside a
heredoc terminates the outer one, and this project's terminal workflow already loses backslash
continuations on paste, so multi-line shell-embedded Python is a known failure mode here.

Reads ``$W0_LIBERO_WAN_DIR/{fw_fz,fw_v}/metrics.json`` and refuses to print a delta when the two
variants did not score the same number of samples -- an unpaired delta is not a comparison.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

VARIANTS = ("fw_fz", "fw_v")


def main() -> int:
    work = os.environ.get("W0_LIBERO_WAN_DIR")
    if not work:
        print("W0_LIBERO_WAN_DIR is not set", file=sys.stderr)
        return 2
    root = Path(work)

    rows: dict[str, dict] = {}
    for variant in VARIANTS:
        path = root / variant / "metrics.json"
        if not path.is_file():
            print(f"    {variant}: metrics.json missing at {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows[variant] = payload
        summary = payload["summary"]
        definition = payload["metric_definition"]
        lpips = summary.get("lpips")
        lpips_text = "n/a" if lpips is None else f"{lpips:.4f}"
        print(
            f"    {variant}: n={summary['count']} psnr={summary['psnr']:.3f} "
            f"ssim={summary['ssim']:.4f} lpips={lpips_text} "
            f"gt={definition['ground_truth_resolution']} "
            f"transform={definition['ground_truth_transform']} "
            f"frames={len(definition['future_frame_indices']) + 1}"
        )

    if len(rows) != len(VARIANTS):
        print("    paired delta skipped: both variants must be scored")
        return 1

    frozen, trained = rows["fw_fz"]["summary"], rows["fw_v"]["summary"]
    if frozen["count"] != trained["count"]:
        print(
            f"    unpaired sample counts {frozen['count']} vs {trained['count']}; "
            "the delta would not be a comparison",
            file=sys.stderr,
        )
        return 1
    if rows["fw_fz"]["manifest"]["sha256"] != rows["fw_v"]["manifest"]["sha256"]:
        print("    variants scored different manifests; refusing to pair", file=sys.stderr)
        return 1

    if frozen["lpips"] is None or trained["lpips"] is None:
        lpips_delta = "n/a"
    else:
        lpips_delta = f"{frozen['lpips'] - trained['lpips']:+.4f}"
    print(
        "    delta psnr={:+.3f} ssim={:+.4f} lpips={} "
        "(psnr/ssim are fw_v-fw_fz, lpips is fw_fz-fw_v, so positive always favours fw_v)".format(
            trained["psnr"] - frozen["psnr"], trained["ssim"] - frozen["ssim"], lpips_delta
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
