#!/usr/bin/env python3
"""Compare W0-Video-40 generated frames against ground truth, visually and numerically.

Diagnostic only. It does not produce or replace any metric in
reports/guides/literature/02_w0_wan_video_generation_preservation.md; use
``--mode=score`` for that. This exists to answer "does the generated video look right at
all", which PSNR alone does not show.

For each requested sample it prints, per frame, the PSNR against the matching ground-truth
frame *and* against the best-matching ground-truth frame. Those two disagreeing is the
signature of a temporal problem: if every generated frame matches ``gt[0]`` best, the sampler
is not advancing in time, which is different from it advancing but predicting badly.

It also writes a contact sheet: one labelled band per sample, ground truth on top and each
supplied variant below, so colour and structure failures are visible directly.

Usage:

    python scripts/experiments/literature/compare_w0_video_frames.py \
      --manifest outputs/.../w0_video_40/manifest.json \
      --videos fw_fz=outputs/.../w0_video_40/fw_fz/videos \
      --videos fw_v=outputs/.../w0_video_40/fw_v/videos \
      --output outputs/.../w0_video_40/frame_compare.png \
      --samples libero_spatial_task00,libero_object_task00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from robot_practice.experiments.literature import w0_video_generation_eval as evaluator

THUMB_WIDTH = 224
THUMB_HEIGHT = 112


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--videos",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="variant label and its videos directory; repeatable",
    )
    parser.add_argument("--output", required=True, help="contact sheet PNG path")
    parser.add_argument("--samples", default="", help="comma-separated sample_id filter")
    parser.add_argument("--limit", type=int, default=3, help="max samples to render (default 3)")
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality when --output ends in .jpg (ignored for .png; default 85)",
    )
    parser.add_argument(
        "--camera",
        choices=("both", "front", "wrist"),
        default="both",
        help="ground truth layout; must match how the videos were generated",
    )
    return parser.parse_args()


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    from PIL import Image

    return np.asarray(
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
            (THUMB_WIDTH, THUMB_HEIGHT), Image.BOX
        ),
        dtype=np.uint8,
    )


def _label_band(text: str, width: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    band = Image.new("RGB", (max(width, 1), 22), (0, 0, 0))
    ImageDraw.Draw(band).text((4, 5), text, fill=(255, 255, 255))
    return np.asarray(band, dtype=np.uint8)


def _write_sheet(sheet: np.ndarray, output: Path, *, quality: int) -> None:
    """Write with Pillow, choosing the encoder from the output extension.

    ``cv2.imwrite`` depends on the codecs the local OpenCV build happens to carry; a build
    without the PNG writer fails with "could not find a writer for the specified extension"
    only after all the work is done. Pillow is already a hard dependency of the evaluator
    module, so using it removes an environment-dependent failure from the last line.

    PNG is the default for diagnostics under ``outputs/``. ``.jpg`` exists because the figures
    embedded in tracked documentation have to be committed to render anywhere, and a lossless
    montage of nine rendered frames per row is several megabytes -- far larger than anything
    else tracked in this repository.
    """
    from PIL import Image

    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(sheet, dtype=np.uint8))
    if output.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output, quality=quality, optimize=True, progressive=True)
    else:
        image.save(output)
    if not output.is_file():
        raise RuntimeError(f"failed to write {output}")


def main() -> int:
    args = parse_args()
    variants: list[tuple[str, Path]] = []
    for item in args.videos:
        if "=" not in item:
            raise SystemExit(f"--videos expects LABEL=DIR, got {item!r}")
        label, _, directory = item.partition("=")
        variants.append((label, Path(directory)))

    manifest = evaluator._read_json(Path(args.manifest))
    samples = list(manifest["samples"])
    wanted = [item for item in args.samples.split(",") if item]
    if wanted:
        samples = [s for s in samples if s["sample_id"] in wanted]
    samples = samples[: max(1, args.limit)]
    if not samples:
        raise SystemExit("no samples selected")

    def strip(frames: list[np.ndarray], count: int) -> np.ndarray:
        thumbs = [_thumbnail(frame) for frame in frames[:count]]
        while len(thumbs) < count:
            thumbs.append(np.zeros((THUMB_HEIGHT, THUMB_WIDTH, 3), np.uint8))
        return np.concatenate(thumbs, axis=1)

    bands: list[np.ndarray] = []
    for sample in samples:
        gt = evaluator._w0_ground_truth(sample, camera=args.camera)
        print(f"\n=== {sample['sample_id']}  ({len(gt)} ground-truth frames)")
        rows = [("GT", gt)]
        for label, directory in variants:
            path = directory / f"{sample['sample_id']}.mp4"
            if not path.is_file():
                print(f"  {label}: MISSING {path}")
                continue
            pred = evaluator._read_video_frames(path)
            print(f"  {label}: {len(pred)} frames, {pred[0].shape if pred else None}")
            if pred and pred[0].shape != gt[0].shape:
                print(
                    f"    SKIP metrics: shape {pred[0].shape} != ground truth "
                    f"{gt[0].shape}; pass --camera to match how it was generated"
                )
                rows.append((label, pred))
                continue
            for index in range(min(len(gt), len(pred))):
                aligned = evaluator._psnr(gt[index], pred[index])
                scores = [evaluator._psnr(frame, pred[index]) for frame in gt]
                best = int(np.argmax(scores))
                flag = "" if best == index else f"  <- best matches gt[{best}]"
                print(f"    frame {index}: {aligned:6.2f} dB{flag}")
            rows.append((label, pred))

        bands.append(
            _label_band(
                f"{sample['sample_id']}  rows: " + " / ".join(label for label, _ in rows),
                THUMB_WIDTH * len(gt),
            )
        )
        bands.extend(strip(frames, len(gt)) for _, frames in rows)

    sheet = np.concatenate(bands, axis=0)
    output = Path(args.output)
    _write_sheet(sheet, output, quality=args.quality)
    print(f"\nwrote {output} {sheet.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
