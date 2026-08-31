#!/usr/bin/env python3
"""Build a review reel and contact sheet from two or four W0 video directories.

Why this exists: the W0 video-generation comparison is not decidable from metrics alone. The same
frame-to-frame change statistic means "saturated noise churning" for one variant and "the zebra
walked across the frame" for another, so any single dynamics number is unreadable without looking
at the clips. Reviewing 21 scenes by opening 42 files is not practical either, so this collapses
them into one labelled reel plus one sheet.

Panels are matched by file name across the directories, so the same prompt always lands in the
same grid position, and only samples present in *every* directory are emitted. Clips with
different frame counts are sampled at the same relative positions, which is what makes a 9-frame
FastWAM clip comparable to an 81-frame official clip.

Deliberately free of OpenCV: writing is done through imageio/ffmpeg and drawing through Pillow,
because ``cv2.imwrite`` has failed on this project's runtimes with "could not find a writer for
the specified extension".

Examples
--------
Two panels, one operating point::

    python scripts/experiments/literature/make_w0_video_review_reel.py \
      --videos "FW-FZ=outputs/.../vbench/fw_fz/videos" \
      --videos "FW-V=outputs/.../vbench/fw_v/videos" \
      --out-reel outputs/.../vbench/review_fastwam.mp4 \
      --out-sheet outputs/.../vbench/review_fastwam.jpg

Four panels, both operating points (after the official run exists)::

    python scripts/experiments/literature/make_w0_video_review_reel.py \
      --videos "FW-FZ @ FastWAM=outputs/.../vbench/fw_fz/videos" \
      --videos "FW-V @ FastWAM=outputs/.../vbench/fw_v/videos" \
      --videos "FW-FZ @ official=outputs/.../vbench/native/fw_fz/videos" \
      --videos "FW-V @ official=outputs/.../vbench/native/fw_v/videos" \
      --out-reel outputs/.../vbench/review_2x2.mp4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


def _font(size: int):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _parse_videos(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        # split at the LAST "=", so a label may contain one: "FW-FZ (frozen = Wan2.2)=path"
        label, _, directory = value.rpartition("=")
        if not label:
            raise SystemExit(f"--videos expects LABEL=DIR, got {value!r}")
        path = Path(directory)
        if not path.is_dir():
            raise SystemExit(f"not a directory: {directory}")
        parsed.append((label.strip(), path))
    if len(parsed) not in (2, 4):
        raise SystemExit(f"pass 2 or 4 --videos entries, got {len(parsed)}")
    return parsed


def _common_samples(sources: list[tuple[str, Path]]) -> list[str]:
    name_sets = []
    for _, directory in sources:
        name_sets.append({p.name for p in sorted(directory.glob("*.mp4"))})
    common = set.intersection(*name_sets)
    missing = {label: sorted(names - common) for (label, _), names in zip(sources, name_sets)}
    for label, names in missing.items():
        if names:
            print(f"  skipping {len(names)} sample(s) absent elsewhere, e.g. {names[0]}"
                  f" (only in {label})")
    return sorted(common)


def _read_frames(path: Path, count: int) -> np.ndarray:
    """Read ``count`` frames at evenly spaced *relative* positions."""
    from decord import VideoReader

    reader = VideoReader(str(path))
    total = len(reader)
    if total == 0:
        raise RuntimeError(f"no frames in {path}")
    if count == 1:
        indices = [0]
    else:
        indices = [round(step * (total - 1) / (count - 1)) for step in range(count)]
    return reader.get_batch(indices).asnumpy()


def _prompt_of(name: str) -> str:
    text = re.sub(r"\.mp4$", "", name)
    text = re.sub(r"^\d+_+", "", text)
    return text.replace("_", " ").strip() or name


def _panel(frame: np.ndarray, *, width: int, label: str, font) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    height = max(1, round(image.height * width / image.width))
    image = image.resize((width, height), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((6, 6, 12 + box[2] - box[0], 12 + box[3] - box[1]), fill=(0, 0, 0))
    draw.text((9, 8), label, font=font, fill=(255, 255, 255))
    return image


def _grid(panels: list[Image.Image], *, columns: int, gap: int = 4) -> Image.Image:
    rows = (len(panels) + columns - 1) // columns
    cell_w = max(p.width for p in panels)
    cell_h = max(p.height for p in panels)
    canvas = Image.new(
        "RGB", (columns * cell_w + (columns - 1) * gap, rows * cell_h + (rows - 1) * gap), "white"
    )
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        canvas.paste(panel, (column * (cell_w + gap), row * (cell_h + gap)))
    return canvas


def _with_caption(image: Image.Image, caption: str, font) -> Image.Image:
    bar = 30
    canvas = Image.new("RGB", (image.width, image.height + bar), "white")
    canvas.paste(image, (0, bar))
    ImageDraw.Draw(canvas).text((8, 7), caption, font=font, fill=(0, 0, 0))
    return canvas


def build_reel(
    sources: list[tuple[str, Path]],
    samples: list[str],
    *,
    out_path: Path,
    panel_width: int,
    fps: int,
    frames_per_sample: int,
) -> int:
    import imageio.v2 as imageio

    columns = 2
    label_font = _font(max(13, panel_width // 26))
    caption_font = _font(19)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=8, macro_block_size=2
    ) as writer:
        for order, name in enumerate(samples, start=1):
            stacks = [
                (label, _read_frames(directory / name, frames_per_sample))
                for label, directory in sources
            ]
            caption = f"{order}/{len(samples)}  {_prompt_of(name)}"
            for step in range(frames_per_sample):
                panels = [
                    _panel(frames[step], width=panel_width, label=label, font=label_font)
                    for label, frames in stacks
                ]
                frame = _with_caption(_grid(panels, columns=columns), caption, caption_font)
                if frame.width % 2 or frame.height % 2:
                    frame = frame.crop((0, 0, frame.width - frame.width % 2,
                                        frame.height - frame.height % 2))
                writer.append_data(np.asarray(frame))
            written += 1
            print(f"  [{order}/{len(samples)}] {name}")
    return written


def build_sheet(
    sources: list[tuple[str, Path]],
    samples: list[str],
    *,
    out_path: Path,
    panel_width: int,
    frames_per_sample: int,
    orientation: str = "row",
) -> None:
    """Contact sheet.

    ``orientation="row"`` puts every source inline on one row per sample: compact, good for
    scanning 21 scenes at once, but it cannot show how a clip degrades over time because each
    source only gets a few adjacent cells.

    ``orientation="column"`` gives each sample a block whose rows are the sources and whose columns
    are the time axis. That is the layout a document figure needs: a variant that is clean at frame
    0 and saturated by frame 40 reads as a row, not as two cells the eye has to pair up.
    """
    if orientation not in {"row", "column"}:
        raise SystemExit(f"--sheet-orientation must be row or column, got {orientation!r}")
    label_font = _font(max(11, panel_width // 20))
    caption_font = _font(15)
    rows: list[Image.Image] = []
    for order, name in enumerate(samples, start=1):
        if orientation == "row":
            panels: list[Image.Image] = []
            for label, directory in sources:
                frames = _read_frames(directory / name, frames_per_sample)
                for step in range(frames_per_sample):
                    tag = f"{label} {step}/{frames_per_sample - 1}" if step == 0 else ""
                    panels.append(
                        _panel(frames[step], width=panel_width, label=tag, font=label_font)
                    )
            strip = _grid(panels, columns=len(panels))
        else:
            blocks: list[Image.Image] = []
            for label, directory in sources:
                frames = _read_frames(directory / name, frames_per_sample)
                cells = [
                    _panel(
                        frames[step],
                        width=panel_width,
                        label=label if step == 0 else "",
                        font=label_font,
                    )
                    for step in range(frames_per_sample)
                ]
                blocks.append(_grid(cells, columns=frames_per_sample))
            strip = _grid(blocks, columns=1)
        rows.append(_with_caption(strip, f"{order}. {_prompt_of(name)}", caption_font))
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + 4 * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for row in rows:
        canvas.paste(row, (0, offset))
        offset += row.height + 4
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=86)
    print(f"  sheet {out_path} {canvas.size}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--out-reel", type=Path)
    parser.add_argument("--out-sheet", type=Path)
    parser.add_argument("--panel-width", type=int, default=560)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--reel-frames",
        type=int,
        default=17,
        help="frames sampled per clip for the reel (relative positions, so clips of different "
        "length stay aligned)",
    )
    parser.add_argument("--sheet-frames", type=int, default=4)
    parser.add_argument(
        "--sheet-orientation",
        choices=("row", "column"),
        default="row",
        help="row packs every source inline per sample; column gives each sample a block whose "
        "rows are the sources and columns are the time axis, which is what shows a clip degrading",
    )
    parser.add_argument(
        "--samples",
        help="comma-separated sample-name substrings to keep, in the given order",
    )
    parser.add_argument("--limit", type=int, default=0, help="only the first N samples")
    args = parser.parse_args(argv)

    if not args.out_reel and not args.out_sheet:
        parser.error("give --out-reel and/or --out-sheet")

    sources = _parse_videos(args.videos)
    samples = _common_samples(sources)
    if not samples:
        raise SystemExit("no sample name is present in every directory")
    if args.samples:
        wanted = [token.strip() for token in args.samples.split(",") if token.strip()]
        picked: list[str] = []
        for token in wanted:
            matches = [name for name in samples if token in name]
            if not matches:
                raise SystemExit(f"no sample matches {token!r}")
            picked.append(matches[0])
        samples = picked
    if args.limit:
        samples = samples[: args.limit]
    print(f"{len(samples)} sample(s) across {len(sources)} directories")

    if args.out_reel:
        count = build_reel(
            sources,
            samples,
            out_path=args.out_reel,
            panel_width=args.panel_width,
            fps=args.fps,
            frames_per_sample=args.reel_frames,
        )
        print(f"  reel {args.out_reel} with {count} sample(s)")
    if args.out_sheet:
        build_sheet(
            sources,
            samples,
            out_path=args.out_sheet,
            panel_width=max(150, args.panel_width // 3),
            frames_per_sample=args.sheet_frames,
            orientation=args.sheet_orientation,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
