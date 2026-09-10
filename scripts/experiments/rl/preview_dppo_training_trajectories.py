#!/usr/bin/env python3
"""Sample and preview complete trajectories from the DPPO Square train.npz."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/raw/dppo/robomimic/square-img/train.npz"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--contact-samples", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def image_hwc(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"expected an RGB CHW image, got shape={image_chw.shape}")
    image = np.transpose(image_chw, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def write_video(
    images_chw: np.ndarray,
    output_path: Path,
    *,
    fps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write trajectory previews")

    height, width = images_chw.shape[2:]
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for image_chw in images_chw:
            process.stdin.write(image_hwc(image_chw).tobytes())
        process.stdin.close()
        process.stdin = None
    except (BrokenPipeError, OSError):
        process.kill()
        process.wait()
        raise
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with return code {return_code}: {output_path}")


def write_contact_sheet(
    images: list[np.ndarray],
    records: list[dict[str, int | str]],
    output_path: Path,
    *,
    sample_count: int,
) -> None:
    tile_size = 128
    label_width = 220
    row_height = tile_size + 28
    sheet = Image.new(
        "RGB",
        (label_width + tile_size * sample_count, row_height * len(images)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, (trajectory, record) in enumerate(zip(images, records)):
        y = row * row_height
        label = (
            f"demo={record['trajectory_index']}\n"
            f"len={record['length']}\n"
            f"start={record['start']}"
        )
        draw.multiline_text((6, y + 6), label, fill=(0, 0, 0), font=font, spacing=3)
        frame_indices = np.linspace(0, len(trajectory) - 1, sample_count).round().astype(int)
        for column, frame_index in enumerate(frame_indices):
            image = Image.fromarray(image_hwc(trajectory[frame_index]))
            image = image.resize((tile_size, tile_size), Image.Resampling.NEAREST)
            x = label_width + column * tile_size
            sheet.paste(image, (x, y))
            draw.text((x + 3, y + 3), str(frame_index), fill=(255, 255, 0), font=font)
    sheet.save(output_path)


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.fps <= 0 or args.contact_samples <= 0:
        raise SystemExit("--fps and --contact-samples must be positive")
    if not args.dataset.is_file() or args.dataset.stat().st_size == 0:
        raise SystemExit(f"dataset is missing or empty: {args.dataset}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.replace:
            raise SystemExit(
                f"refusing to reuse non-empty output directory: {args.output_dir}; "
                "use --replace"
            )
        for entry in args.output_dir.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as data:
        required = {"images", "traj_lengths"}
        missing = sorted(required - set(data.files))
        if missing:
            raise SystemExit("dataset missing keys: " + ", ".join(missing))
        images = data["images"]
        trajectory_lengths = data["traj_lengths"].astype(np.int64, copy=False)
        if images.ndim != 4 or images.shape[1] != 3:
            raise SystemExit(f"expected images with shape [N, 3, H, W], got {images.shape}")
        if int(trajectory_lengths.sum()) != images.shape[0]:
            raise SystemExit(
                "trajectory lengths do not cover images: "
                f"sum={int(trajectory_lengths.sum())} images={images.shape[0]}"
            )
        if args.count > len(trajectory_lengths):
            raise SystemExit(
                f"cannot sample {args.count} trajectories from {len(trajectory_lengths)}"
            )

        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(trajectory_lengths), size=args.count, replace=False))
        records: list[dict[str, int | str]] = []
        selected_images: list[np.ndarray] = []
        offset = 0
        selected_set = set(int(index) for index in selected)
        for trajectory_index, length_value in enumerate(trajectory_lengths):
            length = int(length_value)
            if trajectory_index in selected_set:
                trajectory = images[offset : offset + length]
                video_name = f"trajectory_{len(records):02d}_demo_{trajectory_index:03d}.mp4"
                video_path = args.output_dir / video_name
                write_video(trajectory, video_path, fps=args.fps)
                selected_images.append(trajectory)
                records.append(
                    {
                        "sample_index": len(records),
                        "trajectory_index": trajectory_index,
                        "start": offset,
                        "length": length,
                        "video": video_name,
                    }
                )
                print(
                    f"STATUS completed={len(records)} total={args.count} "
                    f"trajectory={trajectory_index} length={length}",
                    flush=True,
                )
            offset += length

    write_contact_sheet(
        selected_images,
        records,
        args.output_dir / "contact_sheet.png",
        sample_count=args.contact_samples,
    )
    manifest = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "sampling_seed": args.seed,
        "count": args.count,
        "fps": args.fps,
        "image_key": "agentview_image",
        "image_shape": list(images.shape[1:]),
        "trajectories": records,
        "contact_sheet": "contact_sheet.png",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"DONE completed={args.count} total={args.count}")
    print(f"output_dir={args.output_dir.resolve()}")
    print(f"contact_sheet={args.output_dir.joinpath('contact_sheet.png').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
