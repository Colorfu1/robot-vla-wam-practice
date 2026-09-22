"""Compose a native SVG evidence panel without modifying source image pixels."""

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image


def image_uri(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    metrics = json.loads((args.source / "metrics.json").read_text())
    assert metrics["status"] == "complete"
    step = metrics["cases"]["demo87_rgb_once"]["first_success_step"]
    assert step == 113
    parts = [('<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="700" '
              'viewBox="0 0 1300 700">'), '<rect width="1300" height="700" fill="white"/>']

    def text(x, y, content, size=16, color="#17212b"):
        parts.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
                     f'fill="{color}">{html.escape(content)}</text>')

    text(20, 30, "FastWAM / Square: one-shot image intervention (one paired case)", 23)
    text(20, 56, "Same physical state + proprio at rollout step 260. B replaces only the first RGB input.")
    columns = ["Actual start t+0", "Policy input", "PREDICTED t+4", "ACTUAL t+4",
               "ACTUAL t+113", "ACTUAL t+340"]
    for row, variant in enumerate(("original_rgb", "demo87_rgb_once")):
        y = 105 + row * 278
        case = metrics["cases"][variant]
        assert max(case["prefix_errors"].values()) == 0
        label = ("A: original RGB | gripper stays open | task failed" if row == 0 else
                 "B: demo frame 87 RGB once | gripper closes | first task success at t+113")
        text(20, y, label, 18)
        directory = args.source / variant
        images = [Image.open(directory / f).convert("RGB") for f in
                  ("ACTUAL_tplus0.png", "POLICY_INPUT.png", "PRED_tplus4.png", "ACTUAL_tplus4.png")]
        with Image.open(directory / "ACTUAL_rollout.gif") as gif:
            assert gif.n_frames == 341, "Must verify GIF timing before selecting frames"
            for t in range(gif.n_frames):
                gif.seek(t)
                assert gif.info["duration"] == 50, "Unexpected GIF frame timing"
            for t in (113, 340):
                gif.seek(t)
                images.append(gif.convert("RGB").copy())
        for col, (label, image) in enumerate(zip(columns, images)):
            x = 20 + col * 213
            color = "#be123c" if col == 2 else "#17212b"
            text(x, y + 27, label, 16, color)
            parts.append(f'<image x="{x}" y="{y + 40}" width="200" height="200" '
                         f'href="{image_uri(image)}"/>')
    text(20, 647, "Times are control steps after intervention (20 Hz). All later inputs are live observations.")
    text(20, 670, "Prediction is NOT executed video. PRED t+4 is not identical to ACTUAL t+4; success is from simulator metrics.")
    text(20, 692, "Single-case evidence for input sensitivity, not evidence that generated video causes action or that all failures share a cause.", 15)
    parts.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts))
    print(f"DONE figure={args.output}")


if __name__ == "__main__":
    main()
