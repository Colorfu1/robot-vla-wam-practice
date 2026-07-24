#!/usr/bin/env python3
"""Generate deterministic L/O/R Rerun launcher icons."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 256
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "src" / "robot_practice" / "assets"
BACKGROUND = "#171B24"
FOREGROUND = "#F8FAFC"
KINDS = {
    "libero": ("L", "#06B6D4"),
    "offline": ("O", "#F59E0B"),
    "unknown": ("R", "#94A3B8"),
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_robot_arm(draw: ImageDraw.ImageDraw, accent: str) -> None:
    # Base and three-link arm. Wide strokes and joints stay legible at 32–48 px.
    draw.rounded_rectangle((34, 184, 132, 213), radius=12, fill=FOREGROUND)
    draw.line((76, 184, 92, 137, 137, 104, 174, 58), fill=FOREGROUND, width=20)
    for center in ((92, 137), (137, 104), (174, 58)):
        x, y = center
        draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill=accent)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=FOREGROUND)
    draw.line((174, 58, 205, 73), fill=FOREGROUND, width=12)
    draw.line((174, 58, 190, 29), fill=FOREGROUND, width=12)


def generate_icon(name: str, badge: str, accent: str) -> Path:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (7, 7, SIZE - 7, SIZE - 7),
        radius=46,
        fill=BACKGROUND,
        outline=accent,
        width=9,
    )
    _draw_robot_arm(draw, accent)

    badge_box = (151, 145, 236, 230)
    draw.ellipse(badge_box, fill=accent, outline=BACKGROUND, width=7)
    font = _font(55)
    bounds = draw.textbbox((0, 0), badge, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    center_x = (badge_box[0] + badge_box[2]) / 2
    center_y = (badge_box[1] + badge_box[3]) / 2
    draw.text(
        (
            center_x - text_width / 2,
            center_y - text_height / 2 - bounds[1],
        ),
        badge,
        font=font,
        fill=BACKGROUND,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"rerun_{name}.png"
    image.save(output, format="PNG", optimize=True)
    return output


def main() -> None:
    for name, (badge, accent) in KINDS.items():
        print(generate_icon(name, badge, accent))


if __name__ == "__main__":
    main()
