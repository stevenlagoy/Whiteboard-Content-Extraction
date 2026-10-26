"""A tiny rendered test set for smoke tests and demos, so the evaluation runs before real labels exist.

Each sample is a white board with its blocks drawn as lines of printed text. It is not a stand-in
for real handwriting; it only proves the script, loader, and metrics work end to end.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .testset import write_manifest

Spec = tuple[str, float, list[tuple[str, str]]]  # (file name, timestamp, [(kind, text), ...])

DEFAULT_SPECS: list[Spec] = [
    ("board_01.png", 12.0, [
        ("prose", "Definition: a basis is a linearly independent spanning set."),
        ("equation", "det(A - lambda I) = 0"),
    ]),
    ("board_02.png", 47.5, [("equation", "x^2 + y^2 = r^2")]),
    ("board_03.png", 90.0, [
        ("prose", "Homework 4 is due Friday"),
        ("prose", "Quiz next week: chapters 2 and 3"),
        ("equation", "e^(i pi) + 1 = 0"),
    ]),
    ("board_04.png", 120.0, []),  # a blank board: the right answer is no blocks at all
]


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1 has no size argument
        return ImageFont.load_default()


def write_synthetic_test_set(directory: Path, specs: list[Spec] = DEFAULT_SPECS, width: int = 960, height: int = 540) -> Path:
    """Render the specs to PNG files plus labels.jsonl in directory; returns the manifest path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    font = _font(28)
    records = []
    for file_name, timestamp, blocks in specs:
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        for line, (_kind, text) in enumerate(blocks):
            draw.text((40, 40 + 56 * line), text, fill="black", font=font)
        image.save(directory / file_name)
        records.append({
            "image": file_name,
            "timestamp": timestamp,
            "blocks": [{"text": text, "kind": kind} for kind, text in blocks],
        })
    return write_manifest(directory, records)
