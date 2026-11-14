"""Loading a labeled test set: a folder of board images with a labels.jsonl manifest.

Each line of labels.jsonl describes one image::

    {"image": "boards/0012.png", "timestamp": 734.0,
     "blocks": [{"text": "Homework 4 is due Friday", "kind": "prose"},
                {"text": "\\det(A - \\lambda I) = 0", "kind": "equation"}]}

* image: path relative to the folder; a cropped board region as the pipeline would hand to a
  recognizer (anything OpenCV can read).
* timestamp: seconds into the source video (optional, default 0); passed through to the
  recognizer and used to order blocks.
* blocks: what is written on the board, top to bottom, each with kind "prose" or "equation".
  An empty list is a valid label for a blank board.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from whiteboard_extraction.recognition.base import RecognizedBlock

from .metrics import KINDS

MANIFEST_NAME = "labels.jsonl"


class TestSetError(ValueError):
    """The test set folder or its manifest is not usable."""

    __test__ = False  # the name starts with "Test"; keep pytest from collecting it


@dataclass(frozen=True)
class Sample:
    """One labeled board image."""

    name: str
    image_path: Path
    timestamp: float
    blocks: tuple[RecognizedBlock, ...]

    def load_image(self) -> np.ndarray:
        """Read the image as a BGR array, the same layout the video pipeline produces."""
        import cv2  # imported here so the metrics can be used without OpenCV installed

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise TestSetError(f"Could not read image {self.image_path}.")
        return image


def load_test_set(directory: Path) -> list[Sample]:
    """Read every sample listed in <directory>/labels.jsonl, checking that the images exist."""
    directory = Path(directory)
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        raise TestSetError(f"{directory} has no {MANIFEST_NAME}; see docs/EVALUATION.md for the format.")
    try:
        text = manifest.read_text(encoding="utf-8-sig")  # a BOM is accepted silently
    except UnicodeDecodeError:
        raise TestSetError(f"{manifest} is not UTF-8 text; save labels.jsonl as UTF-8.") from None
    samples: list[Sample] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{manifest}:{number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TestSetError(f"{where}: not valid JSON ({error.msg}).") from None
        samples.append(_parse_record(record, directory, where))
    if not samples:
        raise TestSetError(f"{manifest} lists no samples.")
    return samples


def write_manifest(directory: Path, records: list[dict[str, Any]]) -> Path:
    """Write labels.jsonl for records shaped like the lines described in the module docstring."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / MANIFEST_NAME
    manifest.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return manifest


def _parse_record(record: Any, directory: Path, where: str) -> Sample:
    if not isinstance(record, dict):
        raise TestSetError(f"{where}: each line must be a JSON object.")
    image = record.get("image")
    if not isinstance(image, str) or not image:
        raise TestSetError(f"{where}: 'image' must be a file path relative to the test set folder.")
    image_path = directory / image
    if not image_path.is_file():
        raise TestSetError(f"{where}: image {image_path} does not exist.")
    try:
        timestamp = float(record.get("timestamp", 0.0))
    except (TypeError, ValueError):
        raise TestSetError(f"{where}: 'timestamp' must be a number of seconds.") from None
    raw_blocks = record.get("blocks")
    if not isinstance(raw_blocks, list):
        raise TestSetError(f"{where}: 'blocks' must be a list (empty for a blank board).")
    blocks: list[RecognizedBlock] = []
    for index, raw in enumerate(raw_blocks, start=1):
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str) or not raw["text"].strip():
            raise TestSetError(f"{where}: block {index} needs a non-empty 'text'.")
        kind = raw.get("kind", "prose")
        if kind not in KINDS:
            raise TestSetError(f"{where}: block {index} has kind {kind!r}; use one of {', '.join(KINDS)}.")
        blocks.append(RecognizedBlock(text=raw["text"], kind=kind, timestamp=timestamp))
    return Sample(name=image, image_path=image_path, timestamp=timestamp, blocks=tuple(blocks))
