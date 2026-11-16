"""Recognition backend using a pretrained OCR model."""

import cv2
import numpy as np
from PIL import Image

from whiteboard_extraction.recognition.base import RecognizedBlock

_EQUATION_TYPES = {"isolated"}


def _reading_key(region: dict) -> tuple[float, float]:
    """(top, left) of a region's box, so output runs top to bottom, then left to right.

    A region without a position sorts last; the sort is stable, so those keep detector order.
    """
    position = region.get("position")
    if position is None:
        return (float("inf"), float("inf"))
    points = np.asarray(position, dtype=float).reshape(-1, 2)
    return (float(points[:, 1].min()), float(points[:, 0].min()))


class PretrainedOCRRecognizer:
    """Wraps a pretrained OCR engine (Pix2Text by default)."""

    def __init__(self, engine: str = "pix2text"):
        self.engine = engine
        if engine == "pix2text":
            from pix2text import Pix2Text
            self._p2t = Pix2Text.from_config(total_configs={"text_formula": {"languages": ("en",)}}) # Was hallucinating Chinese characters, restrict to English only
        else:
            raise ValueError(f"Unsupported OCR engine {engine!r}.")

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        regions = sorted(self._p2t.recognize(rgb, return_text=False), key=_reading_key)
        
        blocks = []
        for region in regions:
            text = region["text"].strip()
            if not text:
                continue
            kind = "equation" if region["type"] in _EQUATION_TYPES else "prose"
            score = region.get("score")
            blocks.append(RecognizedBlock(
                text=text,
                kind=kind,
                timestamp=timestamp,
                confidence=float(score) if score is not None else None,
            ))
        return blocks