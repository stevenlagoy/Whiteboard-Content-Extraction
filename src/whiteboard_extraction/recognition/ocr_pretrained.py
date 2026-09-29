"""Recognition backend using a pretrained OCR model."""

import numpy as np

from whiteboard_extraction.recognition.base import RecognizedBlock


class PretrainedOCRRecognizer:
    """Wraps a pretrained OCR engine (e.g. Tesseract or EasyOCR)."""

    def __init__(self, engine: str = "tesseract"):
        self.engine = engine

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        raise NotImplementedError("Wire up the chosen OCR engine here.")