"""Recognition backend using a multimodal LLM."""

import numpy as np

from whiteboard_extraction.recognition.base import RecognizedBlock


class MultimodalLLMRecognizer:
    """Sends a keyframe image to a multimodal LLM and parses its response."""

    def __init__(self, model: str):
        self.model = model

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        raise NotImplementedError("Wire up the chosen multimodal API here.")