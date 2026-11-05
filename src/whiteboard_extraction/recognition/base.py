"""Common interface every recognition backend must implement.

A recognition method is any object with ``recognize(timestamp, image)`` that returns a list of
``RecognizedBlock``. The protocol is runtime-checkable, so ``isinstance(obj, Recognizer)`` says
whether an object can be plugged into the pipeline. Methods are chosen by name through
``registry.build_recognizer`` and called only through ``pipeline.recognize_keyframes``, so the
rest of the system never imports a concrete backend.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

KINDS = ("equation", "prose")  # the only values RecognizedBlock.kind may take


@dataclass
class RecognizedBlock:
    """A single recognized piece of content from a keyframe."""
    text: str
    kind: str  # one of KINDS
    timestamp: float
    confidence: float | None = None


@runtime_checkable
class Recognizer(Protocol):
    """Interface implemented by each DL recognition method."""

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        """Recognize content on a single keyframe (a BGR image, as OpenCV reads it)."""
        ...
