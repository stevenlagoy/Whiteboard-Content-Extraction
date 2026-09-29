"""Common interface every recognition backend must implement."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class RecognizedBlock:
    """A single recognized piece of content from a keyframe."""
    text: str
    kind: str  # "equation" or "prose"
    timestamp: float
    confidence: float | None = None


class Recognizer(Protocol):
    """Interface implemented by each DL recognition method."""

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        """Recognize content on a single keyframe."""
        ...