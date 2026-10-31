"""A stand-in engine that reports steady progress and returns sample results."""

from __future__ import annotations

import time
from collections.abc import Callable

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.reconstruction.document import BoardDocument

from .base import ProcessingRequest, ProgressReporter

SAMPLE_BLOCKS = (
    ("prose", "Sample heading: simulated recognition result"),
    ("equation", r"\int_0^1 x^2 \, dx = \frac{1}{3}"),
    ("prose", "Sample note: replace this engine to see real recognition"),
    ("equation", r"A\mathbf{x} = \lambda \mathbf{x}"),
)


class SimulatedEngine:
    name = "Simulated engine"

    def __init__(
        self,
        seconds_per_media_second: float = 0.02,
        minimum_seconds: float = 6.0,
        maximum_seconds: float = 120.0,
        step_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rate = seconds_per_media_second
        self._minimum = minimum_seconds
        self._maximum = maximum_seconds
        self._step = step_seconds
        self._sleep = sleep

    def estimate_seconds(self, clip_seconds: float) -> float:
        return min(max(clip_seconds * self._rate, self._minimum), self._maximum)

    def process(self, request: ProcessingRequest, reporter: ProgressReporter) -> BoardDocument:
        steps = max(round(self.estimate_seconds(request.clip_seconds) / self._step), 1)
        for step in range(1, steps + 1):
            self._sleep(self._step)
            reporter.report(step / steps)
        return BoardDocument(blocks=self._sample_blocks(request))

    @staticmethod
    def _sample_blocks(request: ProcessingRequest) -> list[RecognizedBlock]:
        count = len(SAMPLE_BLOCKS)
        return [
            RecognizedBlock(
                text=text,
                kind=kind,
                timestamp=request.start_seconds + request.clip_seconds * (index + 0.5) / count,
            )
            for index, (kind, text) in enumerate(SAMPLE_BLOCKS)
        ]
