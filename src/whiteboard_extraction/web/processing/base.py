"""The contract between the job queue and whatever turns video into text.

Any object with these members can be handed to the app as its engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from whiteboard_extraction.reconstruction.document import BoardDocument


@dataclass(frozen=True)
class ProcessingRequest:
    video_id: str
    media_path: Path | None  # None when the original file is not stored
    start_seconds: float
    end_seconds: float

    @property
    def clip_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class ProgressReporter(Protocol):
    def report(self, fraction: float) -> None:
        """Report progress from 0.0 to 1.0. Raises JobCancelled if the user cancelled."""
        ...


class ProcessingEngine(Protocol):
    name: str

    def estimate_seconds(self, clip_seconds: float) -> float:
        """Expected wall-clock processing time for a clip of this length."""
        ...

    def process(self, request: ProcessingRequest, reporter: ProgressReporter) -> BoardDocument: ...
