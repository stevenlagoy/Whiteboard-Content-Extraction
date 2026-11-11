"""Engine that runs the repository's real stages: sample, segment, keyframe, recognize."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from whiteboard_extraction.recognition.base import RecognizedBlock, Recognizer
from whiteboard_extraction.reconstruction.document import BoardDocument

from .base import ProcessingRequest, ProgressReporter

Frame = tuple[float, np.ndarray]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class PipelineStages:
    """The swappable stages. Replace any field to change one step."""

    sample: Callable[[Path, float], list[Frame]]
    detect_board: Callable[[np.ndarray], Box]
    crop: Callable[[np.ndarray, Box], np.ndarray]
    select_keyframes: Callable[[list[Frame]], list[Frame]]
    build: Callable[[list[list[RecognizedBlock]]], BoardDocument]

    @classmethod
    def default(cls) -> PipelineStages:
        # Imported here so the web app starts without OpenCV unless this engine is used.
        from whiteboard_extraction.reconstruction.document import build_document
        from whiteboard_extraction.video.keyframes import extract_keyframes
        from whiteboard_extraction.video.sampling import sample_frames
        from whiteboard_extraction.video.segmentation import crop_to_board, detect_board_region

        return cls(sample_frames, detect_board_region, crop_to_board, extract_keyframes, build_document)


class PipelineEngine:
    name = "Pipeline engine"

    def __init__(
        self,
        recognizer: Recognizer,
        stages: PipelineStages | None = None,
        sample_interval_seconds: float = 2.0,
        seconds_per_media_second: float = 0.5,
    ) -> None:
        self._recognizer = recognizer
        self._stages = stages or PipelineStages.default()
        self._interval = sample_interval_seconds
        self._rate = seconds_per_media_second

    def estimate_seconds(self, clip_seconds: float) -> float:
        return clip_seconds * self._rate

    def process(self, request: ProcessingRequest, reporter: ProgressReporter) -> BoardDocument:
        if request.media_path is None:
            raise RuntimeError("The original video file is not available for processing.")
        stages = self._stages

        frames = [
            (timestamp, frame)
            for timestamp, frame in stages.sample(request.media_path, self._interval)
            if request.start_seconds <= timestamp <= request.end_seconds
        ]
        reporter.report(0.3)

        boards: list[Frame] = []
        for index, (timestamp, frame) in enumerate(frames, start=1):
            boards.append((timestamp, stages.crop(frame, stages.detect_board(frame))))
            reporter.report(0.3 + 0.2 * index / len(frames))

        keyframes = stages.select_keyframes(boards)
        recognized: list[list[RecognizedBlock]] = []
        for index, (timestamp, image) in enumerate(keyframes, start=1):
            recognized.append(self._recognizer.recognize(timestamp, image))
            reporter.report(0.5 + 0.5 * index / len(keyframes))

        reporter.report(1.0)
        return stages.build(recognized)
