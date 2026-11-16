"""Reading video files: length and thumbnails."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..domain.errors import MediaError


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float


class MediaProbe(Protocol):
    def probe(self, path: Path) -> MediaInfo:
        """Raises MediaError when the file is not a readable video."""
        ...

    def write_thumbnail(self, path: Path, destination: Path, at_seconds: float = 1.0) -> bool:
        """Save a JPEG frame; returns False when no frame could be read."""
        ...


class OpenCVMediaProbe:
    def probe(self, path: Path) -> MediaInfo:
        cv2 = self._cv2()
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise MediaError("That file could not be read as a video.")
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            capture.release()
        if fps <= 0 or frames <= 0:
            raise MediaError("That video has no readable length.")
        return MediaInfo(duration_seconds=frames / fps)

    def write_thumbnail(self, path: Path, destination: Path, at_seconds: float = 1.0) -> bool:
        cv2 = self._cv2()
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, at_seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            return False
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (320, max(round(320 * height / width), 1)))
        return bool(cv2.imwrite(str(destination), frame))

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as error:
            raise MediaError("OpenCV is required to read video files (pip install opencv-python-headless).") from error
        return cv2
