"""Sample frames from an input lecture video at a fixed interval."""

from pathlib import Path
from collections.abc import Iterator

import cv2
import numpy as np


def sample_frames(video_path: Path | str, interval_seconds: float = 2.0) -> Iterator[tuple[float, np.ndarray]]:
    """
    Pull frames from a video at a fixed time interval.

    Returns a list of (timestamp_seconds, frame) pairs.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps * interval_seconds), 1)
    frame_index = 0
    try:
        while cap.grab():
            if frame_index % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield frame_index / fps, frame
            frame_index += 1
    finally:
        cap.release()
