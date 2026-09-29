"""Sample frames from an input lecture video at a fixed interval."""

from pathlib import Path

import cv2
import numpy as np


def sample_frames(video_path: Path, interval_seconds: float = 2.0) -> list[tuple[float, np.ndarray]]:
    """Pull frames from a video at a fixed time interval.

    Returns a list of (timestamp_seconds, frame) pairs.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps * interval_seconds), 1)

    frames = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            frames.append((frame_index / fps, frame))
        frame_index += 1

    cap.release()
    return frames