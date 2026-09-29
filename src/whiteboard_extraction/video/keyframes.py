"""Reduce a sequence of segmented board frames to stable, timestamped keyframes."""

import numpy as np


def frames_differ(prev: np.ndarray, curr: np.ndarray, threshold: float = 0.02) -> bool:
    """Return True if curr has meaningfully new content compared to prev."""
    diff = np.abs(prev.astype("int16") - curr.astype("int16"))
    return bool((diff > 25).mean() > threshold)


def extract_keyframes(segmented_frames: list[tuple[float, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    """Collapse (timestamp, board_frame) pairs into keyframes.

    A keyframe is kept whenever content has changed enough from the
    last kept keyframe to represent new writing.
    """
    if not segmented_frames:
        return []

    keyframes = [segmented_frames[0]]
    for timestamp, frame in segmented_frames[1:]:
        _, last_frame = keyframes[-1]
        if frames_differ(last_frame, frame):
            keyframes.append((timestamp, frame))
    return keyframes