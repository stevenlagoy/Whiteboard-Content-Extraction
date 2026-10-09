"""Reduce a sequence of segmented board frames to stable, timestamped keyframes."""

from collections.abc import Iterable

import numpy as np


def change_fraction(previous: np.ndarray, current: np.ndarray, pixel_delta: int = 25) -> float:
    """Fraction of pixels whose intensity changed by more than pixel_delta."""
    if previous.shape != current.shape:
        raise ValueError("Frames must have same shape.")
    
    diff = np.abs(previous.astype("int16") - current.astype("int16"))
    return float((diff > pixel_delta).mean())


def frames_differ(previous: np.ndarray, current: np.ndarray, threshold: float = 0.02) -> bool:
    """Return True if current has meaningfully new content compared to previous."""
    return change_fraction(previous, current) > threshold


def extract_keyframes(segmented_frames: Iterable[tuple[float, np.ndarray]], threshold: float = 0.02, quota: int = 0) -> list[tuple[float, np.ndarray]]:
    """Collapse (timestamp, board_frame) pairs into keyframes.

    A keyframe is kept whenever content has changed enough from the
    last kept keyframe to represent new writing.

    If quota > 0, lower the threshold as needed to approach the requested count. Quota is a target and not a guarantee.
    """
    frames = list(segmented_frames)
    if not frames: return []

    if threshold < 0:
        raise ValueError("Threshold must be nonnegative.")
    if quota < 0:
        raise ValueError("Quota must be nonnegative.")

    def select(current_threshold: float) -> list[tuple[float, np.ndarray]]:
        selected = [frames[0]]

        for timestamp, frame in frames[1:]:
            if frames_differ(selected[-1][1], frame, current_threshold):
                selected.append((timestamp, frame))

        return selected

    effective_threshold = threshold
    keyframes = select(effective_threshold)

    while quota > len(keyframes) and effective_threshold > 1e-6:
        effective_threshold /= 2
        keyframes = select(effective_threshold)

    return keyframes