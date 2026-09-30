"""Reduce a sequence of segmented board frames to stable, timestamped keyframes."""

from collections.abc import Iterable

import numpy as np


def change_fraction(prev: np.ndarray, curr: np.ndarray, pixel_delta: int = 25) -> float:
    """Fraction of pixels whose intensity changed by more than pixel_delta."""
    diff = np.abs(prev.astype("int16") - curr.astype("int16"))
    return float((diff > pixel_delta).mean())


def frames_differ(prev: np.ndarray, curr: np.ndarray, threshold: float = 0.02) -> bool:
    """Return True if curr has meaningfully new content compared to prev."""
    return change_fraction(prev, curr) > threshold


def extract_keyframes(segmented_frames: Iterable[tuple[float, np.ndarray]], threshold: float = 0.02, quota: int = 0) -> list[tuple[float, np.ndarray]]:
    """Collapse (timestamp, board_frame) pairs into keyframes.

    A keyframe is kept whenever content has changed enough from the
    last kept keyframe to represent new writing.
    """
    keyframes: list[tuple[float, np.ndarray]] = []
    for timestamp, frame in segmented_frames:
        if not keyframes or frames_differ(keyframes[-1][1], frame, threshold):
            keyframes.append((timestamp, frame))
    if len(keyframes) < quota:
        quota_ratio = 1 - abs(quota - len(keyframes)) / quota
        return extract_keyframes(segmented_frames, threshold * quota_ratio, quota)
    return keyframes