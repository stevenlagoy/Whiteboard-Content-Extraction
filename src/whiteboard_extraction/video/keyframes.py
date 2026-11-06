"""Select stable, timestamped keyframes from board-region images."""

from collections.abc import Iterable

import cv2
import numpy as np


def change_fraction(previous: np.ndarray, current: np.ndarray, pixel_delta: int = 25) -> float:
    """Fraction of pixels whose intensity changed by more than ``pixel_delta``."""
    if previous.shape[:2] != current.shape[:2]:
        raise ValueError("Frames must have same spatial dimensions.")
    previous_gray = _to_gray(previous)
    current_gray = _to_gray(current)
    diff = np.abs(previous_gray.astype(np.int16) - current_gray.astype(np.int16))
    return float((diff > pixel_delta).mean())


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Frames must be grayscale or BGR images.")


def structural_similarity(previous: np.ndarray, current: np.ndarray) -> float:
    """Return SSIM in [-1, 1]; use a safe fallback if scikit-image is absent.

    SSIM is calculated on grayscale images. Very small images are compared by
    normalized absolute error because SSIM requires a minimum window size.
    """
    if previous.shape[:2] != current.shape[:2]:
        raise ValueError("Frames must have same spatial dimensions.")
    first = _to_gray(previous)
    second = _to_gray(current)
    if min(first.shape[:2]) < 7:
        error = np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))) / 255.0
        return float(1.0 - error)
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        error = np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))) / 255.0
        return float(1.0 - error)
    score, _ = ssim(first, second, data_range=255, full=True)
    return float(score)


def frames_differ(
    previous: np.ndarray,
    current: np.ndarray,
    threshold: float = 0.02,
    pixel_delta: int = 25,
) -> bool:
    """Detect meaningful writing changes using pixel change and SSIM.

    The pixel-change test remains the primary guard against tiny changes.
    SSIM adds sensitivity to structural changes in a board crop while avoiding
    triggering on very small uniform intensity shifts.
    """
    changed = change_fraction(previous, current, pixel_delta)
    if changed > threshold:
        return True
    # SSIM is a secondary signal for localized structural changes. Requiring
    # some changed pixels prevents uniform exposure shifts from becoming keys.
    return changed > 0 and structural_similarity(previous, current) < (1.0 - threshold)


def extract_keyframes(
    segmented_frames: Iterable[tuple[float, np.ndarray]],
    threshold: float = 0.02,
    quota: int = 0,
) -> list[tuple[float, np.ndarray]]:
    """Reduce timestamp/image pairs to meaningful keyframes.

    Pass board crops (rather than full video frames) to avoid selecting frames
    solely because a lecturer or background object moved. If ``quota`` is set,
    the threshold is progressively relaxed until the target is reached or the
    threshold reaches its lower bound. The quota is a target, not a guarantee.
    """
    frames = list(segmented_frames)
    if not frames:
        return []
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
