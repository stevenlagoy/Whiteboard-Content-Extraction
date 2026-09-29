"""Detect and crop the whiteboard/blackboard region from a frame."""

import numpy as np


def detect_board_region(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the writing surface within a frame.

    Returns an (x, y, width, height) bounding box.
    """
    raise NotImplementedError("Board detection not yet implemented.")


def crop_to_board(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a frame to the given board bounding box."""
    x, y, w, h = box
    return frame[y:y + h, x:x + w]