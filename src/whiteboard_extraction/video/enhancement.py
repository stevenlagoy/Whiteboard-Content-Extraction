import cv2
import numpy as np


def enhance_board(board: np.ndarray) -> np.ndarray:
    """Improve local contrast while preserving a color output."""
    lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
    luminance, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    luminance = clahe.apply(luminance)

    return cv2.cvtColor(
        cv2.merge((luminance, a, b)),
        cv2.COLOR_LAB2BGR,
    )