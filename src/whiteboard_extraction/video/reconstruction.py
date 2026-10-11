import cv2
import numpy as np


def reconstruct_occlusions(
    current: np.ndarray,
    reference: np.ndarray,
    visible_mask: np.ndarray,
) -> np.ndarray:
    """Fill masked regions using a reference crop."""
    if current.shape != reference.shape:
        raise ValueError("Current and reference images must match.")
    if visible_mask.shape != current.shape[:2]:
        raise ValueError("Mask dimensions must match the images.")

    # 255 = visible; 0 = occluded
    return np.where(
        visible_mask[:, :, None] > 0,
        current,
        reference,
    ).astype(np.uint8)