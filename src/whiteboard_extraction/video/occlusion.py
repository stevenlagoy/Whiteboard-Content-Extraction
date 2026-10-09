"""Segment occluding objects and produce a mask for the board."""

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

_model = None

# COCO classes likely to represent occluders
OCCLUDER_CLASSES = {
    "person",
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "book",
    "scissors",
    "cell phone",
    "remote",
    "keyboard",
    "laptop",
}


def detect_occluding_objects(frame: np.ndarray) -> np.ndarray:
    """Return a uint8 mask: 255 = visible board, 0 = occluded."""
    global _model

    if frame is None or frame.size == 0:
        raise ValueError("Frame is empty.")

    if _model is None:
        _model = YOLO("yolo11n-seg.pt")

    # YOLO expects an image in BGR or RGB; OpenCV frames are BGR
    result = [p for p in _model.predict(
        source=frame,
        verbose=False,
        conf=0.25,
        imgsz=640,
    )][0]
    assert isinstance(result, Results)

    visible_mask = np.full(frame.shape[:2], 255, dtype=np.uint8)

    if result.masks is None or result.boxes is None:
        return visible_mask

    classes = result.boxes.cls.cpu().numpy().astype(int)
    masks = result.masks.data.cpu().numpy()

    for class_id, object_mask in zip(classes, masks):
        class_name = result.names[class_id]
        if class_name not in OCCLUDER_CLASSES:
            continue

        # Model masks may be lower resolution than original frame
        object_mask = cv2.resize(
            object_mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Slight dilation covers object edges missed by segmentation
        object_mask = cv2.dilate(object_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        visible_mask[object_mask > 0] = 0

    return visible_mask