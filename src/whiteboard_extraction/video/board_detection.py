"""Detect and crop whiteboard/blackboard regions."""

import cv2
import numpy as np

def detect_board_region(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Return the (x, y, width, height) bounding box of the board."""
    if frame is None or frame.size == 0:
        raise ValueError("Frame is empty.")

    height, width = frame.shape[:2]
    frame_area = height * width
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Connect broken edges, like board frame interrupted by glare
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.08: # Too small
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: # Too small
            continue

        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon): # Boards are convex rectangles
            continue

        x, y, w, h = cv2.boundingRect(polygon)
        if w <= 0 or h <= 0: # Invalid shape
            continue

        aspect = w / h
        if not (0.5 <= aspect < 3.5): # Bad shape
            continue

        rectangularity = area / (w * h)
        if rectangularity < 0.65: # It's a square
            continue

        # Prefer large, rectangular regions without favoring a specific position
        score = (area / frame_area) * rectangularity
        candidates.append((score, (x, y, w, h)))
    if candidates: # Rank by score and return best
        return max(candidates, key=lambda item: item[0])[1]

    # Fallback: detect large, bright, low-saturation regions
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 100, 255]))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

    count, _, stats, _ = cv2.connectedComponentsWithStats(bright)
    regions = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < frame_area * 0.10: # Too small
            continue
        if w / max(h, 1) < 0.5: # Want width greater than height
            continue
        regions.append((area, (int(x), int(y), int(w), int(h))))
    if regions: # Rank by area and return largest
        return max(regions, key=lambda item: item[0])[1]

    raise ValueError("Board was not detected. Try another model.")


def crop_to_board(
    frame: np.ndarray,
    box: tuple[int, int, int, int]
) -> np.ndarray:
    """Crop to a bounding box clamped to frame dimensions."""
    x, y, w, h = box
    height, width = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Board bounding box is outside the frame.")

    return frame[y1:y2, x1:x2].copy()