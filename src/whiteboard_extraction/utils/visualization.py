"""Helpers for saving and viewing pipeline intermediates."""

from collections.abc import Sequence
from pathlib import Path


import cv2
import numpy as np

Keyframe = tuple[float, np.ndarray]


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_keyframes(keyframes: Sequence[Keyframe], output_dir: Path) -> list[Path]:
    """Write each keyframe as keyframe_001_00-00-12.jpg, and so on."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (timestamp, frame) in enumerate(keyframes, start=1):
        label = format_timestamp(timestamp).replace(":", "-")
        path = output_dir / f"keyframe_{index:03d}_{label}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def make_contact_sheet(
    keyframes: Sequence[Keyframe],
    columns: int = 4,
    tile_width: int = 320,
    max_tiles: int = 48,
) -> np.ndarray:
    """Tile timestamp-labelled thumbnails into one image."""
    if not keyframes:
        raise ValueError("No keyframes to display.")

    if len(keyframes) > max_tiles:
        picks = np.linspace(0, len(keyframes) - 1, max_tiles).astype(int)
        keyframes = [keyframes[i] for i in picks]

    height, width = keyframes[0][1].shape[:2]
    tile_height = max(int(height * tile_width / width), 1)

    tiles = []
    for timestamp, frame in keyframes:
        tile = cv2.resize(frame, (tile_width, tile_height))
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        text = format_timestamp(timestamp)
        cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.599, (0, 0, 0), 3)
        cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)

    rows = -(-len(tiles) // columns)
    tiles += [np.zeros_like(tiles[0])] * (rows * columns - len(tiles))
    return np.vstack([np.hstack(tiles[r * columns:(r + 1) * columns]) for r in range(rows)])
