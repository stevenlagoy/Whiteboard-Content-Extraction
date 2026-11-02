"""Helpers for saving and viewing pipeline intermediates."""

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def make_contact_sheet(
    images: list[tuple[str, np.ndarray]],
    thumbnail_size: tuple[int, int] = (240, 135),
    columns: int = 4,
    label_height: int = 28,
    background_color: tuple[int, int, int] = (35, 35, 35),
) -> np.ndarray:
    """Create a contact sheet from (label, image) pairs."""
    if columns < 1:
        raise ValueError("columns must be at least 1")

    width, height = thumbnail_size
    cell_height = height + label_height
    rows = max(1, (len(images) + columns - 1) // columns)

    sheet = np.full(
        (rows * cell_height, columns * width, 3),
        background_color,
        dtype=np.uint8,
    )

    for index, (label, image) in enumerate(images):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        thumbnail = cv2.resize(image, thumbnail_size)
        row, column = divmod(index, columns)
        x, y = column * width, row * cell_height

        sheet[y:y + height, x:x + width] = thumbnail
        cv2.putText(
            sheet,
            label,
            (x + 6, y + height + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return sheet


def create_directory_contact_sheet(
    directory: str | Path,
    thumbnail_size: tuple[int, int] = (240, 135),
    columns: int = 4,
) -> Path | None:
    """Create contact_sheet.jpg for every image in a directory.

    Uses each image's filename as its label. Excludes existing contact sheets.
    Returns the contact-sheet path, or None if no images are found.
    """
    directory = Path(directory)

    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.lower() != "contact_sheet.jpg"
    )

    if not image_paths:
        return None

    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            images.append((path.stem, image))

    if not images:
        return None

    sheet = make_contact_sheet(
        images,
        thumbnail_size=thumbnail_size,
        columns=columns,
    )

    output_path = directory / "contact_sheet.jpg"
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"Failed to save contact sheet: {output_path}")

    return output_path