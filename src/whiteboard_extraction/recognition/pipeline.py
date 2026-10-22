"""The one place the pipeline calls a recognition method.

Video code produces timestamped keyframes; reconstruction and export code consume lists of
``RecognizedBlock``. Neither knows which method is active, so swapping a method never touches
them (user story 13).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .base import KINDS, RecognizedBlock, Recognizer

Keyframe = tuple[float, np.ndarray]  # (seconds into the video, BGR image)
ProgressCallback = Callable[[int, int], None]  # (keyframes done, keyframes in total)


def recognize_keyframes(
    recognizer: Recognizer,
    keyframes: Iterable[Keyframe],
    on_progress: ProgressCallback | None = None,
) -> list[list[RecognizedBlock]]:
    """Run the active method over every keyframe in order.

    Returns one block list per keyframe (empty for a blank board), which is exactly what
    ``reconstruction.document.build_document`` takes. Keyframes are read into memory first,
    so an iterator is fine and progress reports know the total from the start.
    """
    frames = list(keyframes)
    results: list[list[RecognizedBlock]] = []
    for index, (timestamp, image) in enumerate(frames, start=1):
        returned = recognizer.recognize(timestamp, image)
        try:
            blocks = list(returned)
        except TypeError:
            raise TypeError(
                f"{type(recognizer).__name__}.recognize returned {type(returned).__name__} at {timestamp:.1f}s; "
                "expected a list of RecognizedBlock."
            ) from None
        for block in blocks:
            if block.kind not in KINDS:
                raise ValueError(
                    f"{type(recognizer).__name__} returned kind {block.kind!r} at {timestamp:.1f}s; "
                    f"expected one of {', '.join(KINDS)}."
                )
        results.append(blocks)
        if on_progress is not None:
            on_progress(index, len(frames))
    return results
