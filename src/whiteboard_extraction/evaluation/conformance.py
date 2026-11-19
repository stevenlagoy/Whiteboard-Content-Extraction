"""Check that a recognition method's output has the form the rest of the pipeline expects.

Every method returns ``list[RecognizedBlock]`` for one keyframe. Reconstruction sorts blocks by
``timestamp`` and export branches on ``kind``, so output in the wrong form fails far from the
method that produced it. ``block_problems`` says what is wrong with one keyframe's output in
plain words, and ``tests/test_recognizer_conformance.py`` runs it against every method in the
registry.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from whiteboard_extraction.recognition.base import KINDS, RecognizedBlock


def _is_number(value: Any) -> bool:
    """A finite int or float (NumPy scalars included); bool does not count."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def block_problems(blocks: Any, timestamp: float) -> list[str]:
    """List every way ``blocks`` breaks the output contract; an empty list means it conforms.

    The contract for what ``recognize(timestamp, image)`` returns:

    * a ``list`` of ``RecognizedBlock``, empty for a blank board;
    * ``text`` is a ``str`` with something in it (leave empty blocks out);
    * ``kind`` is one of ``KINDS``, spelled exactly;
    * ``timestamp`` is the keyframe's timestamp, the number that was passed in;
    * ``confidence`` is ``None`` or a number from 0 to 1.
    """
    if not isinstance(blocks, list):
        return [f"returned {type(blocks).__name__}, expected a list of RecognizedBlock"]
    problems: list[str] = []
    for index, block in enumerate(blocks):
        where = f"block {index}"
        if not isinstance(block, RecognizedBlock):
            problems.append(f"{where} is {type(block).__name__}, expected RecognizedBlock")
            continue
        if not isinstance(block.text, str):
            problems.append(f"{where}: text is {type(block.text).__name__}, expected str")
        elif not block.text.strip():
            problems.append(f"{where}: text is blank; leave empty blocks out")
        if not isinstance(block.kind, str) or block.kind not in KINDS:
            problems.append(f"{where}: kind is {block.kind!r}, expected one of {', '.join(KINDS)}")
        if not _is_number(block.timestamp):
            problems.append(f"{where}: timestamp is {block.timestamp!r}, expected a number of seconds")
        elif not math.isclose(block.timestamp, timestamp, abs_tol=1e-6):
            problems.append(f"{where}: timestamp is {block.timestamp!r}, expected the keyframe's {timestamp!r}")
        if block.confidence is not None and not (_is_number(block.confidence) and 0 <= block.confidence <= 1):
            problems.append(f"{where}: confidence is {block.confidence!r}, expected None or a number from 0 to 1")
    return problems
