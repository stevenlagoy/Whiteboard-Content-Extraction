"""Conversion between seconds and the H:MM:SS text people read and type."""

from __future__ import annotations

import re

from .domain.errors import ValidationError

_PATTERN = re.compile(r"\d+(?::\d{1,2}){0,2}(?:\.\d+)?")


def format_timecode(seconds: float) -> str:
    """Format seconds as M:SS, or H:MM:SS from one hour upward."""
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02}:{secs:02}"
    return f"{minutes}:{secs:02}"


def parse_timecode(text: str, field: str = "Time") -> float:
    """Parse 90, 1:30, or 1:02:03 (optionally with decimals) into seconds."""
    cleaned = text.strip()
    if not _PATTERN.fullmatch(cleaned):
        raise ValidationError(f"{field} must look like 90, 1:30, or 1:02:03.")
    parts = cleaned.split(":")
    if any(float(part) >= 60 for part in parts[1:]):
        raise ValidationError(f"{field} needs minutes and seconds below 60.")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total
