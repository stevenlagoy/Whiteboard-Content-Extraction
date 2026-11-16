"""Turning submitted form fields into typed service arguments."""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

from werkzeug.datastructures import MultiDict

from whiteboard_extraction.recognition.base import RecognizedBlock

from ..domain.errors import ValidationError
from ..services.ingest import IngestRequest
from ..timecodes import parse_timecode


def optional_id(raw: str | None) -> str | None:
    """Folder ids arrive as text; an empty value means the top level."""
    return raw or None


def parse_ingest_request(form: Mapping[str, str]) -> IngestRequest:
    trim_start_text = form.get("trim_start", "").strip()
    trim_end_text = form.get("trim_end", "").strip()
    start_at: datetime | None = None
    if form.get("start_mode", "now") == "later":
        try:
            start_at = datetime.fromisoformat(form.get("start_at", "").strip())
        except ValueError:
            raise ValidationError("Choose the date and time to start processing.") from None
    return IngestRequest(
        name=form.get("name", "").strip() or None,
        folder_id=optional_id(form.get("folder_id")),
        trim_start=parse_timecode(trim_start_text, "Trim start") if trim_start_text else 0.0,
        trim_end=parse_timecode(trim_end_text, "Trim end") if trim_end_text else None,
        start_at=start_at,
    )


def parse_ocr_blocks(form: MultiDict) -> list[RecognizedBlock]:
    """Rebuild the edited block list. Rows are parallel fields; removed or blank rows are dropped."""
    timestamps, kinds, texts = form.getlist("timestamp"), form.getlist("kind"), form.getlist("text")
    if not len(timestamps) == len(kinds) == len(texts):
        raise ValidationError("The text blocks could not be read. Reload the page and try again.")
    removed = set(form.getlist("remove"))
    blocks: list[RecognizedBlock] = []
    for index, (stamp, kind, text) in enumerate(zip(timestamps, kinds, texts)):
        if str(index) in removed or not text.strip():
            continue
        blocks.append(RecognizedBlock(text=text, kind=kind, timestamp=parse_timecode(stamp, f"Timestamp {index + 1}")))
    return blocks
