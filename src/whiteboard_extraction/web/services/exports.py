"""Turn a video's recognized text into downloadable documents."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from whiteboard_extraction.export.docx_export import export_docx
from whiteboard_extraction.export.pdf_export import export_pdf
from whiteboard_extraction.reconstruction.document import BoardDocument

from ..domain.errors import ConflictError, NotFoundError
from .library import LibraryService

Writer = Callable[[BoardDocument, Path], Path]


@dataclass(frozen=True)
class ExportFormat:
    key: str
    label: str
    extension: str
    mimetype: str
    write: Writer


@dataclass(frozen=True)
class ExportedFile:
    filename: str
    mimetype: str
    content: bytes


DEFAULT_FORMATS = (
    ExportFormat(
        "docx", "Word document", ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", export_docx,
    ),
    ExportFormat("pdf", "PDF", ".pdf", "application/pdf", export_pdf),
)


def _download_stem(name: str) -> str:
    return re.sub(r"[^\w\-. ]+", "", name).strip() or "lecture"


class ExportService:
    def __init__(self, library: LibraryService, formats: tuple[ExportFormat, ...] = DEFAULT_FORMATS) -> None:
        self._library = library
        self._formats = {fmt.key: fmt for fmt in formats}

    @property
    def formats(self) -> tuple[ExportFormat, ...]:
        return tuple(self._formats.values())

    def export(self, video_id: str, key: str) -> ExportedFile:
        fmt = self._formats.get(key)
        if fmt is None:
            raise NotFoundError("That export format does not exist.")
        video = self._library.get_video(video_id)
        if not video.ocr_blocks:
            raise ConflictError("There is no recognized text to export yet.")
        document = BoardDocument(blocks=list(video.ocr_blocks))
        with tempfile.TemporaryDirectory() as folder:
            content = fmt.write(document, Path(folder) / f"export{fmt.extension}").read_bytes()
        self._library.record_export(video_id)
        return ExportedFile(_download_stem(video.name) + fmt.extension, fmt.mimetype, content)
