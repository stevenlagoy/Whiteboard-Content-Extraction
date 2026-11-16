import io

import pytest
from docx import Document

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.web.domain.errors import ConflictError, NotFoundError
from whiteboard_extraction.web.services.exports import ExportService


@pytest.fixture
def exports(library):
    return ExportService(library)


def video_with_text(library, make_video, name="Video 1", text="Linear maps"):
    video = make_video(name, length=300)
    library.update_ocr(video.id, [RecognizedBlock(text, "prose", 65)])
    return video


def test_docx_export_contains_the_text_and_timestamp(exports, library, make_video):
    video = video_with_text(library, make_video)
    exported = exports.export(video.id, "docx")
    assert exported.filename == "Video 1.docx" and exported.content.startswith(b"PK")
    text = "\n".join(p.text for p in Document(io.BytesIO(exported.content)).paragraphs)
    assert "Linear maps" in text and "01:05" in text


def test_pdf_export_is_a_pdf(exports, library, make_video):
    exported = exports.export(video_with_text(library, make_video).id, "pdf")
    assert exported.filename == "Video 1.pdf" and exported.mimetype == "application/pdf"
    assert exported.content.startswith(b"%PDF")


def test_pdf_export_survives_markup_characters(exports, library, make_video):
    video = video_with_text(library, make_video, text="if x < 5 & y > 2 then <b>unclosed")
    assert exports.export(video.id, "pdf").content.startswith(b"%PDF")


def test_export_records_the_time(exports, library, make_video, clock):
    video = video_with_text(library, make_video)
    clock.advance(hours=1)
    exports.export(video.id, "docx")
    assert library.get_video(video.id).exported_at == clock.now


def test_filenames_are_made_safe(exports, library, make_video):
    video = video_with_text(library, make_video, name='Week 1/2: "Intro"?')
    assert exports.export(video.id, "pdf").filename == "Week 12 Intro.pdf"
    symbols = video_with_text(library, make_video, name="???")
    assert exports.export(symbols.id, "pdf").filename == "lecture.pdf"


def test_export_refusals(exports, library, make_video):
    empty = make_video()
    with pytest.raises(ConflictError):
        exports.export(empty.id, "pdf")
    with pytest.raises(NotFoundError):
        exports.export(video_with_text(library, make_video).id, "odt")
    with pytest.raises(NotFoundError):
        exports.export("missing", "pdf")
    assert library.get_video(empty.id).exported_at is None


def test_formats_can_be_extended(library, make_video):
    from whiteboard_extraction.web.services.exports import DEFAULT_FORMATS, ExportFormat

    def write_text(document, path):
        path.write_text(document.as_plain_text())
        return path

    custom = ExportFormat("txt", "Plain text", ".txt", "text/plain", write_text)
    service = ExportService(library, (*DEFAULT_FORMATS, custom))
    assert [f.key for f in service.formats] == ["docx", "pdf", "txt"]
    assert b"Linear maps" in service.export(video_with_text(library, make_video).id, "txt").content
