import io

from docx import Document

from whiteboard_extraction.recognition.base import RecognizedBlock


def stored_video(svc, content=b"0123456789", **kwargs):
    video = svc.library.add_video(name="Clip", folder_id=None, length_seconds=60, media_file="clip.mp4", **kwargs)
    svc.storage.source_path("clip.mp4").write_bytes(content)
    return video


def test_media_is_served_with_range_support(client, svc):
    video = stored_video(svc)
    full = client.get(f"/videos/{video.id}/media")
    assert full.status_code == 200 and full.data == b"0123456789" and full.mimetype == "video/mp4"
    part = client.get(f"/videos/{video.id}/media", headers={"Range": "bytes=2-5"})
    assert part.status_code == 206 and part.data == b"2345" and part.headers["Content-Range"] == "bytes 2-5/10"


def test_media_404_when_nothing_is_stored(client, svc):
    no_file = svc.library.add_video(name="Demo", folder_id=None, length_seconds=10)
    assert client.get(f"/videos/{no_file.id}/media").status_code == 404
    gone = svc.library.add_video(name="Gone", folder_id=None, length_seconds=10, media_file="missing.mp4")
    assert client.get(f"/videos/{gone.id}/media").status_code == 404
    assert client.get("/videos/unknown/media").status_code == 404


def test_thumbnail_serves_the_image_or_the_placeholder(client, svc):
    with_thumb = svc.library.add_video(name="T", folder_id=None, length_seconds=10, has_thumbnail=True)
    svc.storage.thumbnail_path(with_thumb.id).write_bytes(b"jpeg")
    image = client.get(f"/videos/{with_thumb.id}/thumbnail")
    assert image.status_code == 200 and image.mimetype == "image/jpeg" and image.data == b"jpeg"
    plain = svc.library.add_video(name="P", folder_id=None, length_seconds=10)
    redirect = client.get(f"/videos/{plain.id}/thumbnail")
    assert redirect.status_code == 302 and redirect.headers["Location"].endswith("/static/placeholder_thumb.png")
    lost = svc.library.add_video(name="L", folder_id=None, length_seconds=10, has_thumbnail=True)
    assert client.get(f"/videos/{lost.id}/thumbnail").status_code == 302


def test_rows_use_the_generated_thumbnail_when_there_is_one(client, svc):
    with_thumb = svc.library.add_video(name="T", folder_id=None, length_seconds=10, has_thumbnail=True)
    html = client.get("/").get_data(as_text=True)
    assert f'src="/videos/{with_thumb.id}/thumbnail"' in html
    svc.library.add_video(name="P", folder_id=None, length_seconds=10)
    assert "/static/placeholder_thumb.png" in client.get("/").get_data(as_text=True)


def test_export_downloads(client, svc):
    video = svc.library.add_video(name="Notes", folder_id=None, length_seconds=60)
    svc.library.update_ocr(video.id, [RecognizedBlock("Eigenvalues", "prose", 12)])
    docx = client.get(f"/videos/{video.id}/export/docx")
    assert docx.status_code == 200 and "attachment" in docx.headers["Content-Disposition"] and "Notes.docx" in docx.headers["Content-Disposition"]
    assert "Eigenvalues" in "\n".join(p.text for p in Document(io.BytesIO(docx.data)).paragraphs)
    pdf = client.get(f"/videos/{video.id}/export/pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF") and pdf.mimetype == "application/pdf"
    assert svc.library.get_video(video.id).exported_at is not None
    assert "Exported" in client.get(f"/videos/{video.id}").get_data(as_text=True)


def test_export_refusals(client, svc):
    empty = svc.library.add_video(name="Empty", folder_id=None, length_seconds=60)
    refused = client.get(f"/videos/{empty.id}/export/pdf")
    assert refused.status_code == 409 and "no recognized text" in refused.get_data(as_text=True)
    svc.library.update_ocr(empty.id, [RecognizedBlock("x", "prose", 1)])
    assert client.get(f"/videos/{empty.id}/export/odt").status_code == 404
    assert client.get("/videos/unknown/export/pdf").status_code == 404
