"""Routes that return files: the video itself, thumbnails, and exports."""

from __future__ import annotations

import mimetypes
from io import BytesIO

from flask import Blueprint, abort, redirect, send_file, url_for

from .support import services

files = Blueprint("files", __name__)


@files.get("/videos/<video_id>/media")
def media(video_id: str):
    svc = services()
    video = svc.library.get_video(video_id)
    if video.media_file is None:
        abort(404)
    path = svc.storage.source_path(video.media_file)
    if not path.exists():
        abort(404)
    mimetype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_file(path, mimetype=mimetype, conditional=True)


@files.get("/videos/<video_id>/thumbnail")
def thumbnail(video_id: str):
    svc = services()
    video = svc.library.get_video(video_id)
    path = svc.storage.thumbnail_path(video.id)
    if not video.has_thumbnail or not path.exists():
        return redirect(url_for("static", filename="placeholder_thumb.png"))
    return send_file(path, mimetype="image/jpeg", max_age=3600)


@files.get("/videos/<video_id>/export/<format_key>")
def export(video_id: str, format_key: str):
    exported = services().exports.export(video_id, format_key)
    return send_file(
        BytesIO(exported.content),
        mimetype=exported.mimetype,
        as_attachment=True,
        download_name=exported.filename,
    )
