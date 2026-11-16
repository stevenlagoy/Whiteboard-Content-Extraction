"""Template filters and globals: formatting and lazy lookups used by every page."""

from __future__ import annotations

from datetime import datetime

from flask import Flask, request, url_for

from .container import Services
from .domain.models import Video
from .timecodes import format_timecode


def format_datetime(value: datetime | None) -> str:
    if value is None:
        return "Not yet"
    hour = value.hour % 12 or 12
    return f"{value:%b} {value.day}, {value.year}, {hour}:{value:%M} {'AM' if value.hour < 12 else 'PM'}"


def format_date(value: datetime | None) -> str:
    return "Not yet" if value is None else f"{value:%b} {value.day}, {value.year}"


def format_bytes(count: int) -> str:
    gigabytes = count / 1024**3
    return f"{gigabytes:g} GB" if gigabytes >= 1 else f"{count / 1024**2:g} MB"


def current_url() -> str:
    """The page the user is on. Fragment requests report it in X-Page-URL."""
    reported = request.headers.get("X-Page-URL", "")
    if reported.startswith("/") and not reported.startswith("//"):
        return reported
    query = request.query_string.decode()
    return f"{request.path}?{query}" if query else request.path


def register_presentation(app: Flask, services: Services) -> None:
    app.add_template_filter(format_timecode, "timecode")
    app.add_template_filter(format_datetime, "datetime")
    app.add_template_filter(format_date, "date")

    @app.context_processor
    def template_helpers():
        def thumbnail_url(video: Video) -> str:
            if video.has_thumbnail:
                return url_for("files.thumbnail", video_id=video.id)
            return url_for("static", filename="placeholder_thumb.png")

        return {
            "library_tree": services.library.folder_tree,
            "folder_choices": services.library.folder_choices,
            "location_of": lambda video: services.library.folder_label(video.folder_id),
            "status_of": services.status.status_of,
            "task_panel": services.status.task_panel,
            "thumbnail_url": thumbnail_url,
            "current_url": current_url,
            "engine_name": services.status.engine_name,
        }
