"""Pages that display the library. Every route here is a GET."""

from __future__ import annotations

from flask import Blueprint, render_template, request

from ..presentation import format_bytes
from .support import services

pages = Blueprint("pages", __name__)


@pages.get("/")
def dashboard():
    svc = services()
    return render_template(
        "dashboard.html",
        status=svc.status.dashboard(),
        recents=svc.library.recent(svc.config.recent_limit),
        favorites=svc.library.favorites(),
    )


@pages.get("/folders/<folder_id>")
def folder(folder_id: str):
    svc = services()
    current = svc.library.get_folder(folder_id)
    return render_template(
        "folder.html",
        folder=current,
        active_folder_id=current.id,
        path=svc.library.folder_path(current.id),
        subfolders=svc.library.child_folders(current.id),
        videos=svc.library.videos_in(current.id),
    )


@pages.get("/videos/<video_id>")
def video(video_id: str):
    svc = services()
    current = svc.library.get_video(video_id)
    return render_template(
        "video.html",
        video=current,
        active_video_id=current.id,
        path=svc.library.folder_path(current.folder_id),
        formats=svc.exports.formats,
        has_media=current.media_file is not None,
    )


@pages.get("/upload")
def upload():
    return render_upload_page()


@pages.get("/tasks/panel")
def tasks_panel():
    return render_template("components/_tasks_list.html")


def render_upload_page(error: str | None = None, form=None, status: int = 200):
    svc = services()
    selected = (form or request.args).get("folder_id", request.args.get("folder", ""))
    page = render_template(
        "upload.html",
        error=error,
        form=form or {},
        selected_folder=selected,
        max_upload=format_bytes(svc.config.max_upload_bytes),
        earliest_start=svc.clock().strftime("%Y-%m-%dT%H:%M"),
    )
    return page, status
