"""Routes that change state. Every route here is a POST."""

from __future__ import annotations

from flask import Blueprint, flash, jsonify, redirect, request, url_for

from ..domain.errors import DomainError, ValidationError
from .forms import optional_id, parse_ingest_request, parse_ocr_blocks
from .pages import render_upload_page
from .support import ActionResult, action, services

actions = Blueprint("actions", __name__)


# ---------------------------------------------------------------- folders

@actions.post("/folders")
@action
def create_folder():
    folder = services().library.create_folder(request.form.get("name", ""), optional_id(request.form.get("parent_id")))
    return ActionResult(f'Created folder "{folder.name}".')


@actions.post("/folders/<folder_id>/rename")
@action
def rename_folder(folder_id: str):
    folder = services().library.rename_folder(folder_id, request.form.get("name", ""))
    return ActionResult(f'Renamed the folder to "{folder.name}".')


@actions.post("/folders/<folder_id>/delete")
@action
def delete_folder(folder_id: str):
    folder = services().library.delete_folder(folder_id)
    destination = url_for("pages.folder", folder_id=folder.parent_id) if folder.parent_id else url_for("pages.dashboard")
    return ActionResult(f'Deleted folder "{folder.name}".', destination)


# ----------------------------------------------------------------- videos

@actions.post("/videos/<video_id>/favorite")
@action
def set_favorite(video_id: str):
    favorite = request.form.get("favorite") == "true"
    video = services().library.set_favorite(video_id, favorite)
    verb = "Added" if favorite else "Removed"
    return ActionResult(f'{verb} "{video.name}" {"to" if favorite else "from"} favorites.')


@actions.post("/videos/<video_id>/rename")
@action
def rename_video(video_id: str):
    video = services().library.rename_video(video_id, request.form.get("name", ""))
    return ActionResult(f'Renamed the video to "{video.name}".')


@actions.post("/videos/<video_id>/move")
@action
def move_video(video_id: str):
    svc = services()
    folder_id = optional_id(request.form.get("folder_id"))
    video = svc.library.move_video(video_id, folder_id, optional_id(request.form.get("before_id")))
    return ActionResult(f'Moved "{video.name}" to {svc.library.folder_label(folder_id)}.')


@actions.post("/videos/<video_id>/shift")
@action
def shift_video(video_id: str):
    video = services().library.shift_video(video_id, request.form.get("direction", ""))
    return ActionResult(f'Moved "{video.name}" {request.form.get("direction")}.')


@actions.post("/videos/<video_id>/delete")
@action
def delete_video(video_id: str):
    svc = services()
    video = svc.library.delete_video(video_id)
    svc.storage.remove_files(video.id, video.media_file)
    return ActionResult(f'Deleted "{video.name}".')


@actions.post("/videos/<video_id>/ocr")
@action
def update_ocr(video_id: str):
    video = services().library.update_ocr(video_id, parse_ocr_blocks(request.form))
    return ActionResult(f'Saved the text for "{video.name}".')


@actions.post("/videos/<video_id>/process")
@action
def process_video(video_id: str):
    svc = services()
    video = svc.library.get_video(video_id)
    svc.jobs.schedule(video_id)
    return ActionResult(f'Added "{video.name}" to the processing queue.')


# ------------------------------------------------------------------- jobs

@actions.post("/jobs")
def create_job():
    """Upload a video and queue it. Answers JSON when the client asks for it (the upload page does)."""
    wants_json = request.accept_mimetypes.best == "application/json"
    try:
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            raise ValidationError("Choose a video file to upload.")
        video = services().ingest.ingest(upload, parse_ingest_request(request.form))
    except DomainError as error:
        if wants_json:
            return jsonify(error=error.message), 400
        return render_upload_page(error.message, request.form, 400)
    flash(f'Added "{video.name}" to the processing queue.', "success")
    destination = url_for("pages.dashboard")
    return jsonify(redirect=destination) if wants_json else redirect(destination, code=303)


@actions.post("/jobs/<job_id>/cancel")
@action
def cancel_job(job_id: str):
    services().jobs.cancel(job_id)
    return ActionResult("Cancelled the task.")

