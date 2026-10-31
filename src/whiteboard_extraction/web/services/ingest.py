"""Accepting an uploaded video: save it, read it, add it to the library, queue it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from ..domain.errors import ValidationError
from ..domain.models import Video, new_id
from ..storage import MediaStorage
from .jobs import JobService
from .library import LibraryService
from .media import MediaProbe

ALLOWED_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi")
_PAST_TOLERANCE = timedelta(minutes=1)


class UploadedFile(Protocol):
    filename: str | None

    def save(self, destination: Path) -> None: ...


@dataclass(frozen=True)
class IngestRequest:
    name: str | None = None
    folder_id: str | None = None
    trim_start: float = 0.0
    trim_end: float | None = None
    start_at: datetime | None = None


class IngestService:
    def __init__(
        self,
        library: LibraryService,
        jobs: JobService,
        probe: MediaProbe,
        storage: MediaStorage,
        clock: Callable[[], datetime],
    ) -> None:
        self._library = library
        self._jobs = jobs
        self._probe = probe
        self._storage = storage
        self._clock = clock

    def ingest(self, upload: UploadedFile, request: IngestRequest) -> Video:
        filename = upload.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValidationError("Choose a video file (MP4, MOV, MKV, WEBM, or AVI).")
        if request.start_at is not None and request.start_at < self._clock() - _PAST_TOLERANCE:
            raise ValidationError("Choose a start time in the future, or start now.")
        if request.folder_id is not None:
            self._library.get_folder(request.folder_id)

        video_id = new_id()
        media_file = f"{video_id}{suffix}"
        path = self._storage.source_path(media_file)
        video: Video | None = None
        try:
            upload.save(path)
            info = self._probe.probe(path)
            has_thumbnail = self._probe.write_thumbnail(path, self._storage.thumbnail_path(video_id))
            video = self._library.add_video(
                name=request.name or Path(filename).stem,
                folder_id=request.folder_id,
                length_seconds=info.duration_seconds,
                source_filename=Path(filename).name,
                media_file=media_file,
                has_thumbnail=has_thumbnail,
                trim_start=request.trim_start,
                trim_end=request.trim_end,
                video_id=video_id,
            )
            self._jobs.schedule(video.id, request.start_at)
        except Exception:
            if video is not None:
                self._library.delete_video(video.id)
            self._storage.remove_files(video_id, media_file)
            raise
        return video
