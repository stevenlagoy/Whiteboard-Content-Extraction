"""Where uploaded files and thumbnails live on disk."""

from __future__ import annotations

from pathlib import Path


class MediaStorage:
    def __init__(self, root: Path) -> None:
        self.uploads = root / "uploads"
        self.thumbnails = root / "thumbnails"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.thumbnails.mkdir(parents=True, exist_ok=True)

    def source_path(self, media_file: str) -> Path:
        return self.uploads / media_file

    def thumbnail_path(self, video_id: str) -> Path:
        return self.thumbnails / f"{video_id}.jpg"

    def remove_files(self, video_id: str, media_file: str | None) -> None:
        if media_file:
            self.source_path(media_file).unlink(missing_ok=True)
        self.thumbnail_path(video_id).unlink(missing_ok=True)
