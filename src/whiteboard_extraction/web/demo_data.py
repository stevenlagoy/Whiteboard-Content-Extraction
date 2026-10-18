"""Sample library shown on first run so every screen has something to display."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from whiteboard_extraction.recognition.base import RecognizedBlock

from .domain.models import JobState, ProcessingJob
from .repositories.base import Repository
from .services.library import LibraryService


def _blocks(length: float) -> list[RecognizedBlock]:
    return [
        RecognizedBlock("Linear transformations preserve addition and scaling", "prose", length * 0.1),
        RecognizedBlock(r"T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})", "equation", length * 0.35),
        RecognizedBlock(r"\det(A - \lambda I) = 0", "equation", length * 0.7),
    ]


def seed_demo_library(library: LibraryService, repo: Repository, now: datetime) -> None:
    """Fill an empty library. Videos have no media file, so the player is replaced by a notice."""
    cs = library.create_folder("CS_593_application_deep_learning", folder_id="demo-cs-593").id
    week1 = library.create_folder("Week 1", cs, folder_id="demo-week-1").id
    ma = library.create_folder("MA_375_elem_linear_algebra", folder_id="demo-ma-375").id

    def video(key: str, name: str, folder: str | None, length: float, age: timedelta, **flags):
        created = library.add_video(name=name, folder_id=folder, length_seconds=length, video_id=key)
        processed = flags.pop("processed", True)
        stamped = replace(created, uploaded_at=now - age, favorite=flags.pop("favorite", False))
        if processed:
            stamped = replace(
                stamped,
                ocr_blocks=tuple(_blocks(length)),
                processed_at=now - age + timedelta(minutes=18),
                exported_at=flags.pop("exported_at", None),
            )
        repo.save_video(stamped)

    video("demo-video-3", "3", cs, 61, timedelta(days=20), processed=False)
    video("demo-video-2", "Video 2 with a medium length", cs, 86 * 60 + 44, timedelta(days=130))
    video("demo-video-1", "Video 1", cs, 95 * 60 + 16, timedelta(days=138), favorite=True)
    video("demo-video-intro", "Introduction to the course", week1, 48 * 60 + 5, timedelta(days=140))
    video("demo-video-6", "Video 6", ma, 90 * 60 + 3, timedelta(hours=3), favorite=True)
    video("demo-video-5", "Video 5 with a longer title which exceeds the width of the paragraph element",
          ma, 75 * 60 + 53, timedelta(days=5))
    video("demo-video-4", "Video 4", ma, 12 * 60 + 30, timedelta(days=5, hours=2), processed=False)

    repo.save_job(ProcessingJob(
        id="demo-job-4", video_id="demo-video-4", state=JobState.FAILED,
        created_at=now - timedelta(days=5), estimated_seconds=30, started_at=now - timedelta(days=5),
        finished_at=now - timedelta(days=5), error="Demo failure: the recognition engine stopped at frame 1042.",
    ))
