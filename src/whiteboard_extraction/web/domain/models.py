"""Entities of the video library and processing queue."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from whiteboard_extraction.recognition.base import RecognizedBlock


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Folder:
    id: str
    name: str
    parent_id: str | None = None


class JobState(str, Enum):
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_JOB_STATES = frozenset({JobState.SCHEDULED, JobState.QUEUED, JobState.RUNNING})


@dataclass(frozen=True)
class Video:
    """A lecture video and everything reconized from it."""
    id: str
    name: str
    folder_id: str | None
    length_seconds: float
    uploaded_at: datetime
    source_filename: str = ""
    media_file: str | None = None
    has_thumbnail: bool = False
    favorite: bool = False
    position: int = 0
    trim_start: float = 0.0
    trim_end: float | None = None
    ocr_blocks: tuple[RecognizedBlock, ...] = ()
    edited_at: datetime | None = None
    processed_at: datetime | None = None
    exported_at: datetime | None = None

    @property
    def clip_end(self) -> float:
        return self.length_seconds if self.trim_end is None else self.trim_end

    @property
    def clip_seconds(self) -> float:
        return self.clip_end - self.trim_start

    @property
    def is_trimmed(self) -> bool:
        return self.trim_start > 0 or self.clip_end < self.length_seconds

    @property
    def last_activity_at(self) -> datetime:
        stamps = (self.uploaded_at, self.edited_at, self.processed_at, self.exported_at)
        return max(stamp for stamp in stamps if stamp is not None)


@dataclass(frozen=True)
class ProcessingJob:
    id: str
    video_id: str
    state: JobState
    created_at: datetime
    estimated_seconds: float
    scheduled_for: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: float = 0.0
    error: str | None = None
    cancel_requested: bool = False

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_JOB_STATES
