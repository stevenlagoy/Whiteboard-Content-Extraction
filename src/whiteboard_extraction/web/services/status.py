"""Read-only views of processing state for the dashboard, rows, and task panel."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..domain.models import JobState, ProcessingJob, Video
from ..repositories.base import Repository
from .library import LibraryService

Clock = Callable[[], datetime]

TASK_LINGER = timedelta(seconds=30)  # finished tasks stay visible in the panel this long
RECENT_WINDOW = timedelta(hours=24)
_MIN_PROGRESS_FOR_PROJECTION = 0.05


@dataclass(frozen=True)
class VideoStatus:
    kind: str  # not-processed, scheduled, queued, processing, processed, failed
    label: str
    detail: str | None = None


@dataclass(frozen=True)
class TaskView:
    job_id: str
    video_id: str
    name: str
    clip_seconds: float
    state: JobState
    label: str
    percent: int
    elapsed_seconds: float | None
    remaining_seconds: float | None
    estimated_seconds: float
    starts_at: datetime | None
    error: str | None

    @property
    def can_cancel(self) -> bool:
        return self.state in (JobState.SCHEDULED, JobState.QUEUED, JobState.RUNNING)

    @property
    def is_active(self) -> bool:
        return self.can_cancel


@dataclass(frozen=True)
class TaskPanel:
    tasks: tuple[TaskView, ...]

    @property
    def active_count(self) -> int:
        return sum(1 for t in self.tasks if t.is_active)

    @property
    def signature(self) -> str:
        """Changes whenever a task appears, disappears, or changes state (not on progress)."""
        return "|".join(f"{t.job_id}:{t.state.value}" for t in self.tasks)


@dataclass(frozen=True)
class DashboardStatus:
    processing: int
    waiting: int
    completed_recently: int
    failed: int
    engine_name: str

    @property
    def headline(self) -> str:
        if self.processing and self.waiting:
            return f"Processing {_videos(self.processing)}, with {self.waiting} waiting."
        if self.processing:
            return f"Processing {_videos(self.processing)}."
        if self.waiting:
            return f"{_videos(self.waiting)} waiting to start."
        return "Nothing is processing right now."


def _videos(count: int) -> str:
    return f"{count} video" if count == 1 else f"{count} videos"


_STATUS_LABELS = {
    "not-processed": "Not processed",
    "scheduled": "Scheduled",
    "queued": "Waiting",
    "processing": "Processing",
    "processed": "Processed",
    "failed": "Failed",
}
_TASK_ORDER = {
    JobState.RUNNING: 0, JobState.QUEUED: 1, JobState.SCHEDULED: 2,
    JobState.COMPLETED: 3, JobState.FAILED: 3,
}


class StatusService:
    def __init__(self, library: LibraryService, repo: Repository, engine_name: str, clock: Clock) -> None:
        self._library = library
        self._repo = repo
        self.engine_name = engine_name
        self._clock = clock

    def status_of(self, video: Video) -> VideoStatus:
        job = self._latest_job(video.id)
        if job is not None and job.state is JobState.FAILED:
            return VideoStatus("failed", _STATUS_LABELS["failed"], job.error)
        if job is not None and job.state is JobState.RUNNING:
            return VideoStatus("processing", _STATUS_LABELS["processing"])
        if job is not None and job.state is JobState.QUEUED:
            return VideoStatus("queued", _STATUS_LABELS["queued"])
        if job is not None and job.state is JobState.SCHEDULED:
            return VideoStatus("scheduled", _STATUS_LABELS["scheduled"], f"Starts {job.scheduled_for:%b} {job.scheduled_for.day}, {job.scheduled_for:%H:%M}")
        kind = "processed" if video.processed_at else "not-processed"
        return VideoStatus(kind, _STATUS_LABELS[kind])

    def dashboard(self) -> DashboardStatus:
        now = self._clock()
        videos = self._library.all_videos()
        kinds = [self.status_of(v).kind for v in videos]
        return DashboardStatus(
            processing=kinds.count("processing"),
            waiting=kinds.count("queued") + kinds.count("scheduled"),
            completed_recently=sum(1 for v in videos if v.processed_at and now - v.processed_at <= RECENT_WINDOW),
            failed=kinds.count("failed"),
            engine_name=self.engine_name,
        )

    def task_panel(self) -> TaskPanel:
        now = self._clock()
        views: list[tuple[ProcessingJob, TaskView]] = []
        for job in self._repo.jobs():
            if not self._belongs_in_panel(job, now):
                continue
            video = self._library.find_video(job.video_id)
            if video is not None:
                views.append((job, self._task_view(job, video, now)))
        views.sort(key=lambda pair: (_TASK_ORDER[pair[0].state], pair[0].created_at))
        return TaskPanel(tuple(view for _, view in views))

    # ------------------------------------------------------------ helpers

    def _latest_job(self, video_id: str) -> ProcessingJob | None:
        latest: ProcessingJob | None = None
        for job in self._repo.jobs():  # stored in creation order, so ties go to the newer job
            if job.video_id == video_id and (latest is None or job.created_at >= latest.created_at):
                latest = job
        return latest

    @staticmethod
    def _belongs_in_panel(job: ProcessingJob, now: datetime) -> bool:
        if job.is_active:
            return True
        finished_recently = job.finished_at is not None and now - job.finished_at <= TASK_LINGER
        return job.state in (JobState.COMPLETED, JobState.FAILED) and finished_recently

    @staticmethod
    def _task_view(job: ProcessingJob, video: Video, now: datetime) -> TaskView:
        elapsed: float | None = None
        if job.started_at is not None:
            end = job.finished_at or now
            elapsed = max((end - job.started_at).total_seconds(), 0.0)

        remaining: float | None = None
        if job.state is JobState.RUNNING:
            if elapsed and job.progress >= _MIN_PROGRESS_FOR_PROJECTION:
                remaining = elapsed * (1 - job.progress) / job.progress
            else:
                remaining = max(job.estimated_seconds - (elapsed or 0.0), 0.0)
        elif job.is_active:
            remaining = job.estimated_seconds

        labels = {
            JobState.RUNNING: "Processing",
            JobState.QUEUED: "Waiting for a free worker",
            JobState.SCHEDULED: "Scheduled",
            JobState.COMPLETED: "Done",
            JobState.FAILED: "Failed",
        }
        return TaskView(
            job_id=job.id,
            video_id=video.id,
            name=video.name,
            clip_seconds=video.clip_seconds,
            state=job.state,
            label=labels[job.state],
            percent=100 if job.state is JobState.COMPLETED else round(job.progress * 100),
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            estimated_seconds=job.estimated_seconds,
            starts_at=job.scheduled_for if job.state is JobState.SCHEDULED else None,
            error=job.error,
        )
