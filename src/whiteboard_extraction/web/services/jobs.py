"""The processing queue: scheduling, claiming, progress, and cancellation.

This service holds the rules only. Threads live in processing/runner.py.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime

from whiteboard_extraction.recognition.base import RecognizedBlock

from ..domain.errors import ConflictError, JobCancelled, NotFoundError
from ..domain.models import JobState, ProcessingJob, new_id
from ..repositories.base import Repository
from .library import LibraryService

Clock = Callable[[], datetime]


class JobService:
    def __init__(
        self,
        library: LibraryService,
        repo: Repository,
        estimate_seconds: Callable[[float], float],
        clock: Clock,
        max_concurrent: int = 1,
    ) -> None:
        self._library = library
        self._repo = repo
        self._estimate = estimate_seconds
        self._clock = clock
        self._max_concurrent = max(max_concurrent, 1)

    def get_job(self, job_id: str) -> ProcessingJob:
        job = self._repo.get_job(job_id)
        if job is None:
            raise NotFoundError("That task no longer exists.")
        return job

    def active_job_for(self, video_id: str) -> ProcessingJob | None:
        return next((j for j in self._repo.jobs() if j.video_id == video_id and j.is_active), None)

    def schedule(self, video_id: str, start_at: datetime | None = None) -> ProcessingJob:
        """Queue a video for processing now, or at start_at when that is in the future."""
        with self._repo.lock:
            video = self._library.get_video(video_id)
            if self.active_job_for(video_id) is not None:
                raise ConflictError("This video already has a task waiting or running.")
            now = self._clock()
            later = start_at if start_at is not None and start_at > now else None
            job = ProcessingJob(
                id=new_id(),
                video_id=video_id,
                state=JobState.SCHEDULED if later else JobState.QUEUED,
                created_at=now,
                estimated_seconds=self._estimate(video.clip_seconds),
                scheduled_for=later,
            )
            self._repo.save_job(job)
            return job

    def promote_due(self) -> int:
        """Move scheduled jobs whose start time has arrived into the queue."""
        now = self._clock()
        with self._repo.lock:
            due = [
                j for j in self._repo.jobs()
                if j.state is JobState.SCHEDULED and j.scheduled_for is not None and j.scheduled_for <= now
            ]
            for job in due:
                self._repo.save_job(replace(job, state=JobState.QUEUED))
            return len(due)

    def claim_next(self) -> ProcessingJob | None:
        """Start the next queued job if a worker slot is free."""
        now = self._clock()
        with self._repo.lock:
            jobs = self._repo.jobs()
            if sum(1 for j in jobs if j.state is JobState.RUNNING) >= self._max_concurrent:
                return None
            queued = sorted(
                (j for j in jobs if j.state is JobState.QUEUED),
                key=lambda j: (j.scheduled_for or j.created_at, j.created_at),
            )
            for job in queued:
                if self._library.find_video(job.video_id) is None:
                    self._repo.save_job(replace(job, state=JobState.CANCELLED, finished_at=now))
                    continue
                running = replace(job, state=JobState.RUNNING, started_at=now)
                self._repo.save_job(running)
                return running
            return None

    def report_progress(self, job_id: str, fraction: float) -> None:
        with self._repo.lock:
            job = self.get_job(job_id)
            if job.state is not JobState.RUNNING or job.cancel_requested:
                raise JobCancelled()
            progress = max(job.progress, min(max(fraction, 0.0), 1.0))
            self._repo.save_job(replace(job, progress=progress))

    def complete(self, job_id: str, blocks: Sequence[RecognizedBlock]) -> ProcessingJob:
        with self._repo.lock:
            job = self.get_job(job_id)
            if job.cancel_requested:
                return self.mark_cancelled(job_id)
            finished = replace(job, state=JobState.COMPLETED, progress=1.0, finished_at=self._clock())
            self._repo.save_job(finished)
            self._library.record_processing_result(job.video_id, blocks)
            return finished

    def fail(self, job_id: str, message: str) -> ProcessingJob:
        with self._repo.lock:
            failed = replace(self.get_job(job_id), state=JobState.FAILED, error=message, finished_at=self._clock())
            self._repo.save_job(failed)
            return failed

    def mark_cancelled(self, job_id: str) -> ProcessingJob:
        with self._repo.lock:
            cancelled = replace(self.get_job(job_id), state=JobState.CANCELLED, finished_at=self._clock())
            self._repo.save_job(cancelled)
            return cancelled

    def cancel(self, job_id: str) -> ProcessingJob:
        """Cancel a waiting job at once; ask a running job to stop at its next progress report."""
        with self._repo.lock:
            job = self.get_job(job_id)
            if job.state in (JobState.SCHEDULED, JobState.QUEUED):
                return self.mark_cancelled(job_id)
            if job.state is JobState.RUNNING:
                requested = replace(job, cancel_requested=True)
                self._repo.save_job(requested)
                return requested
            raise ConflictError("That task has already finished.")
