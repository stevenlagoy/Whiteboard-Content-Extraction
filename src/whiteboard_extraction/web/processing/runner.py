"""Background execution of queued jobs."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Executor, ThreadPoolExecutor

from ..domain.errors import JobCancelled, NotFoundError
from ..domain.models import ProcessingJob
from ..services.jobs import JobService
from ..services.library import LibraryService
from ..storage import MediaStorage
from .base import ProcessingEngine, ProcessingRequest

log = logging.getLogger(__name__)


class _JobProgress:
    """Adapts JobService.report_progress to the engine's ProgressReporter."""

    def __init__(self, jobs: JobService, job_id: str) -> None:
        self._jobs = jobs
        self._job_id = job_id

    def report(self, fraction: float) -> None:
        self._jobs.report_progress(self._job_id, fraction)


class JobRunner:
    def __init__(
        self,
        jobs: JobService,
        library: LibraryService,
        engine: ProcessingEngine,
        storage: MediaStorage,
        executor: Executor | None = None,
        max_workers: int = 1,
        poll_seconds: float = 0.5,
    ) -> None:
        self._jobs = jobs
        self._library = library
        self._engine = engine
        self._storage = storage
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="wce-job")
        self._poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    def ensure_started(self) -> None:
        """Start the scheduler thread once; safe to call on every request."""
        if self._thread is not None:
            return
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="wce-scheduler", daemon=True)
                self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def step(self) -> None:
        """Promote due jobs and hand every claimable job to the executor."""
        self._jobs.promote_due()
        while (job := self._jobs.claim_next()) is not None:
            self._executor.submit(self._execute, job)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                log.exception("Job scheduler step failed")
            self._stop.wait(self._poll_seconds)

    def _execute(self, job: ProcessingJob) -> None:
        try:
            video = self._library.get_video(job.video_id)
            media_path = self._storage.source_path(video.media_file) if video.media_file else None
            request = ProcessingRequest(video.id, media_path, video.trim_start, video.clip_end)
            document = self._engine.process(request, _JobProgress(self._jobs, job.id))
            self._jobs.complete(job.id, document.blocks)
        except JobCancelled:
            self._jobs.mark_cancelled(job.id)
        except NotFoundError:
            self._jobs.mark_cancelled(job.id)
        except Exception as error:
            log.exception("Job %s failed", job.id)
            self._jobs.fail(job.id, str(error) or error.__class__.__name__)
