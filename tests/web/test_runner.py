import time

import pytest
from web_fakes import DeferredExecutor, FakeEngine, InlineExecutor

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.web.domain.models import JobState
from whiteboard_extraction.web.processing.runner import JobRunner
from whiteboard_extraction.web.storage import MediaStorage


@pytest.fixture
def storage(tmp_path):
    return MediaStorage(tmp_path)


def runner_for(jobs, library, engine, storage, executor=None):
    return JobRunner(jobs, library, engine, storage, executor or InlineExecutor())


def test_step_runs_a_queued_job_to_completion(jobs, library, storage, make_video):
    engine = FakeEngine(blocks=[RecognizedBlock("answer", "equation", 7.0)])
    video = make_video(length=100, trim_start=10, trim_end=60)
    job = jobs.schedule(video.id)
    runner_for(jobs, library, engine, storage).step()
    assert jobs.get_job(job.id).state is JobState.COMPLETED
    assert [b.text for b in library.get_video(video.id).ocr_blocks] == ["answer"]
    request = engine.requests[0]
    assert (request.start_seconds, request.end_seconds, request.media_path) == (10, 60, None)


def test_the_engine_receives_the_stored_file_path(jobs, library, storage, make_video):
    engine = FakeEngine()
    video = make_video(media_file="abc.mp4")
    jobs.schedule(video.id)
    runner_for(jobs, library, engine, storage).step()
    assert engine.requests[0].media_path == storage.source_path("abc.mp4")


def test_engine_errors_fail_the_job_with_the_message(jobs, library, storage, make_video):
    video = make_video()
    job = jobs.schedule(video.id)
    runner_for(jobs, library, FakeEngine(error=RuntimeError("cannot read frames")), storage).step()
    failed = jobs.get_job(job.id)
    assert failed.state is JobState.FAILED and failed.error == "cannot read frames"
    assert library.get_video(video.id).processed_at is None


def test_an_error_without_a_message_still_explains_itself(jobs, library, storage, make_video):
    job = jobs.schedule(make_video().id)
    runner_for(jobs, library, FakeEngine(error=ValueError()), storage).step()
    assert jobs.get_job(job.id).error == "ValueError"


def test_not_implemented_backends_fail_gracefully(jobs, library, storage, make_video):
    job = jobs.schedule(make_video().id)
    runner_for(jobs, library, FakeEngine(error=NotImplementedError("Wire up the chosen OCR engine here.")), storage).step()
    assert "Wire up" in jobs.get_job(job.id).error


def test_a_cancel_requested_mid_run_ends_as_cancelled(jobs, library, storage, make_video, repo):
    class CancelsItself(FakeEngine):
        def process(self, request, reporter):
            jobs.cancel(next(j.id for j in repo.jobs() if j.state is JobState.RUNNING))
            return super().process(request, reporter)

    job = jobs.schedule(make_video().id)
    runner_for(jobs, library, CancelsItself(), storage).step()
    assert jobs.get_job(job.id).state is JobState.CANCELLED


def test_scheduled_jobs_wait_for_their_time(jobs, library, storage, make_video, clock):
    from datetime import timedelta
    job = jobs.schedule(make_video().id, clock.now + timedelta(minutes=30))
    runner = runner_for(jobs, library, FakeEngine(), storage)
    runner.step()
    assert jobs.get_job(job.id).state is JobState.SCHEDULED
    clock.advance(minutes=30)
    runner.step()
    assert jobs.get_job(job.id).state is JobState.COMPLETED


def test_only_one_job_runs_at_a_time_by_default(jobs, library, storage, make_video):
    first = jobs.schedule(make_video("a").id)
    second = jobs.schedule(make_video("b").id)
    executor = DeferredExecutor()
    runner = runner_for(jobs, library, FakeEngine(), storage, executor)
    runner.step()
    assert len(executor.pending) == 1
    assert jobs.get_job(first.id).state is JobState.RUNNING and jobs.get_job(second.id).state is JobState.QUEUED
    executor.run_all()
    runner.step()
    executor.run_all()
    assert jobs.get_job(second.id).state is JobState.COMPLETED


def test_a_video_deleted_while_queued_is_skipped(jobs, library, storage, make_video, repo):
    video = make_video()
    job = jobs.schedule(video.id)
    repo.delete_video(video.id)
    runner_for(jobs, library, FakeEngine(), storage).step()
    assert jobs.get_job(job.id).state is JobState.CANCELLED


def test_the_background_thread_processes_jobs_and_starts_once(jobs, library, storage, make_video):
    runner = JobRunner(jobs, library, FakeEngine(), storage, poll_seconds=0.02)
    job = jobs.schedule(make_video().id)
    runner.ensure_started()
    thread = runner._thread
    runner.ensure_started()
    assert runner._thread is thread
    try:
        deadline = time.monotonic() + 3
        while jobs.get_job(job.id).state is not JobState.COMPLETED and time.monotonic() < deadline:
            time.sleep(0.02)
        assert jobs.get_job(job.id).state is JobState.COMPLETED
    finally:
        runner.stop()
