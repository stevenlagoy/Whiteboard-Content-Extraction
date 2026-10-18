from datetime import timedelta

import pytest

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.web.domain.errors import ConflictError, JobCancelled, NotFoundError
from whiteboard_extraction.web.domain.models import JobState
from whiteboard_extraction.web.services.jobs import JobService


def test_schedule_now_queues_immediately(jobs, make_video, clock):
    video = make_video()
    job = jobs.schedule(video.id)
    assert job.state is JobState.QUEUED and job.scheduled_for is None
    assert job.created_at == clock.now and job.estimated_seconds == 10.0


def test_schedule_for_later_waits(jobs, make_video, clock):
    video = make_video()
    start = clock.now + timedelta(hours=1)
    job = jobs.schedule(video.id, start)
    assert job.state is JobState.SCHEDULED and job.scheduled_for == start


def test_schedule_with_a_start_time_in_the_past_queues_now(jobs, make_video, clock):
    job = jobs.schedule(make_video().id, clock.now - timedelta(minutes=5))
    assert job.state is JobState.QUEUED and job.scheduled_for is None


def test_estimate_uses_the_trimmed_clip(library, repo, clock, make_video):
    seen = []
    service = JobService(library, repo, lambda seconds: seen.append(seconds) or 42.0, clock)
    job = service.schedule(make_video(length=600, trim_start=60, trim_end=360).id)
    assert seen == [300] and job.estimated_seconds == 42.0


def test_one_active_job_per_video(jobs, make_video):
    video = make_video()
    jobs.schedule(video.id)
    with pytest.raises(ConflictError):
        jobs.schedule(video.id)


def test_schedule_unknown_video(jobs):
    with pytest.raises(NotFoundError):
        jobs.schedule("missing")


def test_can_schedule_again_after_the_job_finished(jobs, make_video):
    video = make_video()
    first = jobs.schedule(video.id)
    jobs.claim_next()
    jobs.fail(first.id, "boom")
    assert jobs.schedule(video.id).state is JobState.QUEUED


def test_promote_due_moves_scheduled_jobs_when_their_time_comes(jobs, make_video, clock):
    job = jobs.schedule(make_video().id, clock.now + timedelta(minutes=10))
    assert jobs.promote_due() == 0
    clock.advance(minutes=10)
    assert jobs.promote_due() == 1
    assert jobs.get_job(job.id).state is JobState.QUEUED


def test_claim_next_starts_the_oldest_queued_job(jobs, make_video, clock):
    first = jobs.schedule(make_video("a").id)
    clock.advance(seconds=1)
    jobs.schedule(make_video("b").id)
    claimed = jobs.claim_next()
    assert claimed.id == first.id and claimed.state is JobState.RUNNING and claimed.started_at == clock.now


def test_claim_next_respects_capacity(jobs, make_video):
    jobs.schedule(make_video("a").id)
    jobs.schedule(make_video("b").id)
    assert jobs.claim_next() is not None
    assert jobs.claim_next() is None


def test_capacity_can_be_raised(library, repo, clock, make_video):
    service = JobService(library, repo, lambda s: 1.0, clock, max_concurrent=2)
    service.schedule(make_video("a").id)
    service.schedule(make_video("b").id)
    assert service.claim_next() and service.claim_next() and service.claim_next() is None


def test_claim_next_with_nothing_queued(jobs):
    assert jobs.claim_next() is None


def test_claim_next_cancels_jobs_whose_video_vanished(jobs, repo, make_video):
    video = make_video()
    job = jobs.schedule(video.id)
    repo.delete_video(video.id)
    assert jobs.claim_next() is None
    assert jobs.get_job(job.id).state is JobState.CANCELLED


def running_job(jobs, make_video):
    video = make_video(length=100)
    jobs.schedule(video.id)
    return video, jobs.claim_next()


def test_progress_is_clamped_and_never_goes_backwards(jobs, make_video):
    _, job = running_job(jobs, make_video)
    jobs.report_progress(job.id, 0.5)
    jobs.report_progress(job.id, 0.2)
    assert jobs.get_job(job.id).progress == 0.5
    jobs.report_progress(job.id, 7)
    assert jobs.get_job(job.id).progress == 1.0


def test_progress_after_cancel_request_raises(jobs, make_video):
    _, job = running_job(jobs, make_video)
    jobs.cancel(job.id)
    with pytest.raises(JobCancelled):
        jobs.report_progress(job.id, 0.5)


def test_progress_on_a_job_that_is_not_running_raises(jobs, make_video):
    job = jobs.schedule(make_video().id)
    with pytest.raises(JobCancelled):
        jobs.report_progress(job.id, 0.1)


def test_complete_stores_the_result_on_the_video(jobs, library, make_video, clock):
    video, job = running_job(jobs, make_video)
    clock.advance(seconds=30)
    done = jobs.complete(job.id, [RecognizedBlock("x", "prose", 3.0)])
    assert done.state is JobState.COMPLETED and done.progress == 1.0 and done.finished_at == clock.now
    stored = library.get_video(video.id)
    assert stored.processed_at == clock.now and [b.text for b in stored.ocr_blocks] == ["x"]


def test_complete_after_a_cancel_request_discards_the_result(jobs, library, make_video):
    video, job = running_job(jobs, make_video)
    jobs.cancel(job.id)
    result = jobs.complete(job.id, [RecognizedBlock("x", "prose", 3.0)])
    assert result.state is JobState.CANCELLED
    assert library.get_video(video.id).ocr_blocks == () and library.get_video(video.id).processed_at is None


def test_fail_records_the_message(jobs, make_video):
    _, job = running_job(jobs, make_video)
    failed = jobs.fail(job.id, "boom")
    assert failed.state is JobState.FAILED and failed.error == "boom" and failed.finished_at


def test_cancel_a_waiting_job_ends_it_at_once(jobs, make_video):
    job = jobs.schedule(make_video().id)
    assert jobs.cancel(job.id).state is JobState.CANCELLED


def test_cancel_a_running_job_asks_it_to_stop(jobs, make_video):
    _, job = running_job(jobs, make_video)
    asked = jobs.cancel(job.id)
    assert asked.state is JobState.RUNNING and asked.cancel_requested


def test_cancel_a_finished_job_conflicts(jobs, make_video):
    _, job = running_job(jobs, make_video)
    jobs.fail(job.id, "x")
    with pytest.raises(ConflictError):
        jobs.cancel(job.id)


def test_unknown_job(jobs):
    with pytest.raises(NotFoundError):
        jobs.get_job("missing")
    assert jobs.active_job_for("missing") is None
