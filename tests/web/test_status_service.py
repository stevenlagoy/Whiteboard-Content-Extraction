from dataclasses import replace
from datetime import timedelta

from whiteboard_extraction.web.domain.models import JobState
from whiteboard_extraction.web.services.status import TASK_LINGER


def test_status_of_a_video_without_jobs(status, library, make_video):
    video = make_video()
    assert status.status_of(video).kind == "not-processed"
    processed = library.record_processing_result(video.id, [])
    assert status.status_of(processed).kind == "processed"


def test_status_follows_the_latest_job(status, jobs, make_video, clock):
    video = make_video()
    first = jobs.schedule(video.id, clock.now + timedelta(hours=1))
    scheduled = status.status_of(video)
    assert scheduled.kind == "scheduled" and "Starts" in scheduled.detail
    jobs.cancel(first.id)
    assert status.status_of(video).kind == "not-processed"
    jobs.schedule(video.id)
    assert status.status_of(video).kind == "queued"
    jobs.claim_next()
    assert status.status_of(video).kind == "processing"


def test_failed_status_carries_the_error(status, jobs, make_video):
    video = make_video()
    job = jobs.schedule(video.id)
    jobs.claim_next()
    jobs.fail(job.id, "frame 7 unreadable")
    shown = status.status_of(video)
    assert (shown.kind, shown.label, shown.detail) == ("failed", "Failed", "frame 7 unreadable")


def test_a_later_success_replaces_an_earlier_failure(status, jobs, library, make_video, clock):
    video = make_video()
    first = jobs.schedule(video.id)
    jobs.claim_next()
    jobs.fail(first.id, "x")
    clock.advance(minutes=1)
    second = jobs.schedule(video.id)
    jobs.claim_next()
    jobs.complete(second.id, [])
    assert status.status_of(library.get_video(video.id)).kind == "processed"


def test_dashboard_counts(status, jobs, repo, library, make_video, clock):
    running, queued, later, failed, done_now, done_old = (make_video(n) for n in "abcdef")
    jobs.schedule(running.id)
    jobs.claim_next()
    jobs.schedule(queued.id)
    jobs.schedule(later.id, clock.now + timedelta(hours=2))
    job = jobs.schedule(failed.id)
    repo.save_job(replace(job, state=JobState.FAILED, error="x"))
    library.record_processing_result(done_now.id, [])
    clock.advance(hours=30)
    library.record_processing_result(done_old.id, [])
    clock.advance(hours=23)  # done_old is 23h old, done_now is 53h old
    summary = status.dashboard()
    assert (summary.processing, summary.waiting, summary.failed) == (1, 2, 1)
    assert summary.completed_recently == 1 and summary.engine_name == "Fake engine"


def test_headlines(status, jobs, make_video):
    assert status.dashboard().headline == "Nothing is processing right now."
    jobs.schedule(make_video("a").id)
    assert status.dashboard().headline == "1 video waiting to start."
    jobs.claim_next()
    assert status.dashboard().headline == "Processing 1 video."
    jobs.schedule(make_video("b").id)
    assert status.dashboard().headline == "Processing 1 video, with 1 waiting."
    jobs.schedule(make_video("c").id)
    jobs.schedule(make_video("d").id)
    assert status.dashboard().headline == "Processing 1 video, with 3 waiting."


def test_running_task_projects_remaining_time_from_progress(status, jobs, make_video, clock):
    video = make_video(length=100)
    job = jobs.schedule(video.id)
    jobs.claim_next()
    clock.advance(seconds=10)
    jobs.report_progress(job.id, 0.5)
    task = status.task_panel().tasks[0]
    assert (task.percent, task.elapsed_seconds, task.remaining_seconds) == (50, 10, 10)
    assert task.name == video.name and task.clip_seconds == 100 and task.can_cancel


def test_early_progress_falls_back_to_the_estimate(status, jobs, make_video, clock):
    job = jobs.schedule(make_video().id)
    jobs.claim_next()
    clock.advance(seconds=4)
    jobs.report_progress(job.id, 0.01)
    assert status.task_panel().tasks[0].remaining_seconds == 6  # estimate 10 minus 4 elapsed


def test_waiting_tasks_show_the_estimate_and_start_time(status, jobs, make_video, clock):
    start = clock.now + timedelta(hours=1)
    jobs.schedule(make_video("later").id, start)
    jobs.schedule(make_video("now").id)
    tasks = {t.name: t for t in status.task_panel().tasks}
    assert tasks["later"].starts_at == start and tasks["later"].elapsed_seconds is None
    assert tasks["now"].remaining_seconds == 10 and tasks["now"].percent == 0


def test_finished_tasks_linger_then_disappear(status, jobs, make_video, clock):
    job = jobs.schedule(make_video().id)
    jobs.claim_next()
    clock.advance(seconds=5)
    jobs.complete(job.id, [])
    done = status.task_panel().tasks[0]
    assert done.state is JobState.COMPLETED and done.percent == 100 and done.elapsed_seconds == 5 and not done.can_cancel
    clock.advance(seconds=TASK_LINGER.seconds + 1)
    assert status.task_panel().tasks == ()


def test_failed_tasks_linger_with_their_error_and_cancelled_ones_do_not(status, jobs, make_video):
    failed = jobs.schedule(make_video("f").id)
    jobs.claim_next()
    jobs.fail(failed.id, "bad")
    cancelled = jobs.schedule(make_video("c").id)
    jobs.cancel(cancelled.id)
    tasks = status.task_panel().tasks
    assert [(t.name, t.error) for t in tasks] == [("f", "bad")]


def test_tasks_are_ordered_running_first(status, jobs, repo, make_video, clock):
    jobs.schedule(make_video("queued").id)
    clock.advance(seconds=1)
    jobs.schedule(make_video("later").id, clock.now + timedelta(hours=1))
    clock.advance(seconds=1)
    running = jobs.schedule(make_video("running").id)
    repo.save_job(replace(running, state=JobState.RUNNING, started_at=clock.now))
    assert [t.name for t in status.task_panel().tasks] == ["running", "queued", "later"]


def test_signature_changes_with_state_but_not_with_progress(status, jobs, make_video):
    job = jobs.schedule(make_video().id)
    before = status.task_panel()
    jobs.claim_next()
    running = status.task_panel()
    jobs.report_progress(job.id, 0.4)
    later = status.task_panel()
    assert before.signature != running.signature and running.signature == later.signature
    assert running.active_count == 1
    assert status.task_panel().signature.startswith(job.id)
