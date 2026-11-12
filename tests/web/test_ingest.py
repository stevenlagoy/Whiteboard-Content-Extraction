from datetime import timedelta

import pytest
from web_fakes import FakeProbe, FakeUpload

from whiteboard_extraction.web.domain.errors import MediaError, NotFoundError, ValidationError
from whiteboard_extraction.web.domain.models import JobState
from whiteboard_extraction.web.services.ingest import IngestRequest, IngestService
from whiteboard_extraction.web.storage import MediaStorage


@pytest.fixture
def storage(tmp_path):
    return MediaStorage(tmp_path)


@pytest.fixture
def make_ingest(library, jobs, storage, clock):
    def build(probe=None):
        return IngestService(library, jobs, probe or FakeProbe(duration=125.0, thumbnail=True), storage, clock)
    return build


def files_on_disk(storage):
    return sorted(p.name for p in [*storage.uploads.iterdir(), *storage.thumbnails.iterdir()])


def test_ingest_saves_probes_adds_and_queues(make_ingest, library, jobs, storage):
    folder = library.create_folder("Course")
    video = make_ingest().ingest(FakeUpload("My Lecture.MP4"), IngestRequest(folder_id=folder.id))
    assert video.name == "My Lecture" and video.folder_id == folder.id
    assert video.length_seconds == 125 and video.source_filename == "My Lecture.MP4" and video.has_thumbnail
    assert video.media_file == f"{video.id}.mp4"
    assert storage.source_path(video.media_file).read_bytes() == b"video-bytes"
    assert storage.thumbnail_path(video.id).exists()
    assert jobs.active_job_for(video.id).state is JobState.QUEUED


def test_ingest_uses_the_given_name_trim_and_schedule(make_ingest, jobs, clock):
    start = clock.now + timedelta(hours=2)
    video = make_ingest().ingest(FakeUpload(), IngestRequest(name="Week 3", trim_start=5, trim_end=100, start_at=start))
    assert (video.name, video.trim_start, video.trim_end) == ("Week 3", 5, 100)
    assert jobs.active_job_for(video.id).state is JobState.SCHEDULED


@pytest.mark.parametrize("filename", [None, "", "notes.txt", "lecture", "clip.exe"])
def test_ingest_rejects_files_that_are_not_videos(make_ingest, library, storage, filename):
    with pytest.raises(ValidationError, match="Choose a video file"):
        make_ingest().ingest(FakeUpload(filename), IngestRequest())
    assert library.all_videos() == [] and files_on_disk(storage) == []


def test_ingest_cleans_up_when_the_file_cannot_be_read(make_ingest, library, storage):
    ingest = make_ingest(FakeProbe(error="That file could not be read as a video."))
    with pytest.raises(MediaError):
        ingest.ingest(FakeUpload(), IngestRequest())
    assert library.all_videos() == [] and files_on_disk(storage) == []


@pytest.mark.parametrize("start, end", [(0, 500), (200, None), (50, 40), (-1, None)])
def test_ingest_cleans_up_when_the_trim_is_invalid(make_ingest, library, repo, storage, start, end):
    with pytest.raises(ValidationError):
        make_ingest().ingest(FakeUpload(), IngestRequest(trim_start=start, trim_end=end))
    assert library.all_videos() == [] and repo.jobs() == [] and files_on_disk(storage) == []


def test_ingest_rejects_a_start_time_in_the_past(make_ingest, storage, clock):
    with pytest.raises(ValidationError, match="future"):
        make_ingest().ingest(FakeUpload(), IngestRequest(start_at=clock.now - timedelta(hours=1)))
    assert files_on_disk(storage) == []


def test_ingest_tolerates_a_start_time_a_few_seconds_old(make_ingest, clock):
    video = make_ingest().ingest(FakeUpload(), IngestRequest(start_at=clock.now - timedelta(seconds=20)))
    assert video.id


def test_unknown_folder_is_refused_before_anything_is_saved(make_ingest, storage):
    with pytest.raises(NotFoundError):
        make_ingest().ingest(FakeUpload(), IngestRequest(folder_id="nope"))
    assert files_on_disk(storage) == []


def test_a_blank_name_falls_back_to_the_file_name(make_ingest):
    assert make_ingest().ingest(FakeUpload("Week 1 notes.webm"), IngestRequest(name=None)).name == "Week 1 notes"


def test_storage_removes_files_it_is_asked_to(storage):
    (storage.uploads / "a.mp4").write_bytes(b"x")
    storage.thumbnail_path("a").write_bytes(b"x")
    storage.remove_files("a", "a.mp4")
    storage.remove_files("a", None)  # missing files are not an error
    assert files_on_disk(storage) == []
