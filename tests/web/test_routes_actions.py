import pytest
from web_fakes import FakeProbe

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.web.config import AppConfig
from whiteboard_extraction.web.domain.models import JobState


def post(client, url, data=None, **kwargs):
    return client.post(url, data=data or {}, **kwargs)


def followed(client, url, data=None, next_url="/"):
    """POST and follow the redirect, returning the page the user lands on."""
    return client.post(url, data={"next": next_url, **(data or {})}, follow_redirects=True).get_data(as_text=True)


@pytest.fixture
def video(svc):
    return svc.library.add_video(name="Lecture", folder_id=None, length_seconds=600)


# ---------------------------------------------------------------- folders

def test_create_folder_redirects_back_and_confirms(client, svc):
    response = post(client, "/folders", {"name": "Algebra_II", "next": "/upload"})
    assert response.status_code == 303 and response.headers["Location"] == "/upload"
    assert [f.name for f in svc.library.child_folders(None)] == ["Algebra_II"]
    html = client.get("/").get_data(as_text=True)
    assert 'Created folder &#34;Algebra_II&#34;.' in html and "Algebra_II" in html


def test_create_nested_folder(client, svc):
    parent = svc.library.create_folder("Parent")
    post(client, "/folders", {"name": "Kid", "parent_id": parent.id})
    assert [f.name for f in svc.library.child_folders(parent.id)] == ["Kid"]


@pytest.mark.parametrize("name, message", [("", "cannot be empty"), ("x" * 200, "characters or fewer")])
def test_create_folder_validation_is_shown_as_an_error(client, name, message):
    html = followed(client, "/folders", {"name": name})
    assert message in html and 'class="message message--error"' in html and 'role="alert"' in html


def test_duplicate_folder_name_is_reported(client, svc):
    svc.library.create_folder("Same")
    assert "already exists" in followed(client, "/folders", {"name": "same"})


def test_rename_folder(client, svc):
    folder = svc.library.create_folder("Old")
    followed(client, f"/folders/{folder.id}/rename", {"name": "New_name"}, f"/folders/{folder.id}")
    assert svc.library.get_folder(folder.id).name == "New_name"


def test_delete_empty_folder_returns_to_its_parent(client, svc):
    parent = svc.library.create_folder("Parent")
    child = svc.library.create_folder("Child", parent.id)
    response = post(client, f"/folders/{child.id}/delete", {"next": f"/folders/{child.id}"})
    assert response.headers["Location"] == f"/folders/{parent.id}"
    top = svc.library.create_folder("Top")
    assert post(client, f"/folders/{top.id}/delete").headers["Location"] == "/"


def test_delete_folder_with_contents_is_refused(client, svc, video):
    folder = svc.library.create_folder("Full")
    svc.library.move_video(video.id, folder.id)
    html = followed(client, f"/folders/{folder.id}/delete", next_url=f"/folders/{folder.id}")
    assert "Move or delete everything inside this folder first." in html
    assert svc.library.get_folder(folder.id)


# ----------------------------------------------------------------- videos

def test_favorite_and_unfavorite(client, svc, video):
    html = followed(client, f"/videos/{video.id}/favorite", {"favorite": "true"})
    assert svc.library.get_video(video.id).favorite and "to favorites." in html
    html = followed(client, f"/videos/{video.id}/favorite", {"favorite": "false"})
    assert not svc.library.get_video(video.id).favorite and "from favorites." in html


def test_favorite_state_is_set_not_toggled(client, svc, video):
    for _ in range(2):  # a stale second tab sends the same request again
        post(client, f"/videos/{video.id}/favorite", {"favorite": "true"})
    assert svc.library.get_video(video.id).favorite


def test_rename_video(client, svc, video):
    followed(client, f"/videos/{video.id}/rename", {"name": "Renamed"}, f"/videos/{video.id}")
    assert svc.library.get_video(video.id).name == "Renamed"
    assert "cannot be empty" in followed(client, f"/videos/{video.id}/rename", {"name": " "}, f"/videos/{video.id}")


def test_move_video_to_a_folder_and_back_to_the_top(client, svc, video):
    folder = svc.library.create_folder("Course")
    html = followed(client, f"/videos/{video.id}/move", {"folder_id": folder.id})
    assert svc.library.get_video(video.id).folder_id == folder.id and "Moved &#34;Lecture&#34; to Course." in html
    followed(client, f"/videos/{video.id}/move", {"folder_id": ""})
    assert svc.library.get_video(video.id).folder_id is None


def test_move_video_before_another(client, svc, video):
    other = svc.library.add_video(name="Other", folder_id=None, length_seconds=10)
    post(client, f"/videos/{other.id}/move", {"folder_id": "", "before_id": video.id})
    assert [v.name for v in svc.library.videos_in(None)] == ["Other", "Lecture"]


def test_move_to_a_missing_folder_is_reported(client, video):
    assert "That folder no longer exists." in followed(client, f"/videos/{video.id}/move", {"folder_id": "gone"})


def test_shift_video(client, svc, video):
    second = svc.library.add_video(name="Second", folder_id=None, length_seconds=10)
    post(client, f"/videos/{second.id}/shift", {"direction": "up"})
    assert [v.name for v in svc.library.videos_in(None)] == ["Second", "Lecture"]
    assert "up or down" in followed(client, f"/videos/{second.id}/shift", {"direction": "left"})


def test_delete_video_removes_it_and_its_files(client, svc):
    video = svc.library.add_video(name="Doomed", folder_id=None, length_seconds=10, media_file="d.mp4")
    svc.storage.source_path("d.mp4").write_bytes(b"x")
    svc.storage.thumbnail_path(video.id).write_bytes(b"x")
    html = followed(client, f"/videos/{video.id}/delete")
    assert svc.library.find_video(video.id) is None and "Deleted &#34;Doomed&#34;." in html
    assert not svc.storage.source_path("d.mp4").exists() and not svc.storage.thumbnail_path(video.id).exists()


def test_delete_video_while_running_is_refused(client, svc, video):
    svc.jobs.schedule(video.id)
    svc.jobs.claim_next()
    assert "Cancel this video&#39;s running task" in followed(client, f"/videos/{video.id}/delete")
    assert svc.library.find_video(video.id)


def test_actions_on_a_deleted_video_report_instead_of_crashing(client):
    for path in ("favorite", "rename", "move", "shift", "delete", "ocr", "process"):
        response = post(client, f"/videos/gone/{path}", {"next": "/"})
        assert response.status_code == 303
    assert "That video no longer exists." in followed(client, "/videos/gone/delete")


# -------------------------------------------------------------------- OCR

def form(rows, removed=()):
    return {
        "timestamp": [r[0] for r in rows], "kind": [r[1] for r in rows], "text": [r[2] for r in rows],
        "remove": [str(i) for i in removed],
    }


def test_ocr_edits_removals_and_additions(client, svc, video):
    svc.library.update_ocr(video.id, [RecognizedBlock("one", "prose", 10), RecognizedBlock("two", "prose", 20)])
    rows = [("0:10", "prose", "one edited"), ("0:20", "prose", "two"), ("1:30", "equation", "x = 1"), ("0:00", "prose", "  ")]
    post(client, f"/videos/{video.id}/ocr", form(rows, removed=[1]))
    saved = svc.library.get_video(video.id)
    assert [(b.text, b.kind, b.timestamp) for b in saved.ocr_blocks] == [("one edited", "prose", 10), ("x = 1", "equation", 90)]
    assert saved.edited_at is not None


def test_ocr_removed_rows_are_not_validated(client, svc, video):
    svc.library.update_ocr(video.id, [RecognizedBlock("one", "prose", 10)])
    post(client, f"/videos/{video.id}/ocr", form([("garbage", "prose", "one")], removed=[0]))
    assert svc.library.get_video(video.id).ocr_blocks == ()


@pytest.mark.parametrize(
    "rows, message",
    [([("nope", "prose", "x")], "Timestamp 1 must look like"), ([("0:10", "chart", "x")], "equation or prose"), ([("99:00", "prose", "x")], "outside the video")],
)
def test_ocr_errors_are_reported_and_nothing_is_saved(client, svc, video, rows, message):
    svc.library.update_ocr(video.id, [RecognizedBlock("keep", "prose", 1)])
    html = followed(client, f"/videos/{video.id}/ocr", form(rows), f"/videos/{video.id}")
    assert message in html
    assert [b.text for b in svc.library.get_video(video.id).ocr_blocks] == ["keep"]


def test_ocr_mismatched_fields_are_rejected(client, video):
    html = followed(client, f"/videos/{video.id}/ocr", {"timestamp": ["0:01"], "kind": ["prose"], "text": []}, f"/videos/{video.id}")
    assert "could not be read" in html


# -------------------------------------------------------------- processing

def test_process_video_queues_it_once(client, svc, video):
    assert "to the processing queue." in followed(client, f"/videos/{video.id}/process")
    assert svc.jobs.active_job_for(video.id).state is JobState.QUEUED
    assert "already has a task" in followed(client, f"/videos/{video.id}/process")


def test_cancel_a_task(client, svc, video):
    job = svc.jobs.schedule(video.id)
    assert "Cancelled the task." in followed(client, f"/jobs/{job.id}/cancel")
    assert svc.jobs.get_job(job.id).state is JobState.CANCELLED
    assert "already finished" in followed(client, f"/jobs/{job.id}/cancel")
    assert "That task no longer exists." in followed(client, "/jobs/gone/cancel")


# ----------------------------------------------------------------- uploads

def upload(client, filename="lecture.mp4", data=b"bytes", headers=None, **fields):
    from io import BytesIO
    payload = {**fields}
    if filename is not None:
        payload["video"] = (BytesIO(data), filename)
    return client.post("/jobs", data=payload, content_type="multipart/form-data", headers=headers or {})


def test_upload_queues_the_video_and_shows_the_dashboard_message(client, svc):
    folder = svc.library.create_folder("Course")
    response = upload(client, name="Week 1", folder_id=folder.id, trim_start="0:10", trim_end="1:00", start_mode="now")
    assert response.status_code == 303 and response.headers["Location"] == "/"
    video = svc.library.all_videos()[0]
    assert (video.name, video.folder_id, video.trim_start, video.trim_end, video.length_seconds) == ("Week 1", folder.id, 10, 60, 600)
    assert svc.jobs.active_job_for(video.id).state is JobState.QUEUED
    assert "to the processing queue." in client.get("/").get_data(as_text=True)


def test_upload_can_be_scheduled_for_later(client, svc, clock):
    upload(client, start_mode="later", start_at="2026-10-18T09:30")
    job = svc.jobs.active_job_for(svc.library.all_videos()[0].id)
    assert job.state is JobState.SCHEDULED and job.scheduled_for.isoformat() == "2026-10-18T09:30:00"


def test_upload_answers_json_for_the_progress_uploader(client, svc):
    response = upload(client, headers={"Accept": "application/json"})
    assert response.status_code == 200 and response.get_json() == {"redirect": "/"}
    assert len(svc.library.all_videos()) == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"filename": None}, "Choose a video file to upload."),
        ({"filename": "notes.txt"}, "Choose a video file (MP4"),
        ({"trim_start": "abc"}, "Trim start must look like"),
        ({"trim_end": "9:00:00"}, "past the end of the video"),
        ({"trim_start": "5:00", "trim_end": "1:00"}, "after the trim start"),
        ({"start_mode": "later", "start_at": ""}, "Choose the date and time"),
        ({"start_mode": "later", "start_at": "2026-10-01T09:00"}, "future"),
        ({"folder_id": "gone"}, "That folder no longer exists."),
    ],
)
def test_upload_errors_rerender_the_form_and_keep_the_input(client, svc, kwargs, message):
    kwargs = {"name": "Typed name", **kwargs}
    response = upload(client, **kwargs)
    html = response.get_data(as_text=True)
    assert response.status_code == 400 and message in html
    assert 'value="Typed name"' in html and "Upload and process a video" in html
    assert svc.library.all_videos() == []
    assert list(svc.storage.uploads.iterdir()) == []


def test_upload_errors_as_json(client):
    response = upload(client, filename="notes.txt", headers={"Accept": "application/json"})
    assert response.status_code == 400 and "Choose a video file" in response.get_json()["error"]


def test_unreadable_video_is_reported(make_app, tmp_path):
    app = make_app(media_probe=FakeProbe(error="That file could not be read as a video."))
    html = upload(app.test_client()).get_data(as_text=True)
    assert "could not be read as a video" in html
    assert list(app.extensions["wce"].storage.uploads.iterdir()) == []


def test_oversized_uploads_get_a_clear_413(make_app, tmp_path):
    app = make_app(AppConfig(data_dir=tmp_path, seed_demo=False, start_runner=False, max_upload_bytes=200))
    client = app.test_client()
    html_response = upload(client, data=b"x" * 5000)
    assert html_response.status_code == 413 and "Uploads can be up to" in html_response.get_data(as_text=True)
    json_response = upload(client, data=b"x" * 5000, headers={"Accept": "application/json"})
    assert json_response.status_code == 413 and "Uploads can be up to" in json_response.get_json()["error"]


# -------------------------------------------------------------- redirects

@pytest.mark.parametrize("target", ["//evil.example", "https://evil.example", "http://evil.example/x", "\\\\evil", "evil.example", ""])
def test_next_cannot_redirect_off_site(client, target):
    response = post(client, "/folders", {"name": "X", "next": target})
    assert response.headers["Location"] == "/"


def test_next_keeps_local_paths_with_queries(client):
    response = post(client, "/folders", {"name": "Y", "next": "/upload?folder=abc"})
    assert response.headers["Location"] == "/upload?folder=abc"


def test_actions_only_accept_post(client, video):
    for url in ("/folders", f"/videos/{video.id}/delete", "/jobs", f"/videos/{video.id}/favorite"):
        assert client.get(url).status_code == 405
