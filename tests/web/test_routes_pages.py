import re

import pytest
from web_html_checks import check

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.web.config import AppConfig


@pytest.fixture
def demo_client(make_app, tmp_path):
    app = make_app(AppConfig(data_dir=tmp_path, seed_demo=True, start_runner=False))
    return app.test_client()


# -------------------------------------------------------------- dashboard

def test_empty_dashboard_explains_itself(client):
    html = client.get("/").get_data(as_text=True)
    for text in ("Nothing is processing right now.", "No videos yet", "No favorites yet", "No folders yet"):
        assert text in html


def test_dashboard_status_tiles_and_engine(client, svc, make_video_in_app):
    make_video_in_app("a")
    html = client.get("/").get_data(as_text=True)
    assert "1 video waiting to start." not in html  # not queued yet
    svc.jobs.schedule(svc.library.all_videos()[0].id)
    html = client.get("/").get_data(as_text=True)
    assert "1 video waiting to start." in html
    for label in ("Processing now", "Waiting to start", "Completed in the last 24 hours", "Failed, need attention"):
        assert label in html
    assert "Recognition engine: Fake engine" in html


def test_dashboard_lists_demo_videos_with_exact_folder_names(demo_client):
    html = demo_client.get("/").get_data(as_text=True)
    assert "CS_593_application_deep_learning" in html and "MA_375_elem_linear_algebra" in html
    assert html.count('class="videoRow"') == 7 + 2  # seven recents, two favorites
    recents = html[html.index('id="recents-heading"'):html.index('id="favorites-heading"')]
    assert recents.index("Video 6") < recents.index("Video 5")  # newest activity first


def test_each_row_carries_every_field_in_one_line(demo_client):
    html = demo_client.get("/").get_data(as_text=True)
    row = html[html.index('class="videoRow"'):]
    row = row[:row.index("</li>")]
    for part in ("rowStar", "rowName", "cellDate", "cellLength", "cellLocation", "cellStatus", "rowActions", "data-move-url"):
        assert part in row


def test_status_badges_use_words_not_just_color(demo_client):
    html = demo_client.get("/").get_data(as_text=True)
    assert "badge--processed" in html and ">Processed<" in html
    assert "badge--failed" in html and ">Failed<" in html and "Demo failure" in html
    assert "badge--not-processed" in html and ">Not processed<" in html


def test_favorite_stars_expose_their_state(demo_client):
    html = demo_client.get("/").get_data(as_text=True)
    assert 'aria-pressed="true"' in html and 'aria-pressed="false"' in html
    assert "Remove from favorites" in html and "Add to favorites" in html


def test_sidebar_tree_is_nested_and_has_both_actions(demo_client):
    html = demo_client.get("/").get_data(as_text=True)
    assert "Week 1" in html and 'aria-label="New folder"' in html and html.count('class="treeToggle"') >= 2
    assert "Upload a video to CS_593_application_deep_learning" in html
    assert 'data-tree-toggle' in html and 'class="treeScroll"' in html


def test_dock_has_the_upload_button_and_a_hidden_empty_task_panel(client):
    html = client.get("/").get_data(as_text=True)
    assert 'class="fab"' in html and 'href="/upload"' in html
    assert re.search(r'<details class="tasksPanel" open hidden', html)


def test_skip_link_and_landmarks(client):
    html = client.get("/").get_data(as_text=True)
    assert 'class="skipLink" href="#main"' in html and '<main id="main"' in html
    assert 'aria-label="My files"' in html and "<aside" in html


# ----------------------------------------------------------------- folders

@pytest.fixture
def make_video_in_app(svc):
    def make(name="Lecture", folder_id=None, **kwargs):
        return svc.library.add_video(name=name, folder_id=folder_id, length_seconds=600, **kwargs)
    return make


def test_folder_page(client, svc, make_video_in_app):
    parent = svc.library.create_folder("Parent_folder")
    child = svc.library.create_folder("Child", parent.id)
    make_video_in_app("In the child", folder_id=child.id)
    page = client.get(f"/folders/{child.id}")
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "Parent_folder" in html and 'aria-current="page"' in html and "In the child" in html
    assert 'videoRows--reorderable' in html and f'data-drop-folder="{child.id}"' in html
    assert f"/upload?folder={child.id}" in html
    assert 'title="Move up"' in html and 'title="Move down"' in html


def test_folder_page_lists_subfolders(client, svc):
    parent = svc.library.create_folder("Parent")
    svc.library.create_folder("Sub A", parent.id)
    html = client.get(f"/folders/{parent.id}").get_data(as_text=True)
    assert "Sub A" in html and 'id="subfolders-heading"' in html


def test_unknown_pages_are_404(client):
    expected = {
        "/folders/nope": "That folder no longer exists.",
        "/videos/nope": "That video no longer exists.",
        "/nothing-here": "That page does not exist.",
    }
    for url, message in expected.items():
        response = client.get(url)
        html = response.get_data(as_text=True)
        assert response.status_code == 404 and message in html and "Back to the dashboard" in html


# ------------------------------------------------------------------ videos

def test_video_page_details_and_forms(client, svc, make_video_in_app):
    folder = svc.library.create_folder("Course")
    video = make_video_in_app("Intro", folder_id=folder.id, source_filename="intro.mp4")
    html = client.get(f"/videos/{video.id}").get_data(as_text=True)
    for label in ("Uploaded", "Edited", "Processed", "Exported", "Original file", "intro.mp4", "10:00"):
        assert label in html
    assert 'id="video-name"' in html and 'id="video-folder"' in html
    assert f'<option value="{folder.id}" selected>Course</option>' in html
    assert "Process this video" in html and "Delete video" in html and "data-confirm" in html


def test_video_page_without_a_stored_file_explains_why(client, make_video_in_app):
    html = client.get(f"/videos/{make_video_in_app().id}").get_data(as_text=True)
    assert "not stored for this entry" in html and "<video" not in html


def test_video_page_with_a_stored_file_has_a_player_and_seek_buttons(client, svc, make_video_in_app):
    video = make_video_in_app(media_file="x.mp4")
    svc.library.update_ocr(video.id, [RecognizedBlock("hello", "prose", 65)])
    html = client.get(f"/videos/{video.id}").get_data(as_text=True)
    assert f'src="/videos/{video.id}/media"' in html and "data-player" in html
    assert 'data-seek="65"' in html and "Play from 1:05" in html


def test_ocr_editor_shows_every_block_and_a_blank_row(client, svc, make_video_in_app):
    video = make_video_in_app()
    svc.library.update_ocr(video.id, [RecognizedBlock("one", "prose", 5), RecognizedBlock("x=1", "equation", 50)])
    html = client.get(f"/videos/{video.id}").get_data(as_text=True)
    assert html.count('name="text"') == 3 and html.count('name="remove"') == 2
    assert 'value="0:05"' in html and 'value="0:50"' in html
    assert re.search(r'<option value="equation" selected>', html)


def test_export_buttons_follow_whether_text_exists(client, svc, make_video_in_app):
    video = make_video_in_app()
    html = client.get(f"/videos/{video.id}").get_data(as_text=True)
    assert "Exports become available" in html and html.count("<button type=\"button\" class=\"button\" disabled>") == 2
    svc.library.update_ocr(video.id, [RecognizedBlock("t", "prose", 1)])
    html = client.get(f"/videos/{video.id}").get_data(as_text=True)
    assert f"/videos/{video.id}/export/docx" in html and f"/videos/{video.id}/export/pdf" in html


def test_failed_video_offers_a_retry_with_the_reason(client, svc, make_video_in_app):
    video = make_video_in_app()
    job = svc.jobs.schedule(video.id)
    svc.jobs.claim_next()
    svc.jobs.fail(job.id, "frame 9 unreadable")
    html = client.get(f"/videos/{video.id}").get_data(as_text=True)
    assert "Retry processing" in html and "frame 9 unreadable" in html


def test_trimmed_video_shows_the_processed_clip(client, make_video_in_app):
    video = make_video_in_app(trim_start=60, trim_end=300)
    assert "1:00 to 5:00 (4:00)" in client.get(f"/videos/{video.id}").get_data(as_text=True)


# ------------------------------------------------------------------ upload

def test_upload_page_fields(client):
    html = client.get("/upload").get_data(as_text=True)
    for hook in ("data-upload-form", 'enctype="multipart/form-data"', 'name="video"', 'name="trim_start"', 'name="trim_end"',
                 'name="name"', 'name="folder_id"', 'name="start_mode"', 'name="start_at"', "up to 4 GB", "data-upload-progress"):
        assert hook in html
    assert 'id="start-now" name="start_mode" value="now" checked' in html


def test_upload_page_preselects_the_folder_from_the_link(client, svc):
    folder = svc.library.create_folder("Target")
    html = client.get(f"/upload?folder={folder.id}").get_data(as_text=True)
    assert f'<option value="{folder.id}" selected>' in html


def test_tasks_panel_fragment(client, svc, make_video_in_app):
    assert 'data-total="0"' in client.get("/tasks/panel").get_data(as_text=True)
    video = make_video_in_app("Queued lecture")
    svc.jobs.schedule(video.id)
    html = client.get("/tasks/panel").get_data(as_text=True)
    assert "Queued lecture" in html and 'role="progressbar"' in html and 'data-active="1"' in html
    assert "/jobs/" in html and "/cancel" in html and "Estimated 0:10" in html


def test_tasks_panel_cancel_returns_to_the_page_being_viewed(client, svc, make_video_in_app):
    svc.jobs.schedule(make_video_in_app().id)
    html = client.get("/tasks/panel", headers={"X-Page-URL": "/folders/abc?x=1"}).get_data(as_text=True)
    assert 'name="next" value="/folders/abc?x=1"' in html
    html = client.get("/tasks/panel", headers={"X-Page-URL": "//evil.example"}).get_data(as_text=True)
    assert "evil.example" not in html


def test_healthz(client):
    assert client.get("/healthz").get_json() == {"status": "ok"}


# --------------------------------------------------------- page structure

def pages_to_check(svc, demo=False):
    folder = svc.library.create_folder("Course")
    video = svc.library.add_video(name='Odd "name" <b>&', folder_id=folder.id, length_seconds=500, media_file="m.mp4")
    svc.library.update_ocr(video.id, [RecognizedBlock("a < b", "equation", 3)])
    svc.jobs.schedule(svc.library.add_video(name="Waiting", folder_id=None, length_seconds=60).id)
    return ["/", f"/folders/{folder.id}", f"/videos/{video.id}", "/upload", "/tasks/panel", "/missing"]


def test_every_page_is_well_formed_and_accessible(client, svc):
    for url in pages_to_check(svc):
        checker = check(client.get(url).get_data(as_text=True))
        assert checker.problems == [], url
        assert checker.images_without_alt == 0, url
        assert checker.unlabeled_controls() == [], url
        if url != "/tasks/panel":
            assert checker.headings.count("h1") == 1, url


def test_user_text_is_escaped(client, svc):
    pages_to_check(svc)
    html = client.get("/").get_data(as_text=True)
    assert "<b>&" not in html and "&lt;b&gt;&amp;" in html


def test_demo_pages_are_well_formed(demo_client):
    for url in ("/", "/videos/demo-video-1", "/videos/demo-video-4", "/videos/demo-video-3"):
        assert check(demo_client.get(url).get_data(as_text=True)).problems == [], url


# ---------------------------------------------------------------- styles

def test_stylesheet_builds_with_valid_row_grid(client):
    client.get("/")
    css = client.get("/static/css/packed.css").get_data(as_text=True)
    assert not re.search(r"\dft\b", css), "a unit typo such as 1ft invalidates the whole declaration"
    assert re.search(r"\.videoRow[^{]*\{[^}]*display: grid", css)
    for declaration in re.findall(r"grid-template-columns:[^;]+;", css):
        assert "ft" not in declaration.replace("minmax", "")
