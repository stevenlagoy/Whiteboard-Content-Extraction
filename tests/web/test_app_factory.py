import time

import pytest
from web_fakes import FakeEngine, FakeProbe

from whiteboard_extraction.web.app import create_app
from whiteboard_extraction.web.config import AppConfig
from whiteboard_extraction.web.repositories.memory import InMemoryRepository


def test_demo_data_is_seeded_with_stable_ids(make_app, tmp_path):
    app = make_app(AppConfig(data_dir=tmp_path, seed_demo=True, start_runner=False))
    library = app.extensions["wce"].library
    labels = [choice.label for choice in library.folder_choices()]
    assert labels == [
        'My files', 'Archive 2025', 'Archive 2025 › Fall', 'Archive 2025 › Fall › Old recordings (do not delete)',
        'CS_593_application_deep_learning', 'CS_593_application_deep_learning › Midterm review',
        'CS_593_application_deep_learning › Week 1', 'CS_593_application_deep_learning › Week 1 › Lab sessions',
        'CS_593_application_deep_learning › Week 1 › Lab sessions › Lab 1 (extra credit)',
        'CS_593_application_deep_learning › Week 1 › Lab sessions › Lab 1 (extra credit) › Raw recordings',
        'CS_593_application_deep_learning › Week 2', 'Edge cases and unusual names', 'Edge cases and unusual names › Duplicate names',
        'Edge cases and unusual names › Empty folder', 'Edge cases and unusual names › Level 1',
        'Edge cases and unusual names › Level 1 › Level 2', 'Edge cases and unusual names › Level 1 › Level 2 › Level 3',
        'Edge cases and unusual names › Level 1 › Level 2 › Level 3 › Level 4',
        'Edge cases and unusual names › Level 1 › Level 2 › Level 3 › Level 4 › Level 5',
        'Edge cases and unusual names › Level 1 › Level 2 › Level 3 › Level 4 › Level 5 › Level 6',
        'Edge cases and unusual names › R&D <draft> "notes" \'v2\'', "Edge cases and unusual names › Séminaire d'été 日本語 ✓",
        'Edge cases and unusual names › Very long folder name Very long folder name Very long folder name Very long folder name Very long folder name Very long!',
        'Empty top-level folder', 'MA_375_elem_linear_algebra', 'MA_375_elem_linear_algebra › Lectures',
        'MA_375_elem_linear_algebra › Lectures › Chapter 1: Systems & matrices',
        'MA_375_elem_linear_algebra › Lectures › Chapter 2: Vector spaces', 'MA_375_elem_linear_algebra › Lectures › Chapter 3: Eigenvalues',
        'MA_375_elem_linear_algebra › Office hours', 'misc notes (lowercase sorts among capitals)', 'PHYS_201_classical_mechanics',
        'PHYS_201_classical_mechanics › Weekly lectures', 'Week 1'
    ]
    assert library.get_folder("demo-week-1").parent_id == "demo-cs-593"
    assert library.get_video("demo-video-1").favorite and library.get_video("demo-video-6").favorite
    assert library.get_video("demo-video-2").ocr_blocks and not library.get_video("demo-video-3").processed_at
    assert len(library.all_videos()) == 94


def test_demo_data_is_not_seeded_over_existing_content(make_app, tmp_path):
    repo = InMemoryRepository()
    app = make_app(AppConfig(data_dir=tmp_path, seed_demo=True, start_runner=False), repository=repo)
    first = len(app.extensions["wce"].library.all_videos())
    again = make_app(AppConfig(data_dir=tmp_path, seed_demo=True, start_runner=False), repository=repo)
    assert len(again.extensions["wce"].library.all_videos()) == first


def test_demo_videos_can_be_processed_end_to_end(make_app, tmp_path):
    app = make_app(AppConfig(data_dir=tmp_path, seed_demo=True, start_runner=False))
    svc = app.extensions["wce"]
    client = app.test_client()
    client.post("/videos/demo-video-3/process")
    svc.runner.step()
    assert client.get("/videos/demo-video-3").get_data(as_text=True).count('name="text"') == 2  # fake block + blank row
    assert svc.library.get_video("demo-video-3").processed_at is not None


def test_each_part_can_be_replaced(make_app):
    engine, probe = FakeEngine(), FakeProbe(duration=42)
    app = make_app(engine=engine, media_probe=probe)
    svc = app.extensions["wce"]
    assert svc.status.engine_name == "Fake engine" and svc.ingest._probe is probe


def test_unknown_engine_name_fails_at_startup(tmp_path):
    with pytest.raises(ValueError, match="Unknown engine"):
        create_app(AppConfig(data_dir=tmp_path, engine="nope", start_runner=False))


def test_the_runner_starts_on_the_first_request_and_processes_jobs(tmp_path, clock):
    config = AppConfig(data_dir=tmp_path, seed_demo=False, start_runner=True)
    app = create_app(config, engine=FakeEngine(), media_probe=FakeProbe(), clock=clock)
    svc = app.extensions["wce"]
    try:
        assert svc.runner._thread is None
        client = app.test_client()
        video = svc.library.add_video(name="Auto", folder_id=None, length_seconds=60)
        client.post(f"/videos/{video.id}/process")
        assert svc.runner._thread is not None
        deadline = time.monotonic() + 5
        while svc.library.get_video(video.id).processed_at is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert svc.library.get_video(video.id).processed_at is not None
    finally:
        svc.runner.stop()


def test_runner_is_not_started_when_disabled(app, client):
    client.get("/")
    assert app.extensions["wce"].runner._thread is None


def test_the_secret_key_comes_from_configuration(tmp_path):
    app = create_app(AppConfig(data_dir=tmp_path, secret_key="fixed", seed_demo=False, start_runner=False), engine=FakeEngine())
    assert app.config["SECRET_KEY"] == "fixed"


def test_stylesheet_rebuilds_when_a_partial_changes(app, client):
    import os
    from pathlib import Path
    client.get("/")
    scss = Path(app.root_path) / "static" / "scss" / "_typography.scss"
    css = Path(app.root_path) / "static" / "css" / "packed.css"
    original = scss.read_text()
    try:
        scss.write_text(original + "\n.rebuild-probe { color: red; }\n")
        stamp = css.stat().st_mtime + 5
        os.utime(scss, (stamp, stamp))
        client.get("/")
        assert ".rebuild-probe" in css.read_text()
    finally:
        scss.write_text(original)
        later = css.stat().st_mtime + 10
        os.utime(scss, (later, later))
        client.get("/")
