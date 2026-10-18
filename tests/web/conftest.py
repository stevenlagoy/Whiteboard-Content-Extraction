"""Fixtures for the web app tests."""

from __future__ import annotations

import pytest
from web_fakes import FakeClock, FakeEngine, FakeProbe, InlineExecutor

from whiteboard_extraction.web.app import create_app
from whiteboard_extraction.web.config import AppConfig
from whiteboard_extraction.web.repositories.memory import InMemoryRepository
from whiteboard_extraction.web.services.jobs import JobService
from whiteboard_extraction.web.services.library import LibraryService
from whiteboard_extraction.web.services.status import StatusService


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def repo() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture
def library(repo, clock) -> LibraryService:
    return LibraryService(repo, clock)


@pytest.fixture
def jobs(library, repo, clock) -> JobService:
    return JobService(library, repo, FakeEngine().estimate_seconds, clock, max_concurrent=1)


@pytest.fixture
def status(library, repo, clock) -> StatusService:
    return StatusService(library, repo, "Fake engine", clock)


@pytest.fixture
def make_video(library):
    def make(name="Lecture", folder_id=None, length=600.0, **kwargs):
        return library.add_video(name=name, folder_id=folder_id, length_seconds=length, **kwargs)
    return make


@pytest.fixture
def make_app(tmp_path, clock):
    """Build an app with every external part faked. Override any argument per test."""

    def build(config: AppConfig | None = None, **overrides):
        parts = dict(engine=FakeEngine(), media_probe=FakeProbe(), clock=clock, executor=InlineExecutor())
        parts.update(overrides)
        config = config or AppConfig(data_dir=tmp_path, seed_demo=False, start_runner=False)
        return create_app(config, **parts)

    return build


@pytest.fixture
def app(make_app):
    return make_app()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def svc(app):
    return app.extensions["wce"]
