"""Fakes shared by the web app tests (imported by conftest.py and the test modules)."""

from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timedelta
from pathlib import Path

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.reconstruction.document import BoardDocument
from whiteboard_extraction.web.domain.errors import MediaError
from whiteboard_extraction.web.services.media import MediaInfo

START = datetime(2026, 10, 17, 12, 0, 0)


class FakeClock:
    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta) -> None:
        self.now += timedelta(**delta)


class InlineExecutor:
    """Runs submitted work immediately, so tests need no threads."""

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:  # pragma: no cover - mirrors a real executor
            future.set_exception(error)
        return future

    def shutdown(self, *args, **kwargs) -> None:
        pass


class DeferredExecutor:
    """Collects work so a test decides when it runs."""

    def __init__(self) -> None:
        self.pending: list = []

    def submit(self, fn, *args, **kwargs) -> Future:
        self.pending.append((fn, args, kwargs))
        return Future()

    def run_all(self) -> None:
        pending, self.pending = self.pending, []
        for fn, args, kwargs in pending:
            fn(*args, **kwargs)


class FakeProbe:
    def __init__(self, duration: float = 600.0, thumbnail: bool = False, error: str | None = None) -> None:
        self.duration = duration
        self.thumbnail = thumbnail
        self.error = error

    def probe(self, path: Path) -> MediaInfo:
        if self.error:
            raise MediaError(self.error)
        return MediaInfo(self.duration)

    def write_thumbnail(self, path: Path, destination: Path, at_seconds: float = 1.0) -> bool:
        if self.thumbnail:
            destination.write_bytes(b"jpeg-bytes")
        return self.thumbnail


class FakeEngine:
    name = "Fake engine"

    def __init__(self, blocks=None, error: Exception | None = None, steps: int = 3) -> None:
        self.blocks = blocks if blocks is not None else [RecognizedBlock("fake text", "prose", 5.0)]
        self.error = error
        self.steps = steps
        self.requests = []

    def estimate_seconds(self, clip_seconds: float) -> float:
        return 10.0

    def process(self, request, reporter) -> BoardDocument:
        self.requests.append(request)
        for step in range(1, self.steps + 1):
            reporter.report(step / self.steps)
        if self.error:
            raise self.error
        return BoardDocument(blocks=list(self.blocks))


class FakeUpload:
    def __init__(self, filename: str | None = "lecture.mp4", content: bytes = b"video-bytes") -> None:
        self.filename = filename
        self.content = content

    def save(self, destination: Path) -> None:
        destination.write_bytes(self.content)
