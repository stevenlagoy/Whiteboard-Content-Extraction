import numpy as np
import pytest

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.reconstruction.document import build_document
from whiteboard_extraction.web.config import AppConfig
from whiteboard_extraction.web.domain.errors import JobCancelled
from whiteboard_extraction.web.processing.base import ProcessingRequest
from whiteboard_extraction.web.processing.pipeline import PipelineEngine, PipelineStages
from whiteboard_extraction.web.processing.registry import build_engine
from whiteboard_extraction.web.processing.simulated import SAMPLE_BLOCKS, SimulatedEngine


class Recorder:
    def __init__(self, stop_after=None):
        self.values = []
        self.stop_after = stop_after

    def report(self, fraction):
        self.values.append(fraction)
        if self.stop_after is not None and len(self.values) >= self.stop_after:
            raise JobCancelled()


def request(start=0.0, end=100.0, path="video.mp4"):
    from pathlib import Path
    return ProcessingRequest("v1", Path(path) if path else None, start, end)


# -------------------------------------------------------------- simulated

def test_simulated_estimate_is_clamped():
    engine = SimulatedEngine(seconds_per_media_second=0.1, minimum_seconds=5, maximum_seconds=50)
    assert engine.estimate_seconds(10) == 5
    assert engine.estimate_seconds(200) == 20
    assert engine.estimate_seconds(10_000) == 50


def test_simulated_reports_steady_progress_and_sleeps_each_step():
    slept = []
    engine = SimulatedEngine(seconds_per_media_second=1, minimum_seconds=2, maximum_seconds=2, step_seconds=0.5, sleep=slept.append)
    reporter = Recorder()
    document = engine.process(request(), reporter)
    assert reporter.values == [0.25, 0.5, 0.75, 1.0] and slept == [0.5] * 4
    assert len(document.blocks) == len(SAMPLE_BLOCKS)


def test_simulated_blocks_fall_inside_the_clip_in_order():
    engine = SimulatedEngine(sleep=lambda s: None)
    blocks = engine.process(request(start=100, end=300), Recorder()).blocks
    stamps = [b.timestamp for b in blocks]
    assert stamps == sorted(stamps) and all(100 < s < 300 for s in stamps)
    assert {b.kind for b in blocks} <= {"equation", "prose"}


def test_simulated_stops_when_cancelled():
    engine = SimulatedEngine(sleep=lambda s: None)
    with pytest.raises(JobCancelled):
        engine.process(request(), Recorder(stop_after=2))


# --------------------------------------------------------------- pipeline

class FakeRecognizer:
    def __init__(self):
        self.calls = []

    def recognize(self, timestamp, image):
        self.calls.append(timestamp)
        return [RecognizedBlock(f"t{timestamp:g}", "prose", timestamp)]


def stages(frames, keep_every=1):
    seen = {}

    def sample(path, interval):
        seen["path"], seen["interval"] = path, interval
        return [(t, np.full((2, 2), t, dtype="uint8")) for t in frames]

    return PipelineStages(
        sample=sample,
        detect_board=lambda frame: (0, 0, 2, 2),
        crop=lambda frame, box: frame,
        select_keyframes=lambda boards: boards[::keep_every],
        build=build_document,
    ), seen


def test_pipeline_runs_every_stage_on_the_trimmed_range():
    fake_stages, seen = stages([0, 10, 20, 30, 40])
    recognizer = FakeRecognizer()
    reporter = Recorder()
    document = PipelineEngine(recognizer, fake_stages, sample_interval_seconds=7).process(request(start=10, end=30), reporter)
    assert recognizer.calls == [10, 20, 30]
    assert [b.text for b in document.blocks] == ["t10", "t20", "t30"]
    assert seen["interval"] == 7 and str(seen["path"]) == "video.mp4"
    assert reporter.values == sorted(reporter.values) and reporter.values[-1] == 1.0


def test_pipeline_recognizes_only_the_selected_keyframes():
    fake_stages, _ = stages([0, 1, 2, 3], keep_every=2)
    recognizer = FakeRecognizer()
    PipelineEngine(recognizer, fake_stages).process(request(end=10), Recorder())
    assert recognizer.calls == [0, 2]


def test_pipeline_with_no_frames_returns_an_empty_document():
    fake_stages, _ = stages([])
    document = PipelineEngine(FakeRecognizer(), fake_stages).process(request(), Recorder())
    assert document.blocks == []


def test_pipeline_needs_the_original_file():
    fake_stages, _ = stages([0])
    with pytest.raises(RuntimeError, match="not available"):
        PipelineEngine(FakeRecognizer(), fake_stages).process(request(path=None), Recorder())


def test_pipeline_cancellation_propagates():
    fake_stages, _ = stages([0, 1, 2])
    with pytest.raises(JobCancelled):
        PipelineEngine(FakeRecognizer(), fake_stages).process(request(end=10), Recorder(stop_after=1))


def test_pipeline_estimate_stays_under_real_time():
    assert PipelineEngine(FakeRecognizer(), stages([])[0]).estimate_seconds(100) < 100


# --------------------------------------------------------------- registry

def test_registry_builds_each_engine():
    assert isinstance(build_engine(AppConfig(engine="simulated")), SimulatedEngine)
    assert isinstance(build_engine(AppConfig(engine="pipeline", recognizer="ocr")), PipelineEngine)
    assert isinstance(build_engine(AppConfig(engine="pipeline", recognizer="llm", llm_model="m")), PipelineEngine)


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown engine"):
        build_engine(AppConfig(engine="nope"))
    with pytest.raises(ValueError, match="Unknown recognizer"):
        build_engine(AppConfig(engine="pipeline", recognizer="nope"))


def test_config_reads_the_environment(monkeypatch, tmp_path):
    for key, value in {"UPLOAD_ROOT": str(tmp_path), "WCE_ENGINE": "pipeline", "WCE_SEED_DEMO": "0", "WCE_MAX_UPLOAD_GB": "0.5", "SECRET_KEY": "k"}.items():
        monkeypatch.setenv(key, value)
    config = AppConfig.from_env()
    assert (config.data_dir, config.engine, config.seed_demo, config.secret_key) == (tmp_path, "pipeline", False, "k")
    assert config.max_upload_bytes == 512 * 1024**2
