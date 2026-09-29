"""Tests for whiteboard_extration.video.sampling."""

import inspect

import cv2
import numpy as np
import pytest

from whiteboard_extraction.video import sampling
from whiteboard_extraction.video.sampling import sample_frames


class FakeCapture:
    """Stand-in for cv2.VideoCapture which records grab/retrieve calls."""

    def __init__(self, n_frames: int = 100, fps: float = 10.0, opened: bool = True):
        self.n_frames = n_frames
        self.fps = fps
        self.opened = opened
        self.grabs = 0
        self.retrieves = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def grab(self):
        if self.grabs >= self.n_frames:
            return False
        self.grabs += 1
        return True

    def retrieve(self):
        self.retrieves += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    """Path cv2.VideoCapture and return the FakeCapture instance it hands out."""

    def install(**kwargs):
        cap = FakeCapture(**kwargs)
        monkeypatch.setattr(sampling.cv2, "VideoCapture", lambda _path: cap)
        return cap

    return install


def test_returns_a_generator(fake_capture):
    fake_capture()
    assert inspect.isgenerator(sample_frames("ignored.mp4"))


def test_samples_at_fixed_interval(fake_capture):
    fake_capture(n_frames=100, fps=10.0) # 10 seconds of video
    result = list(sample_frames("ignored.mp4", interval_seconds=2.0))
    assert [t for t, _ in result] == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_yields_timestamp_and_bgr_frame(fake_capture):
    fake_capture()
    timestamp, frame = next(sample_frames("ignored.mp4"))
    assert timestamp == 0.0
    assert frame.shape == (48, 64, 3)


def test_interval_shorter_than_one_frame_yields_every_frame(fake_capture):
    fake_capture(n_frames=10, fps=10.0)
    assert len(list(sample_frames("ignored.mp4", interval_seconds=0.001))) == 10


def test_only_decodes_kept_frames(fake_capture):
    cap = fake_capture(n_frames=100, fps=10.0)
    list(sample_frames("ignored.mp4", interval_seconds=2.0))
    assert cap.grabs == 100
    assert cap.retrieves == 5 # expensive decode runs only on sampled frames


def test_is_lazy(fake_capture):
    cap = fake_capture(n_frames=100, fps=10.0)
    next(sample_frames("ignored.mp4", interval_seconds=2.0))
    assert cap.retrieves == 1
    assert cap.grabs < 100


def test_zero_fps_falls_back_to_30(fake_capture):
    fake_capture(n_frames=90, fps=0.0)
    timestamps = [t for t, _ in sample_frames("ignored.mp4", interval_seconds=1.0)]
    assert timestamps == [0.0, 1.0, 2.0]


def test_releases_capture_when_exhausted(fake_capture):
    cap = fake_capture(n_frames=10)
    list(sample_frames("ignored.mp4"))
    assert cap.released


def test_releases_capture_when_closed_early(fake_capture):
    cap = fake_capture(n_frames=100)
    gen = sample_frames("ignored.mp4")
    next(gen)
    gen.close()
    assert cap.released


def test_unopenable_video_raises_value_error(fake_capture):
    fake_capture(opened=False)
    with pytest.raises(ValueError, match="Could not open video"):
        next(sample_frames("missing.mp4"))

def test_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        next(sample_frames(tmp_path / "does_not_exist.mp4"))


def test_real_video_file(tmp_path):
    """Integration test against a real file."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write MJPG")
    for _ in range(30): # 3 seconds 10 fps
        writer.write(np.full((48, 64, 3), 255, dtype=np.uint8))
    writer.release()

    result = list(sample_frames(path, interval_seconds=1.0))
    assert [t for t, _ in result] == [0.0, 1.0, 2.0]
    assert all(frame.shape == (48, 64, 3) for _, frame in result)