import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from whiteboard_extraction.web.domain.errors import MediaError
from whiteboard_extraction.web.services.media import OpenCVMediaProbe


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4 files")
    for index in range(50):  # five seconds at 10 frames per second
        writer.write(np.full((48, 64, 3), index * 5 % 255, dtype="uint8"))
    writer.release()
    return path


def test_probe_reads_the_duration(sample_video):
    assert OpenCVMediaProbe().probe(sample_video).duration_seconds == pytest.approx(5.0, abs=0.2)


def test_thumbnail_is_a_jpeg_scaled_to_320_wide(sample_video, tmp_path):
    destination = tmp_path / "thumb.jpg"
    assert OpenCVMediaProbe().write_thumbnail(sample_video, destination, at_seconds=1.0)
    image = cv2.imread(str(destination))
    assert image.shape[1] == 320 and destination.read_bytes()[:2] == b"\xff\xd8"


def test_thumbnail_past_the_end_falls_back_to_the_first_frame(sample_video, tmp_path):
    assert OpenCVMediaProbe().write_thumbnail(sample_video, tmp_path / "t.jpg", at_seconds=999)


def test_files_that_are_not_video_are_rejected(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"this is not a video")
    with pytest.raises(MediaError):
        OpenCVMediaProbe().probe(fake)
    assert OpenCVMediaProbe().write_thumbnail(fake, tmp_path / "t.jpg") is False


def test_missing_files_are_rejected(tmp_path):
    with pytest.raises(MediaError):
        OpenCVMediaProbe().probe(tmp_path / "absent.mp4")
