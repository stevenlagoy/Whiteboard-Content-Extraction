"""Run sampling and keyframe stages on real footage from data/raw/test_footage/."""

import pytest

from whiteboard_extraction.data.datasets import PROCESSED_DIR, TEST_FOOTAGE_DIR
from whiteboard_extraction.utils.visualization import make_contact_sheet, save_keyframes
from whiteboard_extraction.video.keyframes import change_fraction, extract_keyframes
from whiteboard_extraction.video.sampling import sample_frames

import cv2

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
THRESHOLD = 0.02
INTERVAL = 2.0

VIDEOS = (
    sorted(p for p in TEST_FOOTAGE_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if TEST_FOOTAGE_DIR.exists()
    else []
)


@pytest.mark.real_video
@pytest.mark.skipif(not VIDEOS, reason="No videos in data/raw/test_footage/")
@pytest.mark.parametrize("video", VIDEOS, ids=lambda p: p.name)
def test_keyframes_on_real_footage(video):
    sampled = list(sample_frames(video, INTERVAL))
    assert sampled, "no frames were sampled"

    keyframes = extract_keyframes(sampled, threshold=THRESHOLD)

    # Save output to allow manual visual inspection
    out_dir = PROCESSED_DIR / "keyframe_inspection" / video.stem
    save_keyframes(keyframes, out_dir)
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), make_contact_sheet(keyframes))

    timestamps = [t for t, _ in keyframes]
    assert timestamps[0] == sampled[0][0]
    assert timestamps == sorted(set(timestamps))
    assert len(keyframes) <= len(sampled)
    for (_, prev), (_, curr) in zip(keyframes, keyframes[1:]):
        assert change_fraction(prev, curr) > THRESHOLD