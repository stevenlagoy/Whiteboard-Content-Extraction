"""Tests for scripts/extract_board_frames.py on a tiny synthetic video (no network, no model)."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from whiteboard_extraction.recognition.base import RecognizedBlock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_board_frames.py"
spec = importlib.util.spec_from_file_location("extract_board_frames", SCRIPT)
ebf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ebf  # dataclasses in the script look their module up here
spec.loader.exec_module(ebf)


@pytest.fixture
def video(tmp_path):
    """A 3-second, 10 fps, 320x180 video whose frame colour encodes the second."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write MJPG video")
    for index in range(30):
        frame = np.full((180, 320, 3), 255, np.uint8)
        frame[:, :, 0] = min(255, 50 * (index // 10))  # blue channel marks the second
        writer.write(frame)
    writer.release()
    return path


def test_read_manifest_keeps_boxes_only_when_detected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filename,timestamp_seconds,detected,x,y,width,height,error\n"
                        "k0,0.0,True,10,20,100,50,\n"
                        "k1,4.0,False,,,,,Board was not detected.\n")
    specs = ebf.read_manifest(manifest)
    assert [(s.timestamp, s.box) for s in specs] == [(0.0, (10, 20, 100, 50)), (4.0, None)]


def test_every_n_seconds_and_crop_and_downscale():
    assert [s.timestamp for s in ebf.every_n_seconds(25, 10)] == [0.0, 10.0, 20.0]
    frame = np.zeros((100, 200, 3), np.uint8)
    assert ebf.crop(frame, (150, 50, 100, 100)).shape == (50, 50, 3)  # clamped to the frame
    assert ebf.crop(frame, None) is frame
    with pytest.raises(ValueError):
        ebf.crop(frame, (300, 0, 10, 10))
    assert ebf.downscale(frame, 50).shape == (25, 50, 3)
    assert ebf.downscale(frame, 500).shape == (100, 200, 3)


def test_extract_frames_writes_pngs_and_manifest(video, tmp_path):
    out = tmp_path / "frames"
    rows = ebf.extract_frames(video, [ebf.FrameSpec(0.0), ebf.FrameSpec(2.5, (0, 0, 160, 90))], out, max_side=80)
    assert [r["filename"] for r in rows] == ["frame_0000_00-00-000.png", "frame_0001_00-02-500.png"]
    first = cv2.imread(str(out / rows[0]["filename"]))
    second = cv2.imread(str(out / rows[1]["filename"]))
    assert first.shape == (45, 80, 3)   # full 320x180 frame downscaled to 80 wide
    assert second.shape == (45, 80, 3)  # 160x90 crop downscaled to 80 wide
    assert int(second[0, 0, 0]) > int(first[0, 0, 0])  # the later frame carries more blue
    manifest = (out / "frames.csv").read_text().splitlines()
    assert manifest[0].startswith("filename,timestamp_seconds,x,y,width,height")
    assert manifest[2].startswith("frame_0001_00-02-500.png,2.5,0,0,160,90")


def test_recognize_frames_records_blocks_and_errors(video, tmp_path):
    out = tmp_path / "frames"
    rows = ebf.extract_frames(video, [ebf.FrameSpec(0.0), ebf.FrameSpec(1.0)], out, max_side=80)

    class FlakyRecognizer:
        calls = 0

        def recognize(self, timestamp, image):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("model spent its reply on reasoning")
            return [RecognizedBlock(text="x = 1", kind="equation", timestamp=timestamp)]

    results = ebf.recognize_frames(rows, out, FlakyRecognizer())
    assert results[0]["blocks"] == [{"text": "x = 1", "kind": "equation", "timestamp": 0.0, "confidence": None}]
    assert "reasoning" in results[1]["error"]
    assert json.loads((out / "blocks.json").read_text()) == results


def test_main_every_and_limit(video, tmp_path, capsys):
    out = tmp_path / "out"
    assert ebf.main([str(video), "--every", "1", "--limit", "2", "--out", str(out), "--max-side", "64"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["frame_0000_00-00-000.png", "frame_0001_00-01-000.png", "frames.csv"]
    assert ebf.main([str(tmp_path / "missing.mp4")]) == 2


def test_main_stride_takes_every_nth_timestamp(video, tmp_path):
    out = tmp_path / "out"
    assert ebf.main([str(video), "--at", "0", "0.5", "1", "1.5", "2", "--stride", "2", "--out", str(out)]) == 0
    names = sorted(p.name for p in out.iterdir() if p.suffix == ".png")
    assert names == ["frame_0000_00-00-000.png", "frame_0001_00-01-000.png", "frame_0002_00-02-000.png"]
