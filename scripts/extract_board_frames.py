"""Extract board frames from a lecture video, optionally run a recognition backend on them.

Frames are sampled at the timestamps of a video-pipeline manifest (``--manifest``), at explicit
times (``--at``), or every N seconds (``--every``). A manifest row with a detected board box
(``x, y, width, height``) crops the frame to the board; ``--box`` applies one box to every frame;
otherwise the full frame is kept. Frames are downscaled so no side exceeds ``--max-side`` and
written as PNG next to a ``frames.csv`` manifest.

With ``--recognize`` every frame is also passed to ``MultimodalLLMRecognizer`` (provider, model and
timeout from the flags or the WCE_LLM_* variables) and the blocks are written to ``blocks.json``.
A frame that fails is recorded with its error and the run continues.

Examples::

    python scripts/extract_board_frames.py clip.mp4 --every 10 --out data/processed/clip_frames
    python scripts/extract_board_frames.py clip.mp4 --manifest board_detection/manifest.csv --limit 5 \\
        --recognize --provider openai --model qwen3-vl:2b-instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, width, height


@dataclass
class FrameSpec:
    """Where to sample and, optionally, where the board is in that frame."""
    timestamp: float
    box: Box | None = None


def read_manifest(path: Path) -> list[FrameSpec]:
    """Timestamps (and board boxes where ``detected`` is true) from a video-pipeline manifest CSV."""
    specs: list[FrameSpec] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = float(row["timestamp_seconds"])
            box = None
            if row.get("detected", "").strip().lower() == "true" and row.get("width"):
                box = (int(float(row["x"])), int(float(row["y"])), int(float(row["width"])), int(float(row["height"])))
            specs.append(FrameSpec(timestamp, box))
    return specs


def every_n_seconds(duration: float, step: float) -> list[FrameSpec]:
    """Timestamps 0, step, 2*step, ... up to the duration."""
    if step <= 0:
        raise ValueError("--every must be positive")
    count = int(duration // step) + 1
    return [FrameSpec(round(index * step, 3)) for index in range(count) if index * step <= duration]


def grab_frame(capture: cv2.VideoCapture, timestamp: float) -> np.ndarray | None:
    """The BGR frame at ``timestamp`` seconds, or None when the video has no frame there."""
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    return frame if ok else None


def crop(frame: np.ndarray, box: Box | None) -> np.ndarray:
    """Crop ``frame`` to ``box`` (x, y, width, height), clamped to the frame; no box -> the frame."""
    if box is None:
        return frame
    height, width = frame.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {box} lies outside the {width}x{height} frame")
    return frame[y0:y1, x0:x1]


def downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so that neither side exceeds ``max_side``; never enlarges."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def frame_name(index: int, timestamp: float) -> str:
    minutes, seconds = divmod(timestamp, 60)
    return f"frame_{index:04d}_{int(minutes):02d}-{seconds:06.3f}".replace(".", "-") + ".png"


def extract_frames(video: Path, specs: list[FrameSpec], out_dir: Path, max_side: int, box: Box | None = None) -> list[dict]:
    """Write one PNG per spec into ``out_dir`` and return the manifest rows."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    try:
        for index, spec in enumerate(specs):
            frame = grab_frame(capture, spec.timestamp)
            if frame is None:
                print(f"  no frame at {spec.timestamp:.3f}s, skipped", file=sys.stderr)
                continue
            used_box = box or spec.box
            image = downscale(crop(frame, used_box), max_side)
            name = frame_name(index, spec.timestamp)
            cv2.imwrite(str(out_dir / name), image)
            x, y, w, h = used_box if used_box else ("", "", "", "")
            rows.append({"filename": name, "timestamp_seconds": spec.timestamp, "x": x, "y": y, "width": w, "height": h,
                         "image_width": image.shape[1], "image_height": image.shape[0]})
    finally:
        capture.release()
    with (out_dir / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["filename", "timestamp_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def recognize_frames(rows: list[dict], out_dir: Path, recognizer) -> list[dict]:
    """Run ``recognizer`` on every extracted frame; errors are recorded, not raised."""
    results: list[dict] = []
    for row in rows:
        image = cv2.imread(str(out_dir / row["filename"]))
        entry = {"filename": row["filename"], "timestamp_seconds": row["timestamp_seconds"]}
        try:
            blocks = recognizer.recognize(float(row["timestamp_seconds"]), image)
            entry["blocks"] = [asdict(block) for block in blocks]
            summary = "; ".join(f"[{b.kind}] {b.text}" for b in blocks) or "(blank)"
        except RuntimeError as error:
            entry["error"] = str(error)
            summary = f"ERROR: {str(error)[:160]}"
        print(f"  {row['filename']}: {summary}")
        results.append(entry)
    with (out_dir / "blocks.json").open("w") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path, help="lecture video readable by OpenCV")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: data/processed/<video stem>_frames)")
    when = parser.add_mutually_exclusive_group()
    when.add_argument("--manifest", type=Path, help="video-pipeline manifest CSV with timestamp_seconds (and board boxes)")
    when.add_argument("--at", type=float, nargs="+", help="explicit timestamps in seconds")
    when.add_argument("--every", type=float, default=None, help="sample every N seconds (default 10 if nothing else is given)")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="crop every frame to this board box")
    parser.add_argument("--stride", type=int, default=1, help="use every Nth timestamp (default 1: all)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N timestamps, after --stride")
    parser.add_argument("--max-side", type=int, default=640, help="downscale so no side exceeds this (default 640)")
    parser.add_argument("--recognize", action="store_true", help="run MultimodalLLMRecognizer on each frame")
    parser.add_argument("--provider", default=None, help="anthropic or openai (default: WCE_LLM_PROVIDER)")
    parser.add_argument("--model", default="", help="model id (default: WCE_LLM_MODEL or the provider default)")
    parser.add_argument("--timeout", type=float, default=None, help="seconds per recognition request")
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2
    out_dir = args.out or Path("data/processed") / f"{args.video.stem}_frames"

    if args.manifest:
        specs = read_manifest(args.manifest)
    elif args.at:
        specs = [FrameSpec(t) for t in args.at]
    else:
        capture = cv2.VideoCapture(str(args.video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        capture.release()
        specs = every_n_seconds(frames / fps if fps else 0.0, args.every or 10.0)
    if args.stride > 1:
        specs = specs[::args.stride]
    if args.limit:
        specs = specs[:args.limit]

    print(f"{args.video.name}: extracting {len(specs)} frame(s) into {out_dir}")
    rows = extract_frames(args.video, specs, out_dir, args.max_side, tuple(args.box) if args.box else None)
    print(f"wrote {len(rows)} frame(s) and frames.csv")

    if args.recognize:
        from whiteboard_extraction.recognition.llm_multimodal import MultimodalLLMRecognizer

        recognizer = MultimodalLLMRecognizer(args.model, provider=args.provider, timeout=args.timeout,
                                             max_image_side=args.max_side)
        print(f"recognizing with provider={recognizer.provider} model={recognizer.model}")
        results = recognize_frames(rows, out_dir, recognizer)
        failed = sum("error" in entry for entry in results)
        print(f"wrote blocks.json: {len(results) - failed} frame(s) recognized, {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
